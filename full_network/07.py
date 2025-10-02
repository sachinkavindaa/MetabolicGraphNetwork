#!/usr/bin/env python3
import argparse
import sys
import math
from typing import Tuple, List, Dict, Set

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ===== Defaults =====
DEFAULT_CURRENCY = {
    "C00001","C00002","C00003","C00004","C00005","C00006",
    "C00008","C00009","C00010","C00080"
}

# ===== I/O and preprocessing =====
def load_edges(path: str, filter_currency: bool, currency_set: Set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"KO_producer", "KO_consumer", "metabolite"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    if filter_currency and "metabolite" in df.columns:
        before = len(df)
        df = df[~df["metabolite"].isin(currency_set)].copy()
        print(f"[INFO] Removed {before - len(df)} currency-metabolite edges")
    W = (df.groupby(["KO_producer","KO_consumer"]).size().reset_index(name="weight"))
    print(f"[INFO] Collapsed to {len(W)} unique edges")
    return W

def build_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges[["KO_producer","KO_consumer","weight"]].itertuples(index=False, name=None))
    print(f"[INFO] Built DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.weakly_connected_components(G))
    H = G.subgraph(max(comps, key=len)).copy()
    print(f"[INFO] Largest component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges (of {len(comps)} components)")
    return H

def trim_top_degree(H: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
    if H.number_of_nodes() <= max_nodes: return H
    deg = dict(H.degree())
    keep = set(sorted(deg, key=deg.get, reverse=True)[:max_nodes])
    H2 = H.subgraph(keep).copy()
    print(f"[INFO] Limited to top {max_nodes} nodes by total degree ({H.number_of_nodes()} → {H2.number_of_nodes()})")
    return H2

# ===== Analysis =====
def detect_communities_weighted(H: nx.DiGraph) -> List[Set[str]]:
    U = nx.Graph()
    for u, v, d in H.edges(data=True):
        w = d.get("weight", 1)
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(U, weight="weight"))
    if not comms: comms = [set(H.nodes())]
    print(f"[INFO] Detected {len(comms)} communities")
    return comms

def compute_degrees(H: nx.DiGraph) -> pd.DataFrame:
    din  = dict(H.in_degree())
    dout = dict(H.out_degree())
    deg  = {n: din.get(n,0) + dout.get(n,0) for n in H.nodes()}
    df = pd.DataFrame({
        "node": list(H.nodes()),
        "deg_in":  [din[n]  for n in H.nodes()],
        "deg_out": [dout[n] for n in H.nodes()],
        "deg_total": [deg[n] for n in H.nodes()],
    }).sort_values("deg_total", ascending=False, ignore_index=True)
    return df

def compute_centrality(H: nx.DiGraph) -> pd.DataFrame:
    """
    Compute centralities on a directed, weighted graph.
    For shortest-path metrics, use distance = 1/weight (stronger weight = closer).
    Eigenvector centrality is computed per weakly-connected component to avoid
    ambiguity errors on disconnected graphs.
    """
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node","pagerank","betweenness","closeness",
                                     "eigenvector","hub","authority"])

    # distance graph for path-based metrics
    distG = H.copy()
    for u, v, d in distG.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["distance"] = 1.0 / w if w > 0 else 1.0

    # PageRank (weighted) with SciPy→NumPy fallback
    try:
        pr = nx.pagerank(H, weight="weight")
    except ModuleNotFoundError:
        pr = nx.pagerank_numpy(H, weight="weight")

    # Betweenness (directed, weighted distances)
    bt = nx.betweenness_centrality(distG, weight="distance", normalized=True)

    # Closeness (directed, distance)
    cl = nx.closeness_centrality(distG, distance="distance")

    # Eigenvector centrality per weakly connected component (undirected view)
    ev: Dict[str, float] = {}
    Ug = H.to_undirected()
    for comp_nodes in nx.connected_components(Ug):
        subU = Ug.subgraph(comp_nodes)
        # If a component is a single node with no edges, eigenvector is 1.0 by definition
        if subU.number_of_edges() == 0:
            for n in subU.nodes():
                ev[n] = 1.0
            continue
        # Compute eigenvector centrality on this component; weight-aware
        ev_comp = nx.eigenvector_centrality_numpy(subU, weight="weight")
        # Merge back
        ev.update(ev_comp)

    # HITS (hubs/authorities). Works on disconnected graphs; scores are per SCC.
    try:
        hubs, auths = nx.hits(H, max_iter=1000, normalized=True)
    except Exception:
        hubs, auths = {}, {}

    rows = []
    for n in H.nodes():
        rows.append({
            "node": n,
            "pagerank":     pr.get(n, 0.0),
            "betweenness":  bt.get(n, 0.0),
            "closeness":    cl.get(n, 0.0),
            "eigenvector":  ev.get(n, 0.0),
            "hub":          hubs.get(n, 0.0),
            "authority":    auths.get(n, 0.0),
        })
    return pd.DataFrame(rows)


def choose_hubs_by_metric(stats_df: pd.DataFrame, metric: str, quantile: float) -> Set[str]:
    if stats_df.empty: return set()
    col = "deg_total" if metric == "degree" else metric
    vals = stats_df[col].values
    thr = (np.percentile(vals, quantile) if len(vals) > 20 else float(np.max(vals)) * 0.8)
    hubs = set(stats_df.loc[stats_df[col] >= thr, "node"])
    print(f"[INFO] Hub threshold ({col} ≥ {thr:.4g}) → {len(hubs)} hubs")
    return hubs

def normalize_widths(H: nx.DiGraph, base: float = 0.6, span: float = 2.4) -> List[float]:
    w = np.array([H[u][v].get("weight", 1) for u, v in H.edges()], dtype=float)
    if w.size == 0:
        return []
    rng = np.ptp(w)  # NumPy 2.x compatible
    if rng == 0:
        return [base + span/2.0] * len(w)
    w_norm = (w - w.min()) / rng
    return list(base + span * w_norm)

# ===== Layout (robust, version-proof) =====
def compute_layout(G: nx.Graph, layout: str = "auto", k: float = 0.8, seed: int = 7):
    """
    Returns node positions with robust fallbacks.
    Order tried (auto): spring(seed) → spring(RandomState) → kamada_kawai → spectral → random → circular.
    """
    layout = layout.lower()
    def try_spring():
        try:
            return nx.spring_layout(G, k=k, seed=seed), "spring(seed)"
        except Exception:
            pass
        try:
            rng = np.random.RandomState(seed)
            return nx.spring_layout(G, k=k, seed=rng), "spring(RandomState)"
        except Exception:
            pass
        raise RuntimeError("spring_layout failed on this NetworkX build")

    if layout == "spring":
        return try_spring()
    elif layout == "kk":
        return nx.kamada_kawai_layout(G, weight="weight"), "kamada_kawai"
    elif layout == "spectral":
        return nx.spectral_layout(G), "spectral"
    elif layout == "random":
        np.random.seed(seed)
        return nx.random_layout(G), "random"
    elif layout == "circular":
        return nx.circular_layout(G), "circular"
    else:
        try:
            return try_spring()
        except Exception:
            try:
                return nx.kamada_kawai_layout(G, weight="weight"), "kamada_kawai"
            except Exception:
                try:
                    return nx.spectral_layout(G), "spectral"
                except Exception:
                    try:
                        np.random.seed(seed)
                        return nx.random_layout(G), "random"
                    except Exception:
                        return nx.circular_layout(G), "circular"

# ===== Drawing & export =====
def draw_network(H: nx.DiGraph,
                 comms: List[Set[str]],
                 hubs: Set[str],
                 out_png: str,
                 seed: int,
                 layout_choice: str) -> None:
    cmap = plt.get_cmap("tab20", len(comms))  # modern API
    color_map: Dict[str, Tuple[float, float, float, float]] = {}
    for i, c in enumerate(comms):
        for n in c:
            color_map[n] = cmap(i)

    deg = dict(H.degree())
    sizes = [50 + 20 * math.sqrt(deg[n]) for n in H.nodes()]
    node_colors = [color_map.get(n, (0.7, 0.7, 0.9, 1.0)) for n in H.nodes()]
    widths = normalize_widths(H, base=0.6, span=2.4)

    pos, used = compute_layout(H, layout=layout_choice, k=0.8, seed=seed)
    print(f"[INFO] Layout used: {used}")

    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=node_colors,
                           linewidths=0.5, edgecolors="black", alpha=0.85)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="->", arrowsize=14,
                           width=widths, alpha=0.6, edge_color="gray")
    labels = {n: n for n in H.nodes() if n in hubs}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight="bold")

    plt.title(f"Enzyme–Enzyme Metabolic Network\n"
              f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges, {len(comms)} communities",
              fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure → {out_png}")

def export_tables(H: nx.DiGraph,
                  comms: List[Set[str]],
                  stats_df: pd.DataFrame,
                  prefix: str) -> None:
    rows = []
    for i, c in enumerate(comms, start=1):
        for n in c:
            rows.append({"node": n, "community": i})
    comm_df = pd.DataFrame(rows).merge(stats_df, on="node", how="left")
    comm_df.to_csv(f"{prefix}_communities.csv", index=False)

    e_rows = [{"src": u, "dst": v, "weight": d.get("weight", 1)} for u, v, d in H.edges(data=True)]
    pd.DataFrame(e_rows).to_csv(f"{prefix}_edges.csv", index=False)

    stats_df.to_csv(f"{prefix}_degrees.csv", index=False)  # now includes centralities too
    print(f"[INFO] Saved stats: {prefix}_communities.csv, {prefix}_edges.csv, {prefix}_degrees.csv")

# ===== CLI =====
def main():
    p = argparse.ArgumentParser(description="Build and plot enzyme–enzyme network (with centrality metrics).")
    p.add_argument("--edges", default="edges_enzyme_enzyme.csv", help="Edges CSV (KO_producer, KO_consumer, metabolite)")
    p.add_argument("--out", default="enzyme_enzyme_colored.png", help="Output PNG path")
    p.add_argument("--max-nodes", type=int, default=300, help="Max nodes to display")
    p.add_argument("--seed", type=int, default=7, help="Layout seed")
    p.add_argument("--layout", choices=["auto","spring","kk","spectral","random","circular"], default="auto",
                   help="Layout choice; 'auto' tries robust fallbacks")
    p.add_argument("--hub-quantile", type=float, default=95.0, help="Percentile for hub labels (0–100)")
    p.add_argument("--hub-metric", choices=["degree","pagerank"], default="degree",
                   help="Metric used to label hubs on the plot")
    p.add_argument("--filter-currency", dest="filter_currency", action="store_true", default=True)
    p.add_argument("--no-filter-currency", dest="filter_currency", action="store_false")
    p.add_argument("--currency", nargs="*", default=None, help="Override currency metabolite list")
    p.add_argument("--stats-prefix", default="enzyme_enzyme", help="Prefix for CSV stats output")
    args = p.parse_args()

    currency = set(args.currency) if args.currency is not None else DEFAULT_CURRENCY

    print("[INFO] === Enzyme–Enzyme Network Analysis (with centrality) ===")
    print(f"[INFO] networkx {nx.__version__}, numpy {np.__version__}")
    try:
        W = load_edges(args.edges, args.filter_currency, currency)
        G = build_digraph(W)
        H = largest_weak_component(G)
        H = trim_top_degree(H, args.max_nodes)
        comms = detect_communities_weighted(H)

        deg_df  = compute_degrees(H)
        cent_df = compute_centrality(H)
        stats_df = deg_df.merge(cent_df, on="node", how="left")

        hubs = choose_hubs_by_metric(stats_df, metric=args.hub_metric, quantile=args.hub_quantile)

        draw_network(H, comms, hubs, out_png=args.out, seed=args.seed, layout_choice=args.layout)
        export_tables(H, comms, stats_df, prefix=args.stats_prefix)

        density = nx.density(H)
        print("[INFO] === NETWORK STATS ===")
        print(f"Nodes: {H.number_of_nodes()}  Edges: {H.number_of_edges()}  Density: {density:.4f}")

        print("\nTop 10 by PageRank:")
        print(stats_df.sort_values("pagerank", ascending=False).head(10).to_string(index=False))

        print("\nTop 10 by Betweenness:")
        print(stats_df.sort_values("betweenness", ascending=False).head(10).to_string(index=False))

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()
