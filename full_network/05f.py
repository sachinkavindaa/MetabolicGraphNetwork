#!/usr/bin/env python3
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse, sys, math, json
from pathlib import Path
from typing import Tuple, List, Dict, Set, Iterable

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ===== Defaults =====
DEFAULT_CURRENCY = {"C00001","C00002","C00003","C00004","C00005","C00006","C00008","C00009","C00010","C00080"}

# ===== I/O and preprocessing =====
def load_edges(path: str, filter_currency: bool, currency_set: Set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"KO_producer","KO_consumer","metabolite"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    if filter_currency and "metabolite" in df.columns:
        before = len(df)
        df = df[~df["metabolite"].isin(currency_set)].copy()
        print(f"[INFO] Removed {before - len(df)} currency-metabolite edges")
    W = df.groupby(["KO_producer","KO_consumer"]).size().reset_index(name="weight")
    print(f"[INFO] Collapsed to {len(W)} unique edges")
    return W

def build_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges[["KO_producer","KO_consumer","weight"]].itertuples(index=False, name=None))
    print(f"[INFO] Built DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ===== Community detection (resolution knob via Louvain if available) =====
def detect_communities(H: nx.DiGraph, method: str = "greedy", resolution: float = 1.0, seed: int = 7) -> List[Set[str]]:
    """
    method ∈ {"greedy","louvain"}; Louvain needs `python-louvain`. Uses undirected weighted projection.
    Increase `resolution` (Louvain) to split into more/smaller communities.
    """
    U = nx.Graph()
    for u, v, d in H.edges(data=True):
        w = d.get("weight", 1.0)
        if U.has_edge(u, v): U[u][v]["weight"] += w
        else: U.add_edge(u, v, weight=w)

    method = method.lower()
    if method == "louvain":
        try:
            from networkx.algorithms.community import louvain_communities
            comms = list(louvain_communities(U, weight="weight", resolution=resolution, seed=seed))
        except Exception:
            print("[WARN] Louvain unavailable; falling back to greedy modularity.")
            method = "greedy"

    if method == "greedy":
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(U, weight="weight"))

    if not comms:
        comms = [set(H.nodes())]
    print(f"[INFO] Detected {len(comms)} communities via {method} (resolution={resolution})")
    return comms

# ===== Node-level analysis =====
def compute_degrees(H: nx.DiGraph) -> pd.DataFrame:
    din  = dict(H.in_degree())
    dout = dict(H.out_degree())
    deg  = {n: din.get(n,0) + dout.get(n,0) for n in H.nodes()}
    return pd.DataFrame({
        "node": list(H.nodes()),
        "deg_in":  [din[n] for n in H.nodes()],
        "deg_out": [dout[n] for n in H.nodes()],
        "deg_total": [deg[n] for n in H.nodes()],
    }).sort_values("deg_total", ascending=False, ignore_index=True)

def compute_centrality(H: nx.DiGraph) -> pd.DataFrame:
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node","pagerank","betweenness","closeness","eigenvector","hub","authority"])
    # distance = 1/weight for path-based metrics
    distG = H.copy()
    for _, _, d in distG.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["distance"] = 1.0 / w if w > 0 else 1.0

    pr = nx.pagerank(H, weight="weight")
    bt = nx.betweenness_centrality(distG, weight="distance", normalized=True)
    cl = nx.closeness_centrality(distG, distance="distance")

    # Eigenvector on undirected components
    ev: Dict[str, float] = {}
    Ug = H.to_undirected()
    for comp_nodes in nx.connected_components(Ug):
        subU = Ug.subgraph(comp_nodes)
        if subU.number_of_edges() == 0:
            for n in subU.nodes(): ev[n] = 1.0
            continue
        try:
            ev_comp = nx.eigenvector_centrality(subU, weight="weight", max_iter=1000, tol=1e-06)
        except Exception:
            degw = dict(subU.degree(weight="weight")); s = sum(degw.values()) or 1.0
            ev_comp = {n: degw.get(n, 0.0) / s for n in subU.nodes()}
        ev.update(ev_comp)

    # HITS
    try:
        hubs, auths = nx.hits(H, max_iter=500, normalized=True)
    except Exception:
        hubs, auths = {}, {}

    return pd.DataFrame([{
        "node": n,
        "pagerank": pr.get(n, 0.0),
        "betweenness": bt.get(n, 0.0),
        "closeness": cl.get(n, 0.0),
        "eigenvector": ev.get(n, 0.0),
        "hub": hubs.get(n, 0.0),
        "authority": auths.get(n, 0.0),
    } for n in H.nodes()])

def compute_clustering(H: nx.DiGraph, mode: str = "undirected") -> pd.DataFrame:
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node","clustering"])
    if mode == "directed":
        c = nx.clustering(H)  # Fagiolo’s directed clustering
    else:
        Ug = H.to_undirected(); c = nx.clustering(Ug, weight="weight")
    return pd.DataFrame({"node": list(c.keys()), "clustering": list(c.values())})

# ===== All-pairs path lengths =====
def _components_for_paths(Gd: nx.DiGraph, mode: str) -> Iterable[nx.Graph]:
    if mode == "directed":
        for nodes in nx.strongly_connected_components(Gd):
            if len(nodes) >= 2: yield Gd.subgraph(nodes)
    else:
        U = Gd.to_undirected()
        for nodes in nx.connected_components(U):
            if len(nodes) >= 2: yield U.subgraph(nodes)

def compute_all_pair_paths(H: nx.DiGraph, mode: str = "undirected"):
    """
    Yields (src, dst, distance) for all reachable ordered pairs (distance>0).
    Edge distance = 1/weight.
    """
    Gd = H.copy()
    for _, _, d in Gd.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["distance"] = 1.0 / w if w > 0 else 1.0

    for sub in _components_for_paths(Gd, mode):
        for src, dists in nx.all_pairs_dijkstra_path_length(sub, weight="distance"):
            for dst, dist in dists.items():
                if dist > 0:
                    yield (src, dst, float(dist))

def path_stats_and_hist_from_pairs(pairs_iter, bin_width: float = 0.05, round_dp: int = 4):
    """
    Consume (src,dst,dist) iterator and return (stats_dict, histogram_df).
    Histogram bins are [b, b+bin_width).
    """
    total = 0.0; count = 0
    min_d = float("inf"); max_d = 0.0
    hist: Dict[float,int] = {}

    dists_for_median = []  # optional: for exact median; comment out if memory is a concern

    for _, _, d in pairs_iter:
        total += d; count += 1
        if d < min_d: min_d = d
        if d > max_d: max_d = d
        b = math.floor(d / bin_width) * bin_width
        b = round(b, round_dp)
        hist[b] = hist.get(b, 0) + 1
        dists_for_median.append(d)

    if count == 0:
        stats = {"avg": float("nan"), "min": float("nan"), "median": float("nan"),
                 "max": float("nan"), "reachable_pairs": 0}
        return stats, pd.DataFrame(columns=["bin_start","bin_end","count"])

    avg = total / count
    median = float(np.median(dists_for_median)) if dists_for_median else float("nan")

    bins_sorted = sorted(hist.items(), key=lambda x: x[0])
    hist_df = pd.DataFrame(
        [(b, round(b+bin_width, round_dp), c) for b, c in bins_sorted],
        columns=["bin_start","bin_end","count"]
    )
    stats = {"avg": avg, "min": float(min_d), "median": median, "max": float(max_d),
             "reachable_pairs": int(count)}
    return stats, hist_df

# ===== Hubs & drawing =====
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
    if w.size == 0: return []
    rng = np.ptp(w)
    if rng == 0: return [base + span/2.0] * len(w)
    w_norm = (w - w.min()) / rng
    return list(base + span * w_norm)

def compute_layout(G: nx.Graph, layout: str = "auto", k: float = 0.8, seed: int = 7):
    layout = layout.lower()
    def try_spring():
        try:
            return nx.spring_layout(G, k=k, seed=seed), "spring(seed)"
        except Exception: pass
        try:
            rng = np.random.RandomState(seed)
            return nx.spring_layout(G, k=k, seed=rng), "spring(RandomState)"
        except Exception: pass
        raise RuntimeError("spring_layout failed")
    if layout == "spring":   return try_spring()
    if layout == "kk":       return nx.kamada_kawai_layout(G, weight="weight"), "kamada_kawai"
    if layout == "spectral": return nx.spectral_layout(G), "spectral"
    if layout == "random":
        np.random.seed(seed); return nx.random_layout(G), "random"
    if layout == "circular": return nx.circular_layout(G), "circular"
    try: return try_spring()
    except Exception:
        try: return nx.kamada_kawai_layout(G, weight="weight"), "kamada_kawai"
        except Exception:
            try: return nx.spectral_layout(G), "spectral"
            except Exception:
                np.random.seed(seed); return nx.random_layout(G), "random"

def draw_network(H: nx.DiGraph, comms: List[Set[str]], hubs: Set[str], out_png: Path,
                 seed: int, layout_choice: str, show_labels: bool = False) -> None:
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20", len(comms))
    color_map: Dict[str, Tuple[float, float, float, float]] = {}
    for i, c in enumerate(comms):
        for n in c: color_map[n] = cmap(i)
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
    if show_labels and hubs:
        labels = {n: n for n in H.nodes() if n in hubs}
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight="bold")
    plt.title(f"Enzyme–Enzyme Metabolic Network (FULL)\n"
              f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges, {len(comms)} communities",
              fontsize=14)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[INFO] Saved figure → {out_png.resolve()}")

# ===== Export & summary =====
def export_tables(H: nx.DiGraph, comms: List[Set[str]], stats_df: pd.DataFrame,
                  hist_df: pd.DataFrame, out_dir: Path, prefix: str, bin_width: float) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / prefix
    # communities
    rows = []
    for i, c in enumerate(comms, start=1):
        for n in c: rows.append({"node": n, "community": i})
    comm_df = pd.DataFrame(rows).merge(stats_df, on="node", how="left")
    (out_dir / f"{prefix}_communities.csv").write_text(comm_df.to_csv(index=False))
    # edges
    e_rows = [{"src": u, "dst": v, "weight": d.get("weight", 1)} for u, v, d in H.edges(data=True)]
    pd.DataFrame(e_rows).to_csv(out_dir / f"{prefix}_edges.csv", index=False)
    # node stats
    stats_df.to_csv(out_dir / f"{prefix}_node_stats.csv", index=False)
    # path histogram
    hist_df.to_csv(out_dir / f"{prefix}_path_histogram_bw{bin_width}.csv", index=False)

def summarize_run(H: nx.DiGraph, stats_df: pd.DataFrame, out_dir: Path, tag: str,
                  clustering_mode: str, avg_path_mode: str, path_stats: Dict[str, float]) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    density = nx.density(H)
    total_w = float(sum(d.get("weight", 1.0) for _,_,d in H.edges(data=True)))
    avg_w = float(np.mean([d.get("weight",1.0) for _,_,d in H.edges(data=True)]) if H.number_of_edges() else 0.0)
    recip_pairs = {(min(u,v), max(u,v)) for u,v in H.edges() if H.has_edge(v,u)}
    top_pr = stats_df.sort_values("pagerank", ascending=False).head(10)[["node","pagerank"]].to_dict(orient="records")
    top_bt = stats_df.sort_values("betweenness", ascending=False).head(10)[["node","betweenness"]].to_dict(orient="records")
    summary = {
        "tag": tag, "nodes": H.number_of_nodes(), "edges": H.number_of_edges(),
        "density": round(density, 6), "total_edge_weight": total_w, "avg_edge_weight": avg_w,
        "reciprocal_pairs": len(recip_pairs),
        "avg_shortest_path_length": path_stats["avg"],
        "median_shortest_path_length": path_stats["median"],
        "min_shortest_path_length": path_stats["min"],
        "max_shortest_path_length": path_stats["max"],
        "avg_path_mode": avg_path_mode,
        "avg_path_reachable_pairs": path_stats["reachable_pairs"],
        "clustering_mode": clustering_mode,
        "top10_pagerank": top_pr, "top10_betweenness": top_bt
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"[{tag}] NODES={H.number_of_nodes()}  EDGES={H.number_of_edges()}  DENSITY={density:.6f}\n")
        f.write(f"Total edge weight={total_w}  Avg edge weight={avg_w:.4f}  Reciprocal pairs={len(recip_pairs)}\n")
        f.write(f"Shortest paths ({avg_path_mode}) — AVG={path_stats['avg']:.6g}  "
                f"MEDIAN={path_stats['median']:.6g}  MIN={path_stats['min']:.6g}  MAX={path_stats['max']:.6g}  "
                f"[reachable pairs={path_stats['reachable_pairs']}]\n")

def print_manifest(dirpath: Path):
    dirpath = Path(dirpath)
    print(f"\n[INFO] Files in {dirpath.resolve()}:")
    for p in sorted(dirpath.glob("*")):
        try: print(f"   {p.name}\t{p.stat().st_size} bytes")
        except Exception: print(f"   {p.name}")

# ===== CLI =====
def main():
    p = argparse.ArgumentParser(description="FULL enzyme–enzyme network with communities and ALL path lengths.")
    p.add_argument("--edges", default="edges_enzyme_enzyme.csv", help="Edges CSV (KO_producer, KO_consumer, metabolite)")
    p.add_argument("--out", default="enzyme_enzyme_full.png", help="Output PNG filename")
    p.add_argument("--seed", type=int, default=7, help="Layout seed")
    p.add_argument("--layout", choices=["auto","spring","kk","spectral","random","circular"], default="auto")
    p.add_argument("--hub-quantile", type=float, default=95.0)
    p.add_argument("--hub-metric", choices=["degree","pagerank"], default="degree")
    p.add_argument("--filter-currency", dest="filter_currency", action="store_true", default=False)
    p.add_argument("--no-filter-currency", dest="filter_currency", action="store_false")
    p.add_argument("--currency", nargs="*", default=None)
    p.add_argument("--stats-prefix", default="enzyme_enzyme")
    p.add_argument("--out-root", default="All_Data")
    p.add_argument("--show-labels", action="store_true")
    p.add_argument("--clustering-mode", choices=["undirected","directed"], default="undirected")
    # Communities
    p.add_argument("--community-method", choices=["greedy","louvain"], default="greedy")
    p.add_argument("--resolution", type=float, default=1.0)
    # Paths
    p.add_argument("--avg-path-mode", choices=["undirected","directed"], default="undirected",
                   help="Paths across connected components (undirected) vs strongly connected (directed)")
    p.add_argument("--path-bin-width", type=float, default=0.05)
    p.add_argument("--dump-all-paths", action="store_true",
                   help="Also save ALL (src,dst,distance) rows to TSV (can be HUGE)")

    args = p.parse_args()

    currency = set(args.currency) if args.currency is not None else DEFAULT_CURRENCY
    out_root = Path(args.out_root).expanduser().resolve()
    out_dir = (out_root / "full"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Results → {out_dir}")
    print(f"[INFO] networkx {nx.__version__}, numpy {np.__version__}")

    try:
        W = load_edges(args.edges, args.filter_currency, currency)
        G = build_digraph(W)
        H = G  # full graph (no trimming)

        # Communities & node stats
        comms  = detect_communities(H, method=args.community_method, resolution=args.resolution, seed=args.seed)
        deg_df = compute_degrees(H)
        cen_df = compute_centrality(H)
        clu_df = compute_clustering(H, mode=args.clustering_mode)
        stats  = deg_df.merge(cen_df, on="node", how="left").merge(clu_df, on="node", how="left")
        hubs   = choose_hubs_by_metric(stats, metric=args.hub_metric, quantile=args.hub_quantile)

        # --- ALL pairwise shortest paths ---
        pairs_iter = compute_all_pair_paths(H, mode=args.avg_path_mode)
        # For stats + histogram we must consume the iterator; materialize to list once if we also dump pairs
        if args.dump_all_paths:
            pairs = list(pairs_iter)
            path_stats, hist_df = path_stats_and_hist_from_pairs(pairs, bin_width=args.path_bin_width, round_dp=4)
            # dump TSV (src, dst, distance)
            path_tsv = out_dir / f"{args.stats_prefix}_all_pairs_paths_{args.avg_path_mode}.tsv"
            pd.DataFrame(pairs, columns=["src","dst","distance"]).to_csv(path_tsv, sep="\t", index=False)
            print(f"[INFO] Wrote ALL pairwise paths → {path_tsv}")
        else:
            path_stats, hist_df = path_stats_and_hist_from_pairs(pairs_iter, bin_width=args.path_bin_width, round_dp=4)

        # Outputs
        draw_network(H, comms, hubs, out_png=out_dir / Path(args.out).name,
                     seed=args.seed, layout_choice=args.layout, show_labels=args.show_labels)
        export_tables(H, comms, stats, hist_df, out_dir=out_dir, prefix=args.stats_prefix,
                      bin_width=args.path_bin_width)
        summarize_run(H, stats, out_dir=out_dir, tag="full",
                      clustering_mode=args.clustering_mode, avg_path_mode=args.avg_path_mode,
                      path_stats=path_stats)
        print_manifest(out_dir)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}"); raise

if __name__ == "__main__":
    main()


#