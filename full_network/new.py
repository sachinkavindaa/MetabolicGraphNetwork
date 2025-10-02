#!/usr/bin/env python3
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse, sys, math, json
from pathlib import Path
from typing import Tuple, List, Dict, Set, Iterable
from datetime import datetime

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

# ===== Community detection =====
def detect_communities(H: nx.DiGraph, method: str = "greedy", resolution: float = 1.0, seed: int = 7) -> List[Set[str]]:
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
    
    # Calculate only the most important centrality measures for efficiency
    pr = nx.pagerank(H, weight="weight")
    bt = nx.betweenness_centrality(H, weight="weight", normalized=True)
    
    return pd.DataFrame([{
        "node": n,
        "pagerank": pr.get(n, 0.0),
        "betweenness": bt.get(n, 0.0),
    } for n in H.nodes()])

# ===== Filtering important edges and nodes =====
def filter_important_elements(H: nx.DiGraph, top_node_percent: float = 10.0, top_edge_percent: float = 15.0) -> nx.DiGraph:
    """
    Create a filtered graph containing only the most important nodes and edges
    """
    # Calculate node importance (using betweenness centrality)
    betweenness = nx.betweenness_centrality(H, weight="weight", normalized=True)
    
    # Get top nodes by betweenness
    nodes_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_nodes_count = max(5, int(len(nodes_sorted) * top_node_percent / 100))
    important_nodes = {node for node, _ in nodes_sorted[:top_nodes_count]}
    
    # Get edge weights
    edge_weights = [(u, v, d['weight']) for u, v, d in H.edges(data=True)]
    edge_weights_sorted = sorted(edge_weights, key=lambda x: x[2], reverse=True)
    top_edges_count = max(10, int(len(edge_weights_sorted) * top_edge_percent / 100))
    important_edges = {(u, v) for u, v, w in edge_weights_sorted[:top_edges_count]}
    
    # Also include edges between important nodes
    for u, v in H.edges():
        if u in important_nodes and v in important_nodes:
            important_edges.add((u, v))
    
    # Create filtered graph
    filtered_G = nx.DiGraph()
    
    # Add important nodes
    for node in important_nodes:
        filtered_G.add_node(node)
    
    # Add important edges with their weights
    for u, v in important_edges:
        if u in filtered_G.nodes() and v in filtered_G.nodes():
            weight = H[u][v].get('weight', 1)
            filtered_G.add_edge(u, v, weight=weight)
    
    print(f"[INFO] Filtered graph: {filtered_G.number_of_nodes()} nodes, {filtered_G.number_of_edges()} edges")
    return filtered_G

# ===== Drawing =====
def compute_layout(G: nx.Graph, layout: str = "spring", k: float = 1.0, seed: int = 7):
    if layout == "spring":
        return nx.spring_layout(G, k=k, seed=seed, iterations=50), "spring"
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, weight="weight"), "kamada_kawai"
    elif layout == "spectral":
        return nx.spectral_layout(G), "spectral"
    elif layout == "circular":
        return nx.circular_layout(G), "circular"
    else:
        np.random.seed(seed)
        return nx.random_layout(G), "random"

def draw_network(H: nx.DiGraph, comms: List[Set[str]], hubs: Set[str], out_png: Path,
                 seed: int, layout_choice: str, show_labels: bool = True) -> None:
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    
    # Color nodes by community
    cmap = plt.get_cmap("tab20", len(comms))
    color_map = {}
    for i, c in enumerate(comms):
        for n in c: 
            if n in H.nodes():
                color_map[n] = cmap(i)
    
    # Size nodes by degree
    deg = dict(H.degree())
    sizes = [300 + 500 * math.sqrt(deg[n]) for n in H.nodes()]
    
    node_colors = [color_map.get(n, (0.7, 0.7, 0.9, 1.0)) for n in H.nodes()]
    
    # Width of edges by weight
    widths = []
    for u, v in H.edges():
        w = H[u][v].get("weight", 1.0)
        widths.append(0.5 + 3.0 * math.log(1 + w))
    
    # Compute layout
    pos, used = compute_layout(H, layout=layout_choice, k=1.5, seed=seed)
    print(f"[INFO] Layout used: {used}")
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Draw edges
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="->", arrowsize=20,
                           width=widths, alpha=0.7, edge_color="gray")
    
    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=node_colors,
                           linewidths=1.0, edgecolors="black", alpha=0.9)
    
    # Draw labels for all nodes
    if show_labels:
        labels = {n: n for n in H.nodes()}
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight="bold")
    
    plt.title(f"Enzyme–Enzyme Metabolic Network (Important Elements)\n"
              f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges, {len(comms)} communities",
              fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure → {out_png.resolve()}")

# ===== Export =====
def export_tables(H: nx.DiGraph, comms: List[Set[str]], stats_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / prefix
    
    # communities
    rows = []
    for i, c in enumerate(comms, start=1):
        for n in c: 
            if n in H.nodes():
                rows.append({"node": n, "community": i})
    comm_df = pd.DataFrame(rows)
    if not comm_df.empty:
        comm_df = comm_df.merge(stats_df, on="node", how="left")
        comm_df.to_csv(out_dir / f"{prefix}_communities.csv", index=False)
    
    # edges
    e_rows = [{"src": u, "dst": v, "weight": d.get("weight", 1)} for u, v, d in H.edges(data=True)]
    pd.DataFrame(e_rows).to_csv(out_dir / f"{prefix}_edges.csv", index=False)
    
    # node stats
    stats_df.to_csv(out_dir / f"{prefix}_node_stats.csv", index=False)

# ===== CLI =====
def main():
    p = argparse.ArgumentParser(description="Create a focused visualization of the enzyme-enzyme network.")
    p.add_argument("--edges", default="edges_enzyme_enzyme.csv", help="Edges CSV (KO_producer, KO_consumer, metabolite)")
    p.add_argument("--out", default="enzyme_enzyme_focused.png", help="Output PNG filename")
    p.add_argument("--seed", type=int, default=7, help="Layout seed")
    p.add_argument("--layout", choices=["spring","kamada_kawai","spectral","circular"], default="spring")
    p.add_argument("--filter-currency", dest="filter_currency", action="store_true", default=False)
    p.add_argument("--no-filter-currency", dest="filter_currency", action="store_false")
    p.add_argument("--currency", nargs="*", default=None)
    p.add_argument("--stats-prefix", default="enzyme_enzyme_focused")
    p.add_argument("--node-percent", type=float, default=10.0, help="Percentage of top nodes to keep")
    p.add_argument("--edge-percent", type=float, default=15.0, help="Percentage of top edges to keep")
    
    args = p.parse_args()

    currency = set(args.currency) if args.currency is not None else DEFAULT_CURRENCY
    
    # Create a timestamped output directory in the current location
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"enzyme_network_analysis_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {out_dir.resolve()}")
    print(f"[INFO] Results → {out_dir}")

    try:
        # Load and build the full graph
        W = load_edges(args.edges, args.filter_currency, currency)
        G = build_digraph(W)
        
        # Filter to keep only important nodes and edges
        H = filter_important_elements(G, top_node_percent=args.node_percent, top_edge_percent=args.edge_percent)
        
        # Detect communities in the filtered graph
        comms = detect_communities(H, method="louvain", resolution=1.0, seed=args.seed)
        
        # Compute node statistics
        deg_df = compute_degrees(H)
        cen_df = compute_centrality(H)
        stats = deg_df.merge(cen_df, on="node", how="left")
        
        # Identify hubs (top 10% by betweenness)
        if not stats.empty:
            betweenness_vals = stats["betweenness"].values
            threshold = np.percentile(betweenness_vals, 90) if len(betweenness_vals) > 0 else 0
            hubs = set(stats.loc[stats["betweenness"] >= threshold, "node"])
            print(f"[INFO] Identified {len(hubs)} hub nodes")
        else:
            hubs = set()
        
        # Draw the network
        draw_network(H, comms, hubs, out_png=out_dir / Path(args.out).name,
                     seed=args.seed, layout_choice=args.layout, show_labels=True)
        
        # Export data
        export_tables(H, comms, stats, out_dir=out_dir, prefix=args.stats_prefix)
        
        # Create a summary file
        summary = {
            "timestamp": timestamp,
            "input_file": args.edges,
            "filtered_nodes": H.number_of_nodes(),
            "filtered_edges": H.number_of_edges(),
            "communities": len(comms),
            "hubs": len(hubs),
            "node_percent": args.node_percent,
            "edge_percent": args.edge_percent
        }
        
        with open(out_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO] Analysis complete. Files saved to {out_dir.resolve()}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}"); raise

if __name__ == "__main__":
    main()