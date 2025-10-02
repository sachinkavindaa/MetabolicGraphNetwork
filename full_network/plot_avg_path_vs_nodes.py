#!/usr/bin/env python3
"""
Incremental node-addition experiment: average shortest path vs nodes added.

Produces two series:
- Random: nodes added in random order (single-run or averaged via repeats)
- Virtual: nodes added in degree-descending order (highest degree first)

Saves PNG and CSV outputs to the current directory.
"""
from pathlib import Path
import argparse
import random
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graph(edges_csv: Path, filter_currency: bool = True, currency_set=None) -> nx.Graph:
    df = pd.read_csv(edges_csv)
    if filter_currency and currency_set is not None and 'metabolite' in df.columns:
        df = df[~df['metabolite'].isin(currency_set)].copy()
    W = df.groupby(['KO_producer', 'KO_consumer']).size().reset_index(name='weight')
    G = nx.Graph()
    G.add_weighted_edges_from(W[['KO_producer','KO_consumer','weight']].itertuples(index=False, name=None))
    return G


def avg_shortest_path_length_induced(G: nx.Graph) -> float:
    """Compute average shortest path length over reachable unordered pairs in G.
    Memory-efficient: iterate single-source BFS from each node in the largest component and
    sum distances only to nodes with a higher index to count unordered pairs exactly once.
    Returns NaN if fewer than 2 nodes or no reachable pairs.
    """
    if G.number_of_nodes() < 2:
        return float('nan')
    # consider largest connected component
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    nodes = list(H.nodes())
    n = len(nodes)
    if n < 2:
        return float('nan')

    total = 0.0
    count = 0
    # map node -> index for ordering
    idx = {node: i for i, node in enumerate(nodes)}
    for i, src in enumerate(nodes):
        # single-source shortest path lengths (unweighted)
        dists = nx.single_source_shortest_path_length(H, src)
        for dst, d in dists.items():
            j = idx.get(dst)
            if j is None: continue
            if j <= i:  # only count unordered pairs once
                continue
            total += d
            count += 1

    if count == 0:
        return float('nan')
    return total / count


def run_experiment(G_full: nx.Graph, order: List[str]) -> List[float]:
    """Given a node-addition order, return average path length at each addition step."""
    results = []
    G = nx.Graph()
    added = set()
    # Pre-build adjacency map for speed
    adj = {n: set(G_full[n]) for n in G_full.nodes()}
    for i, n in enumerate(order, start=1):
        added.add(n)
        # add edges to existing added nodes
        for nb in adj.get(n, ()): 
            if nb in added:
                w = G_full.get_edge_data(n, nb).get('weight', 1)
                G.add_edge(n, nb, weight=w)
        # compute avg shortest path on current induced subgraph (largest CC)
        apl = avg_shortest_path_length_induced(G)
        results.append(float(apl) if not math.isnan(apl) else float('nan'))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edges', default='edges_enzyme_enzyme.csv')
    p.add_argument('--out-prefix', default='avgpath_nodes')
    p.add_argument('--random-seed', type=int, default=7)
    p.add_argument('--repeat-random', type=int, default=1, help='Number of random reorder repeats to average')
    p.add_argument('--filter-currency', action='store_true', default=False)
    p.add_argument('--max-nodes', type=int, default=350, help='Maximum nodes to evaluate (reduce for speed)')
    p.add_argument('--step', type=int, default=1, help='Evaluate every `step` nodes added')
    args = p.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    edges_csv = Path(args.edges)
    Gfull = load_graph(edges_csv, filter_currency=args.filter_currency, currency_set=None)
    nodes = list(Gfull.nodes())
    N_total = len(nodes)
    N = min(args.max_nodes, N_total)

    # Virtual order: degree-descending (ties broken deterministically)
    degs = dict(Gfull.degree())
    virtual_order = sorted(nodes, key=lambda n: (-degs.get(n,0), n))

    # Random order: single or multiple repeats
    # We'll evaluate at nodes_added = 1..N but only keep every `step` point to reduce work
    nodes_added_points = list(range(1, N+1, args.step))

    random_results = np.zeros((args.repeat_random, len(nodes_added_points)), dtype=float)
    for r in range(args.repeat_random):
        order = nodes.copy()
        random.shuffle(order)
        # we only need first N nodes from order
        order = order[:N]
        res_full = run_experiment(Gfull, order)
        # sample by step
        random_results[r, :] = [res_full[i-1] for i in nodes_added_points]

    random_avg = np.nanmean(random_results, axis=0)

    virtual_order = virtual_order[:N]
    v_full = run_experiment(Gfull, virtual_order)
    virtual_results = [v_full[i-1] for i in nodes_added_points]

    # Save CSVs
    prefix = Path(args.out_prefix)
    pd.DataFrame({'nodes_added': nodes_added_points, 'avg_path_random': random_avg}).to_csv(prefix.with_suffix('.random.csv'), index=False)
    pd.DataFrame({'nodes_added': nodes_added_points, 'avg_path_virtual': virtual_results}).to_csv(prefix.with_suffix('.virtual.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(nodes_added_points, random_avg, marker='o', linestyle='-', markersize=4, label='Random', color='#E24A33', alpha=0.9)
    ax.plot(nodes_added_points, virtual_results, marker='o', linestyle='--', markersize=4, label='Virtual', color='#348ABD', alpha=0.9)
    ax.set_xlabel('Nodes Added')
    ax.set_ylabel('Average Path Length')
    ax.set_xlim(0, N+5)
    ax.legend()
    plt.tight_layout()
    out_png = prefix.with_suffix('.png')
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot → {out_png}")


if __name__ == '__main__':
    main()
