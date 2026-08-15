"""Kernel vs. reference on a grid graph — the argument for the Rust half.

The reference implementation is the correctness oracle, not a competitor: this
measures the gap that justifies having both.

    python benchmarks/bench_dijkstra.py --side 300
"""

from __future__ import annotations

import argparse
import random
import time

import routelab as rl


def grid_graph(side: int, seed: int = 0) -> rl.Graph:
    """A `side` x `side` 4-neighbour grid with random weights — a stand-in for a
    street network: planar, low degree, large."""
    rng = random.Random(seed)
    edges = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                edges.append((node, node + 1, rng.randrange(1, 100)))
                edges.append((node + 1, node, rng.randrange(1, 100)))
            if row + 1 < side:
                edges.append((node, node + side, rng.randrange(1, 100)))
                edges.append((node + side, node, rng.randrange(1, 100)))
    return rl.Graph.from_edges(edges, num_nodes=side * side)


def time_call(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{label:<28} {elapsed * 1000:8.1f} ms")
    return result, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=300, help="grid side length")
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    graph, _ = time_call("build graph", grid_graph, args.side)
    print(f"{'':<28} {graph.num_nodes} nodes, {graph.num_edges} edges\n")

    fast, fast_time = time_call("dijkstra (rust)", rl.dijkstra, graph, 0)
    if args.skip_reference:
        return

    slow, slow_time = time_call("dijkstra (pure python)", rl.reference.dijkstra, graph, 0)
    assert fast.costs == slow.costs, "kernel and reference disagree"
    print(f"\nspeedup: {slow_time / fast_time:.0f}x  (results identical)")


if __name__ == "__main__":
    main()
