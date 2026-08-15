"""What guidance buys, measured in nodes settled rather than guessed at.

Runs one corner-to-corner query on a grid with three planners: Dijkstra, A* with
a zero heuristic (the control — the same search, so it should match Dijkstra
almost exactly), and A* with a Euclidean heuristic.

The `--no-diagonal` flag is the interesting one. It changes nothing about the
algorithm and everything about the result: without diagonal moves you travel in
L1 while the estimate measures L2, the bound is loose by up to sqrt(2), and the
guidance stops paying for itself.

    python benchmarks/bench_astar.py --side 200
    python benchmarks/bench_astar.py --side 200 --no-diagonal
"""

from __future__ import annotations

import argparse
import math
import time

import routelab as rl


def grid_environment(side: int, *, spacing: float = 100.0, diagonal: bool = True):
    """A `side` x `side` grid where every edge costs exactly the ground it covers."""
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if diagonal else [])
    edges = []
    positions = {}
    for row in range(side):
        for col in range(side):
            positions[(row, col)] = (col * spacing, row * spacing)
            for row_step, col_step in steps:
                neighbor = (row + row_step, col + col_step)
                if 0 <= neighbor[0] < side and 0 <= neighbor[1] < side:
                    weight = math.ceil(math.hypot(row_step, col_step) * spacing)
                    edges.append(((row, col), neighbor, weight))
                    edges.append((neighbor, (row, col), weight))

    return rl.Environment(
        rl.ScalarEdges(edges, cost_per_distance=1.0),
        rl.Positions(positions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=200, help="grid side length")
    parser.add_argument(
        "--no-diagonal",
        dest="diagonal",
        action="store_false",
        help="4-connected grid, where a straight-line estimate is a poor bound",
    )
    args = parser.parse_args()

    env = grid_environment(args.side, diagonal=args.diagonal)
    compiled = env.compile()
    origin, destination = (0, 0), (args.side - 1, args.side - 1)
    print(
        f"{compiled.graph.num_nodes} nodes, {compiled.graph.num_edges} edges, "
        f"{'8' if args.diagonal else '4'}-connected\n"
    )

    planners = [
        ("dijkstra", rl.Dijkstra(env)),
        ("astar (zero)", rl.AStar(env, rl.Zero())),
        ("astar (euclidean)", rl.AStar(env, rl.Euclidean())),
    ]

    print(f"{'planner':<20}{'settled':>10}{'of graph':>10}{'ms':>8}{'cost':>10}")
    baseline = None
    for name, planner in planners:
        target = [planner.node_id(destination)]
        start = time.perf_counter()
        result = planner.search(origin, targets=target)
        elapsed = (time.perf_counter() - start) * 1000
        settled = len(result.order)
        cost = result.cost(target[0])

        if baseline is None:
            baseline = cost
        assert cost == baseline, f"{name} disagrees: {cost} != {baseline}"

        share = settled / compiled.graph.num_nodes
        print(f"{name:<20}{settled:>10}{share:>9.0%}{elapsed:>8.1f}{cost:>10}")


if __name__ == "__main__":
    main()
