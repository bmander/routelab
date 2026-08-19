"""Every technique in the library against the same city, on the same trips.

    python benchmarks/bench_contraction.py data/Seattle.osm.pbf

Four rows, one per technique, and three numbers that matter: what preprocessing
cost, how many nodes a query settles, and how long it takes. Costs are checked
against Dijkstra's on every trip, so a row that got faster by getting a
different answer fails here rather than looking good.

The point of running them side by side is that they are not four speeds of the
same thing. A* narrows the search; landmarks narrow it further for a table paid
up front; a contraction hierarchy changes the graph and stops searching the city
at all. The settled-nodes column is where that shows.

    python benchmarks/bench_contraction.py data/Seattle.osm.pbf --trips 50
    python benchmarks/bench_contraction.py extract.osm.pbf --profile walking
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from pathlib import Path

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}

TECHNIQUES = [
    ("Dijkstra", rl.Dijkstra()),
    ("A* (euclidean)", rl.AStar(rl.Euclidean())),
    ("A* (16 landmarks)", rl.AStar(rl.Landmarks(16))),
    ("Contraction hierarchy", rl.ContractionHierarchy()),
]



def isolated() -> None:
    """Start a technique's measurement from a heap the last one has left.

    Techniques are timed in one process so that they can be held to each
    other's answers, which means one row's footprint can tax the next: the row
    after PTL's gigabyte used to read about half a millisecond slow, purely for
    having followed it. Dropping the planner and collecting before the next
    bind is what makes a row a measurement of its own technique.
    """
    gc.collect()

def megabytes(byte_count: int) -> str:
    return "-" if byte_count == 0 else f"{byte_count / 1e6:.0f} MB"


def trips(environment: rl.Environment, count: int, seed: int) -> "list[tuple]":
    """Random origin/destination pairs that are actually connected.

    Drawn from one Dijkstra's settled set, so every pair has an answer — timing
    a technique on unreachable pairs measures how fast it gives up.
    """
    rng = random.Random(seed)
    compiled = environment.compile()
    start = compiled.label(0)
    reachable = [
        compiled.label(node)
        for node in rl.Dijkstra().bind(environment).search(start).order
    ]
    return [(rng.choice(reachable), rng.choice(reachable)) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an .osm.pbf or .osm file")
    parser.add_argument("--profile", default="driving", choices=sorted(PROFILES))
    parser.add_argument("--trips", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    began = time.perf_counter()
    environment = rl.Environment(rl.OSM(args.extract, PROFILES[args.profile]()))
    compiled = environment.compile()
    print(
        f"{args.extract.name}, {args.profile}: "
        f"{compiled.graph.num_nodes:,} nodes, {compiled.graph.num_edges:,} edges "
        f"({time.perf_counter() - began:.1f}s)"
    )

    pairs = trips(environment, args.trips, args.seed)
    print(f"{len(pairs)} random trips\n")
    print(f"{'technique':<22} {'preprocess':>11} {'memory':>8} {'settled':>10} {'query':>10}")
    print("-" * 65)

    truth: "dict[tuple, int]" = {}
    for name, technique in TECHNIQUES:
        isolated()
        began = time.perf_counter()
        planner = technique.bind(environment)
        preprocess = time.perf_counter() - began

        settled, times, costs = [], [], []
        for origin, destination in pairs:
            target = planner.node_id(destination)
            began = time.perf_counter()
            # The goal-directed pair take the one place they aim at or climb
            # toward; Dijkstra takes a set of targets to stop at.
            result = (
                planner.search(origin, target=target)
                if isinstance(
                    planner, (rl.AStarPlanner, rl.ContractionHierarchyPlanner)
                )
                else planner.search(origin, targets=[target])
            )
            times.append((time.perf_counter() - began) * 1000)
            settled.append(result.settled)
            costs.append(result.cost(target))

        # Dijkstra runs first and sets the answer every other row has to match.
        for pair, cost in zip(pairs, costs):
            if truth.setdefault(pair, cost) != cost:
                raise SystemExit(f"{name} disagrees on {pair}: {cost} != {truth[pair]}")

        print(
            f"{name:<22} {preprocess:>10.1f}s {megabytes(planner.footprint):>8} "
            f"{statistics.mean(settled):>10,.0f} "
            f"{statistics.median(times):>8.3f}ms"
        )
        del planner
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
