"""Every timetable technique against the same feed, on the same trips.

    python benchmarks/bench_transit.py data/kcm.zip --date 2026-08-17

Six rows, one per technique, and the numbers that matter: what binding cost,
what it holds, how much of the timetable a query settles, and how long it
takes. Every arrival is checked against every other technique's on every trip
— the paper's thesis for the two Pyrga models, and RAPTOR's only oracle — so a
row that got faster by getting a different answer fails here rather than
looking good.

The point of running them side by side is that they are not five speeds of
the same thing. The time-dependent model keeps one node per stop and pays in
the search; the time-expanded one spends nodes to make the search plain
Dijkstra; RAPTOR builds no graph at all and scans routes in rounds; CSA sorts
every connection into one array and scans that; TripBased precomputes the
transfers between trips at bind and sweeps those; PTL labels every event once,
in minutes and hundreds of megabytes, and then reads a few hundred integers
per query. The settled column is where that shows — read it against the
`nodes` column, because they do not count the same thing.

    python benchmarks/bench_transit.py data/kcm.zip --date 2026-08-17 --pairs 50
    python benchmarks/bench_transit.py data/kcm.zip --date 2026-08-17 --walk 0
"""

from __future__ import annotations

import argparse
import gc
import datetime
import random
import statistics
import time
from pathlib import Path

import routelab as rl

TECHNIQUES = [
    ("TimeDependent", rl.TimeDependent()),
    ("TimeExpanded", rl.TimeExpanded()),
    ("RAPTOR", rl.RAPTOR()),
    ("CSA", rl.CSA()),
    ("TripBased", rl.TripBased()),
    ("PTL", rl.PTL()),
    ("ULTRA", rl.ULTRA(rl.RAPTOR())),
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


def trips(environment: rl.Environment, departing: int, count: int, seed: int) -> "list[tuple]":
    """Random origin/destination pairs that are actually connected at this hour.

    Drawn from one RAPTOR one-to-all search's reached set, so every pair has an
    answer — timing a technique on unreachable pairs measures how fast it gives
    up. RAPTOR because it is the technique here with a one-to-all search.
    """
    rng = random.Random(seed)
    planner = rl.RAPTOR().bind(environment)
    compiled = environment.compile()
    origin = compiled.label(rng.randrange(len(compiled)))
    reached = [compiled.label(stop) for stop, _, _ in planner.search(origin, departing=departing).reached()]
    return [(rng.choice(reached), rng.choice(reached)) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed", type=Path, help="a GTFS .zip or directory")
    parser.add_argument("--date", type=datetime.date.fromisoformat, required=True)
    parser.add_argument("--departing", type=datetime.time.fromisoformat, default=datetime.time(8, 30))
    parser.add_argument("--walk", type=float, default=200.0, help="footpaths within this many metres; 0 for none")
    parser.add_argument("--pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    began = time.perf_counter()
    feed = rl.GTFS(args.feed, args.date)
    environment = rl.Environment(feed)
    if args.walk > 0:
        environment.register(rl.Footpaths(feed, within=args.walk))
    compiled = environment.compile()
    print(
        f"{args.feed.name} on {args.date}: {feed.feed.num_stops:,} stops, "
        f"{feed.feed.num_connections:,} connections, {compiled.graph.num_edges:,} edges "
        f"({time.perf_counter() - began:.1f}s)"
    )
    departing = rl.service_seconds(args.departing)
    pairs = trips(environment, departing, args.pairs, args.seed)
    print(f"{len(pairs)} random pairs, departing {args.departing:%H:%M}\n")
    print(f"{'technique':<14} {'nodes':>16} {'bind':>7} {'memory':>8} {'settled':>9} {'query':>10}")
    print("-" * 70)

    truth: "dict[tuple, int | None]" = {}
    for name, technique in TECHNIQUES:
        isolated()
        began = time.perf_counter()
        planner = technique.bind(environment)
        bound = time.perf_counter() - began

        settled, times = [], []
        for origin, destination in pairs:
            began = time.perf_counter()
            journey = planner.route(origin, destination, departing=departing).routes[0]
            times.append((time.perf_counter() - began) * 1000)
            arrives = None if journey is None else journey.arrives
            if journey is not None:
                settled.append(journey.settled)
            # The first technique sets the answer every other row has to match.
            if truth.setdefault((origin, destination), arrives) != arrives:
                raise SystemExit(
                    f"{name} disagrees on {origin} -> {destination}: "
                    f"{arrives} != {truth[(origin, destination)]}"
                )

        noun, count = planner.searches
        print(
            f"{name:<14} {count:>9,} {noun:<6} {bound:>6.1f}s {megabytes(planner.footprint):>8} "
            f"{statistics.mean(settled):>9,.0f} {statistics.median(times):>8.3f}ms"
        )
        del planner
    print(f"\n{len(pairs)} pairs, all {len(TECHNIQUES)} agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
