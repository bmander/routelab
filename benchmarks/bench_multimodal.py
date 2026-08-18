"""Every multimodal technique against the same city and feed, on the same trips.

    python benchmarks/bench_multimodal.py data/Seattle.osm.pbf data/kcm.zip \
        --date 2026-08-17

Three rows, and they are the three corners of one trade. `LabelConstrained`
precomputes nothing and searches the whole network. `UCCH` contracts each mode's
own subnetwork — never a vertex where they join, so no shortcut can cross a
mode boundary and the language stays a query input — and searches a core.
`ULTRA` works out which walking transfers a journey could ever want, then hands
a stock timetable technique a transfer set with no radius in it.

A query here starts and ends on the *pavement*, not at a stop, which is what
makes it multimodal: walk, cross a link arc, ride, cross back, walk. Every
arrival is checked against every other technique's on every trip, so a row that
got faster by getting a different answer fails here rather than looking good.

    python benchmarks/bench_multimodal.py … --pairs 10 --walk 400
"""

from __future__ import annotations

import argparse
import gc
import datetime
import json
import random
import statistics
import time
from pathlib import Path

import routelab as rl

TECHNIQUES = [
    ("LabelConstrained", rl.LabelConstrained()),
    ("UCCH", rl.UCCH()),
    ("ULTRA(RAPTOR)", rl.ULTRA(rl.RAPTOR())),
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


def doorsteps(feed, streets, environment, departing: int, count: int, seed: int):
    """Random pairs of *street* nodes near stops that can actually reach each other.

    Drawn from one RAPTOR one-to-all search over the stops, so every pair has a
    transit answer, and then snapped out onto the pavement — timing a technique
    on unreachable pairs measures how fast it gives up.
    """
    rng = random.Random(seed)
    stops = rl.Environment(feed, rl.Footpaths(feed, within=200))
    planner = rl.RAPTOR().bind(stops)
    compiled = stops.compile()
    origin = compiled.label(rng.randrange(len(compiled)))
    reached = [compiled.label(stop) for stop, _, _ in planner.search(origin, departing=departing).reached()]
    places = feed.coordinates()
    known = set(environment.compile().labels)
    corners = [streets.nearest(*places[stop]) for stop in rng.sample(reached, min(len(reached), 4 * count))]
    corners = [corner for corner in corners if corner in known]
    if not corners:
        raise SystemExit("no reachable stop snapped to a street node; try a larger --walk")
    return [(rng.choice(corners), rng.choice(corners)) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an OpenStreetMap .osm.pbf")
    parser.add_argument("feed", type=Path, help="a GTFS .zip or directory")
    parser.add_argument("--date", type=datetime.date.fromisoformat, required=True)
    parser.add_argument("--departing", type=datetime.time.fromisoformat, default=datetime.time(8, 30))
    parser.add_argument("--walk", type=float, default=400.0, help="how far a stop may be from a street")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the measurements here")
    arguments = parser.parse_args()

    began = time.perf_counter()
    feed = rl.GTFS(arguments.feed, arguments.date)
    streets = rl.OSM(arguments.extract, rl.Walking())
    environment = rl.Environment(feed, streets, rl.Access(feed, streets, within=arguments.walk))
    compiled = environment.compile()
    read = time.perf_counter() - began
    print(
        f"{arguments.extract.name} + {arguments.feed.name} on {arguments.date}: "
        f"{len(compiled):,} nodes, {compiled.graph.num_edges:,} edges ({read:.1f}s)"
    )

    departing = rl.service_seconds(arguments.departing)
    pairs = doorsteps(feed, streets, environment, departing, arguments.pairs, arguments.seed)
    print(f"{len(pairs)} random pavement-to-pavement trips, departing {arguments.departing:%H:%M}\n")
    print(f"{'technique':<18} {'preprocess':>10} {'memory':>8} {'query':>11}")
    print("-" * 51)

    truth: "dict[tuple, int | None]" = {}
    rows = []
    for name, technique in TECHNIQUES:
        isolated()
        began = time.perf_counter()
        planner = technique.bind(environment)
        bound = time.perf_counter() - began

        times = []
        for origin, destination in pairs:
            began = time.perf_counter()
            journey = planner.route(origin, destination, departing=departing)
            times.append((time.perf_counter() - began) * 1000)
            arrives = None if journey is None else journey.arrives
            # The first technique sets the answer every other row has to match.
            if truth.setdefault((origin, destination), arrives) != arrives:
                raise SystemExit(
                    f"{name} disagrees on {origin} -> {destination}: "
                    f"{arrives} != {truth[(origin, destination)]}"
                )

        query = statistics.median(times)
        rows.append({"technique": name, "preprocess_s": bound, "query_ms": query,
                     "memory_bytes": planner.footprint})
        print(f"{name:<18} {bound:>9.1f}s {megabytes(planner.footprint):>8} {query:>9.3f}ms")
        del planner

    print(f"\n{len(pairs)} trips, all {len(TECHNIQUES)} agree")
    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
