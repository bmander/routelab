"""One transit trip, routed by both of Pyrga et al.'s models.

    python demos/route_transit.py kcm.zip --date 2026-08-17

Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable Information
in Public Transportation Systems* (ACM JEA 12, Article 2.4, 2007) is not a paper
about an algorithm. It is a paper about **modelling**: a timetable is a set of
departures rather than a network with weights, and there are two ways to make
one into something a shortest-path algorithm can read.

- **Time-dependent** — a node per stop, and a search that asks each edge what
  leaves next. Small, and the search is bespoke.
- **Time-expanded** — a node per event, and then it is an ordinary static graph
  that plain Dijkstra routes unchanged. General, and large.

They must return the same arrival time. That claim is the paper's thesis and
this demo's point, so the two run side by side and the table prints both.

    python demos/route_transit.py kcm.zip --date 2026-08-17 \\
        --from 47.6062,-122.3321 --to 47.6588,-122.3120 --departing 08:30
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import routelab as rl
from routelab import GTFS

#: Downtown Seattle to the University District, which is a real trip somebody
#: takes. Any two coordinates in the feed's area will do.
DOWNTOWN = (47.6062, -122.3321)
U_DISTRICT = (47.6588, -122.3120)


def coordinate(text: str) -> "tuple[float, float]":
    lat, lon = text.split(",")
    return float(lat), float(lon)


def service_time(text: str) -> datetime.time:
    return datetime.time.fromisoformat(text)


def clock(seconds: int) -> str:
    """`HH:MM`, and past 24:00 rather than wrapping — a service day is a line."""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("feed", type=Path, help="a GTFS .zip or directory")
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        required=True,
        help="the service day to read, YYYY-MM-DD",
    )
    parser.add_argument("--from", dest="origin", type=coordinate, default=DOWNTOWN)
    parser.add_argument("--to", dest="target", type=coordinate, default=U_DISTRICT)
    parser.add_argument("--departing", type=service_time, default=datetime.time(8, 30))
    args = parser.parse_args(argv)

    if not args.feed.exists():
        parser.error(f"no GTFS feed at {args.feed}")

    started = time.perf_counter()
    feed = GTFS(args.feed, args.date)
    env = rl.Environment(feed)
    compiled = env.compile()
    read = time.perf_counter() - started

    print(f"{args.feed.name} on {args.date} ({args.date:%A})")
    print(
        f"  {feed.feed.num_stops:,} stops, {feed.feed.num_connections:,} connections, "
        f"{compiled.graph.num_edges:,} stop pairs, read in {read:.1f}s"
    )
    if feed.unimplemented:
        print(f"  {feed.unimplemented:,} trips this reader cannot represent, dropped")
    if not compiled.timetable:
        print("  nothing runs on this date — try another, or check the calendar")
        return 1

    names = feed.names()
    origin = feed.nearest(*args.origin)
    target = feed.nearest(*args.target)
    print(f"\n{names[origin]} → {names[target]}, leaving {args.departing:%H:%M}\n")

    header = f"  {'model':<16} {'bind':>7} {'memory':>9} {'query':>9} {'settled':>9}  arrives"
    print(header)
    print("  " + "-" * (len(header) - 2))

    journeys = {}
    for technique in (rl.TimeDependent(), rl.TimeExpanded()):
        name = type(technique).__name__

        started = time.perf_counter()
        planner = technique.bind(env)
        bound = time.perf_counter() - started

        # Timed a second time, because the first query of a run pays for pages
        # the bind step only just wrote.
        planner.route(origin, target, departing=args.departing)
        started = time.perf_counter()
        journey = planner.route(origin, target, departing=args.departing)
        query = (time.perf_counter() - started) * 1000

        if journey is None:
            print(f"  {name:<16} {bound:6.1f}s {'':>9} {query:8.1f}ms  no journey")
            return 1
        journeys[name] = journey
        size = planner.footprint / 1e6
        print(
            f"  {name:<16} {bound:6.1f}s {size:8.1f}M {query:8.1f}ms "
            f"{journey.settled:9,}  {clock(journey.arrives)}"
        )

    arrivals = {journey.arrives for journey in journeys.values()}
    print(
        f"\n  the two models agree: {clock(arrivals.pop())}"
        if len(arrivals) == 1
        else f"\n  THE MODELS DISAGREE: {arrivals} — one of them is wrong"
    )

    journey = journeys["TimeDependent"]
    print(
        f"  {journey.cost // 60} minutes, {journey.moving // 60} riding and "
        f"{journey.waiting // 60} waiting, {journey.transfers} transfers\n"
    )
    for departs, arrives, route, boarded, alighted, _ in _rides(feed, journey):
        print(
            f"  {clock(departs)}–{clock(arrives)}  route {route:<8} "
            f"{boarded} → {alighted}"
        )
    return 0


def _rides(feed: GTFS, journey: rl.Journey) -> "list[list]":
    """The journey's legs collapsed to the vehicles it boarded.

    A leg is one stop to the next, so a bus ridden for twelve stops is twelve
    legs; what a rider wants is the four buses.
    """
    trips = feed.trips()
    names = feed.names()
    rides: "list[list]" = []
    for leg in journey.legs:
        route = trips[leg.trip][1]
        if rides and rides[-1][2] == route and rides[-1][5] == leg.trip:
            rides[-1][1] = leg.arrives
            rides[-1][4] = names[leg.head]
        else:
            rides.append(
                [leg.departs, leg.arrives, route, names[leg.tail], names[leg.head], leg.trip]
            )
    return rides


if __name__ == "__main__":
    sys.exit(main())
