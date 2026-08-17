"""One transit trip, routed by every timetable technique here.

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
this demo's point — and RAPTOR (Delling, Pajor & Werneck, 2012), which builds
no graph at all, CSA (Dibbelt, Pajor, Strasser & Wagner, 2013), which sorts
the connections into one array and scans it, TripBased (Witt, 2015), which
precomputes the transfers between trips and sweeps those, and PTL (Delling,
Dibbelt, Pajor & Werneck, 2015), which labels every event once and never
searches again, must agree with all of them. The six run side by side and the
table prints each; RAPTOR and TripBased also print their whole front, one
journey per number of changes, and CSA, TripBased and PTL their profiles for
the next two hours, one journey per departure worth taking — CSA's and PTL's
the same list, arrived at from opposite ends.

All six are given the paper's *foot-edges* — walks between stops within
``--walk`` metres of each other — because a real feed says nothing about the
northbound stop and the southbound one across the street being the same place,
and a model that cannot cross the street cannot make most real trips. Pass
``--walk 0`` for the plain model and watch the trip fall apart.

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

#: Every timetable technique, in the order the literature produced them.
MODELS = [rl.TimeDependent(), rl.TimeExpanded(), rl.RAPTOR(), rl.CSA(), rl.TripBased(), rl.PTL()]

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
    parser.add_argument(
        "--walk",
        type=float,
        default=200.0,
        help="join stops within this many metres by a walk; 0 for none (default 200)",
    )
    args = parser.parse_args(argv)

    if not args.feed.exists():
        parser.error(f"no GTFS feed at {args.feed}")

    started = time.perf_counter()
    feed = GTFS(args.feed, args.date)
    env = rl.Environment(feed)
    if args.walk > 0:
        env.register(rl.Footpaths(feed, within=args.walk))
    compiled = env.compile()
    read = time.perf_counter() - started

    print(f"{args.feed.name} on {args.date} ({args.date:%A})")
    print(
        f"  {feed.feed.num_stops:,} stops, {feed.feed.num_connections:,} connections, "
        f"{compiled.graph.num_edges:,} stop pairs, read in {read:.1f}s"
    )
    if feed.unimplemented:
        print(f"  {feed.unimplemented:,} trips this reader cannot represent, dropped")
    if rl.Departures.missing_from(compiled):
        print("  nothing runs on this date — try another, or check the calendar")
        return 1

    names = feed.names()
    origin = feed.nearest(*args.origin)
    target = feed.nearest(*args.target)
    print(f"\n{names[origin]} → {names[target]}, leaving {args.departing:%H:%M}\n")

    header = (
        f"  {'technique':<14} {'nodes':>16} {'bind':>7} {'memory':>9} {'query':>9} "
        f"{'settled':>9}  arrives"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    journeys = {}
    planners = {}
    for technique in MODELS:
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
            print(f"  {name:<14} {'':>16} {bound:6.1f}s {'':>9} {query:8.1f}ms  no journey")
            return 1
        journeys[name] = journey
        planners[name] = planner
        size = planner.footprint / 1e6
        noun, count = planner.searches
        print(
            f"  {name:<14} {count:>9,} {noun:<6} {bound:6.1f}s {size:8.1f}M {query:8.1f}ms "
            f"{journey.settled:9,}  {clock(journey.arrives)}"
        )

    arrivals = {journey.arrives for journey in journeys.values()}
    print(
        f"\n  all {len(journeys)} agree: {clock(arrivals.pop())}"
        if len(arrivals) == 1
        else f"\n  THE TECHNIQUES DISAGREE: {arrivals} — one of them is wrong"
    )
    # The extra readouts, each asked of the techniques that have the verb.
    for name in ("RAPTOR", "TripBased"):
        planner = planners.get(name)
        if planner is None:
            continue
        front = planner.frontier(origin, target, departing=args.departing)
        if len(front) > 1:
            print(f"  {name}'s front: " + " · ".join(
                f"{j.transfers} change{'' if j.transfers == 1 else 's'} arrives {clock(j.arrives)}"
                for j in front
            ))

    # CSA's profile is over departure and arrival; TripBased's over changes as
    # well, so the same window holds more and each of its steps says how many.
    def step(journey: rl.Journey, changes: bool) -> str:
        leg = f"leave {clock(journey.departs)}, arrive {clock(journey.arrives)}"
        return f"{leg} ({journey.transfers} ch.)" if changes else leg

    for name in ("CSA", "TripBased", "PTL"):
        planner = planners.get(name)
        if planner is None:
            continue
        # Two hours on the service-day clock, which runs past midnight rather
        # than wrapping at it — a 23:30 departure asks about 25:30.
        until = rl.service_seconds(args.departing) + 2 * 3600
        started = time.perf_counter()
        profile = planner.profile(origin, target, departing=args.departing, until=until)
        took = (time.perf_counter() - started) * 1000
        print(
            f"  {name}'s profile to {clock(until)} ({took:.1f}ms): "
            f"{len(profile)} departures worth taking — "
            + " · ".join(step(j, name == "TripBased") for j in profile[:4])
            + (" · …" if len(profile) > 4 else "")
        )

    journey = journeys["TimeDependent"]
    print(
        f"  {journey.cost // 60} minutes, {(journey.moving - journey.walking) // 60} riding, "
        f"{journey.walking // 60} walking and {journey.waiting // 60} waiting, "
        f"{journey.transfers} transfers\n"
    )
    for departs, arrives, route, boarded, alighted, _ in _rides(feed, journey):
        print(
            f"  {clock(departs)}–{clock(arrives)}  {route:<15} "
            f"{boarded} → {alighted}"
        )
    return 0


def _rides(feed: GTFS, journey: rl.Journey) -> "list[list]":
    """The journey's legs collapsed to the vehicles it boarded and the walks
    between them.

    A leg is one stop to the next, so a bus ridden for twelve stops is twelve
    legs; what a rider wants is the four buses — and the walk across the street
    between two of them, which is a leg with no trip.
    """
    trips = feed.trips()
    names = feed.names()
    rides: "list[list]" = []
    for leg in journey.legs:
        route = "walk" if leg.trip is None else f"route {trips[leg.trip][1]}"
        if rides and rides[-1][5] == leg.trip:
            rides[-1][1] = leg.arrives
            rides[-1][4] = names[leg.head]
        else:
            rides.append(
                [leg.departs, leg.arrives, route, names[leg.tail], names[leg.head], leg.trip]
            )
    return rides


if __name__ == "__main__":
    sys.exit(main())
