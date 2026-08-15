"""The same trip at every hour, over a network that is not always open.

    python demos/route_by_clock.py seattle.osm.pbf

OpenStreetMap records when a way may be used, not only whether — a park path
locked overnight, a lane that runs the other way before eleven. `Dijkstra`
ignores all of that and routes the always-open network; `TimeDependentDijkstra`
reads the clock. This prints the same journey under both, hour by hour, so the
difference is a column rather than a claim.

The default trip crosses the Hiram M. Chittenden Locks in Seattle, whose
walkway is tagged `access:conditional=yes @(07:00-21:00)`. Any extract and any
pair of points will do; a trip over nothing scheduled simply reports the same
answer twenty-four times, which is itself worth seeing.

    python demos/route_by_clock.py extract.osm.pbf --profile driving \\
        --from 47.6220,-122.3364 --to 47.6902,-122.3180
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}

#: Ballard, and Commodore Park on the far side of the ship canal.
BALLARD = (47.6720, -122.4000)
MAGNOLIA = (47.6560, -122.3960)

#: A Monday, for `--day` to count forward from. Which Monday does not matter:
#: the clock a schedule is written on is weekly, so nothing about a departure
#: survives except its weekday and its time.
WEEK_START = datetime.date(2026, 1, 5)


def coordinate(text: str) -> "tuple[float, float]":
    lat, _, lon = text.partition(",")
    return float(lat), float(lon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an .osm.pbf (or .osm) file")
    parser.add_argument("--from", dest="origin", type=coordinate, default=BALLARD)
    parser.add_argument("--to", dest="destination", type=coordinate, default=MAGNOLIA)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="walking")
    parser.add_argument("--day", type=int, default=0, help="0 is Monday, 6 is Sunday")
    args = parser.parse_args()

    if not args.extract.is_file():
        print(f"no extract at {args.extract}", file=sys.stderr)
        print("download one from https://download.geofabrik.de", file=sys.stderr)
        return 1

    layer = rl.OSM(args.extract, PROFILES[args.profile]())
    env = rl.Environment(layer)
    compiled = env.compile()
    print(
        f"{compiled.graph.num_nodes:,} nodes, {compiled.graph.num_edges:,} edges "
        f"({args.profile})"
    )

    # A technique declares what it needs, and the environment declares what it
    # has, so this is answerable before anything is built.
    technique = rl.TimeDependentDijkstra()
    missing = technique.missing_from(compiled)
    if missing:
        print(f"\nnothing to show: this extract provides no {', '.join(missing)}.")
        print("Try a profile whose access tags the ways actually carry.")
        return 1

    print(
        f"{len(compiled.calendar):,} of {compiled.graph.num_edges:,} edges are scheduled; "
        f"{layer.unreadable_schedules} schedules could not be read"
    )

    origin = layer.nearest(*args.origin)
    destination = layer.nearest(*args.destination)
    clock = technique.bind(env)
    always_open = rl.Dijkstra().bind(env)

    baseline = always_open.route(origin, destination)
    if baseline is None:
        print("\nthose points are not connected in this profile")
        return 1
    print(f"\nignoring the clock: {baseline.cost / 60:.1f} min over {len(baseline.legs)} legs\n")

    print(f"{'depart':>7}  {'moving':>7}  {'waiting':>8}  {'legs':>5}  {'scheduled':>9}")
    print("-" * 44)
    for hour in range(24):
        departing = datetime.datetime.combine(
            WEEK_START + datetime.timedelta(days=args.day), datetime.time(hour)
        )
        journey = clock.route(origin, destination, departing=departing)
        if journey is None:
            print(f"{hour:>5}:00  {'no route at this hour':>33}")
            continue
        # How many legs are scheduled is the number that explains the rest: a
        # route over edges nobody scheduled cannot care what time it is.
        scheduled = sum(
            1 for leg in journey if compiled.calendar.is_restricted(leg.edge)
        )
        print(
            f"{hour:>5}:00  {journey.moving / 60:>6.1f}m  {journey.waiting / 60:>7.0f}m  "
            f"{len(journey.legs):>5}  {scheduled:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
