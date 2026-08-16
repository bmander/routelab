"""An interactive routing demo: wire up a planner, click two points, watch it search.

    python demos/serve.py data/Seattle.osm.pbf
    python demos/serve.py data/Seattle.osm.pbf --gtfs data/kcm.zip --date 2026-08-17

Then open http://localhost:8000. The panel below the map is a **node board**, and
it is the library's own three steps drawn as a graph: layers feed an
`Environment`, a technique binds to one and becomes a planner, and a `Query`
asks it something. Drag the nodes, unplug the wires, plug them in somewhere
else. There is no second representation of what the controls mean — the board
*is* the query, so a wire that goes nowhere is a missing argument and says so.

The map is a node too, and the graph runs through it and back: it hands the
query an `origin` and a `destination`, and takes a `route` and a `space` in
return. Which makes those wires worth pulling on. Cross the two points and the
trip really does reverse. Unplug `space` and no search space is built at all —
it is ten megabytes of GeoJSON that nothing was listening for. Unplug `route`
and the query still runs and still reports what it cost; there is simply
nothing drawing it.

Wiring is also what makes the refusals visible. Put a GTFS layer into Dijkstra
and the node turns red with the library's own words: *Dijkstra cannot route
over timetable layers; it accepts scalar*. Put `A* (euclidean)` on a timetable
and the heuristic says it has no rate to price a distance against. Those are the
errors this project exists to raise, and they are easier to believe when you
cause one yourself.

A dropdown in the board's toolbar loads a starting point — the same query wired
seven different ways. They are places to begin, not modes: the pins stay where
they are across a change, so the same two points can be routed by A*, by a
contraction hierarchy, and by each of the timetable techniques in turn.

Click once on the map to drop an origin, again for a destination, and the route
draws over the search tree that found it — the part a routing engine normally
throws away. Branch widths follow the total hanging off each one, so the tree
reads like a river network: thick where the whole search flowed, thinning to
capillaries where it gave up.

Either endpoint can then be dragged, and the route follows the cursor. The
search space does not: it is up to sixty thousand branches, so it is drawn once
when the drag stops. Dragging is also the quickest way to feel the difference
between the techniques — the route keeps up under a contraction hierarchy and
visibly does not under Dijkstra.

Everything the board builds is cached by a canonical spelling of the node and
everything upstream of it, so rewiring back to a shape you had before is free.
That matters because the expensive things are exactly the ones worth comparing:
reading Seattle is six seconds, sixteen landmarks a second and 33 MB, a
contraction hierarchy six seconds and 19 MB.

Which nodes those are is not left to be guessed at. A node whose configuration
the board has never had an answer for greys out and spins until it does, and
because a node's identity includes everything upstream of it, the spinners
spread exactly as far as the rebuild does and no further — swap the ordering
under a hierarchy and the hierarchy waits, while the extract it was built from
does not. A change that costs nothing shows nothing.

The board is a drawer: the first button on its toolbar folds it down to the
toolbar and back, and on a phone it starts folded, the map having the screen.
Nodes, wires and the grip answer to a finger as they do to a mouse.

Nothing here is a production server: it is the standard library's `http.server`,
bound to localhost, with no cache and no rate limiting. It is a way to look at
what the algorithms do.
"""

from __future__ import annotations

import argparse
import datetime
import json
from http.server import ThreadingHTTPServer
from pathlib import Path


from board.catalogue import PROFILES
from board.handler import Handler
from board.router import Router
from board.wiring import Board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an .osm.pbf file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="driving")
    parser.add_argument(
        "--gtfs", type=Path, help="a GTFS .zip or directory, to offer transit too"
    )
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        help="the service day to read the feed for, YYYY-MM-DD",
    )
    args = parser.parse_args()

    if not args.extract.is_file():
        print(f"no extract at {args.extract}")
        print("city extracts: https://download.bbbike.org/osm/bbbike/")
        return 1
    if args.gtfs and not args.gtfs.exists():
        parser.error(f"no GTFS feed at {args.gtfs}")
    # A feed covers months and its trips run on different days, so which day is
    # not something to guess at — the same reason `GTFS` takes a date.
    if bool(args.gtfs) != bool(args.date):
        parser.error("--gtfs and --date go together: a feed is read one day at a time")

    router = Router(args.extract, args.gtfs, args.date)

    # Read eagerly, so the several seconds each file costs happen at startup
    # rather than under the first wire someone plugs in.
    print(f"reading {args.extract.name}")
    board = Board.parse(json.dumps({"nodes": [{"id": "l", "type": "OSM",
                                               "params": {"profile": args.profile}}]}))
    layer = router.build(board, "l")
    min_lat, min_lon, max_lat, max_lon = layer.bounds
    if args.gtfs:
        print(f"reading {args.gtfs.name} for {args.date}")
        feed = router.build(Board.parse('{"nodes":[{"id":"f","type":"GTFS"}]}'), "f")
        if feed.unimplemented:
            print(f"  {feed.unimplemented:,} trips this reader cannot represent")

    Handler.router = router
    Handler.center = ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nhttp://localhost:{args.port}  — wire up a planner, click two points\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
