"""An interactive routing demo: wire up a planner, click two points, watch it search.

    python demos/serve.py data/Seattle.osm.pbf
    python demos/serve.py data/Seattle.osm.pbf --gtfs data/kcm.zip --date 2026-08-17

Then open http://localhost:8000. The panel below the map is a **node board**, and
it is the library's own three steps drawn as a graph: layers feed an
`Environment`, a technique binds to one and becomes a planner, and a `Query`
asks it something. Drag the nodes, unplug the wires, plug them in somewhere
else. There is no second representation of what the controls mean — the board
*is* the query, so a wire that goes nowhere is a missing argument and says so.

Which is also what makes the refusals visible. Wire a GTFS layer into Dijkstra
and the board turns red with the library's own words: *Dijkstra cannot route
over timetable layers; it accepts scalar*. Wire `A* (euclidean)` into a
timetable and the heuristic says it has no rate to price a distance against.
Those are the errors this project exists to raise, and they are easier to
believe when you cause one yourself.

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
reading Seattle is six seconds, sixteen landmarks a second and 33 MB.

Nothing here is a production server: it is the standard library's `http.server`,
bound to localhost, with no cache and no rate limiting. It is a way to look at
what the algorithms do.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}

#: What each node type builds, and what has to be plugged into it first.
#:
#: This is the whole vocabulary of the board, and it is deliberately a
#: transcription of the library's own three steps rather than a menu of
#: features: layers make an environment, a technique binds to one and becomes a
#: planner, a planner takes a query. Adding a node is adding an entry here and
#: its shape in `static/board.js` — nothing between the two knows what the nodes
#: mean.
#:
#: `inputs` maps a port name to the kind of value it accepts. `kind` is what the
#: node produces, and the two together are all the wiring rules there are: a
#: port takes what it says it takes.
NODES: "dict[str, dict]" = {
    "OSM": {"kind": "layer", "inputs": {}},
    "GTFS": {"kind": "layer", "inputs": {}},
    "Environment": {"kind": "environment", "inputs": {"layers": "layer"}},
    "Euclidean": {"kind": "heuristic", "inputs": {}},
    "Landmarks": {"kind": "heuristic", "inputs": {}},
    "EdgeDifference": {"kind": "ordering", "inputs": {}},
    "RandomOrder": {"kind": "ordering", "inputs": {}},
    "Dijkstra": {"kind": "planner", "inputs": {"environment": "environment"}},
    "BFS": {"kind": "planner", "inputs": {"environment": "environment"}},
    "AStar": {
        "kind": "planner",
        "inputs": {"environment": "environment", "heuristic": "heuristic"},
    },
    "ContractionHierarchy": {
        "kind": "planner",
        "inputs": {"environment": "environment", "ordering": "ordering"},
    },
    "TimeDependentDijkstra": {"kind": "planner", "inputs": {"environment": "environment"}},
    "TimeDependent": {"kind": "planner", "inputs": {"environment": "environment"}},
    "TimeExpanded": {"kind": "planner", "inputs": {"environment": "environment"}},
    "Query": {"kind": "query", "inputs": {"planner": "planner"}},
}


def wants_time(planner: rl.Planner) -> bool:
    """Does this technique want a departure time?

    Asked of the technique rather than hardcoded here: a planner that needs a
    schedule or a timetable declares it, and that is the same declaration
    `missing_from` uses to say a dataset cannot support it.
    """
    return bool(getattr(planner, "requires", frozenset()) & {"schedule", "timetable"})


def rides_transit(planner: rl.Planner) -> bool:
    """Does this technique route a timetable rather than a network?

    The one question that decides which world a query happens in — street
    corners or bus stops — and it is the technique's own declaration that
    answers it.
    """
    return "timetable" in getattr(planner, "requires", frozenset())


class Board(NamedTuple):
    """A graph as the page draws it: nodes by id, and the wires between them.

    The board *is* the query. There is no second representation of what the
    controls mean — evaluating this is how a route gets found, so a wire that is
    not plugged in is not a validation failure but a missing argument, and it
    says so in the same words the library would.
    """

    nodes: "dict[str, dict]"
    links: "list[dict]"

    @classmethod
    def parse(cls, text: str) -> "Board":
        raw = json.loads(text)
        nodes = {str(node["id"]): node for node in raw.get("nodes", [])}
        links = [dict(link) for link in raw.get("links", [])]
        return cls(nodes, links)

    def sources(self, node_id: str, port: str) -> "list[str]":
        """Ids feeding one input port, in the order they were wired."""
        return [
            link["from"]
            for link in self.links
            if link.get("to") == node_id and link.get("port") == port
        ]

    def only(self, kind: str) -> "str | None":
        """The one node of a kind, if there is exactly one. How the query is found."""
        found = [
            node_id
            for node_id, node in self.nodes.items()
            if NODES.get(node["type"], {}).get("kind") == kind
        ]
        return found[0] if len(found) == 1 else None


class Unwired(ValueError):
    """A node whose input has nothing plugged into it.

    Carries the node so the board can point at it. Everything else that can go
    wrong — a technique refusing a cost model, a heuristic with no positions —
    is the library's own error, raised in the library's own words, and this
    exists only because "nothing is connected" is a question the library never
    gets asked.
    """

    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id


class Router:
    """The files this demo was given, and everything built from them so far.

    Values are cached by **signature** — a canonical spelling of a node and
    everything upstream of it — rather than by which node drew it. Rewiring a
    graph back to a shape you had before therefore costs nothing, which matters
    because the expensive things here are exactly the ones you want to compare:
    reading Seattle is six seconds, sixteen landmarks a second and 33 MB, a
    contraction hierarchy six seconds and 19 MB.
    """

    def __init__(
        self, path: Path, feed: "Path | None" = None, date: "datetime.date | None" = None
    ):
        self.path = path
        self.feed = feed
        self.date = date
        self._built: "dict[str, object]" = {}
        self._reachable: "dict[tuple[str, object], tuple]" = {}

    @property
    def catalogue(self) -> dict:
        """What this demo can offer, for the board to build its menu from.

        The page knows what the node types look like; only the server knows
        which of them have a file behind them today.
        """
        return {
            "extract": self.path.name,
            "profiles": sorted(PROFILES),
            "feed": self.feed.name if self.feed else None,
            "date": self.date.isoformat() if self.date else None,
        }

    def signature(self, board: Board, node_id: str) -> str:
        """A canonical spelling of a node and everything upstream of it.

        Two graphs that would build the same object spell the same, so the cache
        hits across rewiring, across page reloads, and across which node happens
        to be selected. It reads like the call it stands for, which is also what
        makes a cache miss legible in the log.
        """
        node = board.nodes[node_id]
        kind = node["type"]
        params = node.get("params", {})
        shown = ", ".join(f"{key}={params[key]!r}" for key in sorted(params))
        wired = ", ".join(
            f"{port}=[{', '.join(self.signature(board, s) for s in board.sources(node_id, port))}]"
            for port in sorted(NODES[kind]["inputs"])
        )
        inner = ", ".join(part for part in (shown, wired) if part)
        return f"{kind}({inner})"

    def build(self, board: Board, node_id: str) -> object:
        """Evaluate a node, and everything it depends on, once.

        The recursion *is* the DSL: an environment is built from its layers, a
        technique binds to an environment, and each step is the same call
        somebody would write by hand.
        """
        signature = self.signature(board, node_id)
        if signature in self._built:
            return self._built[signature]

        node = board.nodes[node_id]
        kind = node["type"]
        params = node.get("params", {})

        def wired(port: str) -> list:
            return [self.build(board, source) for source in board.sources(node_id, port)]

        def one(port: str) -> object:
            values = wired(port)
            if not values:
                raise Unwired(
                    node_id, f"{kind} has nothing plugged into its {port} input"
                )
            return values[0]

        started = time.perf_counter()
        if kind == "OSM":
            value: object = rl.OSM(self.path, PROFILES[params.get("profile", "driving")]())
        elif kind == "GTFS":
            if self.feed is None:
                raise Unwired(
                    node_id,
                    "this demo was started without a feed — pass --gtfs FEED "
                    "--date YYYY-MM-DD to route a timetable",
                )
            value = rl.GTFS(self.feed, self.date)
        elif kind == "Environment":
            layers = wired("layers")
            if not layers:
                raise Unwired(node_id, "Environment has no layers plugged into it")
            value = rl.Environment(*layers)
            value.compile()
        elif kind == "Euclidean":
            value = rl.Euclidean()
        elif kind == "Landmarks":
            value = rl.Landmarks(int(params.get("count", 16)))
        elif kind == "EdgeDifference":
            value = rl.EdgeDifference()
        elif kind == "RandomOrder":
            value = rl.RandomOrder(seed=int(params.get("seed", 0)))
        elif kind == "Query":
            raise Unwired(node_id, "a query is asked, not built")
        else:
            # A technique node. Configuring and binding are one step here
            # because the board has both halves in hand — but they are still the
            # two calls the library makes, and the signature says so.
            technique = self._technique(kind, params, one)
            environment = one("environment")
            value = technique.bind(environment)

        elapsed = time.perf_counter() - started
        if elapsed > 0.1:
            print(f"  {signature}: ready in {elapsed:.1f}s", flush=True)
        self._built[signature] = value
        return value

    @staticmethod
    def _technique(kind: str, params: dict, one) -> rl.Planner:
        """The unbound technique a node stands for."""
        if kind == "Dijkstra":
            return rl.Dijkstra()
        if kind == "BFS":
            return rl.BFS()
        if kind == "AStar":
            return rl.AStar(one("heuristic"))
        if kind == "ContractionHierarchy":
            return rl.ContractionHierarchy(one("ordering"))
        if kind == "TimeDependentDijkstra":
            return rl.TimeDependentDijkstra(params.get("waiting", "unrestricted"))
        if kind == "TimeDependent":
            return rl.TimeDependent()
        if kind == "TimeExpanded":
            return rl.TimeExpanded()
        raise ValueError(f"no such node type: {kind}")

    @staticmethod
    def _snappable(planner: rl.Planner):
        """The layer a click resolves against.

        The first one that can answer "what is nearest here", which with the
        usual one-layer environment is the only one. An environment of several
        would want a rule about which wins; there is no such environment on this
        board yet, and inventing the rule before the case exists would be
        guessing at it.
        """
        for source in planner.environment.sources:
            if hasattr(source, "nearest"):
                return source
        raise ValueError("no layer here can say which node is nearest a click")

    def route(
        self,
        board: Board,
        origin,
        destination,
        branches: "int | None" = None,
        explore: bool = True,
    ) -> dict:
        """Evaluate the board and ask its query.

        `explore` is what makes dragging an endpoint feel live. The route is a
        few hundred points; the search space behind it is up to sixty thousand
        branches and ten megabytes of GeoJSON, which is worth building once when
        the drag stops and not sixty times a second while it is moving.
        """
        query = board.only("query")
        if query is None:
            return {"error": "the board needs exactly one Query node"}
        planners = board.sources(query, "planner")
        if not planners:
            return {
                "error": "nothing is plugged into the query",
                "node": query,
            }
        planner = self.build(board, planners[0])
        params = board.nodes[query].get("params", {})

        layer = self._snappable(planner)
        start = layer.nearest(*origin)
        end = layer.nearest(*destination)

        # Only the clock-reading techniques understand a departure, and the
        # others rightly refuse one, so it goes to whoever asked for it. The two
        # clocks are not the same: a timetable runs on the service day fixed
        # when its feed was loaded, and a street schedule repeats weekly.
        minute = int(params.get("minute", 480))
        if not wants_time(planner):
            when = {}
        elif rides_transit(planner):
            when = {"departing": minute * 60}
        else:
            when = {"departing": int(params.get("day", 0)) * 86400 + minute * 60}

        if rides_transit(planner):
            return self._ride(planner, layer, start, end, when)
        return self._search(planner, layer, start, end, when, branches, explore)

    def _search(self, planner, layer, start, end, when, branches, explore) -> dict:
        """A static or time-dependent search over a network, and its search space."""
        compiled = planner.compiled
        target = planner.node_id(end)

        # Snap plainly, and work out what is actually connected only if that
        # turns out to have failed. Restricting the snap up front costs a
        # quarter of a second per request — more than any of these algorithms
        # spends routing, and the same quarter-second for all of them, which
        # would flatten the difference this demo exists to show. Paying it on
        # the rare failure keeps a drag answering at the speed of the search.
        began = time.perf_counter()
        result = planner.search(start, targets=[target], **when)
        elapsed = (time.perf_counter() - began) * 1000

        if result.cost(target) is None:
            end = layer.nearest(end, within=self.reachable(planner, start))
            target = planner.node_id(end)
            began = time.perf_counter()
            result = planner.search(start, targets=[target], **when)
            elapsed = (time.perf_counter() - began) * 1000

        if result.cost(target) is None:
            return {"error": "no route between those points"}
        # Built from the result already in hand rather than by asking the
        # planner to route again — the same query twice is the one thing a drag
        # cannot afford.
        journey = rl.Journey.from_result(compiled, result, end)

        coordinates = layer.coordinates()
        answer = {
            "route": journey.geometry,
            "snapped": [coordinates[start], coordinates[end]],
            "seconds": journey.cost,
            "waiting": journey.waiting,
            "legs": len(journey.legs),
            # How many edges this profile has a schedule for, and whether the
            # chosen technique is reading it. A schedule quietly ignored is the
            # failure nobody can see, so the page is told and says so.
            "scheduled_edges": 0 if compiled.calendar is None else len(compiled.calendar),
            "reads_clock": bool(when),
            # How many legs of *this* route are scheduled. The number that
            # answers "why did changing the hour do nothing": a route over
            # edges nobody scheduled cannot care what time it is.
            "scheduled_legs": 0
            if compiled.calendar is None
            else sum(1 for leg in journey.legs if compiled.calendar.is_restricted(leg.edge)),
            "settled_count": result.settled,
            "graph_nodes": compiled.graph.num_nodes,
            "ms": round(elapsed, 1),
        }
        if explore:
            # The search space, as the planner reports it. Every branch is drawn
            # by default — keeping only the heaviest keeps the trunk and throws
            # away the crown, which is exactly the part that shows how far the
            # search reached. A city-wide Dijkstra is ~58k branches and about
            # 10 MB of GeoJSON, which localhost and a canvas renderer both take
            # in stride.
            space = planner.explored(result)
            answer["tree"] = space.geojson(limit=branches)
            answer["peak"] = space.peak
            answer["branch_count"] = len(space)
        return answer

    def reachable(self, planner: rl.Planner, origin) -> tuple:
        """Labels reachable from `origin` — what a destination may snap to.

        Extracts are full of stubs that connect to nothing under a given
        profile, and snapping a click to one produces "no route" for reasons
        that have nothing to do with routing.

        A quarter of a second on a city: a whole Dijkstra, and a label for every
        node it settled. Which is why nothing calls this until a route has
        already failed — see `_search`.
        """
        environment = planner.environment
        key = (str(id(environment)), origin)
        if key not in self._reachable:
            compiled = environment.compile()
            result = rl.Dijkstra().bind(environment).search(origin)
            self._reachable = dict(list(self._reachable.items())[-7:])
            self._reachable[key] = tuple(compiled.label(node) for node in result.order)
        return self._reachable[key]

    @staticmethod
    def _ride(planner, layer, start, end, when) -> dict:
        """A timetable query, which answers with an itinerary.

        Short, and short for a reason: there is no cost per node to hand back,
        so no search space to draw and no result to rebuild a journey from.
        `planner.route` twice is the thing `_search` goes out of its way to
        avoid; here it is the only call there is.
        """
        began = time.perf_counter()
        journey = planner.route(start, end, **when)
        elapsed = (time.perf_counter() - began) * 1000

        coordinates = layer.coordinates()
        if journey is None:
            return {
                "error": f"no journey from {layer.names()[start]} at that hour",
                "snapped": [coordinates[start], coordinates[end]],
            }

        # No `shapes.txt` yet, so a leg has no geometry and the route is drawn
        # stop to stop. Straight lines, and honestly so — a bus does not fly,
        # but neither does this claim to know which streets it took.
        return {
            "route": [coordinates[label] for label in journey.nodes],
            "snapped": [coordinates[start], coordinates[end]],
            "seconds": journey.cost,
            "waiting": journey.waiting,
            "legs": len(journey.legs),
            "transfers": journey.transfers,
            "arrives": journey.arrives,
            "settled_count": journey.settled,
            # The model's own node set, not the environment's: the time-expanded
            # model settles events, and reporting its share against the stop
            # count would put it over 100%.
            "graph_nodes": getattr(planner, "num_events", None)
            or planner.compiled.graph.num_nodes,
            "nodes_are": "events" if hasattr(planner, "num_events") else "stops",
            "ms": round(elapsed, 1),
            "transit": True,
        }


class Handler(BaseHTTPRequestHandler):
    router: Router
    center: "tuple[float, float]"

    #: The page and the board's own code, read from disk beside this file. The
    #: editor is several hundred lines of JavaScript, which is a thing to keep
    #: in a `.js` file where it can be read and linted rather than in a Python
    #: string where it cannot.
    STATIC = Path(__file__).resolve().parent / "static"
    TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        url = urlparse(self.path)
        if url.path == "/":
            self.respond(200, "text/html; charset=utf-8", self.page().encode("utf-8"))
        elif url.path == "/route":
            self.respond(200, "application/json", self.route(parse_qs(url.query)))
        elif url.path.startswith("/static/"):
            self.static(url.path[len("/static/") :])
        else:
            self.respond(404, "text/plain", b"not found")

    def static(self, name: str) -> None:
        """Serve one file from `static/`, and nothing outside it."""
        path = (self.STATIC / name).resolve()
        if not path.is_file() or self.STATIC not in path.parents:
            self.respond(404, "text/plain", b"not found")
            return
        kind = self.TYPES.get(path.suffix, "application/octet-stream")
        self.respond(200, f"{kind}; charset=utf-8", path.read_bytes())

    def route(self, query: "dict[str, list[str]]") -> bytes:
        def point(name: str) -> "tuple[float, float]":
            lat, _, lon = query[name][0].partition(",")
            return float(lat), float(lon)

        try:
            branches = query.get("branches", [""])[0]
            payload = self.router.route(
                Board.parse(query["board"][0]),
                point("from"),
                point("to"),
                int(branches) if branches else None,
                explore=query.get("explore", ["1"])[0] != "0",
            )
        except Unwired as error:
            # The board can point at the node that is missing an argument.
            payload = {"error": str(error), "node": error.node_id}
        except (KeyError, TypeError, ValueError) as error:
            # Everything else is the library refusing the graph in its own
            # words — a technique that cannot route a cost model, a heuristic
            # with no positions to work from — and those messages are better
            # than any this could write over them.
            payload = {"error": str(error)}
        # Compact separators: the payload is mostly numbers, and the default
        # ", " / ": " padding is about a tenth of ten megabytes.
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def page(self) -> str:
        return (self.STATIC / "index.html").read_text().replace(
            "__SETUP__",
            json.dumps({"center": list(self.center), **self.router.catalogue}),
        )

    def respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Quieter than the default, which logs every tile-less request."""
        if "/route" in str(args[0] if args else ""):
            print(f"  {str(args[0])[:110]}", flush=True)


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
