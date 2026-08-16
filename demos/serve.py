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
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

import routelab as rl
from routelab import _routelab

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}

#: How many built things to keep, and how many reachable-label sets.
#:
#: A cap rather than none. The cache is keyed by signature, and
#: `Landmarks(count=…)` and `RandomOrder(seed=…)` are each a whole family of
#: keys worth tens of megabytes apiece — nudge a spinbox through a dozen values
#: and an uncapped cache holds every one of them for the life of the process.
#: Eight is enough that rewiring back to a shape you had a moment ago is free,
#: which is the only promise being made.
REMEMBERED = 8

#: "No value passed", for a cache that may legitimately hold `None`.
_UNSET = object()

#: What each node type builds, and what has to be plugged into it first.
#:
#: This is the whole vocabulary of the board, and it is deliberately a
#: transcription of the library's own three steps rather than a menu of
#: features: layers make an environment, a technique binds to one and becomes a
#: planner, a planner takes a query. Adding a node is adding an entry here and
#: its shape in `static/board.js` — nothing between the two knows what the nodes
#: mean.
#:
#: `inputs` maps a port name to the kind of value it accepts. `kind` is what a
#: node is for, which decides how it is built; the two together are all the
#: wiring rules there are, and a port takes what it says it takes.
#:
#: What is *not* a port is as deliberate as what is. A thing is a wire iff it
#: is a choice: a heuristic or an ordering has alternatives and knobs, so it is
#: wired in. The calendar a `TimeDependentDijkstra` reads, or the coordinates
#: `Euclidean` prices, have one possible construction from the layers already
#: plugged into the environment, so the technique derives them at bind and no
#: wire is drawn — a wire that could not change anything would be decoration.
#:
#: Shipped to the page as-is (see `Router.catalogue`), which is why it is data
#: rather than code: `static/board.js` adds titles, fields and colours to these
#: entries and never restates the wiring.
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
    # `clock` names the clock a technique's departure time is read on, for
    # the ones that take one: "week" repeats (a street schedule), "day" is the
    # service day a feed was read for. Absent for a technique that has no hour.
    "TimeDependentDijkstra": {
        "kind": "planner",
        "inputs": {"environment": "environment"},
        "clock": "week",
    },
    "TimeDependent": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "TimeExpanded": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "RAPTOR": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    # Walks between nearby stops, made from another layer's coordinates. A
    # layer that takes a layer: its output is edges like any other layer's,
    # and how far a rider will walk is the knob — which is why it is a node
    # and not something the environment does behind your back.
    "Footpaths": {"kind": "layer", "inputs": {"stops": "layer"}},
    # A query takes its endpoints from wires like everything else, so "where
    # from" is an argument rather than an ambient fact about the page. Its two
    # answers are separate outputs because they cost different amounts: the
    # route is a few hundred points, the search space up to ten megabytes.
    "Query": {
        "kind": "query",
        "inputs": {"planner": "planner", "origin": "point", "destination": "point"},
        "outputs": {"route": "route", "space": "space"},
    },
    # The map, as a node. A source of two points and a sink for two answers —
    # and not a cycle, because where the pins are owes nothing to what is drawn.
    "Map": {
        "kind": "map",
        "inputs": {"route": "route", "space": "space"},
        "outputs": {"origin": "point", "destination": "point"},
        # Its outputs owe nothing to its inputs, so its inputs are not
        # arguments and nothing upstream of it is upstream of anything. Which
        # is what stops a signature walking Map -> Query -> Map for ever.
        "terminal": True,
    },
}


#: What each technique node stands for in the library — the class whose
#: `options` say which query knobs the board should show under it. Declared
#: by the technique, read here; the board learns what a technique takes from
#: the library rather than from a second table.
TECHNIQUES: "dict[str, type]" = {
    "Dijkstra": rl.Dijkstra,
    "BFS": rl.BFS,
    "AStar": rl.AStar,
    "ContractionHierarchy": rl.ContractionHierarchy,
    "TimeDependentDijkstra": rl.TimeDependentDijkstra,
    "TimeDependent": rl.TimeDependent,
    "TimeExpanded": rl.TimeExpanded,
    "RAPTOR": rl.RAPTOR,
}
for _kind, _cls in TECHNIQUES.items():
    # `departing` is covered by `clock`; the rest are the knobs a Query node
    # shows when this technique is wired into it.
    NODES[_kind]["options"] = sorted(_cls.options - {"departing"})


def rides_transit(planner: rl.Planner) -> bool:
    """Does this technique route a timetable rather than a network?

    The one question that decides which world a query happens in — street
    corners or bus stops.
    """
    return isinstance(planner, rl.kernels.TimetablePlanner)


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
        return cls(nodes, raw.get("links", []))

    def sources(self, node_id: str, port: str) -> "list[tuple[str, str]]":
        """`(id, output port)` feeding one input, in the order they were wired."""
        return [
            (link["from"], link.get("fromPort", ""))
            for link in self.links
            if link.get("to") == node_id and link.get("toPort", link.get("port")) == port
        ]

    def listeners(self, node_id: str, port: str) -> "list[str]":
        """Ids wired to one *output* — who is listening for it, if anyone.

        Which is a question worth being able to ask: nothing is listening for a
        search space means nothing has to build one.
        """
        return [
            link["to"]
            for link in self.links
            if link.get("from") == node_id and link.get("fromPort") == port
        ]

    def upstream(self, node_id: str, found: "list[str] | None" = None) -> "list[str]":
        """`node_id` and everything it takes as an argument, transitively.

        The one place the argument-port rule is written: a terminal node has no
        arguments — what comes out of the map does not depend on what is drawn
        on it — which is both what stops Map -> Query -> Map recurring for ever
        and what `signature` and `settled` are agreeing about when they agree.
        """
        found = [] if found is None else found
        if node_id in found:
            return found
        found.append(node_id)
        kind = self.nodes[node_id]["type"]
        if not NODES[kind].get("terminal"):
            for port in NODES[kind]["inputs"]:
                for source, _ in self.sources(node_id, port):
                    self.upstream(source, found)
        return found

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
        self._built: "OrderedDict[str, object]" = OrderedDict()
        #: Reachable-label sets, keyed by the environment's *signature* rather
        #: than its identity. `id()` is reused after a collection, so an
        #: identity key can hand one environment another's labels; a signature
        #: says what the environment is.
        self._reachable: "OrderedDict[tuple[str, object], tuple]" = OrderedDict()
        #: Calendars derived from an environment, keyed by its signature. The
        #: page reports how many edges are scheduled under *every* technique —
        #: a schedule quietly ignored is the failure nobody can see — so the
        #: calendar has to be had without a clock-reading planner in hand. The
        #: library keeps no such table on the environment, deliberately; the
        #: board is the one consumer that rebinds, so the board remembers.
        self._calendars: "OrderedDict[str, object]" = OrderedDict()
        #: What is being built right now, by node id. Read by `/progress` from
        #: another request entirely, which is the whole reason the counter
        #: inside is an atomic rather than a callback.
        self._working: "dict[str, dict]" = {}
        #: Bumped for each build started, so a watcher can tell which of several
        #: nested builds is the innermost. Elapsed cannot: an `AStar`, the
        #: `Environment` it waits on and the `OSM` under that all start in the
        #: same instant.
        self._sequence = 0

    def working(self) -> dict:
        """What is under construction, and how far along.

        `fraction` is `None` for work with no honest measure of its own — a
        parser that yields no counts, a graph being compiled — and those report
        elapsed seconds alone. Better than a bar that guesses: a progress bar
        which lies once is a progress bar nobody reads again.

        `done` and `total` come too, and they are not redundant. Contraction is
        wildly non-linear in time: the last half-percent of a walking network's
        nodes are its most connected, and contracting them takes a third of the
        wall clock with the percentage pinned at 100. A count that keeps
        climbing is what actually answers "is this hung", which is the question
        any of this was added for.
        """
        now = time.perf_counter()
        report = {}
        for node_id, job in list(self._working.items()):
            phase, done, total = job["progress"].read()
            report[node_id] = {
                "what": job["what"],
                "seq": job["seq"],
                "elapsed": round(now - job["since"], 1),
                "phase": phase,
                "done": done,
                "total": total,
                "fraction": done / total if total else None,
            }
        return report

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
            # The wiring rules themselves, so the page does not keep a second
            # copy of them. What a port accepts, and what depends on what, is
            # one fact — and the board's whole claim is that what it draws is
            # what the server does, which a table that could drift would quietly
            # stop being true. The page adds only how they look.
            "nodes": NODES,
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
        # Only ports that are arguments — the same rule `Board.upstream` walks,
        # for the same reason: a node with more than one output is never
        # upstream of anything, so which socket a wire left by cannot change
        # what gets built, and a terminal node has no arguments at all.
        wired = "" if NODES[kind].get("terminal") else ", ".join(
            f"{port}=[{', '.join(self.signature(board, s) for s, _ in board.sources(node_id, port))}]"
            for port in sorted(NODES[kind]["inputs"])
        )
        inner = ", ".join(part for part in (shown, wired) if part)
        return f"{kind}({inner})"

    def settled(self, board: Board, node_id: str) -> "dict[str, str]":
        """Ids at or upstream of `node_id` whose value is in hand, and its spelling.

        Deliberately a walk of the graph and **not** a record of what a build
        happened to touch. A cache hit short-circuits the recursion, so a
        request whose planner was already built would visit only the planner —
        and the board, told that, would leave every node under it looking as
        though it had never been built and spin them on the next slow query.
        That is the bug this exists to have fixed.
        """
        return {
            node_id: self.signature(board, node_id)
            for node_id in board.upstream(node_id)
            if self.signature(board, node_id) in self._built
        }

    def build(self, board: Board, node_id: str) -> object:
        """Evaluate a node, and everything it depends on, once.

        The recursion *is* the DSL: an environment is built from its layers, a
        technique binds to an environment, and each step is the same call
        somebody would write by hand.
        """
        signature = self.signature(board, node_id)
        if signature in self._built:
            return self._remember(self._built, signature)

        node = board.nodes[node_id]
        kind = node["type"]
        params = node.get("params", {})

        def wired(port: str) -> list:
            return [self.build(board, source) for source, _ in board.sources(node_id, port)]

        def one(port: str) -> object:
            values = wired(port)
            if not values:
                raise Unwired(
                    node_id, f"{kind} has nothing plugged into its {port} input"
                )
            return values[0]

        # A counter for whoever asked, whether or not this kind of work has
        # anything to write into it. Handing one to everything and letting
        # `fraction` come back `None` beats keeping a list here of which
        # techniques report — the list would be one release out of date.
        progress = _routelab.Progress()
        self._sequence += 1
        self._working[node_id] = {
            "since": time.perf_counter(),
            "progress": progress,
            "what": kind,
            "seq": self._sequence,
        }
        started = time.perf_counter()
        try:
            value = self._make(node_id, kind, params, one, wired, progress)
        finally:
            self._working.pop(node_id, None)

        elapsed = time.perf_counter() - started
        if elapsed > 0.1:
            print(f"  {signature}: ready in {elapsed:.1f}s", flush=True)
        return self._remember(self._built, signature, value)

    @staticmethod
    def _remember(cache: "OrderedDict", key, value=_UNSET):
        """Note `key` as freshly used in `cache` — storing `value` if given —
        and forget the least recently used beyond `REMEMBERED`."""
        if value is not _UNSET:
            cache[key] = value
        cache.move_to_end(key)
        while len(cache) > REMEMBERED:
            cache.popitem(last=False)
        return cache[key]

    def _make(self, node_id, kind, params, one, wired, progress) -> object:
        """Build one node, having established what it depends on."""
        if kind == "OSM":
            # Read here rather than left to whoever first asks. Both layers
            # are lazy, which is right for a library and wrong for a board: the
            # six seconds would be charged to the `Environment` that happened
            # to touch it first, and the node naming the file would never so
            # much as blink.
            value: object = rl.OSM(self.path, PROFILES[params.get("profile", "driving")]()).load()
        elif kind == "GTFS":
            if self.feed is None:
                raise Unwired(
                    node_id,
                    "this demo was started without a feed — pass --gtfs FEED "
                    "--date YYYY-MM-DD to route a timetable",
                )
            value = rl.GTFS(self.feed, self.date).load()
        elif kind == "Footpaths":
            value = rl.Footpaths(one("stops"), within=float(params.get("within", 200))).load()
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
            value = technique.bind(environment, progress=progress)
        return value

    @staticmethod
    def _technique(kind: str, params: dict, one) -> rl.Planner:
        """The unbound technique a node stands for.

        Only the ones that take something get a line: a wire, or a field. The
        rest are the class in `TECHNIQUES` called with nothing, which is what
        a technique with no choices to make is.
        """
        if kind == "AStar":
            return rl.AStar(one("heuristic"))
        if kind == "ContractionHierarchy":
            return rl.ContractionHierarchy(one("ordering"))
        if kind == "TimeDependentDijkstra":
            return rl.TimeDependentDijkstra(params.get("waiting", "unrestricted"))
        try:
            return TECHNIQUES[kind]()
        except KeyError:
            raise ValueError(f"no such node type: {kind}") from None

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
        explore: bool = True,
    ) -> dict:
        """Evaluate the board and ask its query.

        `explore` is what makes dragging an endpoint feel live. The route is a
        few hundred points; the search space behind it is up to sixty thousand
        branches and ten megabytes of GeoJSON, which is worth building once when
        the drag stops and not sixty times a second while it is moving. It is a
        second answer for a second reason too: nobody wired to the query's
        `space` output is nobody who needs one built, and the wire says so.
        """
        query = board.only("query")
        if query is None:
            return {"error": "the board needs exactly one Query node"}
        planners = board.sources(query, "planner")
        if not planners:
            return {"error": "nothing is plugged into the query", "node": query}
        try:
            planner = self.build(board, planners[0][0])
        except (Unwired, KeyError, TypeError, ValueError) as error:
            # A failure is an answer like any other here, and it says the same
            # things: the library's own sentence, the node to point at, and
            # what did get built. That last matters — a graph that fails at its
            # last node still built everything before it, and a board that kept
            # those spinning would claim work that is already done.
            return {
                "error": str(error),
                "built": self.settled(board, planners[0][0]),
                **({"node": error.node_id} if isinstance(error, Unwired) else {}),
            }
        params = board.nodes[query].get("params", {})

        # Where from and where to are arguments now, wired in like everything
        # else — which means they can be crossed, and crossing them really does
        # reverse the trip.
        pins = {"origin": origin, "destination": destination}
        ends = []
        for port in ("origin", "destination"):
            wires = board.sources(query, port)
            if not wires:
                return {
                    "error": f"the query has no {port} plugged into it — wire one "
                             f"in from a Map",
                    "node": query,
                }
            ends.append(pins.get(wires[0][1], origin))

        layer = self._snappable(planner)
        start = layer.nearest(*ends[0])
        end = layer.nearest(*ends[1])
        explore = explore and bool(board.listeners(query, "space"))
        drawn = bool(board.listeners(query, "route"))

        # Only the clock-reading techniques understand a departure, and the
        # others rightly refuse one, so it goes to whoever asked for it. The two
        # clocks are not the same: a timetable runs on the service day fixed
        # when its feed was loaded, and a street schedule repeats weekly.
        minute = int(params.get("minute", 480))
        spec = NODES[board.nodes[planners[0][0]]["type"]]
        clock = spec.get("clock")
        if clock is None:
            when = {}
        elif clock == "day":
            when = {"departing": minute * 60}
        else:
            when = {"departing": int(params.get("day", 0)) * 86400 + minute * 60}
        # The other knobs the wired technique takes, from the Query node —
        # blank means "not set", the technique's own default.
        for option in spec.get("options", ()):
            value = params.get(option)
            if value not in (None, ""):
                when[option] = int(value)

        answer = self._ask(board, planners[0][0], layer, start, end, when, explore)
        # Nothing is listening for the route, so there is nothing to draw. Said
        # rather than silently drawn anyway: a wire that changes nothing is not
        # a wire, it is decoration.
        answer["drawn"] = drawn
        answer["built"] = self.settled(board, planners[0][0])
        return answer

    def _ask(self, board, planner_id, layer, start, end, when, explore) -> dict:
        """Ask the planner, and answer in one vocabulary whatever it is.

        Every technique is asked the same way, and asked once: `search` for the
        cost table, `journey` to read the answer off it, `journeys` for the
        rest of a front if it keeps one, `explored` for the space behind it. A
        technique that keeps no table says so in the library's own words and is
        asked for its journey instead; one that keeps no space says so too, and
        that sentence is what the page shows where a space would have gone.
        """
        planner = self.build(board, planner_id)
        compiled = planner.compiled
        coordinates = layer.coordinates()
        target = planner.node_id(end)

        def ask(target):
            began = time.perf_counter()
            try:
                result = planner.search(start, targets=[target], **when)
                journey = planner.journey(result, planner.label(target))
                settled = result.settled
            except NotImplementedError:
                # A technique that keeps no cost table: ask it for the journey
                # it does answer with. Its refusal for a search space comes
                # later, from `explored`, and only if a space is wanted.
                result = None
                journey = planner.route(start, planner.label(target), **when)
                settled = journey.settled if journey is not None else 0
            return result, journey, settled, (time.perf_counter() - began) * 1000

        # Snap plainly, and work out what is actually connected only if that
        # turns out to have failed. Restricting the snap up front costs a
        # quarter of a second per request — more than any of these algorithms
        # spends routing, and the same quarter-second for all of them, which
        # would flatten the difference this demo exists to show. Paying it on
        # the rare failure keeps a drag answering at the speed of the search.
        result, journey, settled, elapsed = ask(target)
        if journey is None and not rides_transit(planner):
            end = layer.nearest(end, within=self.reachable(board, planner_id, start))
            result, journey, settled, elapsed = ask(planner.node_id(end))
        if journey is None:
            error = (
                f"no journey from {layer.names()[start]} at that hour"
                if rides_transit(planner)
                else "no route between those points"
            )
            return {"error": error, "snapped": [coordinates[start], coordinates[end]]}

        # Drawn in pieces where the legs are scheduled: a piece per vehicle
        # boarded and per walk between them, so the page can tell a change of
        # bus from a stop passed through, and a walk across the street from
        # either. A street journey draws as one line of real road.
        segments = None
        if any(leg.trip is not None for leg in journey.legs):
            segments = []
            for leg in journey.legs:
                if segments and segments[-1]["trip"] == leg.trip:
                    segments[-1]["points"].append(coordinates[leg.head])
                else:
                    segments.append({
                        "trip": leg.trip,
                        "walk": leg.trip is None,
                        "points": [coordinates[leg.tail], coordinates[leg.head]],
                    })

        # The search space, as the planner reports it, if anything is
        # listening for one. Every branch is drawn by default — keeping only
        # the heaviest keeps the trunk and throws away the crown, which is
        # exactly the part that shows how far the search reached. A technique
        # that keeps none refuses in its own words, and those are what the
        # page shows where a space would have gone.
        space = space_kind = space_size = note = None
        if explore:
            try:
                explored = planner.explored(result)
                space, space_kind, space_size = explored.geojson(), explored.kind, len(explored)
            except NotImplementedError as nothing_to_draw:
                note = str(nothing_to_draw)

        calendar = self.calendar(board, planner_id)
        noun, of = planner.searches
        # Read off the search already in hand rather than asking for another:
        # a second front costs a second full set of rounds, and a drag cannot
        # afford the query twice.
        frontier = None
        if result is not None and hasattr(planner, "journeys"):
            frontier = [
                {"arrives": alternative.arrives, "transfers": alternative.transfers}
                for alternative in planner.journeys(result, end)
            ]
        return {
            # No `shapes.txt` yet, so a transit leg has no geometry and the
            # route is drawn stop to stop. Straight lines, and honestly so — a
            # bus does not fly, but neither does this claim to know its streets.
            "route": journey.geometry or [coordinates[label] for label in journey.nodes],
            "segments": segments,
            "snapped": [coordinates[start], coordinates[end]],
            "seconds": journey.cost,
            "waiting": journey.waiting,
            "walked": journey.walking,
            "legs": len(journey.legs),
            "transfers": journey.transfers,
            "arrives": journey.arrives,
            "clock": NODES[board.nodes[planner_id]["type"]].get("clock"),
            "settled": settled,
            "searches": noun,
            "of": of,
            "ms": round(elapsed, 1),
            "space": space,
            "space_kind": space_kind,
            "space_size": space_size,
            "space_note": note,
            "rounds": getattr(result, "rounds", None),
            "frontier": frontier,
            # How many edges this profile has a schedule for, and whether the
            # chosen technique is reading it. A schedule quietly ignored is the
            # failure nobody can see, so the page is told and says so — and
            # told which technique here would read it.
            "scheduled_edges": 0 if calendar is None else len(calendar),
            "reads_clock": bool(when.get("departing") is not None),
            "scheduled_legs": 0
            if calendar is None
            else sum(1 for leg in journey.legs if calendar.is_restricted(leg.edge)),
            "clock_reader": rl.clock_readers(compiled),
        }

    def calendar(self, board: Board, planner_id: str) -> "_routelab.Calendar | None":
        """The schedule under `planner_id`'s environment, or `None` if it has none.

        Derived once per environment and remembered, capped like `_reachable`;
        one walk of the layers, refused or not.
        """
        planner = self.build(board, planner_id)
        environment_id = board.sources(planner_id, "environment")[0][0]
        key = self.signature(board, environment_id)
        if key in self._calendars:
            return self._remember(self._calendars, key)
        try:
            calendar = rl.Schedule().bind(planner.compiled)
        except ValueError:
            calendar = None
        return self._remember(self._calendars, key, calendar)

    def reachable(self, board: Board, planner_id: str, origin) -> tuple:
        """Labels reachable from `origin` — what a destination may snap to.

        Extracts are full of stubs that connect to nothing under a given
        profile, and snapping a click to one produces "no route" for reasons
        that have nothing to do with routing.

        A quarter of a second on a city: a whole Dijkstra, and a label for every
        node it settled. Which is why nothing calls this until a route has
        already failed — see `_search`.
        """
        environment = self.build(board, planner_id).environment
        key = (self.signature(board, planner_id), origin)
        if key in self._reachable:
            return self._remember(self._reachable, key)
        compiled = environment.compile()
        result = rl.Dijkstra().bind(environment).search(origin)
        # `labels` is a list, so index it rather than calling `label` half a
        # million times.
        labels = compiled.labels
        return self._remember(
            self._reachable, key, tuple(labels[node] for node in result.order)
        )


class Handler(BaseHTTPRequestHandler):
    router: Router
    center: "tuple[float, float]"

    #: Keep-alive. Every response here sets `Content-Length`, so this is safe,
    #: and a drag makes tens of requests a second — each one otherwise a fresh
    #: connection and a fresh thread.
    protocol_version = "HTTP/1.1"

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
        elif url.path == "/progress":
            # Answered while another request is mid-build, which is the only
            # reason it can say anything. `ThreadingHTTPServer`, an atomic
            # counter, and no lock between them.
            self.respond(
                200,
                "application/json",
                json.dumps(self.router.working(), separators=(",", ":")).encode("utf-8"),
                cache=False,
            )
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
        # Never cached. This is a demo you are meant to edit, and a stale
        # board.js served out of the browser's cache is an afternoon.
        self.respond(200, f"{kind}; charset=utf-8", path.read_bytes(), cache=False)

    def route(self, query: "dict[str, list[str]]") -> bytes:
        def point(name: str) -> "tuple[float, float]":
            lat, _, lon = query[name][0].partition(",")
            return float(lat), float(lon)

        try:
            payload = self.router.route(
                Board.parse(query["board"][0]),
                point("from"),
                point("to"),
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

    def respond(
        self, status: int, content_type: str, body: bytes, cache: bool = True
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
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
