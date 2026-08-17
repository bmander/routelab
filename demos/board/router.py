"""Building what the board asks for, and asking it.

Holds the loaded files and caches what preprocessing produced, because binding
a technique to a city is seconds and a drag of the mouse is not.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from collections import OrderedDict

import routelab as rl
from routelab import _routelab

from .catalogue import NODES, PROFILES, TECHNIQUES
from .wiring import Board, Unwired

REMEMBERED = 8

#: "No value passed", for a cache that may legitimately hold `None`.
_UNSET = object()


def rides_transit(planner: rl.Planner) -> bool:
    """Does this technique route a timetable rather than a network?

    The one question that decides which world a query happens in — street
    corners or bus stops.
    """
    return isinstance(planner, rl.kernels.TimetablePlanner)


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
            value = self._make(
                node_id, kind, params, one, wired, progress, self._unbound(board, node_id)
            )
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

    def _make(self, node_id, kind, params, one, wired, progress, unbound) -> object:
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
        elif kind == "Access":
            value = rl.Access(
                one("stops"), one("streets"), within=float(params.get("within", 400))
            ).load()
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
            technique = self._technique(kind, params, one, unbound)
            environment = one("environment")
            value = technique.bind(environment, progress=progress)
        return value

    def _unbound(self, board: Board, node_id: str):
        """Resolve a wired *technique* port to a technique nobody has bound.

        A heuristic or an ordering arrives at its planner as a specification —
        `Euclidean()`, not a heuristic bound to anything — because that is what
        `AStar(...)` takes. A planner node's value, by contrast, is already
        bound, since the board has the environment in hand and builds in one
        step. A technique that wraps another wants the specification, so this
        reaches past the built value to the node and configures it again.
        """

        def resolve(port: str) -> rl.Planner:
            wires = board.sources(node_id, port)
            if not wires:
                raise Unwired(
                    node_id,
                    f"{board.nodes[node_id]['type']} has nothing plugged into "
                    f"its {port} input",
                )
            inner = board.nodes[wires[0][0]]
            return self._technique(
                inner["type"],
                inner.get("params", {}),
                lambda _port: self.build(board, wires[0][0]),
                self._unbound(board, wires[0][0]),
            )

        return resolve

    def _technique(self, kind: str, params: dict, one, unbound) -> rl.Planner:
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
        if kind == "ULTRA":
            return rl.ULTRA(unbound("technique"))
        try:
            return TECHNIQUES[kind]()
        except KeyError:
            raise ValueError(f"no such node type: {kind}") from None

    @staticmethod
    def _snappable(planner: rl.Planner):
        """The layer a click resolves against.

        The first one that can answer "what is nearest here". With a single
        layer that is the only one; with a feed and a street network both
        registered, it is whichever was wired into the environment first, and
        that is the rule rather than an accident — a multimodal board wants
        clicks to land on the pavement, so the streets go in first and a
        stop-to-stop board puts the feed there instead.
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
        # Every layer that places its own nodes, not just the one a click snaps
        # against: a multimodal journey walks a street, boards at a stop and
        # gets off at another, so the labels along it come from more than one
        # layer and a table built from one of them cannot draw it. Labels are
        # disjoint across layers — a stop id is not a street id — so this is a
        # merge and not a precedence.
        coordinates: "dict" = {}
        for source in planner.environment.sources:
            places = getattr(source, "coordinates", None)
            if places is not None:
                coordinates.update(places())
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
            # Named where the layer has names — a feed does, a street network
            # does not, and a multimodal environment snaps against whichever
            # was registered first.
            names = getattr(layer, "names", None)
            here = names().get(start, start) if names else start
            error = (
                f"no journey from {here} at that hour"
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
