"""Baum, Buchhold, Sauer, Wagner & Zündorf, *UnLimited TRAnsfers for
multi-modal route planning* (2019) — ULTRA."""

from __future__ import annotations

import copy
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

from .. import _routelab
from ..model.environment import CompiledEnvironment, Environment
from ..model.journey import Journey, Leg, _edge_between, _leg
from ..model.search import Result
from ..model.searchspace import SearchSpace
from ..util.clock import service_seconds
from .departures import Departures
from .planner import Origins, Planner, TimetablePlanner, names, techniques

__all__ = ["ULTRA", "Transfers"]

#: What a vertex that is not a stop renumbers to — the kernel's own `NO_NODE`,
#: which `footpaths_renumbered` reads as "drop this end".
_NOT_A_STOP = 0xFFFFFFFF


class Transfers:
    """The scalar edges between stops, as a graph rather than a closure.

        >>> Transfers()
        Transfers()

    The same edges :class:`~routelab.Walks` gathers, read the other way. A
    timetable technique needs its walks as *one-hop transfers*, so `Walks`
    closes them under composition — walk A→B and B→C and it writes A→C — and
    that closure is what forces a radius: two hundred metres of King County
    Metro closes to five times its own size, four hundred to a hundred and
    thirty times.

    ULTRA does not need the closure, so this does not build one. It hands back
    the links exactly as the layers gave them, as a graph a search can walk
    without limit: a long walk is a path of short hops rather than an edge
    somebody had to write down in advance.

    Each edge remembers the compiled edge it came from, in
    :attr:`Transfers.edges`, so a walk found here can be told in the caller's
    own edges — with the layer it came from and the shape it follows.
    """

    #: The word :meth:`missing_from` answers with.
    name = "transfers"

    def __init__(self, max_degree: float = 12.0):
        self.max_degree = float(max_degree)

    @staticmethod
    def _stops(compiled: CompiledEnvironment) -> "Set[Hashable]":
        """The labels a vehicle calls at."""
        stops: "Set[Hashable]" = set()
        for _, _, source in compiled.spans:
            if source.cost_model == "timetable":
                for tail, head, _ in source.edges():
                    stops.add(tail)
                    stops.add(head)
        return stops

    @staticmethod
    def _positions(compiled: CompiledEnvironment) -> "List[int]":
        """Input positions of every scalar edge.

        All of them, not only the ones between stops the way
        :meth:`~routelab.Walks.bind` takes: a street is a leg of a transfer
        even though neither of its ends is a stop, and the whole point of
        walking without a radius is that the long walks are paths.
        """
        found: "List[int]" = []
        for start, stop, source in compiled.spans:
            if source.cost_model == "scalar":
                found.extend(range(start, stop))
        return found

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Nothing, ever — the same answer :class:`~routelab.Walks` gives.

        An environment where nothing walks is the plain model, not a refusal:
        there is simply no intermediate transfer to work out, and the
        technique underneath runs as it would have anyway.
        """
        return frozenset()

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "Transfers":
        """Build the graph, keeping which compiled edge each of its edges was.

        ``progress`` is accepted for parity with every other ``bind`` and left
        alone: this is a filter over edges the environment already has.
        """
        positions = self._positions(compiled)
        by_input = {compiled.graph.input_index(e): e for e in range(compiled.graph.num_edges)}
        links: "List[Tuple[int, int, int]]" = []
        self.edges: "List[int]" = []
        for position in positions:
            edge = by_input[position]
            tail, head, weight = compiled.graph.edge(edge)
            links.append((tail, head, weight))
            self.edges.append(edge)
        self.graph = _routelab.Graph(len(compiled), links)

        # The core, and why there are two graphs. ULTRA's preprocessing
        # searches this once per departure event, so a city's pavements are out
        # of reach — Core-CH contracts every vertex that is not a stop and
        # leaves a graph a hundredth the size with the same distances between
        # stops, which is all an intermediate transfer ever runs between. The
        # full graph stays for the query, whose walks start and end wherever
        # the caller stood rather than at a stop.
        stops = self._stops(compiled)
        #: The vertices a vehicle calls at. Kept because a query has to ask
        #: every one of them where it would get off, and on a street network
        #: those are a few thousand against half a million corners.
        self.stops: "List[int]" = sorted(compiled.node_id(label) for label in stops)
        keep = self.stops
        streets = len(compiled) - len(stops)
        if streets > 0 and keep:
            self.contraction = _routelab.CoreHierarchy.build(
                self.graph, keep, self.max_degree, progress=progress
            )
            self.core = self.contraction.core
        else:
            # Nothing to contract: every vertex is already a stop, which is
            # what a footpath layer on its own gives.
            self.contraction = None
            self.core = self.graph
        return self

    def compiled_edge(self, edge: int) -> int:
        """The compiled environment's edge behind one of this graph's — which
        is what makes a walk found here tellable as legs.

        One direction only: every walk a query tells comes back from the
        hierarchy already pointing the way it is travelled, so the reversed
        copy this used to keep — a second graph the size of the pavement —
        went with the backward search that needed it.
        """
        return self.edges[self.graph.input_index(edge)]

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        inside = "" if self.max_degree == 12.0 else f"max_degree={self.max_degree:g}"
        return f"Transfers({inside})"


class ULTRA(TimetablePlanner):
    """Unlimited transfers, worked out once (Baum et al., 2019).

        ULTRA(RAPTOR()).bind(env).route(origin, target, departing=time(8, 30))

    Not a routing algorithm but a **preprocessing** one, so it wraps the
    technique that does the routing — the paper's own *ULTRA-Query family*.
    Every timetable technique here takes its walks as one-hop transfers between
    stops, closed under composition, which is why
    :class:`~routelab.Footpaths` has a radius: the closure is what does not
    scale, not the walking. ULTRA computes instead the few walks that some
    optimal journey actually needs — the *intermediate* transfers, between two
    vehicles — and those go into the wrapped technique unchanged. It walks
    without a radius and the technique underneath does not know.

    The walks at either end of a journey are not shortcuts and cannot be: one
    endpoint of each is the query's own source or target, which preprocessing
    has never seen. The paper's §4.1 answers them with two one-to-many searches
    at query time, accelerated by **Bucket-CH** over the transfer graph — which
    is the third thing :meth:`preprocess` builds, and why a query over a city's
    pavement costs milliseconds rather than the two half-million-vertex
    Dijkstras the same question asked directly would.

    Only a technique that keeps a label per stop can be wrapped, since the
    query has to read every stop's arrival to find the best place to get off.
    :class:`RAPTOR` and :class:`CSA` are those, and are the two the paper
    names; anything else is refused by name.
    """

    options = frozenset({"departing", "max_transfers"})

    def __init__(self, technique: "Optional[TimetablePlanner]" = None):
        self.technique = technique if technique is not None else _default()
        readers = _tabled()
        if type(self.technique) not in readers:
            raise TypeError(
                f"ULTRA works the transfers out for a technique that keeps a label "
                f"per stop, and {type(self.technique).__name__}() keeps none — its "
                f"query could not say where to get off. Wrap {names(readers)}."
            )

    def __repr__(self) -> str:
        return self._describe(repr(self.technique))

    def walks(self) -> Any:
        """The shortcuts, which is what the wrapped technique reads instead of
        a closure — see :meth:`TimetablePlanner.walks`.

        Renumbered into the transit network's own vertices, since that is what
        the technique underneath was bound to.
        """
        return _Shortcuts(self.shortcuts, self._slots, self._inner_stops)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Work the shortcuts out, then bind the technique onto them.

        The wrapped technique is bound to the **timetable layers alone**, not
        to the environment this was given. That is Algorithm 2's last line —
        the black box runs on the public transit network, not on the merged
        one — and it is the difference between a technique whose tables are a
        row per stop and one whose tables are a row per street corner. On
        Seattle's pavement with King County Metro that is 6,313 against
        560,706, nine rounds of it per query.

        The cost is a second numbering, and :attr:`_inner_of` is the seam.
        Nothing outside this class sees it: what goes in and comes out of
        :meth:`route` is the caller's own labels, as everywhere else.
        """
        compiled = self._bound()
        self.timetable = Departures().bind(compiled, progress)
        self.transfers = Transfers().bind(compiled, progress)
        self.shortcuts = _routelab.Ultra.compute(self.timetable, self.transfers.core, progress)
        # The third of the paper's three preprocessing steps (§4.1): a core
        # graph, the shortcuts computed over it, and Bucket-CH over the
        # *original* transfer graph, which is what answers the walks at either
        # end of a journey.
        self.buckets = _routelab.Buckets.build(
            self.transfers.graph, self.transfers.stops, progress
        )

        # The transit network on its own, and the two translations between its
        # numbering and the caller's.
        riding = [
            source for source in self.environment.sources
            if source.cost_model == "timetable"
        ]
        inner_environment = Environment(*riding)
        inner_compiled = inner_environment.compile()
        self._inner_of = {
            compiled.node_id(label): inner_compiled.node_id(label)
            for label in inner_compiled.labels
        }
        self._outer_of = {inner: outer for outer, inner in self._inner_of.items()}
        # Indexed by *this* graph's vertices, for renumbering the shortcuts:
        # NO_NODE wherever a vertex is a street corner rather than a stop.
        self._slots = [_NOT_A_STOP] * len(compiled)
        for outer, inner in self._inner_of.items():
            self._slots[outer] = inner
        self._inner_stops = len(inner_compiled)

        # The technique underneath reads the shortcuts rather than a closure,
        # which is the whole of the integration: `walks()` is the seam.
        inner = copy.copy(self.technique)
        inner.walks = self.walks  # type: ignore[method-assign]
        self.inner = inner.bind(inner_environment, progress)
        self.footpaths = self.inner.footpaths

    def _footprint(self) -> int:
        return self.inner.footprint + self.shortcuts.footprint + self.buckets.footprint

    @property
    def num_shortcuts(self) -> int:
        """Shortcuts kept: the transfer set a query walks, against the closure
        it stands in for."""
        self._bound()
        return self.shortcuts.num_shortcuts

    @property
    def searches(self) -> "Tuple[str, int]":
        return self.inner.searches

    # --- the query ----------------------------------------------------------

    def _walk_legs(self, path: "List[int]", at: int) -> "List[Leg]":
        """A walk through the transfer graph, told as the caller's own edges.

        A shortcut, and an access or egress walk, is a *path* of hops rather
        than one edge — so each hop becomes a leg, with the layer it came from
        and the shape it follows, the way a hierarchy unpacks its shortcuts
        before anyone sees them. ``path`` is what
        :meth:`~routelab._routelab.Buckets.path` unpacked, already in order.
        """
        compiled = self._bound()
        if not path:
            return []
        hops = [self.transfers.compiled_edge(edge) for edge in path]
        legs: "List[Leg]" = []
        clock = at
        for edge in hops:
            tail, head, weight = compiled.graph.edge(edge)
            source, position = compiled.locate(edge)
            legs.append(
                Leg(
                    tail=compiled.label(tail),
                    head=compiled.label(head),
                    weight=weight,
                    source=source,
                    edge=edge,
                    position=position,
                    departs=clock,
                    arrives=clock + weight,
                    trip=None,
                )
            )
            clock += weight
        return legs

    def _itinerary_legs(self, itinerary: Any) -> "List[Leg]":
        """The wrapped technique's answer, told in the caller's edges.

        A ride is one edge and comes back as one leg. A walk is a *shortcut* —
        a path through the transfer graph that no single edge stands for — so
        it is searched again and told hop by hop, which is the same obligation
        a contraction hierarchy meets by unpacking a shortcut before anyone
        sees it. There are a handful per journey, over a graph of stops.
        """
        compiled = self._bound()
        outer_of = self._outer_of
        legs: "List[Leg]" = []
        for trip, tail, head, departs, arrives in itinerary.legs():
            # The technique answers in the transit network's numbering; every
            # leg is told against the environment the caller handed in.
            tail, head = outer_of[tail], outer_of[head]
            if trip is not None:
                legs.append(
                    _leg(compiled, _edge_between(compiled, tail, head, "timetable"),
                         departs, arrives, trip)
                )
                continue
            found = self.buckets.path([(tail, 0)], head)
            legs.extend(self._walk_legs([] if found is None else found[1], departs))
        return legs

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        """Not this: ULTRA answers a query it has to bracket with two searches
        of its own, so a raw one-to-all result would be the wrapped
        technique's and not this one's."""
        raise NotImplementedError(
            f"{type(self).__name__} brackets its technique's search with a walk "
            f"out of the origin and a walk into the target, so there is no one "
            f"table to hand back. Use route(origin, destination, ...), or bind "
            f"{type(self.technique).__name__}() directly to search its stops."
        )

    def route(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> Optional[Journey]:
        """The earliest arrival at ``destination``, walking without a radius.

        The paper's Algorithm 2. One hierarchy query gives the direct walk
        from origin to destination and the two search spaces; scanning the
        buckets of those spaces gives every initial and final transfer worth
        having, which is every one that beats walking the whole way. The
        wrapped technique then runs on the stops that survived, and the best
        place to get off is read from the final transfers.
        """
        options = self._options(options)
        # The technique underneath has the last word on the knobs: this one
        # advertises what any of them takes, and only it knows which.
        self.inner._options(options)  # noqa: SLF001 - the seam this technique is
        compiled = self._bound()
        at = service_seconds(options["departing"])
        target = self.node_id(destination)
        starts = self._origin_ids(origin)

        # Algorithm 2, lines 1-3: the direct walk, and the initial and final
        # transfers that beat it. Two bucket scans over a hierarchy's search
        # space, not two searches of the network.
        found = self.buckets.endpoints(list(starts.items()), target)
        # Into the transit network's own numbering, which is what the wrapped
        # technique was bound to.
        inner_of = self._inner_of
        out = {inner_of[stop]: cost for stop, cost in found.to_stops if stop in inner_of}

        # A journey that boards nothing: the walk, which needs no vehicle and
        # so is the floor every ride has to beat — and, being the floor, is
        # also what the two scans above pruned themselves by.
        arrives = None if found.direct is None else at + found.direct
        aboard: "Optional[Any]" = None
        alight: "Optional[int]" = None

        if out:
            result = self.inner._search(out, **options)  # noqa: SLF001
            # Where to get off: any stop that walks to the target sooner than
            # the target can be walked to directly. That is the final
            # transfer, and it has nothing to do with where the journey
            # started walking.
            for stop, walk in found.from_stops:
                inner = inner_of.get(stop)
                reached = None if inner is None else result.cost(inner)
                if reached is None:
                    continue
                landed = reached + walk
                if arrives is None or landed < arrives:
                    # `alight` is kept in the caller's numbering: it is where
                    # the final walk starts, and that walk is the transfer
                    # graph's business rather than the timetable's.
                    arrives, aboard, alight = landed, result, stop

        if arrives is None:
            return None
        if aboard is None:
            source, legs = self._access(starts, at, target)
            return Journey(
                origin=compiled.label(source),
                destination=destination,
                cost=arrives - at,
                legs=tuple(legs),
                settled=0,
            )

        itinerary = aboard.itinerary(self._inner_of[alight])
        ridden = self._itinerary_legs(itinerary)
        # Where the riding began: the first leg's tail, or the alighting stop
        # itself when the technique found nothing to ride.
        boarded = compiled.node_id(ridden[0].tail) if ridden else alight
        source, legs = self._access(starts, at, boarded)
        legs += ridden
        egress = self.buckets.path([(alight, 0)], target)
        legs += self._walk_legs([] if egress is None else egress[1], itinerary.arrives)
        return Journey(
            origin=compiled.label(source),
            destination=destination,
            cost=arrives - at,
            legs=tuple(legs),
            settled=itinerary.settled,
        )

    def _access(
        self, starts: "Dict[int, int]", at: int, to: int
    ) -> "Tuple[int, List[Leg]]":
        """Where the walk that starts a journey left from, and its legs.

        A query may stand at several places, each already some seconds along,
        so the hierarchy is asked which one it was worth leaving from, and the
        walk departs at that origin's own head start.
        """
        found = self.buckets.path(list(starts.items()), to)
        if found is None:
            return to, []
        source, path = found
        return source, self._walk_legs(path, at + starts.get(source, 0))

    # No `explored`: ULTRA keeps no table of its own, since the search whose
    # space could be drawn is the wrapped technique's and is bracketed by two
    # walks that are this one's. The family's refusal already says exactly
    # that, and names the techniques that do keep one.


class _Shortcuts:
    """A derivation that hands back an already-computed transfer set.

    What :meth:`ULTRA.walks` returns, so the wrapped technique's `preprocess`
    reads the shortcuts through the same seam every other technique reads
    :class:`~routelab.Walks` through, and nothing about it has to change.
    """

    name = "walks"

    def __init__(self, shortcuts: Any, slots: "List[int]", stops: int):
        self.shortcuts = shortcuts
        self.slots = slots
        self.stops = stops

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        return frozenset()

    def bind(self, compiled: CompiledEnvironment, progress: Any = None) -> Any:
        return self.shortcuts.footpaths_renumbered(self.slots, self.stops)

    def __repr__(self) -> str:
        return "Shortcuts()"


def _tabled() -> "List[type]":
    """The timetable techniques that keep a label per stop.

    Read off the shelf rather than listed: a technique keeps a table if it
    reports a search space, which is the same test
    :meth:`TimetablePlanner.explored` uses to name who to ask.
    """
    return [
        cls
        for cls in techniques()
        if issubclass(cls, TimetablePlanner)
        and cls.explored is not TimetablePlanner.explored
        and cls._search is not Planner._search
    ]


def _default() -> "TimetablePlanner":
    """The technique ULTRA wraps when nobody says: the first on the shelf that
    keeps a table, so `ULTRA()` is a sentence rather than an error."""
    return _tabled()[0]()
