"""Environments: the world you are routing over, assembled from layers.

An :class:`Environment` is a bag of registered layers — a street network, a
timetable, a bikeshare fleet — that knows how to compile itself into the dense
integer graph the kernels want. Two things live here that the kernel deliberately
does not know about:

**Labels.** Real networks are keyed by stop ids, OSM node ids, ``('bike', 42)``
tuples — anything hashable. The kernel wants dense ``u32``. The environment owns
that translation so nothing above it has to.

**Provenance.** Every edge remembers which layer produced it, which is how a
journey can later say "walk, then the 14 bus, then bike" rather than just naming
a cost. This is what ``Graph.input_index`` exists for.

Layers declare a ``cost_model``, and planners declare which cost models they can
route over. Today there is one — ``"scalar"``, a fixed cost per edge — so the
check always passes. It is here because the interesting case is the one where it
fails: Dijkstra cannot route over a timetable, and should say so when a timetable
layer arrives rather than quietly returning a wrong answer.

Compilation checks the same attribute for a different reason. Flattening every
layer into one fixed-cost graph is what makes a scalar environment cheap, and it
is exactly what a schedule cannot survive — so :class:`CompiledEnvironment`
refuses cost models it would have to lie about, rather than dropping the
time-dependence on the floor. That refusal is the seam CSA and RAPTOR plug into.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Tuple

from . import _routelab
from .graph import Graph

__all__ = [
    "CompiledEnvironment",
    "EdgeSource",
    "Environment",
    "Positions",
    "ScalarEdges",
]

#: An edge as a layer emits it: labelled endpoints and an integer cost.
LabelledEdge = Tuple[Hashable, Hashable, int]

#: Cost models that flatten into a single fixed-cost graph. A time-dependent
#: layer does not, and compiling one as if it did would quietly throw away the
#: thing that makes it time-dependent — so :class:`CompiledEnvironment` refuses.
COMPILABLE_COST_MODELS = frozenset({"scalar"})


def windows_of(source: "EdgeSource", index: int) -> "Optional[List[Tuple[int, int]]]":
    """When a layer's ``index``-th edge may be travelled, or ``None`` for always.

    The sibling of :func:`shape_of`, and optional in the same way: a layer that
    knows nothing about time simply has no hook, and everything it contributes
    is open at every hour.
    """
    getter = getattr(source, "windows", None)
    return None if getter is None else getter(index)


def shape_of(source: "EdgeSource", index: int) -> "Optional[List[Tuple[float, float]]]":
    """The shape of a layer's ``index``-th edge, in that layer's own numbering.

    Separate from :meth:`CompiledEnvironment.geometry` because two callers need
    it from different directions: an environment that has a graph edge id and
    must work out which layer it came from, and a journey leg that already knows.
    """
    # `register` asks only for `edges`, so a layer need not inherit EdgeSource
    # and need not have the hook at all.
    getter = getattr(source, "geometry", None)
    return None if getter is None else getter(index)


class EdgeSource:
    """A layer of the world that contributes edges to an :class:`Environment`.

    Subclasses implement :meth:`edges` and, if they are not simple fixed-cost
    edges, set :attr:`cost_model`.

    A layer may also declare material that algorithms need but edges cannot
    express: where its nodes are (:meth:`positions`) and how cheaply it covers
    ground (:attr:`cost_per_distance`). Both are optional, and a layer that
    declares neither simply cannot be used by anything that needs them.
    """

    #: What kind of cost this layer's edges carry. Planners refuse cost models
    #: they cannot handle.
    cost_model = "scalar"

    #: The least this layer can charge to cover one unit of distance — 1/speed if
    #: costs are seconds and positions are metres. ``None`` means "unknown", which
    #: is not the same as "free": a layer that might be arbitrarily fast makes any
    #: distance-based lower bound unsafe, so it disables them rather than being
    #: quietly assumed slow.
    #:
    #: One case this does not yet distinguish: a layer whose edges cover no ground
    #: at all, like a transfer penalty inside a station. Its rate is not unknown,
    #: it is undefined — but leaving it ``None`` disables distance bounds for the
    #: whole environment rather than just for itself. Erring toward the safe answer
    #: is right; telling the two apart is work for whoever adds such a layer.
    cost_per_distance: Optional[float] = None

    def edges(self) -> Iterable[LabelledEdge]:
        """Yield ``(tail, head, weight)`` with labelled endpoints."""
        raise NotImplementedError

    def positions(self) -> Mapping[Hashable, "Tuple[float, float]"]:
        """Where this layer's nodes are, as ``{label: (x, y)}``.

        Planar coordinates in whatever unit :attr:`cost_per_distance` is priced
        against. Empty by default — most layers know their topology and not their
        geometry.
        """
        return {}

    def geometry(self, index: int) -> "Optional[List[Tuple[float, float]]]":
        """The shape of the edge this layer emitted at ``index``, if it has one.

        An edge is a straight line between two labels as far as the graph is
        concerned; on the ground it may be a winding street. Indices are this
        layer's own, counted in the order :meth:`edges` yielded them —
        :meth:`CompiledEnvironment.geometry` does the translation from a graph
        edge id. ``None`` by default: a layer that knows only topology has no
        shape to give.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ScalarEdges(EdgeSource):
    """Fixed-cost edges between labelled nodes.

        >>> ScalarEdges(("a", "b", 1), ("b", "c", 15))
        ScalarEdges(2 edges)

    A tuple argument is one edge; any other iterable is a collection of edges, so
    both ``ScalarEdges(*triples)`` and ``ScalarEdges(triples)`` work.
    """

    cost_model = "scalar"

    def __init__(
        self,
        *edges: Any,
        bidirectional: bool = False,
        cost_per_distance: Optional[float] = None,
    ):
        if len(edges) == 1 and not isinstance(edges[0], tuple):
            edges = tuple(edges[0])
        self.cost_per_distance = cost_per_distance
        self._edges: List[LabelledEdge] = [
            (tail, head, int(weight)) for tail, head, weight in edges
        ]
        if bidirectional:
            self._edges += [(head, tail, weight) for tail, head, weight in self._edges]

    def edges(self) -> Iterator[LabelledEdge]:
        return iter(self._edges)

    def __len__(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return f"ScalarEdges({len(self._edges)} edges)"


class Positions(EdgeSource):
    """Where nodes are, as a layer of its own.

        >>> Positions({"a": (0.0, 0.0), "b": (300.0, 40.0)})
        Positions(2 nodes)

    Geometry is node data, not edge data, and usually arrives from a different
    place than the topology — a shapes file, a stops table — so it registers
    separately and contributes no edges. Coordinates for labels that no edge
    mentions are ignored; a node nothing connects to is not a place you can route
    through.
    """

    def __init__(self, positions: Mapping[Hashable, "Tuple[float, float]"]):
        self._positions = {
            label: (float(x), float(y)) for label, (x, y) in positions.items()
        }

    def edges(self) -> Iterator[LabelledEdge]:
        return iter(())

    def positions(self) -> Mapping[Hashable, "Tuple[float, float]"]:
        return self._positions

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        return f"Positions({len(self._positions)} nodes)"


class Environment:
    """What you are routing over: one or more registered layers.

        >>> env = Environment()
        >>> env.register(ScalarEdges(("a", "b", 1), ("b", "c", 15)))
        Environment(1 layer)

    Layers are kept as given and compiled on demand; registering another layer
    discards the compiled form so the next query rebuilds it.
    """

    def __init__(self, *sources: EdgeSource):
        self._sources: List[EdgeSource] = []
        self._compiled: Optional[CompiledEnvironment] = None
        self.register(*sources)

    def register(self, *sources: EdgeSource) -> "Environment":
        """Add layers. Returns the environment, so calls can be chained."""
        for source in sources:
            if not hasattr(source, "edges"):
                raise TypeError(
                    f"{source!r} is not an EdgeSource: it has no edges() method"
                )
            self._sources.append(source)
        self._compiled = None
        return self

    @property
    def sources(self) -> "Tuple[EdgeSource, ...]":
        return tuple(self._sources)

    @property
    def cost_models(self) -> "frozenset[str]":
        """The distinct cost models present, which planners check against."""
        return frozenset(source.cost_model for source in self._sources)

    def compile(self) -> "CompiledEnvironment":
        """Flatten the layers into a graph, assigning dense ids to labels.

        Cached: repeated calls return the same object until a layer is registered.
        """
        if self._compiled is None:
            self._compiled = CompiledEnvironment(self._sources)
        return self._compiled

    def __repr__(self) -> str:
        count = len(self._sources)
        return f"Environment({count} layer{'' if count == 1 else 's'})"


class CompiledEnvironment:
    """An environment flattened into a graph, with the label bookkeeping kept.

    Labels are numbered in the order they are first seen, so a given set of
    layers always compiles to the same graph — necessary for results to be
    comparable across runs and across planners.
    """

    def __init__(self, sources: Iterable[EdgeSource]):
        sources = list(sources)
        # Dicts keep insertion order, so the keys of `index` are the labels in
        # first-seen order — no second list to hold in lockstep.
        index: Dict[Hashable, int] = {}
        edges: List[Tuple[int, int, int]] = []
        # Where each layer's edges sit in the input list, noted as we go. Layers
        # emit contiguous runs, so a handful of spans replaces a per-edge table:
        # provenance costs a bisect over the layers rather than memory per edge.
        # A layer with no run at all also has no say in how fast the environment
        # can move.
        spans: "List[Tuple[int, int, EdgeSource]]" = []

        for source in sources:
            if source.cost_model not in COMPILABLE_COST_MODELS:
                raise NotImplementedError(
                    f"{source!r} is a {source.cost_model!r} layer, which does not "
                    f"flatten into a fixed-cost graph. This is the seam a "
                    f"schedule-based algorithm plugs into: a layer whose cost "
                    f"depends on when you arrive needs a compiled form this class "
                    f"does not build yet."
                )
            before = len(edges)
            for tail, head, weight in source.edges():
                for label in (tail, head):
                    if label not in index:
                        index[label] = len(index)
                edges.append((index[tail], index[head], int(weight)))
            if len(edges) > before:
                spans.append((before, len(edges), source))

        # Straight to the kernel constructor: `from_edges` re-coerces every
        # triple to int for callers who might hand it anything, and these are
        # already dense ints. At a million edges that pass is pure waste.
        self.graph = Graph(len(index), edges)
        self.labels: "Tuple[Hashable, ...]" = tuple(index)
        self._index = index
        #: The cost models of the layers that went into this graph. Always
        #: ``{"scalar"}`` today, since compilation refuses anything else — but
        #: carried so that a technique can be asked whether it handles what is
        #: actually here, rather than what used to be.
        self.cost_models: "frozenset[str]" = frozenset(
            source.cost_model for source in sources
        )
        # Spans index the *input* edge list, so a CSR edge id is looked up
        # through Graph.input_index rather than assumed to line up.
        self._spans: "Tuple[Tuple[int, int, EdgeSource], ...]" = tuple(spans)
        self._span_starts: "Tuple[int, ...]" = tuple(start for start, _, _ in spans)

        #: When each edge may be travelled, for the techniques that ask. Built
        #: from the layers' own ``windows`` hooks and keyed by *input* position,
        #: which the kernel translates — an edge id is a CSR position and the
        #: two are not the same. ``None`` when no layer schedules anything.
        self.calendar = self._gather_calendar(spans)

        #: Coordinates per node id, ``None`` where a node has none.
        self.positions: "Tuple[Optional[Tuple[float, float]], ...]" = self._gather_positions(
            sources, index
        )
        #: The cheapest rate any edge-contributing layer charges per unit of
        #: distance, or ``None`` if one of them did not say. See
        #: :meth:`_gather_cost_per_distance` for why one silent layer spoils it.
        self.cost_per_distance = self._gather_cost_per_distance(
            [source for _, _, source in spans]
        )

    def _gather_calendar(
        self, spans: "List[Tuple[int, int, EdgeSource]]"
    ) -> "Optional[_routelab.Calendar]":
        """Collect every layer's schedules into one calendar, or ``None``.

        Walks the input edge list rather than the graph, because that is the
        numbering a layer speaks: ``locate`` runs the other way and would need
        the answer to ask the question.
        """
        windows = []
        for start, stop, source in spans:
            for position in range(stop - start):
                schedule = windows_of(source, position)
                if schedule:
                    windows.append((start + position, schedule))
        if not windows:
            return None
        return _routelab.Calendar.from_windows(self.graph, windows)

    @staticmethod
    def _gather_positions(
        sources: "List[EdgeSource]", index: "Dict[Hashable, int]"
    ) -> "Tuple[Optional[Tuple[float, float]], ...]":
        """Collect every layer's coordinates into one node-indexed table.

        Later layers win, so a dedicated geometry layer can correct coordinates a
        topology layer guessed at.
        """
        positions: "List[Optional[Tuple[float, float]]]" = [None] * len(index)
        for source in sources:
            for label, point in source.positions().items():
                node_id = index.get(label)
                if node_id is not None:
                    positions[node_id] = (float(point[0]), float(point[1]))
        return tuple(positions)

    @staticmethod
    def _gather_cost_per_distance(
        contributing: "List[EdgeSource]",
    ) -> Optional[float]:
        """The lower bound on cost per unit distance across the whole environment.

        The *minimum*, because a bound has to hold for every path, and a path may
        use the fastest layer for its whole length. Add a train to a walking
        network and the bound has to drop to the train's rate, or a straight-line
        estimate priced at walking speed starts overestimating — which is how an
        admissible heuristic silently becomes an inadmissible one.

        One layer that declares nothing means no bound at all: it might be
        arbitrarily fast, so nothing can be ruled out. Only layers that actually
        contributed edges count, which is what lets a geometry-only layer stay out
        of the arithmetic without a special case.
        """
        rates: "List[float]" = []
        for source in contributing:
            if source.cost_per_distance is None:
                return None
            rates.append(source.cost_per_distance)
        return min(rates) if rates else None

    @property
    def provides(self) -> "frozenset[str]":
        """What this environment can supply beyond a graph, by name.

        The counterpart to :attr:`routelab.Heuristic.requires`: between them a
        caller can ask which techniques a given dataset can support without
        building any of them and reading the errors. An OSM extract provides all
        three; a hand-written edge list usually provides none.

        - ``"positions"`` — every node has coordinates
        - ``"cost_per_distance"`` — every edge-contributing layer declared a rate
        - ``"geometry"`` — at least one layer knows the shape of its edges
        """
        provided = set()
        if self.positions and all(point is not None for point in self.positions):
            provided.add("positions")
        if self.cost_per_distance is not None:
            provided.add("cost_per_distance")
        if any(source.geometry(0) is not None for _, _, source in self._spans):
            provided.add("geometry")
        if self.calendar is not None:
            provided.add("schedule")
        return frozenset(provided)

    def node_id(self, label: Hashable) -> int:
        """The dense id of ``label``."""
        try:
            return self._index[label]
        except KeyError:
            raise KeyError(f"{label!r} is not a node in this environment") from None

    def label(self, node_id: int) -> Hashable:
        """The label of a dense id."""
        return self.labels[node_id]

    def source_of(self, edge_id: int) -> EdgeSource:
        """The layer that contributed ``edge_id``."""
        return self.locate(edge_id)[0]

    def geometry(self, edge_id: int) -> "Optional[List[Tuple[float, float]]]":
        """The shape of ``edge_id``, if its layer keeps one.

        An edge in the graph is a straight line between two labels; on the ground
        it may be a winding street. Layers that know the difference expose a
        ``geometry(index)``, and this finds the right layer and asks it in that
        layer's own numbering. ``None`` when the layer has no shape to give.

        For an edge you got from a :class:`~routelab.Journey`, prefer
        ``leg.geometry`` — a leg already knows its layer and its place in it, and
        should not send you back to the environment to find out its own shape.
        """
        return shape_of(*self.locate(edge_id))

    def locate(self, edge_id: int) -> "Tuple[EdgeSource, int]":
        """The layer that produced ``edge_id``, and its position within it.

        Public because a :class:`~routelab.Leg` is built from it: a leg keeps
        both, so it can answer for its own shape without being handed the
        environment back.
        """
        input_index = self.graph.input_index(edge_id)
        position = bisect.bisect_right(self._span_starts, input_index) - 1
        start, _, source = self._spans[position]
        return source, input_index - start

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"CompiledEnvironment(num_nodes={self.graph.num_nodes}, "
            f"num_edges={self.graph.num_edges})"
        )
