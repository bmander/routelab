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

And that is all the compiled form is: the *merge* — one numbering, one graph,
and which layer each edge came from. Everything else a technique might want
from the layers — a calendar of opening hours, a timetable of departures, a
coordinates table, a rate to price a distance against — is not merged here.
The technique that reads it derives it, from :attr:`CompiledEnvironment.spans`
and :attr:`CompiledEnvironment.sources`, and refuses if the layers cannot
supply it: see :class:`~routelab.Schedule`, :class:`~routelab.Departures`,
:class:`~routelab.Plane` and :class:`~routelab.Pace`. That is what keeps the
next technique from touching this file: it brings its own derivation, and the
environment does not need to know what it is for.

Layers declare a ``cost_model``, and planners declare which cost models they can
route over. Today there are two — ``"scalar"``, a fixed cost per edge, and
``"timetable"``, edges that are lower bounds standing in for departures — and
the check exists for the case where it fails: Dijkstra cannot route over a
timetable, and should say so when a timetable layer arrives rather than quietly
returning a wrong answer. This one *is* the environment's to carry, because it
describes the graph's own weights.

Compilation checks the same attribute for a different reason. Flattening every
layer into one fixed-cost graph is what makes a scalar environment cheap, and it
is exactly what a schedule cannot survive — so :class:`CompiledEnvironment`
refuses cost models it would have to lie about, rather than dropping the
time-dependence on the floor. That refusal is the seam RAPTOR and CSA plug into.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Tuple

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

#: Cost models a :class:`CompiledEnvironment` knows how to carry.
#:
#: ``"scalar"`` flattens into the graph itself. ``"timetable"`` does not — its
#: edges get a lower-bound weight and the schedule travels beside the graph, so
#: what compiles is a graph a planner must not route as if the weights told the
#: whole story. :attr:`~routelab.kernels.Technique.accepts` is what enforces that.
#:
#: A cost model absent from here is one nothing could carry without lying about
#: it, which is the seam a new schedule-based algorithm plugs into.
COMPILABLE_COST_MODELS = frozenset({"scalar", "timetable"})


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

    def load(self) -> "EdgeSource":
        """Read whatever this layer defers, now, and return it.

        Layers read lazily: an OSM extract is six seconds and nobody should pay
        for one they never route over. But laziness moves *when* the cost lands
        as well as whether, and something timing the steps — or drawing a
        spinner over the one it is waiting on — needs to be able to say "not
        later, now" without knowing which attribute happens to trigger it.

        Nothing to do for a layer that was never lazy, which is why this is
        here and not on each subclass.
        """
        return self

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
        self._sources: "Tuple[EdgeSource, ...]" = tuple(sources)
        #: The cost models of the layers that went into this graph, carried so
        #: that a technique can be asked whether it handles what is actually
        #: here, rather than what used to be.
        self.cost_models: "frozenset[str]" = frozenset(
            source.cost_model for source in sources
        )
        # Spans index the *input* edge list, so a CSR edge id is looked up
        # through Graph.input_index rather than assumed to line up.
        self._spans: "Tuple[Tuple[int, int, EdgeSource], ...]" = tuple(spans)
        self._span_starts: "Tuple[int, ...]" = tuple(start for start, _, _ in spans)

    @property
    def sources(self) -> "Tuple[EdgeSource, ...]":
        """Every layer that went into this graph, in registration order —
        including ones that contributed no edges, like :class:`Positions`."""
        return self._sources

    @property
    def spans(self) -> "Tuple[Tuple[int, int, EdgeSource], ...]":
        """``(start, stop, layer)`` runs in the *input* edge list, one per
        edge-contributing layer.

        This is the numbering a layer's own hooks speak — ``windows(i)``,
        ``connections(i)``, ``geometry(i)`` count from ``start`` — and what the
        kernel's ``Calendar.from_windows`` and ``Timetable.from_connections``
        translate to edge ids. Public so that whatever a technique derives from
        the layers can walk them without the environment having to know what
        is being derived.
        """
        return self._spans

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

    def coordinates(self) -> "Dict[Hashable, Tuple[float, float]]":
        """``{label: (lat, lon)}`` from every layer that knows where its nodes
        are on the ground — the ``coordinates()`` hook a feed and an extract
        both have.

        Here rather than in whoever wants to draw, because an environment is
        the thing that knows it is a merge: a multimodal journey walks a street,
        boards at a stop and gets off at another, so its labels come from more
        than one layer and one layer's table cannot place them. Later layers
        win. Not ``positions()``, which is whatever planar unit a distance
        bound is priced in.
        """
        points: "Dict[Hashable, Tuple[float, float]]" = {}
        for source in self.sources:
            getter = getattr(source, "coordinates", None)
            if getter is not None:
                points.update(getter())
        return points

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
