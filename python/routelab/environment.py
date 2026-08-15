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

from typing import Any, Dict, Hashable, Iterable, Iterator, List, Optional, Tuple

from .graph import Graph

__all__ = ["CompiledEnvironment", "EdgeSource", "Environment", "ScalarEdges"]

#: An edge as a layer emits it: labelled endpoints and an integer cost.
LabelledEdge = Tuple[Hashable, Hashable, int]

#: Cost models that flatten into a single fixed-cost graph. A time-dependent
#: layer does not, and compiling one as if it did would quietly throw away the
#: thing that makes it time-dependent — so :class:`CompiledEnvironment` refuses.
COMPILABLE_COST_MODELS = frozenset({"scalar"})


class EdgeSource:
    """A layer of the world that contributes edges to an :class:`Environment`.

    Subclasses implement :meth:`edges` and, if they are not simple fixed-cost
    edges, set :attr:`cost_model`.
    """

    #: What kind of cost this layer's edges carry. Planners refuse cost models
    #: they cannot handle.
    cost_model = "scalar"

    def edges(self) -> Iterable[LabelledEdge]:
        """Yield ``(tail, head, weight)`` with labelled endpoints."""
        raise NotImplementedError

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

    def __init__(self, *edges: Any, bidirectional: bool = False):
        if len(edges) == 1 and not isinstance(edges[0], tuple):
            edges = tuple(edges[0])
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
        # Dicts keep insertion order, so the keys of `index` are the labels in
        # first-seen order — no second list to hold in lockstep.
        index: Dict[Hashable, int] = {}
        edges: List[Tuple[int, int, int]] = []
        edge_sources: List[EdgeSource] = []

        for source in sources:
            if source.cost_model not in COMPILABLE_COST_MODELS:
                raise NotImplementedError(
                    f"{source!r} is a {source.cost_model!r} layer, which does not "
                    f"flatten into a fixed-cost graph. This is the seam a "
                    f"schedule-based algorithm plugs into: a layer whose cost "
                    f"depends on when you arrive needs a compiled form this class "
                    f"does not build yet."
                )
            for tail, head, weight in source.edges():
                for label in (tail, head):
                    if label not in index:
                        index[label] = len(index)
                edges.append((index[tail], index[head], int(weight)))
                edge_sources.append(source)

        # Straight to the kernel constructor: `from_edges` re-coerces every
        # triple to int for callers who might hand it anything, and these are
        # already dense ints. At a million edges that pass is pure waste.
        self.graph = Graph(len(index), edges)
        self.labels: "Tuple[Hashable, ...]" = tuple(index)
        self._index = index
        # Parallel to the *input* edge list, so a CSR edge id is looked up
        # through Graph.input_index rather than assumed to line up.
        self._edge_sources: "Tuple[EdgeSource, ...]" = tuple(edge_sources)

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
        return self._edge_sources[self.graph.input_index(edge_id)]

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"CompiledEnvironment(num_nodes={self.graph.num_nodes}, "
            f"num_edges={self.graph.num_edges})"
        )
