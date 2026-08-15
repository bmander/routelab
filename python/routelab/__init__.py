"""routelab — reference implementations of routing algorithms.

The Python layer is the showroom: friendly constructors, flexible arguments,
docstrings, and a pure-Python reference implementation of every kernel to check
the fast one against. The Rust layer (``routelab._routelab``) is the kernel:
CSR graphs and searches whose constant factors are the point.

    >>> import routelab as rl
    >>> graph = rl.Graph.from_edges([(0, 1, 60), (1, 3, 120), (0, 2, 90), (2, 3, 30)])
    >>> result = rl.dijkstra(graph, 0)
    >>> result.cost(3)
    120
    >>> result.path(3)
    [0, 2, 3]
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

from . import _routelab, reference
from ._args import Nodes, Sources, normalize_nodes, normalize_sources
from ._routelab import SearchResult

__all__ = ["Graph", "Nodes", "SearchResult", "Sources", "bfs", "dijkstra", "reference"]

__version__ = _routelab.__version__


class Graph(_routelab.Graph):
    """An immutable directed graph with non-negative integer weights.

    Weights are ``u32`` and conventionally seconds — the resolution the transit
    routing literature works in. Parallel edges and self-loops are allowed and
    nothing is deduplicated.

    Edges are stored in CSR order (grouped by tail, input order within a tail),
    so edge ids are *not* positions in the list you passed. Use
    :meth:`input_index` to get back to your own per-edge data.
    """

    __slots__ = ()

    @classmethod
    def from_edges(
        cls,
        edges: Iterable["tuple[int, int, int]"],
        *,
        num_nodes: Optional[int] = None,
        directed: bool = True,
    ) -> "Graph":
        """Build a graph from ``(tail, head, weight)`` triples.

        Args:
            edges: The edge list. Weights must be non-negative integers.
            num_nodes: Node count. Defaults to ``max(node id) + 1``, which means
                trailing isolated nodes disappear unless you say how many there are.
            directed: If false, each triple is inserted in both directions, so the
                graph has twice as many edges as triples.
        """
        edges = [(int(tail), int(head), int(weight)) for tail, head, weight in edges]
        if not directed:
            edges = [edge for tail, head, weight in edges for edge in ((tail, head, weight), (head, tail, weight))]
        if num_nodes is None:
            num_nodes = 1 + max((max(tail, head) for tail, head, _ in edges), default=-1)
        return cls(num_nodes, edges)

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Mapping[int, Iterable["tuple[int, int]"]],
        *,
        num_nodes: Optional[int] = None,
    ) -> "Graph":
        """Build a graph from ``{tail: [(head, weight), ...]}``."""
        edges = [
            (tail, head, weight)
            for tail, out_edges in adjacency.items()
            for head, weight in out_edges
        ]
        return cls.from_edges(edges, num_nodes=num_nodes)


def dijkstra(
    graph: _routelab.Graph,
    sources: Sources,
    *,
    targets: Optional[Nodes] = None,
    max_cost: Optional[int] = None,
) -> SearchResult:
    """Shortest paths by cost, from one or more sources.

    Args:
        graph: The graph to search.
        sources: Where the search starts. Either node ids, which start at cost 0,
            or ``(node, initial_cost)`` pairs — the shape multimodal routing wants,
            where reaching each transit stop already costs an access walk.
        targets: Stop once all of these are settled. Everything settled before then
            has its final cost; nodes merely touched do not.
        max_cost: Do not settle nodes costing more than this (inclusive) — an
            isochrone, and the usual way to keep a one-to-all search local.

    Returns:
        A :class:`SearchResult` holding the shortest-path tree.
    """
    return _routelab.dijkstra(
        graph,
        normalize_sources(sources),
        targets=normalize_nodes(targets),
        max_cost=max_cost,
    )


def bfs(
    graph: _routelab.Graph,
    sources: Nodes,
    *,
    targets: Optional[Nodes] = None,
    max_depth: Optional[int] = None,
) -> SearchResult:
    """Fewest-hops paths, ignoring edge weights.

    Every source starts at depth 0 — a FIFO queue is only correct when the
    frontier enters at a single depth, so unlike :func:`dijkstra` this takes no
    initial costs.

    Args:
        graph: The graph to search.
        sources: Node ids to start from.
        targets: Stop once all of these are settled.
        max_depth: Do not expand past this hop count (inclusive).
    """
    return _routelab.bfs(
        graph,
        normalize_nodes(sources) or [],
        targets=normalize_nodes(targets),
        max_depth=max_depth,
    )
