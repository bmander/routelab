"""The kernels, called directly on integer node ids.

These are the low road: no labels, no environment, no preprocessing — the
searches as the literature states them. :mod:`routelab.planners` is the high
road built on top.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from .. import _routelab
from ..util._args import Nodes, Sources, normalize_nodes, normalize_sources
from .._routelab import SearchResult

__all__ = ["EdgeResult", "Result", "SearchResult", "astar", "bfs", "dijkstra"]


@runtime_checkable
class Result(Protocol):
    """What anything downstream of a search actually needs from it.

    :class:`~routelab.SearchResult` is one implementation and the common one,
    but it is not the only shape a search comes in: a bidirectional search
    produces two trees and a meeting point, and a round-based transit search
    produces a label per stop per round; no amount of widening `SearchResult`
    would make either one. What the planners consume is this much and no more
    — a cost, a path, and how much work it took — so this is the contract, and
    a technique is free to satisfy it however its algorithm demands.

    Note what is *not* here: `order`. A search that settles in more than one
    direction has no single settle sequence, which is exactly why the count
    worth comparing across algorithms is :attr:`settled` rather than the length
    of a list.
    """

    def cost(self, node: int) -> Optional[int]:
        """The cheapest cost to ``node``, or ``None`` if it has none."""

    def path(self, node: int) -> "Optional[List[int]]":
        """Node ids from the source to ``node``, source first."""

    @property
    def settled(self) -> int:
        """How many nodes the search settled: the work it did."""


@runtime_checkable
class EdgeResult(Result, Protocol):
    """A result that walked the environment's own edges, and can say which.

    Everything a graph search returns is one of these, and it is what
    :meth:`~routelab.Journey.from_result` needs: a leg is an edge, and an edge
    is where provenance and geometry hang. A search that never touched the
    graph — a round-based one over routes and trips — is a :class:`Result` and
    not this, and answers with an itinerary instead.
    """

    def edge_path(self, node: int) -> "Optional[List[int]]":
        """Edge ids from the source to ``node`` — in the *caller's* graph."""


def astar(
    graph: _routelab.Graph,
    sources: Sources,
    target: int,
    heuristic: "_routelab.Heuristic",
    *,
    max_cost: Optional[int] = None,
) -> SearchResult:
    """Cheapest paths to ``target``, guided by ``heuristic``.

    Dijkstra ordered by cost-so-far plus estimated-cost-remaining. The result
    records real costs, not the estimates the queue was sorted by.

    Args:
        graph: The graph to search.
        sources: Node ids, or ``(node, initial_cost)`` pairs.
        target: The node to search toward. A* needs exactly one.
        heuristic: A kernel heuristic, from
            :meth:`routelab.heuristics.Heuristic.bind`. It must be admissible —
            never estimating more than the true remaining cost — or the paths
            returned will not be the cheapest, quietly.
        max_cost: Bounds the real cost, exactly as for :func:`dijkstra`.
    """
    return _routelab.astar(
        graph,
        normalize_sources(sources),
        target,
        heuristic,
        max_cost=max_cost,
    )


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
