"""The kernels, called directly on integer node ids.

These are the low road: no labels, no environment, no preprocessing — the
searches as the literature states them. :mod:`routelab.planners` is the high
road built on top.
"""

from __future__ import annotations

from typing import Optional

from . import _routelab
from ._args import Nodes, Sources, normalize_nodes, normalize_sources
from ._routelab import SearchResult

__all__ = ["SearchResult", "astar", "bfs", "dijkstra"]


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
