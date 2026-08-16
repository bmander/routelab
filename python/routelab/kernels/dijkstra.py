"""Dijkstra, *A note on two problems in connexion with graphs* (1959)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..util._args import Nodes, Sources, normalize_nodes, normalize_sources
from .planner import Planner

__all__ = ["Dijkstra", "dijkstra"]


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


class Dijkstra(Planner):
    """Cheapest-cost routing over fixed-cost edges."""

    options = frozenset({"max_cost"})

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        return dijkstra(self._bound().graph, starts, **options)
