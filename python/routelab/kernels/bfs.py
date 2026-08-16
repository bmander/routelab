"""Moore, *The shortest path through a maze* (1959) — breadth-first search."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.search import Result
from ..util._args import Nodes, Sources, normalize_nodes, normalize_sources
from .planner import Planner

__all__ = ["BFS", "bfs"]


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


class BFS(Planner):
    """Fewest-hops routing, ignoring edge costs.

    Origins all start at depth 0: a hop count cannot express "you are already
    part of the way there", so an initial cost has nowhere to go.
    """

    options = frozenset({"max_depth"})

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        priced = {self.label(node) for node, cost in starts.items() if cost}
        if priced:
            raise ValueError(
                f"BFS counts hops, so origins cannot carry an initial cost: "
                f"{', '.join(repr(label) for label in sorted(priced, key=repr))}"
            )
        return bfs(self._bound().graph, list(starts), **options)
