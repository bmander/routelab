"""Moore, *The shortest path through a maze* (1959) — breadth-first search."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.answer import Answer
from ..model.environment import Environment
from ..util._args import Nodes, normalize_nodes
from .planner import Origins, Technique, TreePlanner

__all__ = ["BFS", "BFSPlanner", "bfs"]


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


class BFS(Technique):
    """Fewest-hops routing, ignoring edge costs.

    Origins all start at depth 0: a hop count cannot express "you are already
    part of the way there", so an initial cost has nowhere to go.
    """

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "BFSPlanner":
        return BFSPlanner(self, environment, self._compile(environment), progress)


class BFSPlanner(TreePlanner):
    """:class:`BFS` over one environment."""

    def route(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        max_depth: Optional[int] = None,
    ) -> Answer:
        """The journey with the fewest hops from ``origin`` to ``destination``.

        Args:
            origin: A label or several labels. Not a mapping of head starts —
                see :meth:`search`.
            destination: The label to route to.
            max_depth: Do not expand past this hop count (inclusive).
        """
        target = self.node_id(destination)
        return self._answer(
            self.search(origin, targets=[target], max_depth=max_depth), destination
        )

    def search(
        self,
        origins: Origins,
        *,
        targets: "Optional[Nodes]" = None,
        max_depth: Optional[int] = None,
    ) -> SearchResult:
        """Run the search and return the raw, id-keyed result."""
        starts = self._origin_ids(origins)
        priced = {self.label(node) for node, cost in starts.items() if cost}
        if priced:
            raise ValueError(
                f"BFS counts hops, so origins cannot carry an initial cost: "
                f"{', '.join(repr(label) for label in sorted(priced, key=repr))}"
            )
        return bfs(
            self.compiled.graph, list(starts), targets=targets, max_depth=max_depth
        )

