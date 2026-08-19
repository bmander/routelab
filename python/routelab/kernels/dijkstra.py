"""Dijkstra, *A note on two problems in connexion with graphs* (1959)."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.answer import Answer
from ..model.environment import Environment
from ..util._args import Nodes, Sources, normalize_nodes, normalize_sources
from .planner import Origins, Technique, TreePlanner

__all__ = ["Dijkstra", "DijkstraPlanner", "dijkstra"]


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


class Dijkstra(Technique):
    """Cheapest-cost routing over fixed-cost edges."""

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "DijkstraPlanner":
        return DijkstraPlanner(self, environment, self._compile(environment), progress)


class DijkstraPlanner(TreePlanner):
    """:class:`Dijkstra` over one environment."""

    def route(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        max_cost: Optional[int] = None,
    ) -> Answer:
        """The cheapest journey from ``origin`` to ``destination``.

        Args:
            origin: A label, several labels, or ``{label: initial_cost}`` — the
                last being how a multimodal query starts, with each entry point
                already costing an access walk.
            destination: The label to route to.
            max_cost: Do not settle anything costing more than this, so a
                destination beyond it is simply not reached.

        Returns:
            An :class:`~routelab.Answer`: the route — empty if nothing was
            reachable — the search space behind it, and the kernel's own table.
        """
        target = self.node_id(destination)
        return self._answer(
            self._run(self._origin_ids(origin), [target], max_cost), destination
        )

    def search(
        self,
        origins: Origins,
        *,
        targets: "Optional[Nodes]" = None,
        max_cost: Optional[int] = None,
    ) -> SearchResult:
        """Run the search and return the raw, id-keyed result.

        The escape hatch: everything the kernel computed, without the journey
        packaging. Node ids here are dense, so use :meth:`node_id`/:meth:`label`
        to get between them and your labels. ``targets`` stops the search once
        all of them are settled.
        """
        return self._run(self._origin_ids(origins), targets, max_cost)

    def _run(
        self,
        starts: "Dict[int, int]",
        targets: "Optional[Nodes]",
        max_cost: Optional[int],
    ) -> SearchResult:
        return dijkstra(self.compiled.graph, starts, targets=targets, max_cost=max_cost)
