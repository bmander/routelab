"""Geisberger, Sanders, Schultes & Delling, *Contraction hierarchies: faster
and simpler hierarchical routing in road networks* (2008)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from ..model.environment import CompiledEnvironment
from ..model.search import Result
from ..model.searchspace import MeetingTrees, SearchSpace
from .orderings import EdgeDifference, Ordering
from .planner import Planner

__all__ = ["ContractionHierarchy"]


class ContractionHierarchy(Planner):
    """Exact routing by rewriting the graph, then only ever climbing it.

        ContractionHierarchy().bind(env).route("a", "b")

    Geisberger, Sanders, Schultes and Delling. Preprocessing contracts nodes one
    at a time, least important first, inserting a **shortcut** wherever removing a
    node would otherwise have lengthened a shortest path. What comes out is the
    original graph plus shortcuts and a rank per node — and a query that searches
    upward from the source and upward from the target and meets above the trip,
    never looking sideways at the thousands of streets in between.

    The answers are exact. Not approximately exact: the tests hold every distance
    to Dijkstra's on every instance, because a routing technique that is usually
    right is not a routing technique.

    Unlike every other technique here, this one searches a graph the environment
    has never seen. Its answers are unpacked back into the environment's own
    edges before anyone sees them, so journeys, geometry and provenance work
    exactly as they do for Dijkstra — a technique may search whatever it likes,
    but it answers in the caller's terms.

    A hierarchy query takes no bounds: its search is over the contracted graph,
    where a cost bound would cut off paths that are still cheap in the original.

    Args:
        ordering: Which node to contract next; see :mod:`routelab.kernels.orderings`.
            A policy, never a correctness choice — every ordering gives the same
            distances, and a bad one just builds a bigger hierarchy.
    """

    def __init__(self, ordering: Optional[Ordering] = None):
        self.ordering = ordering if ordering is not None else EdgeDifference()

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the ordering needs, on top of what any planner needs."""
        return super().missing_from(compiled) | self.ordering.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Contract the graph. The expensive step, and the whole technique."""
        self.hierarchy = self.ordering.bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.hierarchy.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        """Run the bidirectional query. Requires exactly one target.

        Returns a :class:`~routelab._routelab.MeetingSearch` rather than a
        `SearchResult`: two searches met in the middle, and neither half alone
        is the answer. It reports costs and paths in the environment's own edges,
        which is all :class:`~routelab.Journey` ever asked of a result.
        """
        target = self._single_target(options, "A hierarchy")
        return self.hierarchy.query(list(starts.items()), target)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """The two halves of the search, and where they met."""
        self._no_other(options, "meeting trees")
        return MeetingTrees(self._bound(), result)

    def __repr__(self) -> str:
        return self._describe(repr(self.ordering))
