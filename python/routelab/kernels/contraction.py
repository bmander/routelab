"""Geisberger, Sanders, Schultes & Delling, *Contraction hierarchies: faster
and simpler hierarchical routing in road networks* (2008)."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

from .. import _routelab
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..model.searchspace import MeetingTrees
from .orderings import EdgeDifference, Ordering
from .planner import GraphPlanner, Origins, Technique

__all__ = ["ContractionHierarchy", "ContractionHierarchyPlanner"]


class ContractionHierarchy(Technique):
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
        """Whatever the ordering needs, on top of what any technique needs."""
        return super().missing_from(compiled) | self.ordering.missing_from(compiled)

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "ContractionHierarchyPlanner":
        return ContractionHierarchyPlanner(
            self, environment, self._compile(environment), progress
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.ordering))


class ContractionHierarchyPlanner(GraphPlanner):
    """A contracted graph, and the bidirectional query over it."""

    def __init__(
        self,
        technique: ContractionHierarchy,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        #: Contracting the graph. The expensive step, and the whole technique.
        self.hierarchy = technique.ordering.bind(compiled, progress)

    @property
    def footprint(self) -> int:
        return self.hierarchy.footprint

    def route(self, origin: Origins, destination: Hashable) -> Answer:
        """The cheapest journey, climbing from both ends and meeting above it.

        No bounds: the search is over the contracted graph, where a cost bound
        would cut off paths that are still cheap in the original.
        """
        target = self.node_id(destination)
        return self._answer(self._run(self._origin_ids(origin), target), destination)

    def search(self, origins: Origins, *, target: int) -> "_routelab.MeetingSearch":
        """Run the bidirectional query toward one target.

        Returns a :class:`~routelab._routelab.MeetingSearch` rather than a
        `SearchResult`: two searches met in the middle, and neither half alone
        is the answer. It reports costs and paths in the environment's own edges,
        which is all :class:`~routelab.Journey` ever asked of a result. A
        hierarchy climbs *toward* somewhere, so the target is required.
        """
        return self._run(self._origin_ids(origins), target)

    def _run(self, starts: "Dict[int, int]", target: int) -> "_routelab.MeetingSearch":
        return self.hierarchy.query(list(starts.items()), target)

    def explored(self, result: "_routelab.MeetingSearch") -> MeetingTrees:
        """The two halves of the search, and where they met."""
        return MeetingTrees(self.compiled, result)
