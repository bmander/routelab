"""Hart, Nilsson & Raphael, *A formal basis for the heuristic determination
of minimum cost paths* (1968)."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..util._args import Sources, normalize_sources
from .heuristics import Heuristic
from .planner import Origins, Technique, TreePlanner

__all__ = ["AStar", "AStarPlanner", "astar"]


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
            :meth:`routelab.kernels.heuristics.Heuristic.bind`. It must be admissible —
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


class AStar(Technique):
    """Cheapest-cost routing, guided toward the destination by a heuristic.

        AStar(Euclidean()).bind(env).route("a", "b")

    Returns exactly what :class:`Dijkstra` returns, by settling fewer nodes — how
    many fewer is the whole question, and ``len(result.order)`` is how you answer
    it.

    The heuristic is required. A* whose heuristic quietly fell back to zero is
    Dijkstra wearing its name, which is the one thing a benchmark must never be
    unable to detect — so :class:`~routelab.kernels.heuristics.Zero` has to be asked for
    out loud.
    """

    def __init__(self, heuristic: Heuristic):
        self.heuristic_spec = heuristic

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the heuristic needs, on top of what any technique needs."""
        return super().missing_from(compiled) | self.heuristic_spec.missing_from(compiled)

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "AStarPlanner":
        return AStarPlanner(self, environment, self._compile(environment), progress)

    def __repr__(self) -> str:
        return self._describe(repr(self.heuristic_spec))


class AStarPlanner(TreePlanner):
    """:class:`AStar` over one environment, its heuristic already built."""

    def __init__(
        self,
        technique: AStar,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        #: Where a landmark table, and any preprocessing after it, gets built.
        self.heuristic = technique.heuristic_spec.bind(compiled, progress)

    @property
    def footprint(self) -> int:
        return super().footprint + self.heuristic.footprint

    def route(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        max_cost: Optional[int] = None,
    ) -> Answer:
        """The cheapest journey to ``destination``, guided toward it."""
        target = self.node_id(destination)
        return self._answer(
            self.search(origin, target=target, max_cost=max_cost), destination
        )

    def search(
        self, origins: Origins, *, target: int, max_cost: Optional[int] = None
    ) -> SearchResult:
        """Run the guided search toward one target.

        A* is goal-directed: the estimate is an estimate *to somewhere*, so the
        target is a required argument rather than a bound on the search. There
        is nothing a heuristic could be a bound on for several of them.
        """
        return astar(
            self.compiled.graph,
            self._origin_ids(origins),
            target,
            self.heuristic,
            max_cost=max_cost,
        )

