"""Hart, Nilsson & Raphael, *A formal basis for the heuristic determination
of minimum cost paths* (1968)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.search import Result
from ..util._args import Nodes, Sources, normalize_nodes, normalize_sources
from .heuristics import Heuristic
from .planner import Planner

__all__ = ["AStar", "astar"]


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


class AStar(Planner):
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

    options = frozenset({"max_cost"})

    def __init__(self, heuristic: Heuristic):
        self.heuristic_spec = heuristic

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the heuristic needs, on top of what any planner needs."""
        return super().missing_from(compiled) | self.heuristic_spec.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Bind the heuristic to this environment — where a landmark table,
        and any preprocessing after it, gets built."""
        self.heuristic = self.heuristic_spec.bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.heuristic.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        """Run the guided search. Requires exactly one target.

        A* is goal-directed: the estimate is an estimate *to somewhere*. Without
        a target there is nothing to aim at, and with several there is no single
        thing the heuristic could be a bound on.
        """
        target = self._single_target(options, "A*")
        return astar(self._bound().graph, starts, target, self.heuristic, **options)

    def __repr__(self) -> str:
        return self._describe(repr(self.heuristic_spec))
