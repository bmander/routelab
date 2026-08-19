"""Dreyfus, *An appraisal of some shortest-path algorithms* (1969) —
time-dependent Dijkstra over scheduled restrictions."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

from .. import _routelab
from .._routelab import SearchResult
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..util._args import Nodes, normalize_nodes
from ..util.clock import Departure, weekly_seconds
from .planner import Origins, Technique, TreePlanner
from .schedule import Schedule

__all__ = ["TimeDependentDijkstra", "TimeDependentDijkstraPlanner"]


class TimeDependentDijkstra(Technique):
    """Cheapest arrival when the network is not always open.

        TimeDependentDijkstra().bind(env).route("a", "b", departing=time(8, 30))

    Dreyfus, *An Appraisal of Some Shortest-Path Algorithms* (1969): Dijkstra's
    algorithm generalises to a time-dependent network unchanged, provided
    arrival is non-decreasing in departure. Here it is, because travel times are
    constant and only availability varies — a gate is shut, a lane runs the
    other way — so leaving later cannot arrive earlier.

    This is a *different technique*, not :class:`Dijkstra` with an extra
    argument, which is what keeps a schedule from being ignored by accident.
    Ask for `Dijkstra` and you get the always-open network, honestly and
    knowingly; ask for this one and you get the clock. A departure time is
    required, and there is no default for it: a time-dependent query without a
    time is not a query with a sensible fallback, it is a different question.

    Args:
        waiting: ``"unrestricted"`` waits at a shut edge and pays the wait as
            travel time — arriving five minutes early beats an hour's detour,
            and a ten-hour wait loses to one, without anyone deciding which.
            ``"forbidden"`` treats a shut edge as absent, which is the control
            that shows waiting is doing real work.
    """

    WAITING = ("unrestricted", "forbidden")

    def __init__(self, waiting: str = "unrestricted"):
        if waiting not in self.WAITING:
            raise ValueError(
                f"unknown waiting policy {waiting!r}; expected "
                f"{' or '.join(repr(name) for name in self.WAITING)}"
            )
        self.waiting = waiting

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """A schedule to read, on top of what any technique needs. Without one
        this is Dijkstra with extra steps, and this says so before anything
        is built."""
        return super().missing_from(compiled) | Schedule.missing_from(compiled)

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "TimeDependentDijkstraPlanner":
        return TimeDependentDijkstraPlanner(
            self, environment, self._compile(environment), progress
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.waiting) if self.waiting != "unrestricted" else "")


class TimeDependentDijkstraPlanner(TreePlanner):
    """:class:`TimeDependentDijkstra` over one environment and its calendar."""

    def __init__(
        self,
        technique: TimeDependentDijkstra,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        #: The environment does not keep a calendar; this technique is what
        #: reads one, so this technique derives it — and refuses here, at bind,
        #: if there is nothing to derive.
        self.calendar = Schedule().bind(compiled, progress)
        self.waiting = technique.waiting

    @property
    def footprint(self) -> int:
        return self.calendar.footprint

    def route(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        departing: Departure,
        max_cost: Optional[int] = None,
    ) -> Answer:
        """The cheapest journey leaving at ``departing``, on the weekly clock.

        There is no default departure: a time-dependent query without a time
        is not a query with a sensible fallback, it is a different question —
        the one :class:`~routelab.Dijkstra` answers.
        """
        target = self.node_id(destination)
        return self._answer(
            self._run(self._origin_ids(origin), departing, [target], max_cost),
            destination,
        )

    def search(
        self,
        origins: Origins,
        *,
        departing: Departure,
        targets: "Optional[Nodes]" = None,
        max_cost: Optional[int] = None,
    ) -> SearchResult:
        """Run the search and return the raw, id-keyed result."""
        return self._run(self._origin_ids(origins), departing, targets, max_cost)

    def _run(
        self,
        starts: "Dict[int, int]",
        departing: Departure,
        targets: "Optional[Nodes]",
        max_cost: Optional[int],
    ) -> SearchResult:
        return _routelab.time_dependent_dijkstra(
            self.compiled.graph,
            self.calendar,
            list(starts.items()),
            weekly_seconds(departing),
            waiting=self.waiting,
            targets=normalize_nodes(targets),
            max_cost=max_cost,
        )
