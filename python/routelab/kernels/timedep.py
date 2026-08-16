"""Dreyfus, *An appraisal of some shortest-path algorithms* (1969) —
time-dependent Dijkstra over scheduled restrictions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from ..model.search import Result, SearchResult
from ..util.clock import weekly_seconds
from .planner import Planner
from .schedule import Schedule

__all__ = ["TimeDependentDijkstra"]


class TimeDependentDijkstra(Planner):
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

    options = frozenset({"departing", "max_cost"})
    required = frozenset({"departing"})

    WAITING = ("unrestricted", "forbidden")

    def __init__(self, waiting: str = "unrestricted"):
        if waiting not in self.WAITING:
            raise ValueError(
                f"unknown waiting policy {waiting!r}; expected "
                f"{' or '.join(repr(name) for name in self.WAITING)}"
            )
        self.waiting = waiting

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """A schedule to read, on top of what any planner needs. Without one
        this is Dijkstra with extra steps, and this says so before anything
        is built."""
        return super().missing_from(compiled) | Schedule.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Gather the layers' schedules into the calendar every query reads.

        The environment does not keep one; this technique is what reads it,
        so this technique derives it — and refuses here, at bind, if there is
        nothing to derive.
        """
        self.calendar = Schedule().bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.calendar.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        departing = options.pop("departing")
        compiled = self._bound()
        return _routelab.time_dependent_dijkstra(
            compiled.graph,
            self.calendar,
            list(starts.items()),
            weekly_seconds(departing),
            waiting=self.waiting,
            **options,
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.waiting) if self.waiting != "unrestricted" else "")
