"""Schedules: when each edge of a compiled environment may be travelled.

A :class:`Schedule` is a specification in the same sense a
:class:`~routelab.Heuristic` is: you write ``Schedule()``, and :meth:`bind`
turns it into a kernel :class:`~routelab._routelab.Calendar` against one
compiled environment — walking the layers, asking each for the hours its edges
are open, and refusing if none of them says anything.

It lives here rather than on the environment because the environment is a
merge — labels, one graph, provenance — and a calendar is not part of that
merge. It is something one technique reads. Keeping it beside its reader means
adding the next thing a technique reads does not touch the environment at all.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from . import _routelab
from .model.environment import CompiledEnvironment, EdgeSource

__all__ = ["Schedule"]


def windows_of(source: EdgeSource, index: int) -> "Optional[List[Tuple[int, int]]]":
    """When a layer's ``index``-th edge may be travelled, or ``None`` for always.

    Optional in the same way :func:`~routelab.environment.shape_of` is: a layer
    that knows nothing about time simply has no hook, and everything it
    contributes is open at every hour.
    """
    getter = getattr(source, "windows", None)
    return None if getter is None else getter(index)


class Schedule:
    """The hours every edge is open, gathered from the layers into one calendar.

        >>> Schedule()
        Schedule()

    Built by :class:`~routelab.TimeDependentDijkstra` at bind time. Walks the
    input edge list rather than the graph, because that is the numbering a
    layer speaks; the kernel translates to edge ids.
    """

    #: The word :meth:`missing_from` answers with.
    name = "schedule"

    @staticmethod
    def _windows(compiled: CompiledEnvironment) -> "Iterator[Tuple[int, List[Tuple[int, int]]]]":
        """``(input position, windows)`` for every edge that has any."""
        for start, stop, source in compiled.spans:
            if getattr(source, "windows", None) is None:
                continue  # a layer with no clock: skip its run, not each edge
            for position in range(stop - start):
                schedule = windows_of(source, position)
                if schedule:
                    yield start + position, schedule

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """``{"schedule"}`` unless some edge here is open only at certain hours.

        Exact, and stops at the first scheduled edge it finds; a network with
        no schedule at all is walked once to say so.
        """
        if any(True for _ in cls._windows(compiled)):
            return frozenset()
        return frozenset({cls.name})

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Calendar":
        """Collect every layer's windows into one calendar, or explain.

        ``progress`` is accepted for parity with every other ``bind`` and left
        alone: a walk of the layers has no honest measure of its own.

        Raises:
            ValueError: If no layer schedules anything.
        """
        windows = list(self._windows(compiled))
        if not windows:
            raise ValueError(
                "nothing here is scheduled: no layer says when its edges are "
                "open, so there is no clock to read. routelab.OSM(path, profile) "
                "reads OpenStreetMap's :conditional tags; without a schedule, use "
                "Dijkstra() to route the always-open network knowingly."
            )
        return _routelab.Calendar.from_windows(compiled.graph, windows)

    def __repr__(self) -> str:
        return "Schedule()"
