"""Pyrga, Schulz, Wagner & Zaroliagis, *Efficient models for timetable
information in public transportation systems* (2007) — the two models."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _routelab
from .planner import TimetablePlanner

__all__ = ["TimeDependent", "TimeExpanded"]


class TimeDependent(TimetablePlanner):
    """A node per stop, and a search that reads the clock (Pyrga et al. §4).

        TimeDependent().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    The graph stays the size of the network. Relaxing an edge is not reading a
    weight but asking what leaves next along it, which is a binary search over
    that edge's departures — so the model is small and the search is bespoke.
    Only the timetable and the walks are built at bind, which is the other half
    of the trade.

    Changing vehicles is instantaneous here, which is the paper's *simple*
    model. Its *realistic* one charges a minimum change time and is not
    expressible with one label per stop: staying in your seat must not be
    charged, so the cost of boarding depends on which vehicle you arrived on.
    The paper's answer is more nodes (§4.2); RAPTOR's rounds are another.
    """

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        return self.timetable.earliest_arrival(sources, target, self.footpaths)


class TimeExpanded(TimetablePlanner):
    """A node per event, and then it is just a graph (Pyrga et al. §3).

        TimeExpanded().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    Every departure and every arrival becomes its own node; riding and waiting
    become edges. What comes out is an ordinary static graph, so
    :func:`~routelab.dijkstra` routes it unchanged — and so would A*, or
    landmarks, or a contraction hierarchy. That is the model's whole appeal.

    Its cost is size. A city's weekday is a few thousand stops and hundreds of
    thousands of events, so the graph is built once at bind time and
    :attr:`footprint` is worth reading before you build one.
    """

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._expanded = _routelab.TimeExpanded.build(self.timetable, self.footpaths)

    def _footprint(self) -> int:
        return super()._footprint() + self._expanded.footprint

    @property
    def num_events(self) -> int:
        """Nodes in the expanded graph — the number the model is judged on."""
        self._bound()
        return self._expanded.num_events

    @property
    def searches(self) -> "Tuple[str, int]":
        return ("events", self.num_events)

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        return self._expanded.earliest_arrival(sources, target)
