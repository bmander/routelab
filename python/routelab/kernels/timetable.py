"""Pyrga, Schulz, Wagner & Zaroliagis, *Efficient models for timetable
information in public transportation systems* (2007) — the two models."""

from __future__ import annotations

from typing import Hashable, Optional, Tuple

from .. import _routelab
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..util.clock import Departure
from .planner import Origins, TimetablePlanner, TimetableTechnique

__all__ = [
    "TimeDependent",
    "TimeDependentPlanner",
    "TimeExpanded",
    "TimeExpandedPlanner",
]


class TimeDependent(TimetableTechnique):
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
    The paper's answer is more nodes (§4.2); RAPTOR's rounds are another,
    CSA's trip flags a third, and TripBased's labels on trips a fourth.
    """

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "TimeDependentPlanner":
        return TimeDependentPlanner(
            self, environment, self._compile(environment), progress
        )


class TimeDependentPlanner(TimetablePlanner):
    """:class:`TimeDependent` over one feed.

    Keeps no table: a query is a search toward one stop, so what it can answer
    is a journey and there is neither a cost per stop to hand back nor a space
    to draw. Which is why it has no ``search`` and no ``explored`` at all.
    """

    def route(
        self, origin: Origins, destination: Hashable, *, departing: Departure
    ) -> Answer:
        """The earliest arrival at ``destination``, leaving at ``departing``."""
        sources, at = self._sources(self._origin_ids(origin), departing)
        itinerary = self.timetable.earliest_arrival(
            sources, self.node_id(destination), self.footpaths
        )
        return self._answer_itinerary(itinerary, destination, at)


class TimeExpanded(TimetableTechnique):
    """A node per event, and then it is just a graph (Pyrga et al. §3).

        TimeExpanded().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    Every departure and every arrival becomes its own node; riding and waiting
    become edges. What comes out is an ordinary static graph, so
    :func:`~routelab.dijkstra` routes it unchanged — and so would A*, or
    landmarks, or a contraction hierarchy. That is the model's whole appeal.

    Its cost is size. A city's weekday is a few thousand stops and hundreds of
    thousands of events, so the graph is built once at bind time and
    :attr:`~TimeExpandedPlanner.footprint` is worth reading before you build one.
    """

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "TimeExpandedPlanner":
        return TimeExpandedPlanner(
            self, environment, self._compile(environment), progress
        )


class TimeExpandedPlanner(TimetablePlanner):
    """:class:`TimeExpanded` over one feed, its event graph already built."""

    def __init__(
        self,
        technique: TimeExpanded,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        self._expanded = _routelab.TimeExpanded.build(self.timetable, self.footpaths)

    @property
    def footprint(self) -> int:
        return super().footprint + self._expanded.footprint

    @property
    def num_events(self) -> int:
        """Nodes in the expanded graph — the number the model is judged on."""
        return self._expanded.num_events

    @property
    def searches(self) -> "Tuple[str, int]":
        return ("events", self.num_events)

    def route(
        self, origin: Origins, destination: Hashable, *, departing: Departure
    ) -> Answer:
        """The earliest arrival at ``destination``, leaving at ``departing``."""
        sources, at = self._sources(self._origin_ids(origin), departing)
        itinerary = self._expanded.earliest_arrival(sources, self.node_id(destination))
        return self._answer_itinerary(itinerary, destination, at)
