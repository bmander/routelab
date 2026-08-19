"""Delling, Pajor & Werneck, *Round-based public transit routing* (2012)."""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

from .. import _routelab
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..model.searchspace import Rounds
from ..util.clock import Departure
from .planner import Front, Origins, TimetablePlanner, TimetableTechnique

__all__ = ["RAPTOR", "RAPTORPlanner"]


class RAPTOR(TimetableTechnique):
    """Round-based public transit routing (Delling, Pajor & Werneck, 2012).

        RAPTOR().bind(env).route(origin, target, departing=time(8, 30))

    No graph. Round `k` scans, once each, every route touched in round `k-1`
    and rides the earliest trip that can be caught, so after `k` rounds every
    stop holds its earliest arrival with at most `k-1` changes. That is
    one-to-all by construction — a label per stop per round — which is why,
    like :class:`~routelab.CSA` and unlike the two Pyrga models, this one has a
    real ``search`` and something to draw. It is Pareto by construction too:
    arrival against changes, one incomparable journey per round that improved
    something, which is what :meth:`~routelab.kernels.Front.journeys` hands back.

    ``max_transfers`` is a query option, not a constructor argument, for the
    reason ``max_cost`` is: it bounds one question, not the technique.
    Changing vehicles is instantaneous, as it is for the two Pyrga models,
    CSA and TripBased it is checked against; a minimum change time is a new
    kernel ``Transfer`` constructor and would land in all five at once, so it
    is not a knob here.
    """

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "RAPTORPlanner":
        return RAPTORPlanner(self, environment, self._compile(environment), progress)


class RAPTORPlanner(Front, TimetablePlanner):
    """:class:`RAPTOR` over one feed, its routes and trips already indexed."""

    def __init__(
        self,
        technique: RAPTOR,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        self._raptor = _routelab.Raptor.build(self.timetable, self.footpaths)

    @property
    def footprint(self) -> int:
        return super().footprint + self._raptor.footprint

    @property
    def num_routes(self) -> int:
        """Routes in the paper's sense — distinct stop sequences whose trips
        never overtake — which is more than a feed's own count of routes."""
        return self._raptor.num_routes

    @property
    def num_trips(self) -> int:
        return self._raptor.num_trips

    def route(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        departing: Departure,
        max_transfers: Optional[int] = None,
    ) -> Answer:
        """Every journey worth having to ``destination``, earliest arrival first.

        One entry per number of changes: the front the rounds produce, reversed
        so that ``routes[0]`` is the journey every other timetable technique
        here would have given.
        """
        return self._answer_front(
            self._run(
                self._origin_ids(origin),
                self.node_id(destination),
                departing,
                max_transfers,
            ),
            destination,
        )

    def search(
        self,
        origins: Origins,
        *,
        departing: Departure,
        target: Optional[int] = None,
        max_transfers: Optional[int] = None,
    ) -> "_routelab.RaptorSearch":
        """Run the rounds — toward one target if given, else to every stop."""
        return self._run(self._origin_ids(origins), target, departing, max_transfers)

    def _run(
        self,
        starts: "Dict[int, int]",
        target: Optional[int],
        departing: Departure,
        max_transfers: Optional[int],
    ) -> "_routelab.RaptorSearch":
        sources, at = self._sources(starts, departing)
        return self._search_stops(sources, at, max_transfers, target)

    def _search_stops(
        self,
        sources: "List[Tuple[int, int]]",
        departing: int,
        max_transfers: Optional[int],
        target: Optional[int] = None,
    ) -> "_routelab.RaptorSearch":
        """The rounds, from stops already on the service-day clock.

        The seam :class:`~routelab.ULTRA` reaches through: it has worked out
        which stops the query can reach and when, and wants the table over
        those and nothing else — no labels, no head starts, no journey. `k`
        changes is `k + 1` trips is `k + 1` rounds.
        """
        changes = self._changes(max_transfers)
        return self._raptor.search(
            sources,
            target=target,
            max_rounds=None if changes is None else changes + 1,
            departing=departing,
        )

    def explored(self, result: "_routelab.RaptorSearch") -> Rounds:
        """Every stop the rounds reached, by the round that first got there."""
        return Rounds(self.compiled, result)
