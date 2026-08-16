"""Delling, Pajor & Werneck, *Round-based public transit routing* (2012)."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional

from .. import _routelab
from ..model.journey import Journey
from ..model.search import Result
from ..model.searchspace import Rounds, SearchSpace
from .planner import Origins, Planner, TimetablePlanner

__all__ = ["RAPTOR"]


class RAPTOR(TimetablePlanner):
    """Round-based public transit routing (Delling, Pajor & Werneck, 2012).

        RAPTOR().bind(env).route(origin, target, departing=time(8, 30))

    No graph. Round `k` scans, once each, every route touched in round `k-1`
    and rides the earliest trip that can be caught, so after `k` rounds every
    stop holds its earliest arrival with at most `k-1` changes. That is
    one-to-all by construction — a label per stop per round — which is why,
    like :class:`CSA` and unlike the two Pyrga models, this one has a real
    :meth:`search` and something to draw. It is Pareto by construction too: arrival against
    changes, one incomparable journey per round that improved something, which
    is what :meth:`frontier` hands back.

    ``max_transfers`` is a query option, not a constructor argument, for the
    reason ``max_cost`` is: it bounds one question, not the technique.
    Changing vehicles is instantaneous, as it is for the two Pyrga models and
    CSA it is checked against; a minimum change time is a new kernel
    ``Transfer`` constructor and would land in all four at once, so it is not
    a knob here.
    """

    options = frozenset({"departing", "max_transfers"})

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._raptor = _routelab.Raptor.build(self.timetable, self.footpaths)

    def _footprint(self) -> int:
        return super()._footprint() + self._raptor.footprint

    @property
    def num_routes(self) -> int:
        """Routes in the paper's sense — distinct stop sequences whose trips
        never overtake — which is more than a feed's own count of routes."""
        self._bound()
        return self._raptor.num_routes

    @property
    def num_trips(self) -> int:
        self._bound()
        return self._raptor.num_trips

    @staticmethod
    def _rounds(max_transfers: Optional[int]) -> Optional[int]:
        """`k` changes is `k + 1` trips is `k + 1` rounds."""
        if max_transfers is None:
            return None
        if max_transfers < 0:
            raise ValueError(
                f"max_transfers counts changes of vehicle, so it cannot be {max_transfers}"
            )
        return int(max_transfers) + 1

    # RAPTOR keeps a label per stop per round, so it has a cost table like any
    # graph search and `Planner.route` — search, then read the journey off the
    # result — is the whole implementation; the family's `journey` reads an
    # itinerary off it. The itinerary hook the two Pyrga models need is not
    # used here.
    _route = Planner._route

    def _search(self, starts: "Dict[int, int]", **options: Any) -> "_routelab.RaptorSearch":
        """Run the rounds — toward one target if given, else to every stop."""
        sources, at = self._sources(starts, options)
        target = None
        if options.get("targets"):
            target = self._single_target(options, "RAPTOR, pruning toward a target,")
        return self._raptor.search(
            sources, target, self._rounds(options.get("max_transfers")), at
        )

    def journeys(self, result: Result, destination: Hashable) -> "List[Journey]":
        """Every journey a kept search holds for ``destination``: the earliest
        arrival for each number of changes, fewest changes first, none
        dominated by another.

        The counterpart to :meth:`journey`, and what :meth:`frontier` is in
        terms of — so a caller holding a search never pays for a second one.
        """
        compiled = self._bound()
        target = self.node_id(destination)
        return [
            Journey.from_itinerary(compiled, itinerary, destination, result.departing)
            for itinerary in result.itineraries(target)  # type: ignore[attr-defined]
        ]

    def frontier(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> "List[Journey]":
        """Every journey worth having, in one call: the Pareto front over
        arrival time and changes, where :meth:`route` returns only its last
        entry.
        """
        options = self._options(options)
        starts = self._origin_ids(origin)
        result = self._search(starts, targets=[self.node_id(destination)], **options)
        return self.journeys(result, destination)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """Every stop the rounds reached, by the round that first got there."""
        self._no_other(options, "rounds")
        return Rounds(self._bound(), result)
