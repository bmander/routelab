"""Dibbelt, Pajor, Strasser & Wagner, *Intriguingly simple and fast transit
routing* (2013) — CSA."""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

from .. import _routelab
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..model.journey import Journey
from ..model.searchspace import Scan
from ..util.clock import Departure
from .planner import Origins, TimetablePlanner, TimetableTechnique

__all__ = ["CSA", "CSAPlanner"]


class CSA(TimetableTechnique):
    """Connection scan (Dibbelt, Pajor, Strasser & Wagner, 2013).

        CSA().bind(env).route(origin, target, departing=time(8, 30))

    No graph, and no routes either: every connection sorted by departure into
    one array, a label per stop and a flag per trip, and one linear pass. A
    connection is reachable if its trip is flagged or its departure stop is
    labelled in time; a reachable one flags its trip and, if it improves its
    arrival stop, walks that stop's footpaths. With a target the scan stops
    once the array reaches the target's label; without one every stop holds
    its earliest arrival — which is why, like RAPTOR, this technique has a
    real ``search`` and something to draw.

    The same array read backwards is a **profile**:
    :meth:`~CSAPlanner.profile` hands back every journey worth leaving on
    within a window, one per Pareto-optimal (departure, arrival) pair. That is
    the question a rider with a flexible morning asks, and the one rRAPTOR
    would answer for the round-based technique; here it is the paper's §4.

    Changing vehicles is instantaneous, as it is for the four techniques it
    is checked against; a minimum change time is a new kernel ``Transfer``
    constructor and would land in all five at once, so it is not a knob here.
    """

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "CSAPlanner":
        return CSAPlanner(self, environment, self._compile(environment), progress)


class CSAPlanner(TimetablePlanner):
    """:class:`CSA` over one feed, its connections already sorted."""

    def __init__(
        self,
        technique: CSA,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        self._csa = _routelab.ConnectionScan.build(self.timetable, self.footpaths)

    @property
    def footprint(self) -> int:
        return super().footprint + self._csa.footprint

    @property
    def num_trips(self) -> int:
        """Trips in the paper's sense: one per unbroken chain of connections."""
        return self._csa.num_trips

    @property
    def num_connections(self) -> int:
        """The length of the array a query scans."""
        return self._csa.num_connections

    def route(
        self, origin: Origins, destination: Hashable, *, departing: Departure
    ) -> Answer:
        """The earliest arrival at ``destination``, and the table it came off."""
        return self._answer(
            self.search(origin, departing=departing, target=self.node_id(destination)),
            destination,
        )

    def search(
        self, origins: Origins, *, departing: Departure, target: Optional[int] = None
    ) -> "_routelab.ScanSearch":
        """Scan — toward one target if given, else to every stop."""
        sources, at = self._sources(self._origin_ids(origins), departing)
        return self._search_stops(sources, at, None, target)

    def _search_stops(
        self,
        sources: "List[Tuple[int, int]]",
        departing: int,
        max_transfers: Optional[int],
        target: Optional[int] = None,
    ) -> "_routelab.ScanSearch":
        """The scan, from stops already on the service-day clock.

        The seam :class:`~routelab.ULTRA` reaches through, which is the only
        way a ``max_transfers`` can arrive here at all — a caller writing
        ``CSA()`` never sees this argument, because a scan counts nothing but
        time and would have to ignore it.
        """
        if max_transfers is not None:
            raise TypeError(
                "CSA tells journeys apart by when they arrive and counts no "
                "changes, so it has no max_transfers to cap; the round-based "
                "techniques do — ULTRA(RAPTOR())."
            )
        return self._csa.search(sources, target=target, departing=departing)

    def profile(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        departing: Departure,
        until: Departure,
    ) -> "List[Journey]":
        """Every journey worth leaving on between ``departing`` and ``until``:
        one per Pareto-optimal pair of departure and arrival, earliest
        departure first, none dominated by another.

        A departure is the *latest* moment you can leave and still make that
        arrival — the paper's profile is a step function and these are its
        steps. Several origins with head starts are read out each on their
        own clock and merged; a journey the direct walk from origin to
        destination would beat is left out, since a walk leaves whenever you
        do and a profile of pairs cannot hold it.
        """
        opens, closes = self._window(departing, until)
        starts = self._origin_ids(origin)
        target = self.node_id(destination)
        # The paper's one-to-one pruning wants the one stop the question is
        # about; with several origins on several clocks there is no one
        # profile to prune against, and the scan simply keeps everything.
        prune = next(iter(starts)) if len(starts) == 1 else None
        profile = self._csa.profile(target, opens, prune)
        found: "List[Tuple[int, _routelab.Itinerary]]" = []
        for stop, head_start in starts.items():
            for departs, itinerary in profile.journeys(
                stop, opens + head_start, closes + head_start
            ):
                found.append((departs - head_start, itinerary))
        return self._merge_departures(found, destination)

    def explored(self, result: "_routelab.ScanSearch") -> Scan:
        """Every stop the scan labelled, by when it was reached."""
        return Scan(self.compiled, result)
