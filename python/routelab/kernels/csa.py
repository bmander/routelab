"""Dibbelt, Pajor, Strasser & Wagner, *Intriguingly simple and fast transit
routing* (2013) — CSA."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

from .. import _routelab
from ..model.journey import Journey
from ..model.search import Result
from ..model.searchspace import Scan, SearchSpace
from .planner import Origins, Planner, TimetablePlanner

__all__ = ["CSA"]


class CSA(TimetablePlanner):
    """Connection scan (Dibbelt, Pajor, Strasser & Wagner, 2013).

        CSA().bind(env).route(origin, target, departing=time(8, 30))

    No graph, and no routes either: every connection sorted by departure into
    one array, a label per stop and a flag per trip, and one linear pass. A
    connection is reachable if its trip is flagged or its departure stop is
    labelled in time; a reachable one flags its trip and, if it improves its
    arrival stop, walks that stop's footpaths. With a target the scan stops
    once the array reaches the target's label; without one every stop holds
    its earliest arrival — which is why, like RAPTOR, this technique has a
    real :meth:`search` and something to draw.

    The same array read backwards is a **profile**: :meth:`profile` hands
    back every journey worth leaving on within a window, one per
    Pareto-optimal (departure, arrival) pair. That is the question a rider with
    a flexible morning asks, and the one rRAPTOR would answer for the
    round-based technique; here it is the paper's §4.

    Changing vehicles is instantaneous, as it is for the four techniques it
    is checked against; a minimum change time is a new kernel ``Transfer``
    constructor and would land in all five at once, so it is not a knob here.
    """

    options = frozenset({"departing"})
    #: ``until`` is real, but it is :meth:`profile`'s and not :meth:`route`'s —
    #: declared so the refusal a route() with one earns can say so itself.
    verbs = {"until": "profile"}

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._csa = _routelab.ConnectionScan.build(self.timetable, self.footpaths)

    def _footprint(self) -> int:
        return super()._footprint() + self._csa.footprint

    @property
    def num_trips(self) -> int:
        """Trips in the paper's sense: one per unbroken chain of connections."""
        self._bound()
        return self._csa.num_trips

    @property
    def num_connections(self) -> int:
        """The length of the array a query scans."""
        self._bound()
        return self._csa.num_connections

    # CSA keeps a label per stop, so it has a cost table like any graph search
    # and `Planner.route` — search, then read the journey off the result — is
    # the whole implementation; the family's `journey` reads an itinerary off
    # it. The itinerary hook the two Pyrga models need is not used here.
    _route = Planner._route

    def _search(self, starts: "Dict[int, int]", **options: Any) -> "_routelab.ScanSearch":
        """Scan — toward one target if given, else to every stop."""
        sources, at = self._sources(starts, options)
        target = None
        if options.get("targets"):
            target = self._single_target(options, "CSA, stopping at a target,")
        return self._csa.search(sources, target, at)

    def profile(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        departing: Any = None,
        until: Any = None,
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
        self._bound()
        starts = self._origin_ids(origin)
        target = self.node_id(destination)
        # The paper's one-to-one pruning wants the one stop the question is
        # about; with several origins on several clocks there is no one
        # profile to prune against, and the scan simply keeps everything.
        prune = next(iter(starts)) if len(starts) == 1 else None
        profile = self._csa.profile(target, opens, prune)
        found: "List[Tuple[int, _routelab.Itinerary]]" = []
        for stop, head_start in starts.items():
            for departs, itinerary in profile.journeys(stop, opens + head_start, closes + head_start):
                found.append((departs - head_start, itinerary))
        return self._merge_departures(found, destination)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """Every stop the scan labelled, by when it was reached."""
        self._no_other(options, "a scan")
        return Scan(self._bound(), result)
