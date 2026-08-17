"""Delling, Dibbelt, Pajor & Werneck, *Public transit labeling* (2015) — PTL."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

from .. import _routelab
from ..model.journey import Journey
from .planner import Origins, TimetablePlanner

__all__ = ["PTL"]


class PTL(TimetablePlanner):
    """Public transit labeling (Delling, Dibbelt, Pajor & Werneck, 2015).

        PTL().bind(env).route(origin, target, departing=time(8, 30))

    RAPTOR and CSA stopped building a graph; PTL builds the biggest one of
    all — a vertex per event, an arc for every wait, ride and walk — and then
    never searches it. Every arc points forward in time, so "is there a
    journey from this event to that one" is reachability in a DAG, and 2-hop
    **hub labels** answer that by intersecting two sorted lists: each event
    gets a forward label of hubs it reaches and a backward label of hubs that
    reach it, such that any reachable pair shares one. A query is a handful of
    those intersections; the search happened at :meth:`bind`, once, for every
    query at once — which is why binding takes minutes and hundreds of
    megabytes on a city, and :attr:`footprint` and the board's progress bar
    are worth watching.

    Earliest arrival is the paper's event-label query: enter at the first
    event at the origin after departing, binary-search the target's events
    for the first one reachable. :meth:`profile` is its stop-label sweep — a
    label per stop that keeps, per hub, only the latest departure that
    reaches it and the earliest arrival from it — and hands back every
    journey worth leaving on within a window, the same question CSA's
    profile answers. Walks at either end are the paper's *superlabels*,
    assembled during the query from the neighbouring stops' labels.

    The labels are computed by pruned labeling (Akiba, Iwata & Yoshida,
    2013) rather than the paper's RXL, which it uses as a black box; the
    labels are larger and every query is the paper's own. What is not here:
    its multicriteria labels (``max_transfers`` names RAPTOR), and a minimum
    change time, which no timetable technique here honours yet. Like the two
    Pyrga models, PTL keeps no table, so :meth:`search` and :meth:`explored`
    say so and point at the techniques that do.
    """

    options = frozenset({"departing"})
    #: ``until`` is real, but it is :meth:`profile`'s and not :meth:`route`'s —
    #: declared so the refusal a route() with one earns can say so itself.
    verbs = {"until": "profile"}

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._ptl = _routelab.PublicTransitLabeling.build(self.timetable, self.footpaths, progress)

    def _footprint(self) -> int:
        return super()._footprint() + self._ptl.footprint

    @property
    def num_events(self) -> int:
        """Vertices of the event graph the labels are over: distinct
        (stop, time) pairs."""
        self._bound()
        return self._ptl.num_events

    @property
    def num_hubs(self) -> int:
        """Entries in the event labels, both directions — the size of the
        labeling, and the number a query's work is a share of."""
        self._bound()
        return self._ptl.num_hubs

    @property
    def hubs_per_label(self) -> float:
        """Average hubs per event label — the paper's own figure of merit
        for a labeling."""
        self._bound()
        return self._ptl.hubs_per_label

    @property
    def searches(self) -> "Tuple[str, int]":
        """What a query reads: label entries, against the size of the
        labeling.

        Unlike every other technique here, a journey's ``settled`` is not a
        count of distinct things and so not a share of this one: an entry is
        re-read on every probe of the binary search, at every stop the target
        can be walked to from. It is the paper's own per-query hubs figure,
        and the denominator is what a labeling of this network costs.
        """
        return ("hubs", self.num_hubs)

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        return self._ptl.earliest_arrival(sources, target)

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
        departure first, none dominated by another — the same answer
        :meth:`CSA.profile` gives, from the stop labels instead of a scan.

        A departure is the *latest* moment you can leave and still make that
        arrival. Several origins with head starts are read out each on their
        own clock and merged; a journey the direct walk from origin to
        destination would beat is left out, since a walk leaves whenever you
        do and a profile of pairs cannot hold it.
        """
        opens, closes = self._window(departing, until)
        self._bound()
        starts = self._origin_ids(origin)
        target = self.node_id(destination)
        found: "List[Tuple[int, _routelab.Itinerary]]" = []
        for stop, head_start in starts.items():
            for departs, itinerary in self._ptl.profile(stop, target, opens + head_start, closes + head_start):
                found.append((departs - head_start, itinerary))
        return self._merge_departures(found, destination)
