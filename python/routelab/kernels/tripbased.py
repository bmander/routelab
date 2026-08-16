"""Witt, *Trip-based public transit routing* (2015)."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

from .. import _routelab
from ..model.journey import Journey
from ..model.search import Result
from ..model.searchspace import SearchSpace, Segments
from .planner import Front, Origins, Planner, TimetablePlanner

__all__ = ["TripBased"]


class TripBased(Front, TimetablePlanner):
    """Trip-based public transit routing (Witt, 2015).

        TripBased().bind(env).route(origin, target, departing=time(8, 30))

    No graph, and no labels on stops either: RAPTOR and CSA label stops,
    this technique labels **trips**. Binding computes, for every trip and
    every stop it reaches, which trips a rider alighting there could change
    onto — the *transfer set* — and then drops every transfer that never
    reaches anywhere sooner, which is most of them. A query is a
    breadth-first sweep over trip segments: round ``n`` scans every trip
    reached with ``n`` changes, checks whether it reaches the target, and
    follows its transfers into round ``n+1``. That is Pareto over arrival
    and changes by construction, like RAPTOR — :meth:`frontier` hands the
    front back — and it is point-to-point by construction, so
    :meth:`search` needs its target and refuses without one; what it keeps
    to draw is the segments it scanned, by round.

    ``reduce`` is the paper's own control: ``TripBased(reduce=False)`` keeps
    every transfer and answers identically, more slowly. A policy, never a
    correctness choice — the same footing as ``RandomOrder()`` for a
    contraction hierarchy. ``max_transfers`` is a query option, as it is for
    RAPTOR: it bounds one question, not the technique.

    :meth:`profile` runs the same sweep once per moment the origin offers a
    departure in a window, latest first, keeping the labels between runs
    (the paper's §3.3), and hands back every journey worth leaving on: one
    per Pareto-optimal (departure, arrival, changes).

    Changing vehicles is instantaneous, as it is for the four techniques it
    is checked against; a minimum change time is a new kernel ``Transfer``
    constructor and would land in all five at once, so it is not a knob here.
    """

    options = frozenset({"departing", "max_transfers"})
    #: ``until`` is real, but it is :meth:`profile`'s and not :meth:`route`'s —
    #: declared so the refusal a route() with one earns can say so itself.
    verbs = {"until": "profile"}

    def __init__(self, reduce: bool = True):
        self.reduce = bool(reduce)

    def __repr__(self) -> str:
        return self._describe(*(() if self.reduce else ("reduce=False",)))

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._trips = _routelab.TripBased.build(
            self.timetable, self.footpaths, self.reduce, progress
        )

    def _footprint(self) -> int:
        return super()._footprint() + self._trips.footprint

    @property
    def num_lines(self) -> int:
        """Lines in the paper's sense — distinct stop sequences whose trips
        never overtake — which is more than a feed's own count of routes."""
        self._bound()
        return self._trips.num_lines

    @property
    def num_trips(self) -> int:
        self._bound()
        return self._trips.num_trips

    @property
    def num_transfers(self) -> int:
        """Transfers kept: the set a query scans."""
        self._bound()
        return self._trips.num_transfers

    @property
    def num_initial_transfers(self) -> int:
        """Transfers computed before reduction dropped the ones never needed."""
        self._bound()
        return self._trips.num_initial_transfers

    @property
    def searches(self) -> "Tuple[str, int]":
        """This technique labels trips, so a settled count is a share of
        them — the paper's own footnote on how to compare it with the
        stop-labelling techniques."""
        return ("trips", self.num_trips)

    # The query keeps a table — one journey per number of changes at its
    # target — so `Planner.route`, which searches and reads the journey off
    # the result, is the whole implementation; the family's `journey` reads
    # an itinerary off it. The itinerary hook the two Pyrga models need is
    # not used here.
    _route = Planner._route

    def _search(self, starts: "Dict[int, int]", **options: Any) -> "_routelab.TripBasedSearch":
        """Sweep the trip segments toward the one target."""
        sources, at = self._sources(starts, options)
        target = self._single_target(options, "TripBased, sweeping trips toward a target,")
        return self._trips.search(sources, target, self._changes(options.get("max_transfers")), at)

    def profile(
        self,
        origin: Origins,
        destination: Hashable,
        *,
        departing: Any = None,
        until: Any = None,
    ) -> "List[Journey]":
        """Every journey worth leaving on between ``departing`` and ``until``:
        one per Pareto-optimal (departure, arrival, changes), earliest
        departure first and, within one, fewest changes first.

        A departure is the *latest* moment you can leave and still make that
        journey. Several origins with head starts are read out each on their
        own clock and merged; a journey the direct walk from origin to
        destination would beat is left out, since a walk leaves whenever you
        do and a profile of departures cannot hold it.
        """
        opens, closes = self._window(departing, until)
        compiled = self._bound()
        starts = self._origin_ids(origin)
        target = self.node_id(destination)
        found: "List[Tuple[int, int, int, _routelab.Itinerary]]" = []
        for stop, head_start in starts.items():
            profile = self._trips.profile(stop, target, opens + head_start, closes + head_start)
            for departs, itinerary in profile.journeys():
                found.append((departs - head_start, itinerary.arrives, itinerary.transfers, itinerary))
        # One origin needs no merge: the kernel already answered with a Pareto
        # set, in the order this returns. Several are merged on the query's
        # clock — latest departure first, keeping each journey no
        # later-leaving one arrives no later than with no more changes.
        kept = found
        if len(starts) > 1:
            found.sort(key=lambda entry: (entry[0], entry[2], -entry[1]))
            kept = []
            for entry in reversed(found):
                left, arrives, transfers, _ = entry
                dominated = any(
                    a <= arrives and n <= transfers and (d, a, n) != (left, arrives, transfers)
                    for d, a, n, _ in kept
                )
                if not dominated:
                    kept.append(entry)
            kept.reverse()
        # The journey is built from the survivors only, since building one asks
        # the environment for an edge per leg.
        return [
            Journey.from_itinerary(compiled, itinerary, destination, left)
            for left, _, _, itinerary in kept
        ]

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """Every trip segment the sweep scanned, by the round it was reached in."""
        self._no_other(options, "trip segments")
        return Segments(self._bound(), result)
