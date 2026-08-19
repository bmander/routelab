"""What one query found: the routes, the space behind them, and the raw table.

A search works out more than the one journey a caller usually wants — every
stop's arrival, the rounds that reached them, the front over changes — and
throwing that away means searching twice for anyone who wanted a picture as
well as an answer. So a query answers with this, and everything else is read
off the search already run.

Three things, and no verbs beyond the one that fetches the space:

* :attr:`routes` — every journey worth having, best first. Usually one.
* :meth:`searchspace` — what the search looked at, drawn.
* :attr:`raw` — everything the kernel computed, on its own dense ids.

**A single route is a Pareto set of one.** That is not a convenience: a front
is the set of journeys no other journey beats on every criterion, and a
technique that tells journeys apart only by when they arrive has exactly one
such journey. So :class:`~routelab.Dijkstra` hands back a list of one and
:class:`~routelab.RAPTOR` hands back one entry per number of changes, and
neither is a special case of the other — the difference between the techniques
shows up in the length of the list rather than in a method one of them refuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Optional

from .journey import Journey
from .search import Result
from .searchspace import SearchSpace


@dataclass(frozen=True)
class Answer:
    """One query's answer, and the search it was read off.

        answer = planner.route("home", "work")
        answer.routes[0]                 # the cheapest journey
        answer.searchspace()             # the space behind it, same search
    """

    #: The planner that was asked — what knows the labels.
    planner: Any
    #: The destination asked about, in the caller's own labels.
    destination: Hashable
    #: Every journey worth having, **best first**: the Pareto set over whatever
    #: criteria this technique tells journeys apart by. One entry for a
    #: technique that knows only arrival time; one per number of changes for a
    #: round-based one, the earliest arrival first and each later entry trading
    #: arrival for fewer changes. Empty where nothing was reachable under the
    #: bounds given, so this is a list always and never ``None``.
    routes: "List[Journey]" = field(default_factory=list)
    #: Everything the kernel computed, on dense ids — ``None`` from a technique
    #: that answers with an itinerary and keeps no table. The escape hatch.
    raw: Optional[Result] = None

    def searchspace(self, magnitude: Optional[str] = None) -> SearchSpace:
        """What the search looked at, in a form something can draw.

        A method rather than an attribute because the shape takes an option —
        ``magnitude="nodes"`` counts settled nodes where the default
        accumulates travel time — and because building it is real work over
        what was settled, which a caller who only wanted a route should not
        pay.

        ``magnitude`` means something only to a technique that grows a tree,
        and is passed on only when it was asked for, so that the ones exploring
        some other shape refuse it in their own signature rather than here.
        """
        if self.raw is None:
            raise NotImplementedError(
                f"{type(self.planner).__name__} answers with a journey and keeps "
                f"no search space, so there is nothing to draw."
            )
        if magnitude is None:
            return self.planner.explored(self.raw)
        return self.planner.explored(self.raw, magnitude=magnitude)

    def __repr__(self) -> str:
        held = "no table" if self.raw is None else repr(self.raw)
        if not self.routes:
            return f"Answer(nothing reachable, {held})"
        best = repr(self.routes[0])
        more = "" if len(self.routes) == 1 else f" +{len(self.routes) - 1} more"
        return f"Answer({best}{more}, {held})"
