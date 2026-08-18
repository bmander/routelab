"""What one query found, kept so it can be asked more than one thing.

A search is the expensive part and it computes more than the one journey a
caller usually wants: every stop's arrival, the rounds that reached them, the
front over changes. :meth:`~routelab.kernels.Planner.route` throws all of that
away and returns the journey, which is right when the journey is all you
wanted and wasteful the moment it is not — drawing the search behind a route
used to mean searching twice, once for the answer and once for the picture.

:meth:`~routelab.kernels.Planner.ask` returns this instead: the journey, and
the working space it was read off, so every other question is answered from
the search already run. Reading is not free — a path walks parent pointers and
a search space is built from what was settled — but nothing here searches
again.

Every technique answers `ask`, including the ones that keep no table. For those
:attr:`result` is ``None`` and the questions that need one refuse in the
technique's own words, naming who to ask instead. That is the difference the
shelf is partly about, and it survives the uniformity rather than being papered
over by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, List, Optional

from .journey import Journey
from .search import Result
from .searchspace import SearchSpace


@dataclass(frozen=True)
class Answer:
    """One query's journey, and the search it came from.

        answer = planner.ask("home", "work")
        answer.journey                      # what route() returns
        answer.explored()                   # the space behind it, same search
    """

    #: The planner that was asked, which is what knows the labels.
    planner: Any
    #: The destination asked about, in the caller's own labels.
    destination: Hashable
    #: The cheapest journey, or ``None`` where the destination cannot be
    #: reached under the bounds given.
    journey: Optional[Journey]
    #: Everything the kernel computed, on dense ids — ``None`` from a technique
    #: that answers with an itinerary and keeps no table. The escape hatch, and
    #: what the readers below are answered from.
    result: Optional[Result]

    def explored(self, **options: Any) -> SearchSpace:
        """The search space behind this answer, in a form something can draw.

        The technique's own :meth:`~routelab.kernels.Planner.explored`, asked
        of the search already run. One that keeps no space says so and names
        the ones that do.
        """
        return self.planner.explored(self.result, **options)

    def frontier(self) -> "List[Journey]":
        """Every journey worth having: the Pareto front over arrival time and
        changes, of which :attr:`journey` is the last entry.

        Read off the same search, so a caller who wanted both pays for one. A
        technique that keeps no front refuses here in its own words, naming the
        ones that do.
        """
        return self.planner.journeys(self.result, self.destination)

    def __repr__(self) -> str:
        held = "no table" if self.result is None else repr(self.result)
        return f"Answer({self.journey!r}, {held})"
