"""What a search answers with.

The result types every kernel returns, as protocols so that two
implementations of one problem are substitutable and a caller can say what it
needs. The searches themselves live beside the paper each one comes from, in
:mod:`routelab.kernels`.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from .._routelab import SearchResult

__all__ = ["EdgeResult", "Result", "SearchResult"]


@runtime_checkable
class Result(Protocol):
    """What anything downstream of a search actually needs from it.

    :class:`~routelab.SearchResult` is one implementation and the common one,
    but it is not the only shape a search comes in: a bidirectional search
    produces two trees and a meeting point, and a round-based transit search
    produces a label per stop per round; no amount of widening `SearchResult`
    would make either one. What the planners consume is this much and no more
    — a cost, a path, and how much work it took — so this is the contract, and
    a technique is free to satisfy it however its algorithm demands.

    Note what is *not* here: `order`. A search that settles in more than one
    direction has no single settle sequence, which is exactly why the count
    worth comparing across algorithms is :attr:`settled` rather than the length
    of a list.
    """

    def cost(self, node: int) -> Optional[int]:
        """The cheapest cost to ``node``, or ``None`` if it has none."""

    def path(self, node: int) -> "Optional[List[int]]":
        """Node ids from the source to ``node``, source first."""

    @property
    def settled(self) -> int:
        """How many nodes the search settled: the work it did."""


@runtime_checkable
class EdgeResult(Result, Protocol):
    """A result that walked the environment's own edges, and can say which.

    Everything a graph search returns is one of these, and it is what
    :meth:`~routelab.Journey.from_result` needs: a leg is an edge, and an edge
    is where provenance and geometry hang. A search that never touched the
    graph — a round-based one over routes and trips — is a :class:`Result` and
    not this, and answers with an itinerary instead.
    """

    def edge_path(self, node: int) -> "Optional[List[int]]":
        """Edge ids from the source to ``node`` — in the *caller's* graph."""
