"""Which node to contract next, and how hard to look before adding a shortcut.

Contraction removes nodes one at a time, and the order decides everything about
the hierarchy that comes out. Contract a city's ring road early and every route
across town needs a shortcut to stand in for it; contract the cul-de-sacs first
and most of them cost nothing at all.

Ordering is a policy, so it is a spec, the same shape as
:mod:`routelab.heuristics` member for member: declare what it needs with
``requires``, check it against an environment with ``missing_from``, and turn it
into the artifact with ``bind``. The parallel is exact — ``Landmarks(16,
selection="farthest")`` is also a policy for choosing nodes, and binding it runs
the selection and hands back the table. Here binding runs the contraction and
hands back the hierarchy.

Both orderings here produce **identical distances**. Ordering is a performance
choice and never a correctness one: a hierarchy built by throwing darts answers
exactly what a carefully built one answers, it just carries far more shortcuts
to do it.
"""

from __future__ import annotations

from typing import Optional

from . import _routelab
from .environment import CompiledEnvironment

__all__ = ["EdgeDifference", "Ordering", "RandomOrder"]


class Ordering:
    """A specification for a contraction order, not yet applied to a graph."""

    #: What this ordering needs an environment to supply, by name. Empty for
    #: both orderings here, which measure the graph itself — but an ordering
    #: that cut the map geometrically would want ``"positions"``, and this is
    #: where it would say so. The mirror of
    #: :attr:`routelab.heuristics.Heuristic.requires`.
    requires: "frozenset[str]" = frozenset()

    def __init__(self, max_settled: int = 500, max_hops: int = 5):
        """
        Args:
            max_settled: How many nodes a witness search may settle before
                giving up, and
            max_hops: how many hops it may take.

        Giving up early is safe: it means a shortcut is added that may not have
        been needed, which costs space and query time. It never costs accuracy,
        which is why these are tuning knobs and not correctness ones.
        """
        self.max_settled = max_settled
        self.max_hops = max_hops

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """What this ordering needs that ``compiled`` does not provide."""
        return cls.requires - compiled.provides

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.ContractionHierarchy":
        """Contract ``compiled``'s graph in this order, and return the result."""
        raise NotImplementedError


class EdgeDifference(Ordering):
    """Contract whichever node costs least to lose.

        >>> EdgeDifference()
        EdgeDifference(deleted_neighbours=True)

    The paper's practical recipe. A node's priority is how many shortcuts its
    contraction would add minus how many edges it would remove, so nodes whose
    neighbours already reach each other go first and the ones everything routes
    through go last. Priorities go stale as the graph changes around them, so
    they are recomputed lazily — on the way out of the queue, and put back if the
    node turns out no longer to be the cheapest.

    Args:
        deleted_neighbours: Also discourage contracting next to nodes already
            contracted, which stops the hierarchy piling up in one corner of the
            map before spreading.
        max_settled: How many nodes a witness search may settle.
        max_hops: How many hops a witness search may take.
    """

    def __init__(
        self,
        deleted_neighbours: bool = True,
        max_settled: int = 500,
        max_hops: int = 5,
    ):
        super().__init__(max_settled, max_hops)
        self.deleted_neighbours = deleted_neighbours

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.ContractionHierarchy":
        return _routelab.ContractionHierarchy.edge_difference(
            compiled.graph,
            deleted_neighbours=self.deleted_neighbours,
            max_settled=self.max_settled,
            max_hops=self.max_hops,
            progress=progress,
        )

    def __repr__(self) -> str:
        return f"EdgeDifference(deleted_neighbours={self.deleted_neighbours})"


class RandomOrder(Ordering):
    """Contract in a fixed arbitrary order.

        >>> RandomOrder(seed=1)
        RandomOrder(seed=1)

    The control, exactly as ``Landmarks(selection="random")`` is for landmarks.
    It answers the same distances and builds a far worse hierarchy, which is the
    measurement that says the ordering heuristic is doing real work rather than
    decorating a result that would have happened anyway.
    """

    def __init__(self, seed: int = 0, max_settled: int = 500, max_hops: int = 5):
        super().__init__(max_settled, max_hops)
        self.seed = seed

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.ContractionHierarchy":
        return _routelab.ContractionHierarchy.random(
            compiled.graph,
            seed=self.seed,
            max_settled=self.max_settled,
            max_hops=self.max_hops,
            progress=progress,
        )

    def __repr__(self) -> str:
        return f"RandomOrder(seed={self.seed})"
