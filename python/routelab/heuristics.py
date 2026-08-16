"""Heuristics: turning what the layers know into a bound a search can use.

A heuristic answers "how much is left?" — and for A* to stay correct, it must
never answer too much. The estimate has to be **admissible**: a lower bound on
the true remaining cost. Overestimate and A* still returns a path, still quickly,
just not the cheapest one, with nothing in the result to say so. That silence is
why the checking happens here, up front, rather than being left to whoever reads
the answer.

The objects in this module are specifications, not estimates. You write
``Euclidean()``, and :meth:`Heuristic.bind` turns it into a kernel heuristic
against a particular compiled environment — gathering coordinates, finding the
rate, and refusing the job if the layers cannot support it. Binding happens once
per planner; the bound object is reused for every query, whatever its target.
"""

from __future__ import annotations

from typing import Hashable, List, Optional

from . import _routelab
from .environment import CompiledEnvironment

__all__ = ["Euclidean", "Heuristic", "Landmarks", "Zero"]

#: How many missing labels to name in an error before trailing off.
_MAX_REPORTED = 5


class Heuristic:
    """A specification for an estimate, not yet attached to an environment."""

    #: What this heuristic needs an environment to supply, by name. Declared
    #: rather than discovered: :meth:`bind` raises a good error when something
    #: is missing, but that only answers the question after you have committed
    #: to asking it. This answers "which of these can I even run here?" — which
    #: is the question worth asking when a dozen techniques are on the shelf.
    #:
    #: The mirror of :attr:`routelab.Planner.accepts`, which does the same for
    #: cost models.
    requires: "frozenset[str]" = frozenset()

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """What this heuristic needs that ``compiled`` does not provide.

        Empty means it can be bound. Cheap, and it touches nothing — no
        preprocessing runs to find out.
        """
        return cls.requires - compiled.provides

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        """Build the kernel heuristic for ``compiled``, or explain what is missing.

        Raises:
            ValueError: If the environment lacks what this heuristic needs.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Zero(Heuristic):
    """Estimate nothing, which makes A* into Dijkstra.

        >>> Zero()
        Zero()

    Trivially admissible, and the control every other heuristic is measured
    against: same answers, and the node count A* has to beat.
    """

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        return _routelab.Heuristic.zero()


class Euclidean(Heuristic):
    """Straight-line distance, priced at the fastest rate in the environment.

        >>> Euclidean()
        Euclidean()

    Needs two things from the layers: coordinates for every node, and a
    ``cost_per_distance`` on every layer that contributes edges. The bound uses
    the *smallest* of those rates, because a path may ride the fastest layer the
    whole way — which is why one layer that declines to declare a rate disables
    the heuristic entirely rather than being assumed slow.

    Args:
        cost_per_distance: Override the rate taken from the layers. Useful for
            deliberately weakening the bound in an experiment; if you set it
            higher than some layer actually charges, the bound stops being
            admissible and A* stops returning cheapest paths.
    """

    requires = frozenset({"positions", "cost_per_distance"})

    def __init__(self, cost_per_distance: Optional[float] = None):
        self.cost_per_distance = cost_per_distance

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        rate = self.cost_per_distance
        if rate is None:
            rate = compiled.cost_per_distance
        if rate is None:
            raise ValueError(
                "Euclidean needs to know how cheaply this environment covers "
                "distance, and at least one layer that contributes edges did not "
                "say. Give every such layer a cost_per_distance=..., pass one to "
                "Euclidean(...) to override, or use Zero()."
            )

        xs: List[float] = []
        ys: List[float] = []
        missing: List[Hashable] = []
        for node_id, point in enumerate(compiled.positions):
            if point is None:
                missing.append(compiled.label(node_id))
            else:
                xs.append(point[0])
                ys.append(point[1])

        if missing:
            shown = ", ".join(repr(label) for label in missing[:_MAX_REPORTED])
            if len(missing) > _MAX_REPORTED:
                shown += f", and {len(missing) - _MAX_REPORTED} more"
            raise ValueError(
                f"Euclidean needs a position for every node; {len(missing)} have "
                f"none ({shown}). Register Positions({{...}}) covering them, or "
                f"use Zero()."
            )

        return _routelab.Heuristic.euclidean(xs, ys, float(rate))

    def __repr__(self) -> str:
        if self.cost_per_distance is None:
            return "Euclidean()"
        return f"Euclidean(cost_per_distance={self.cost_per_distance})"


class Landmarks(Heuristic):
    """Distances measured from a few fixed nodes, combined by triangle inequality.

        >>> Landmarks(16)
        Landmarks(16, selection='farthest')

    Where :class:`Euclidean` assumes — that nothing crosses ground faster than
    the fastest layer — this one *measures*. For every landmark it precomputes
    the distance to and from every node, and a bound follows from the triangle
    inequality. Because those distances were measured through the network, the
    bound already knows about one-way streets, dead ends, water, and the fact
    that most roads are not motorways.

    That last point is what makes it worth the memory on a road network. A
    straight-line bound must be priced at the network's top speed, so on a
    graph spanning 5 to 31 m/s it is roughly three times too optimistic
    everywhere. A landmark bound has no such problem.

    It also asks nothing of the environment — no coordinates, no declared
    speeds — which makes it the heuristic for networks whose geometry is
    unknown or meaningless.

    What it costs is paid at :meth:`bind` time: two full searches of the graph
    per landmark, and two ``u32`` per landmark per node held for as long as the
    planner lives. Sixteen landmarks over a 250,000-node city is a few seconds
    and about 32 MB.

    Args:
        count: How many landmarks. More is a sharper bound and more memory,
            with the usual diminishing returns; 16 is the conventional default.
        selection: ``"farthest"`` spreads them to the edges of the network,
            which is where they inform. ``"random"`` is the control — a
            landmark set that cannot beat random is not earning its memory.
        seed: Fixes the choice, so a benchmark can be repeated.
    """

    #: Nothing: a landmark bound is measured from the graph itself. What it
    #: wants instead is time and memory, which no environment can supply.
    requires = frozenset()

    def __init__(self, count: int = 16, selection: str = "farthest", seed: int = 0):
        self.count = count
        self.selection = selection
        self.seed = seed

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        if self.count < 1:
            raise ValueError(f"a landmark heuristic needs at least one landmark, got {self.count}")
        return _routelab.Heuristic.landmarks(
            compiled.graph, self.count, self.selection, self.seed, progress=progress
        )

    def __repr__(self) -> str:
        return f"Landmarks({self.count}, selection={self.selection!r})"
