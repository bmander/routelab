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

The gathering is done by two small specifications of the same shape,
:class:`Plane` and :class:`Pace`, which live here because a straight-line bound
is the only thing that reads them. The environment does not keep a coordinates
table or a rate; a technique that needs one derives it, and says what is
missing if it cannot.
"""

from __future__ import annotations

from typing import Hashable, List, Optional, Tuple

from . import _routelab
from .model.environment import CompiledEnvironment

__all__ = ["Euclidean", "Heuristic", "Landmarks", "Pace", "Plane", "Zero"]

#: How many missing labels to name in an error before trailing off.
_MAX_REPORTED = 5


class Plane:
    """Where every node is, as one dense table of planar coordinates.

        >>> Plane()
        Plane()

    Gathered from every layer's ``positions()``; later layers win, so a
    dedicated geometry layer can correct coordinates a topology layer guessed
    at. Every node must be covered — a straight-line bound to a node with no
    position is not a bound — and :meth:`bind` names the ones that are not.
    """

    #: The word :meth:`missing_from` answers with.
    name = "positions"

    @staticmethod
    def _gather(compiled: CompiledEnvironment) -> "List[Optional[Tuple[float, float]]]":
        """One entry per node id, ``None`` where no layer placed it."""
        positions: "List[Optional[Tuple[float, float]]]" = [None] * len(compiled)
        for source in compiled.sources:
            for label, point in source.positions().items():
                # Coordinates for labels no edge mentions are ignored: a node
                # nothing connects to is not a place you can route through.
                try:
                    node_id = compiled.node_id(label)
                except KeyError:
                    continue
                positions[node_id] = (float(point[0]), float(point[1]))
        return positions

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """``{"positions"}`` unless every node here has coordinates.

        Coverage only — which labels are placed — without building the table.
        """
        placed = set()
        for source in compiled.sources:
            placed.update(source.positions().keys())
        if compiled.labels and all(label in placed for label in compiled.labels):
            return frozenset()
        return frozenset({cls.name})

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "Tuple[List[float], List[float]]":
        """The table as parallel ``xs, ys`` per node id, or name the nodes
        nothing placed. ``progress`` is accepted for parity and left alone.

        Raises:
            ValueError: If some node has no position.
        """
        missing: List[Hashable] = []
        xs: List[float] = []
        ys: List[float] = []
        for node_id, point in enumerate(self._gather(compiled)):
            if point is None:
                missing.append(compiled.label(node_id))
            else:
                xs.append(point[0])
                ys.append(point[1])
        if missing or not xs:
            shown = ", ".join(repr(label) for label in missing[:_MAX_REPORTED])
            if len(missing) > _MAX_REPORTED:
                shown += f", and {len(missing) - _MAX_REPORTED} more"
            raise ValueError(
                f"Euclidean needs a position for every node; {len(missing)} have "
                f"none ({shown}). Register Positions({{...}}) covering them, or "
                f"use Zero()."
            )
        return xs, ys

    def __repr__(self) -> str:
        return "Plane()"


class Pace:
    """The least any edge-contributing layer charges per unit of distance.

        >>> Pace()
        Pace()

    The *minimum*, because a bound has to hold for every path, and a path may
    use the fastest layer for its whole length. Add a train to a walking
    network and the bound has to drop to the train's rate, or a straight-line
    estimate priced at walking speed starts overestimating — which is how an
    admissible heuristic silently becomes an inadmissible one.

    One layer that declares nothing means no bound at all: it might be
    arbitrarily fast, so nothing can be ruled out. Only layers that actually
    contributed edges count, which is what lets a geometry-only layer stay out
    of the arithmetic without a special case.
    """

    #: The word :meth:`missing_from` answers with.
    name = "cost_per_distance"

    @staticmethod
    def _gather(compiled: CompiledEnvironment) -> Optional[float]:
        rates: List[float] = []
        for _, _, source in compiled.spans:
            if source.cost_per_distance is None:
                return None
            rates.append(source.cost_per_distance)
        return min(rates) if rates else None

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """``{"cost_per_distance"}`` unless every edge-contributing layer
        declared a rate."""
        return frozenset() if cls._gather(compiled) is not None else frozenset({cls.name})

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> float:
        """The rate, or an explanation of which layer withheld it.
        ``progress`` is accepted for parity and left alone.

        Raises:
            ValueError: If some edge-contributing layer declared no rate.
        """
        rate = self._gather(compiled)
        if rate is None:
            raise ValueError(
                "Euclidean needs to know how cheaply this environment covers "
                "distance, and at least one layer that contributes edges did not "
                "say. Give every such layer a cost_per_distance=..., pass one to "
                "Euclidean(...) to override, or use Zero()."
            )
        return rate

    def __repr__(self) -> str:
        return "Pace()"


class Heuristic:
    """A specification for an estimate, not yet attached to an environment."""

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """What this heuristic needs that ``compiled`` cannot supply, by name.

        Empty means it can be bound. Answered without building anything, so
        a study can ask "which of these can I even run here?" before spending
        preprocessing to find out — the mirror of :attr:`routelab.Planner.accepts`,
        which does the same for cost models. Nothing, for a heuristic that
        measures the graph itself.
        """
        return frozenset()

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

    def __init__(self, cost_per_distance: Optional[float] = None):
        self.cost_per_distance = cost_per_distance

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Coordinates and a rate — whichever of the two the layers withheld.

        Answered for the class, so an instance's ``cost_per_distance=``
        override is not consulted: the question is what the *environment*
        can support, not what one configuration papered over.
        """
        return Plane.missing_from(compiled) | Pace.missing_from(compiled)

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        rate = self.cost_per_distance
        if rate is None:
            rate = Pace().bind(compiled)
        xs, ys = Plane().bind(compiled)
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

    SELECTIONS = ("farthest", "random")

    def __init__(self, count: int = 16, selection: str = "farthest", seed: int = 0):
        # Refused where it is written, not at bind: a technique is a value you
        # can put in a list and compare, and a value that is wrong should say
        # so when it is made.
        if count < 1:
            raise ValueError(f"a landmark heuristic needs at least one landmark, got {count}")
        if selection not in self.SELECTIONS:
            names = " or ".join(repr(name) for name in self.SELECTIONS)
            raise ValueError(f"unknown landmark selection {selection!r}; expected {names}")
        self.count = count
        self.selection = selection
        self.seed = seed

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Heuristic":
        return _routelab.Heuristic.landmarks(
            compiled.graph, self.count, self.selection, self.seed, progress=progress
        )

    def __repr__(self) -> str:
        return f"Landmarks({self.count}, selection={self.selection!r})"
