"""Departures and walks: what a timetable technique reads beyond the graph.

A :class:`Departures` is a specification, the way :class:`~routelab.Schedule`
is: ``bind`` walks the layers, asks each for the connections along its edges,
and builds one kernel :class:`~routelab._routelab.Timetable` — or refuses,
naming the layer that would have supplied one.

The distinction from a schedule is the distinction between the two ways a
network can depend on the clock: a window says *when an edge is open*, a
connection says *when a vehicle leaves*. Both are read by techniques and by
nothing else, which is why neither lives on the environment.

:class:`Walks` is the third thing the timetable techniques read, and the one
that makes them usable on a real feed: every ``"scalar"`` edge in the
environment, taken as a link a rider may walk at any time for its weight —
the paper's foot-edges. It refuses nothing; a timetable with no walks is the
paper's plain model, and an empty table says so.
"""

from __future__ import annotations

from typing import Hashable, Iterator, List, Optional, Set, Tuple

from . import _routelab
from .model.environment import CompiledEnvironment, EdgeSource

__all__ = ["Departures", "Walks"]


def connections_of(
    source: EdgeSource, index: int
) -> "Optional[List[Tuple[int, int, int]]]":
    """What runs along a layer's ``index``-th edge, or ``None`` for a layer with
    no timetable."""
    getter = getattr(source, "connections", None)
    return None if getter is None else getter(index)


class Departures:
    """Every layer's departures, gathered into one timetable keyed by edge.

        >>> Departures()
        Departures()

    Built by the timetable techniques at bind time. A connection's pair of
    stops comes from the edge it is filed under rather than from the layer, so
    there is nothing here to pair up wrongly.
    """

    #: The word :meth:`missing_from` answers with.
    name = "timetable"

    @staticmethod
    def _connections(compiled: CompiledEnvironment) -> "Iterator[Tuple[int, List[Tuple[int, int, int]]]]":
        """``(input position, connections)`` for every edge that has any."""
        for start, stop, source in compiled.spans:
            if getattr(source, "connections", None) is None:
                continue  # a layer with no timetable: skip its run, not each edge
            for position in range(stop - start):
                running = connections_of(source, position)
                if running:
                    yield start + position, running

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """``{"timetable"}`` unless some edge here is served by departures.

        Exact, and stops at the first connection it finds.
        """
        if any(True for _ in cls._connections(compiled)):
            return frozenset()
        return frozenset({cls.name})

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Timetable":
        """Collect every layer's departures into one timetable, or explain.

        ``progress`` is accepted for parity with every other ``bind`` and left
        alone: a walk of the layers has no honest measure of its own.

        Raises:
            ValueError: If no layer keeps a timetable, or the one that does
                runs nothing on the day it was read for.
        """
        connections = list(self._connections(compiled))
        if not connections:
            raise ValueError(
                "a timetable technique needs a timetable and this environment "
                "has none: no layer keeps departures, or the one that does runs "
                "nothing on the service day it was read for. Register a layer "
                "that does — routelab.GTFS(path, date)."
            )
        return _routelab.Timetable.from_connections(compiled.graph, connections)

    def __repr__(self) -> str:
        return "Departures()"


class Walks:
    """The scalar edges between a timetable's stops, as footpaths.

        >>> Walks()
        Walks()

    A timetable technique accepts ``"scalar"`` layers beside its timetable, and
    this is what it does with them: a scalar edge **joining two stops the
    timetable serves** is a walk a rider may take at any time for its weight —
    the paper's foot-edge, by definition. Register :class:`~routelab.Footpaths`
    to have walks between nearby stops made for you; register a hand-written
    :class:`~routelab.ScalarEdges` to say exactly which stops join.

    Only edges between stops, and exactly those: a scalar layer that also
    carries streets, or access links to places no vehicle serves, is not swept
    in — which is what keeps this a derivation with one possible construction
    rather than a guess that happens to hold while the only scalar neighbour
    of a feed is a footpath layer.

    The kernel closes the set under composition — walk A→B and B→C and you may
    walk A→C — for the reason :class:`~routelab._routelab.Footpaths` gives:
    the timetable techniques chain walks differently and must still agree.
    """

    #: The word every derivation carries. Never answered by :meth:`missing_from`
    #: — an environment with no walks is the plain model, not a refusal — but a
    #: derivation with no name would be the one that reads differently.
    name = "walks"

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Nothing, ever: an environment with no walks is the plain model."""
        return frozenset()

    def bind(
        self, compiled: CompiledEnvironment, progress: "Optional[_routelab.Progress]" = None
    ) -> "_routelab.Footpaths":
        """Gather every scalar edge between two timetable stops into one
        kernel footpath table.

        Empty when no such edge exists, which is a legitimate environment
        rather than a refusal. ``progress`` is accepted for parity and left
        alone.
        """
        stops: "Set[Hashable]" = set()
        for _, _, source in compiled.spans:
            if source.cost_model == "timetable":
                for tail, head, _ in source.edges():
                    stops.add(tail)
                    stops.add(head)
        positions = []
        for start, _, source in compiled.spans:
            if source.cost_model != "scalar":
                continue
            for offset, (tail, head, _) in enumerate(source.edges()):
                if tail in stops and head in stops:
                    positions.append(start + offset)
        return _routelab.Footpaths.from_edges(compiled.graph, positions)

    def __repr__(self) -> str:
        return "Walks()"
