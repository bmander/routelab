"""What a query returns: a journey, in labels, with each leg's provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterator, List, NamedTuple, Optional, Tuple

from .environment import CompiledEnvironment, EdgeSource, shape_of
from .search import Result

__all__ = ["Journey", "Leg"]

#: A point on the ground, `(latitude, longitude)`.
Point = Tuple[float, float]


class Leg(NamedTuple):
    """One edge of a journey, with the layer that supplied it."""

    tail: Hashable
    head: Hashable
    weight: int
    source: EdgeSource
    #: The graph edge this leg crossed, for asking the environment about it.
    edge: int
    #: Where this leg sits within its own layer, counted in the order that layer
    #: emitted its edges. Carried rather than looked up because resolving the
    #: layer already computed it, and because it is what makes a leg able to
    #: answer for itself. Not `index`, which is a method every tuple already has.
    position: int

    @property
    def geometry(self) -> "Optional[List[Point]]":
        """The `(lat, lon)` shape of this leg, if its layer keeps one.

        A leg between two labels is a straight line as far as the graph is
        concerned; on the ground it may be a winding street. ``None`` from a
        layer that knows only topology.
        """
        return shape_of(self.source, self.position)


@dataclass(frozen=True)
class Journey:
    """A path through an environment, with the cost the planner minimized.

    Iterating a journey gives its legs, which is what a journey mostly is::

        for leg in journey:
            leg.head, leg.weight, leg.geometry

    A record rather than a tuple, precisely so that can be true: a
    :class:`~typing.NamedTuple` that iterated its legs would be lying about its
    own length everywhere else.

    ``cost`` is what the planner was optimizing, which is not always the sum of
    the leg weights: :class:`~routelab.planners.BFS` counts hops, so its journeys
    report a hop count while their legs still carry real edge weights.
    """

    origin: Hashable
    destination: Hashable
    cost: int
    legs: "Tuple[Leg, ...]"

    @property
    def nodes(self) -> "List[Hashable]":
        """The labels along the journey, origin first."""
        return [self.origin] + [leg.head for leg in self.legs]

    @property
    def waiting(self) -> int:
        """Seconds of the journey spent not moving — a shut gate, a locked path.

        Whatever the planner counted that no leg accounts for. That makes it
        work for any result, including a bidirectional one that can only report
        a cost at its target, and it is why this is a total rather than a figure
        per leg: the legs cannot each be asked when a search never priced them
        individually.

        Zero for a search with no clock, and clamped at zero for one whose cost
        is not seconds at all — :class:`~routelab.planners.BFS` counts hops, and
        the difference between a hop count and a sum of seconds is not a wait.
        """
        return max(0, self.cost - sum(leg.weight for leg in self.legs))

    @property
    def moving(self) -> int:
        """Seconds actually spent travelling: :attr:`cost` less :attr:`waiting`."""
        return self.cost - self.waiting

    @property
    def geometry(self) -> "List[Point]":
        """The whole route as one polyline, origin first.

        Legs are stitched rather than concatenated — each one starts where the
        last ended, and repeating that point would put a duplicate vertex at
        every corner. Legs whose layer has no shape contribute nothing, so a
        journey through a layer that knows only topology comes back empty rather
        than partially drawn.
        """
        points: "List[Point]" = []
        for leg in self.legs:
            shape = leg.geometry
            if shape is None:
                continue
            points.extend(shape[1:] if points else shape)
        return points

    @classmethod
    def from_result(
        cls,
        compiled: CompiledEnvironment,
        result: Result,
        destination: Hashable,
    ) -> "Journey":
        """Rebuild a journey from a search result. Assumes the target was reached."""
        node_id = compiled.node_id(destination)
        legs = []
        for edge_id in result.edge_path(node_id):
            tail, head, weight = compiled.graph.edge(edge_id)
            source, position = compiled.locate(edge_id)
            legs.append(
                Leg(
                    tail=compiled.label(tail),
                    head=compiled.label(head),
                    weight=weight,
                    source=source,
                    edge=edge_id,
                    position=position,
                )
            )
        return cls(
            # Where the journey starts, which with several origins is whichever
            # of them won rather than the first one the caller named.
            origin=compiled.label(result.path(node_id)[0]),
            destination=destination,
            cost=result.cost(node_id),
            legs=tuple(legs),
        )

    def __iter__(self) -> "Iterator[Leg]":
        """The legs. Deliberately not `__len__`: a journey from a place to
        itself has no legs and is still an answer, and `len(journey) == 0` would
        make `if journey:` mean something other than "a route was found" —
        which is what `route()` already says by returning None when one was not.
        """
        return iter(self.legs)

    def __repr__(self) -> str:
        arrow = " → ".join(repr(label) for label in self.nodes)
        return f"Journey({arrow}, cost={self.cost})"
