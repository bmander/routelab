"""What a query returns: a journey, in labels, with each leg's provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterator, List, NamedTuple, Optional, Tuple

from . import _routelab
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
    #: When this leg leaves and lands, in the clock its planner reads — seconds
    #: since the service day's midnight for a timetable. ``None`` from a search
    #: with no clock, which is most of them: a static edge has a duration and no
    #: hour, and inventing one would make a journey look scheduled when it is
    #: only priced.
    departs: Optional[int] = None
    arrives: Optional[int] = None
    #: The vehicle run this leg rides, for a leg that rides one. Two consecutive
    #: legs sharing a trip are one bus, and the difference is a transfer — which
    #: is the thing a transit answer is actually judged on.
    trip: Optional[int] = None

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
    #: Nodes the search settled finding this, when the planner reports it.
    #:
    #: ``None`` from a static search, and not because the number is unknown —
    #: :meth:`Planner.explored` answers it from the result, in a form something
    #: can draw. A timetable query has no result to hand back, so the one number
    #: worth comparing the two models on travels with the answer instead.
    settled: Optional[int] = None

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
    def walking(self) -> int:
        """Seconds of :attr:`moving` spent on foot between stops — the legs
        that ride no vehicle. Zero for a journey with no clock, whose legs
        have no trip to be without."""
        return sum(leg.weight for leg in self.legs if leg.trip is None)

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

    @property
    def departs(self) -> Optional[int]:
        """When the journey leaves, if its legs are scheduled."""
        return self.legs[0].departs if self.legs else None

    @property
    def arrives(self) -> Optional[int]:
        """When the journey lands, if its legs are scheduled."""
        return self.legs[-1].arrives if self.legs else None

    @property
    def transfers(self) -> int:
        """How many times you change vehicles.

        Zero for a journey you never get off and for one with no vehicles at
        all. Counted from the legs rather than carried, because it is a fact
        about them: a leg knows its trip, so nothing else has to be told. A
        walk between two vehicles is how the change is made, not a second one,
        so only the rides are compared.
        """
        trips = [leg.trip for leg in self.legs if leg.trip is not None]
        return sum(1 for here, then in zip(trips, trips[1:]) if here != then)

    @classmethod
    def from_itinerary(
        cls,
        compiled: CompiledEnvironment,
        itinerary: "_routelab.Itinerary",
        origin: Hashable,
        departing: int,
    ) -> "Journey":
        """Rebuild a journey from a timetable query's answer.

        The counterpart to :meth:`from_result` for the searches that answer with
        rides rather than with a cost per node. Legs carry the ride they
        actually made, so :attr:`waiting` comes out as time spent at stops —
        which for a transit journey is the number a rider cares about and the
        static case can only ever report as a residual.
        """
        # A ride's edge and a walk's edge may join the same two stops; the cost
        # model tells them apart, and a walk comes back hop by hop already —
        # the kernel tells a closed walk as the given links it chains.
        legs = [
            _leg(
                compiled,
                _edge_between(compiled, tail, head, "scalar" if trip is None else "timetable"),
                departs,
                arrives,
                trip,
            )
            for trip, tail, head, departs, arrives in itinerary.legs()
        ]
        return cls(
            origin=origin,
            destination=legs[-1].head if legs else origin,
            # Elapsed time, so that `waiting` is the part of it spent standing
            # at a stop rather than moving. An arrival time on its own would
            # make the cost depend on when midnight was.
            cost=itinerary.arrives - departing,
            legs=tuple(legs),
            settled=itinerary.settled,
        )

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


def _edge_between(
    compiled: CompiledEnvironment, tail: int, head: int, cost_model: Optional[str] = None
) -> int:
    """The graph edge from ``tail`` to ``head``, from a layer of ``cost_model``
    if one is named.

    A timetable query answers in stops, not in edges, and a leg needs an edge to
    trace its own provenance. Scanned rather than indexed: an itinerary has
    tens of legs and a stop has a handful of neighbours, so a table over the
    whole graph would cost more to build than every query it ever served. The
    cost model tells a ride's edge from a walk's when both join the same stops.
    """
    graph = compiled.graph
    for edge_id in graph.out_edges(tail):
        if graph.edge(edge_id)[1] != head:
            continue
        if cost_model is None or compiled.source_of(edge_id).cost_model == cost_model:
            return edge_id
    raise KeyError(
        f"{compiled.label(tail)!r} → {compiled.label(head)!r} is not an edge in "
        f"this environment, so the itinerary did not come from it"
    )


def _leg(
    compiled: CompiledEnvironment,
    edge_id: int,
    departs: int,
    arrives: int,
    trip: Optional[int],
) -> Leg:
    """A scheduled leg over ``edge_id``, with its provenance looked up."""
    tail, head, _ = compiled.graph.edge(edge_id)
    source, position = compiled.locate(edge_id)
    return Leg(
        tail=compiled.label(tail),
        head=compiled.label(head),
        weight=arrives - departs,
        source=source,
        edge=edge_id,
        position=position,
        departs=departs,
        arrives=arrives,
        trip=trip,
    )
