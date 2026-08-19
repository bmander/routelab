"""The step from a stop onto the pavement outside it.

    env = Environment(feed, streets, Access(feed, streets))

A feed says where its stops are; a street network says where its corners are;
neither says that the two are the same place. Until something joins them a
timetable and a map registered together are two graphs sharing one numbering
and no edges, and a rider can no more walk to a bus than fly.

This layer supplies the join: an edge each way between every stop and the
street node nearest it, costing the walk at ``speed``. That is all it is —
which is why it is a layer and not something the environment does for you.
How far a rider will go to reach the pavement is a modelling choice, and a
stop further than ``within`` from any street is one this leaves stranded
rather than attaching to somewhere implausible.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterator, List, Mapping, Optional, Tuple

from .. import _routelab
from ..model.environment import EdgeSource, LabelledEdge, Located, Snapping

__all__ = ["Access"]


class Access(EdgeSource):
    """Links from each stop to the street node nearest it.

        Access(feed, streets)    # Access(6,313 stops to OSM(...), unbuilt)

    Args:
        stops: The layer whose stops these attach — anything with a
            ``coordinates()`` mapping labels to ``(lat, lon)``, which is what
            :class:`~routelab.GTFS` has.
        streets: The layer they attach to — anything with ``nearest(lat, lon)``
            returning one of its own labels, which is what
            :class:`~routelab.OSM` has.
        within: How far, in metres, a stop may be from the street network and
            still be joined to it. A stop further than this from anything is
            left unattached, which is honest: it is a stop the map does not
            reach.
        speed: Walking speed in metres per second; the edge weight is the
            distance at this speed, rounded up to whole seconds.
    """

    cost_model = "scalar"

    #: What this layer's arcs are, for a technique that constrains the sequence
    #: of transport modes — Dibbelt, Pajor & Wagner's `link`, the arcs that join
    #: two networks and the only place a journey may change mode. See
    #: :class:`~routelab.LabelConstrained`.
    mode = "link"

    def __init__(
        self,
        stops: Located,
        streets: Snapping,
        within: float = 400.0,
        speed: float = 1.4,
    ):
        if within <= 0:
            raise ValueError(f"within must be a positive distance in metres, got {within}")
        if speed <= 0:
            raise ValueError(f"speed must be positive metres per second, got {speed}")
        if not hasattr(stops, "coordinates"):
            raise TypeError(
                f"{stops!r} has no coordinates() to place stops by; Access needs a "
                f"layer that knows where its stops are, like GTFS"
            )
        if not hasattr(streets, "nearest"):
            raise TypeError(
                f"{streets!r} has no nearest() to attach a stop to; Access needs a "
                f"layer that can say which of its nodes is closest, like OSM"
            )
        self.stops = stops
        self.streets = streets
        self.within = float(within)
        self.speed = float(speed)
        self._edges: "Optional[List[LabelledEdge]]" = None
        self._stranded = 0

    @property
    def cost_per_distance(self) -> float:
        return 1.0 / self.speed

    def load(self) -> "Access":
        self._build()
        return self

    def _build(self) -> "List[LabelledEdge]":
        if self._edges is not None:
            return self._edges
        stops: Mapping[Hashable, Tuple[float, float]] = self.stops.coordinates()
        streets: Mapping[Hashable, Tuple[float, float]] = self.streets.coordinates()
        edges: "List[LabelledEdge]" = []
        stranded = 0
        for label, (lat, lon) in stops.items():
            try:
                node = self.streets.nearest(lat, lon)
            except ValueError:
                stranded += 1
                continue
            there = streets.get(node)
            metres = None if there is None else _routelab.haversine(lat, lon, there[0], there[1])
            if metres is None or metres > self.within:
                # Further from the map than a rider would walk to reach it —
                # a stop outside the extract, most often. Left alone rather
                # than attached to somewhere implausible.
                stranded += 1
                continue
            seconds = max(1, math.ceil(metres / self.speed))
            edges.append((label, node, seconds))
            edges.append((node, label, seconds))
        self._edges = edges
        self._stranded = stranded
        return edges

    @property
    def stranded(self) -> int:
        """Stops no street node was close enough to reach.

        Reported rather than hidden: a feed whose stops sit outside the
        extract, or a walking profile that excludes the road they stand on,
        shows up here as a number instead of as journeys that quietly cannot
        be made.
        """
        self._build()
        return self._stranded

    def edges(self) -> Iterator[LabelledEdge]:
        return iter(self._build())

    def positions(self) -> Mapping[Hashable, "Tuple[float, float]"]:
        """Nothing of its own: both layers already place their own nodes."""
        return {}

    def geometry(self, index: int) -> "Optional[List[Tuple[float, float]]]":
        """A straight line from the stop to the street, as `(lat, lon)`."""
        tail, head, _ = self._build()[index]
        here = self.stops.coordinates()
        there = self.streets.coordinates()
        start = here.get(tail, there.get(tail))
        end = there.get(head, here.get(head))
        return None if start is None or end is None else [start, end]

    def __len__(self) -> int:
        return len(self._build())

    def __repr__(self) -> str:
        state = "unbuilt" if self._edges is None else f"{len(self._edges) // 2} joined"
        return f"Access({self.stops!r} to {self.streets!r}, {state})"
