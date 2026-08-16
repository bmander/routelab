"""Walks between nearby stops, as a layer of fixed-cost edges.

    env = Environment(feed, Footpaths(feed, within=200))

A feed says where its stops are and what leaves them. It rarely says that the
northbound stop and the southbound one across the street are, for a rider,
the same place — King County Metro's has no ``transfers.txt`` and no parent
stations at all — and a timetable technique that cannot cross the street has
no answer for most real trips. This layer supplies the walks: an edge each way
between every pair of stops within ``within`` metres, costing the walk at
``speed``.

They are the paper's *foot-edges*, and here they are ordinary ``"scalar"``
edges. That is deliberate. A timetable technique already accepts a scalar layer
alongside its timetable; from now on it *reads* one, as the links a rider may
take at any time for their weight — see :class:`~routelab.Walks`. Nothing about
the environment had to change to carry them, and a journey's walk legs get
their provenance and geometry through the same machinery as every other leg.

How far a rider will walk is a modelling choice, which is why it is a knob on
this layer and not a fact the feed is asked for. Two hundred metres joins the
two sides of a street and the bays of a transit centre; a kilometre starts
inventing transfers a rider would not make.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Hashable, Iterator, List, Mapping, Optional, Tuple

from .. import _routelab
from ..environment import EdgeSource, LabelledEdge

__all__ = ["Footpaths"]

#: Metres per degree of latitude on the sphere the kernel measures with — the
#: same radius an OSM edge's length is priced by, so a walk and a street agree
#: about how long a metre is.
_METRES_PER_DEGREE = _routelab.EARTH_RADIUS * math.pi / 180.0


class Footpaths(EdgeSource):
    """Walks between every pair of stops within reach of each other.

        Footpaths(feed, within=200)    # Footpaths(within 200 m of GTFS(...), unbuilt)

    Args:
        stops: The layer whose stops these connect — anything with a
            ``coordinates()`` mapping labels to ``(lat, lon)``, which is what
            :class:`~routelab.GTFS` has. Labels are that layer's, so the walks
            join the same nodes its connections do.
        within: How far apart, in metres, two stops may be and still be a walk.
        speed: Walking speed in metres per second; the edge weight is the
            distance at this speed, rounded up to whole seconds.
    """

    cost_model = "scalar"

    def __init__(self, stops: EdgeSource, within: float = 200.0, speed: float = 1.4):
        if within <= 0:
            raise ValueError(f"within must be a positive distance in metres, got {within}")
        if speed <= 0:
            raise ValueError(f"speed must be positive metres per second, got {speed}")
        if not hasattr(stops, "coordinates"):
            raise TypeError(
                f"{stops!r} has no coordinates() to place stops by; Footpaths needs a "
                f"layer that knows where its stops are, like GTFS"
            )
        self.stops = stops
        self.within = float(within)
        self.speed = float(speed)
        self._edges: "Optional[List[LabelledEdge]]" = None

    #: Seconds per metre at the walking speed, so a distance bound priced
    #: against this layer knows how slowly it moves.
    @property
    def cost_per_distance(self) -> float:  # type: ignore[override]
        return 1.0 / self.speed

    def load(self) -> "Footpaths":
        """Build the walks now. See :meth:`~routelab.EdgeSource.load`."""
        self._build()
        return self

    def _build(self) -> "List[LabelledEdge]":
        if self._edges is not None:
            return self._edges
        coordinates: Mapping[Hashable, Tuple[float, float]] = self.stops.coordinates()
        # Bucket stops into a grid a little coarser than `within`, so each stop
        # is compared with its own cell and the eight around it rather than
        # with every stop in the city — six thousand stops squared is a number
        # worth not computing.
        lat_step = self.within / _METRES_PER_DEGREE
        # One longitude cell width for the whole set, sized at the highest
        # latitude present so a cell is at least `within` metres wide
        # everywhere — the invariant that lets neighbours be found in the
        # adjacent cells alone.
        widest = max((abs(lat) for lat, _ in coordinates.values()), default=0.0)
        lon_step = lat_step / max(0.05, math.cos(math.radians(widest)))
        buckets: "Dict[Tuple[int, int], List[Hashable]]" = defaultdict(list)
        cells: "Dict[Hashable, Tuple[int, int]]" = {}
        for label, (lat, lon) in coordinates.items():
            cell = (int(math.floor(lat / lat_step)), int(math.floor(lon / lon_step)))
            buckets[cell].append(label)
            cells[label] = cell

        edges: "List[LabelledEdge]" = []
        for label, here in coordinates.items():
            row, column = cells[label]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for other in buckets.get((row + dr, column + dc), ()):
                        if other == label:
                            continue
                        there = coordinates[other]
                        metres = _routelab.haversine(here[0], here[1], there[0], there[1])
                        if metres <= self.within:
                            edges.append((label, other, max(1, math.ceil(metres / self.speed))))
        self._edges = edges
        return edges

    def edges(self) -> Iterator[LabelledEdge]:
        return iter(self._build())

    def positions(self) -> Mapping[Hashable, "Tuple[float, float]"]:
        """Nothing of its own: the stops layer already places every node."""
        return {}

    def geometry(self, index: int) -> "Optional[List[Tuple[float, float]]]":
        """A straight line between the two stops, as `(lat, lon)` pairs."""
        tail, head, _ = self._build()[index]
        coordinates = self.stops.coordinates()
        return [coordinates[tail], coordinates[head]]

    def __len__(self) -> int:
        return len(self._build())

    def __repr__(self) -> str:
        state = "unbuilt" if self._edges is None else f"{len(self._edges)} walks"
        return f"Footpaths(within {self.within:g} m of {self.stops!r}, {state})"
