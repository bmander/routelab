"""A GTFS feed as a routable layer.

    env = Environment().register(GTFS("kcm.zip", date(2026, 8, 17)))

A timetable is not a network with weights on it. Its edges do not have a cost —
they have *departures*, and what one costs depends on when you arrive at it. So
this layer declares ``cost_model = "timetable"``, and that declaration is
load-bearing: :class:`~routelab.Dijkstra` refuses to bind to an environment
containing one, because routing a bus network as though every edge were always
traversable at its shortest ride is a wrong answer that looks like a right one.

What the layer *does* contribute as edges is one per pair of adjacent stops,
weighted by the shortest ride anyone makes along it. That is a genuine lower
bound rather than a cost — enough to give every stop a label, to snap a
coordinate to one, and to price a distance heuristic — and the schedule itself
is what a timetable technique derives from this layer at bind, as a
:class:`~routelab._routelab.Timetable` keyed by edge (see
:class:`~routelab.Departures`).

One service day at a time. GTFS writes ``25:30:00`` for a bus that leaves before
midnight and arrives after it, so a service day is a line rather than a cycle,
and :mod:`routelab.util.clock`'s weekly seconds are a different clock — which is why a
walking layer and this one cannot yet share an environment.
"""

from __future__ import annotations

import datetime as _datetime
import math
import os
from typing import Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Tuple

from .. import _routelab
from ..model.environment import EdgeSource, LabelledEdge

__all__ = ["GTFS"]


class GTFS(EdgeSource):
    """A transit timetable read from a GTFS feed, for one service day.

        GTFS("kcm.zip", date(2026, 8, 17))    # GTFS('kcm.zip', 2026-08-17, unread)

    Nodes are labelled with GTFS `stop_id`s, so an itinerary names the stops a
    rider would recognise. Reading is deferred until the layer is first used,
    but the path is checked immediately.

    Args:
        path: A `.zip` feed, or a directory of `.txt` files.
        date: Which service day to read. Required, and deliberately: a feed
            covers months, its trips run on different days, and "today" would
            make the same code mean something different tomorrow.
    """

    cost_model = "timetable"

    def __init__(self, path: "str | os.PathLike[str]", date: _datetime.date):
        self.path = os.fspath(path)
        self.date = date
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"no GTFS feed at {self.path!r}")
        self._feed: "Optional[_routelab.Feed]" = None
        self._edges: "Optional[List[Tuple[int, int, int]]]" = None
        self._coordinates: "Optional[Mapping[str, Tuple[float, float]]]" = None

    @property
    def feed(self) -> "_routelab.Feed":
        """The service day's connections, read on first use."""
        if self._feed is None:
            self._feed = _routelab.load_gtfs(
                self.path, self.date.year, self.date.month, self.date.day
            )
        return self._feed

    def load(self) -> "GTFS":
        """Read the feed now. See :meth:`~routelab.EdgeSource.load`."""
        self.feed
        return self

    def _pairs(self) -> "List[Tuple[int, int, int]]":
        """The stop pairs, in the order :meth:`edges` yields them."""
        edges = self._edges
        if edges is None:
            edges = self._edges = self.feed.edges()
        return edges

    def edges(self) -> Iterator[LabelledEdge]:
        stop_ids = self.feed.stop_ids
        for tail, head, shortest in self._pairs():
            yield stop_ids[tail], stop_ids[head], shortest

    def connections(self, index: int) -> "List[Tuple[int, int, int]]":
        """What runs along this layer's ``index``-th edge, in departure order.

        ``(trip, departs, arrives)``, with times in seconds since the service
        day's midnight and possibly past 86400. The hook mirrors
        :meth:`~routelab.data.osm.OSM.windows`: indices are this layer's own,
        and :class:`~routelab.Departures` — the derivation the timetable
        techniques bind — does the translation into the graph's numbering.
        """
        return self.feed.connections(index)

    def positions(self) -> "Mapping[Hashable, Tuple[float, float]]":
        """Stop ids to `(lat, lon)`.

        Degrees, not metres — so a distance heuristic priced in seconds per
        metre has nothing to stand on here, which is why
        :attr:`cost_per_distance` stays ``None``. Transit's cost is the
        timetable's, not the ground's.
        """
        positions: "Dict[Hashable, Tuple[float, float]]" = {
            stop: place for stop, place in self.coordinates().items()
        }
        return positions

    def coordinates(self) -> "Mapping[str, Tuple[float, float]]":
        """Stop ids to `(lat, lon)`, built once and kept."""
        if self._coordinates is None:
            feed = self.feed
            lats, lons = feed.coordinates()
            self._coordinates = dict(zip(feed.stop_ids, zip(lats, lons)))
        return self._coordinates

    def names(self) -> "Mapping[str, str]":
        """Stop ids to the names a rider would read on a sign."""
        feed = self.feed
        return dict(zip(feed.stop_ids, feed.stop_names))

    def trips(self) -> "Mapping[int, Tuple[str, str]]":
        """Dense trip index to its GTFS `(trip_id, route_id)`.

        The other half of an itinerary's answer: a ride names a trip by number,
        and this is what turns that into the run a rider would recognise.
        """
        trip_ids, routes = self.feed.trips()
        return dict(enumerate(zip(trip_ids, routes)))

    def nearest(
        self, lat: float, lon: float, within: Optional[Iterable[str]] = None
    ) -> str:
        """The stop id closest to a coordinate.

        Straight-line over degrees scaled by latitude, which is accurate enough
        at the scale of "which stop is that" and wrong enough that nothing else
        should use it.
        """
        coordinates = self.coordinates()
        candidates = coordinates if within is None else list(within)
        scale = math.cos(math.radians(lat))
        best, distance = None, float("inf")
        for stop in candidates:
            point = coordinates.get(stop)
            if point is None:
                continue
            offset = (point[0] - lat) ** 2 + ((point[1] - lon) * scale) ** 2
            if offset < distance:
                best, distance = stop, offset
        if best is None:
            raise ValueError(f"{self!r} has no stops to be near {lat}, {lon}")
        return best

    @property
    def unimplemented(self) -> int:
        """Trips this reader could not represent, and so dropped.

        Reported rather than hidden, in the manner of
        :attr:`~routelab.data.osm.OSM.unreadable_schedules`. Frequency-based
        service is the big one; a feed built on it would come back nearly empty
        and should say so rather than look sparse.
        """
        return self.feed.unimplemented

    def __repr__(self) -> str:
        state = (
            "unread"
            if self._feed is None
            else f"{self._feed.num_connections} connections"
        )
        return f"GTFS({os.path.basename(self.path)!r}, {self.date}, {state})"
