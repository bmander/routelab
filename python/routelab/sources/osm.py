"""OpenStreetMap extracts as a routable layer.

    env = Environment().register(OSM("seattle.osm.pbf", Driving()))

OSM describes the world rather than a graph, and the same file is three
different networks depending on who is travelling: a footway is a road to a
pedestrian and a wall to a driver. A :class:`Profile` is that decision, made
explicitly — which `highway` classes count, how fast each one goes, and whether
one-way restrictions bind.

The reading and cutting happen in Rust (`crates/routelab-osm`): filter the ways a
profile can travel, cut them where they meet, price each piece, and keep the
nodes in between as the edge's shape rather than as graph nodes. What lives here
is the part worth reading and changing — the profiles themselves, and the
translation into a layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Hashable, Iterable, Iterator, List, Mapping, Optional, Tuple

from .. import _routelab
from ..environment import EdgeSource, LabelledEdge

__all__ = ["OSM", "Cycling", "Driving", "Profile", "Walking"]

#: Metres per second, by `highway` class. Anything absent is not routable for
#: that profile — which is how a motorway stays out of a walking network.
WALKING_SPEEDS = {
    "footway": 1.4,
    "path": 1.4,
    "pedestrian": 1.4,
    "steps": 0.5,
    "living_street": 1.4,
    "residential": 1.4,
    "service": 1.4,
    "unclassified": 1.4,
    "tertiary": 1.4,
    "secondary": 1.4,
    "primary": 1.4,
    "track": 1.2,
    "cycleway": 1.4,
}

CYCLING_SPEEDS = {
    "cycleway": 5.0,
    "path": 3.5,
    "living_street": 4.0,
    "residential": 4.5,
    "service": 4.0,
    "unclassified": 4.5,
    "tertiary": 5.0,
    "secondary": 5.0,
    "primary": 5.0,
    "track": 3.0,
    "footway": 1.4,
}

DRIVING_SPEEDS = {
    "motorway": 31.0,
    "motorway_link": 18.0,
    "trunk": 25.0,
    "trunk_link": 15.0,
    "primary": 20.0,
    "primary_link": 12.0,
    "secondary": 17.0,
    "secondary_link": 11.0,
    "tertiary": 14.0,
    "tertiary_link": 9.0,
    "unclassified": 11.0,
    "residential": 8.0,
    "living_street": 5.0,
    "service": 5.0,
}


@dataclass(frozen=True)
class Profile:
    """Who is travelling, and therefore what counts as a road.

    Args:
        name: For readable reprs and errors.
        speeds: `highway` class to metres per second. A class left out is not
            routable at all for this profile.
        respect_oneway: Whether `oneway` tags bind. They do for vehicles; a
            pedestrian walks up a one-way street without breaking anything.
        access_keys: Which access tags speak for this traveller, most specific
            first — a walker reads `foot` then `access`, a driver
            `motor_vehicle` then `vehicle` then `access`. This is also what
            decides whose `:conditional` schedules are read: a gate tagged
            `foot:conditional` shuts a footpath and says nothing to a car.
            Empty means access tags are ignored entirely.
        use_maxspeed: Whether a way's own `maxspeed` overrides its class speed.
            A posted limit above :attr:`max_speed` is capped there — see below.
    """

    name: str
    speeds: Mapping[str, float] = field(default_factory=dict)
    respect_oneway: bool = True
    use_maxspeed: bool = True
    access_keys: "Tuple[str, ...]" = ()

    @property
    def max_speed(self) -> float:
        """The fastest anything travels under this profile.

        Load-bearing beyond documentation: it becomes the layer's
        ``cost_per_distance``, so a distance-based heuristic is priced against
        it. That is why a posted `maxspeed` above this value is clamped rather
        than honoured — one edge quietly beating the profile's own top speed
        would make the bound an overestimate, and A* would stop returning
        cheapest paths without saying so.
        """
        return max(self.speeds.values(), default=1.0)

    def but(self, **changes: object) -> "Profile":
        """A copy with fields replaced: ``Driving().but(respect_oneway=False)``."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"{self.name}()"


def Walking() -> Profile:  # noqa: N802 - reads as a constructor at the call site
    """On foot: paths and streets alike, no one-way restrictions, 1.4 m/s."""
    return Profile(
        "Walking", WALKING_SPEEDS, respect_oneway=False, access_keys=("foot", "access")
    )


def Cycling() -> Profile:  # noqa: N802
    """By bicycle: cycleways and most streets, one-way restrictions respected."""
    return Profile(
        "Cycling",
        CYCLING_SPEEDS,
        respect_oneway=True,
        access_keys=("bicycle", "vehicle", "access"),
    )


def Driving() -> Profile:  # noqa: N802
    """By car: the road hierarchy at class speeds, posted limits honoured."""
    return Profile(
        "Driving",
        DRIVING_SPEEDS,
        respect_oneway=True,
        access_keys=("motor_vehicle", "vehicle", "access"),
    )


class OSM(EdgeSource):
    """A routable layer read from an OpenStreetMap extract.

        OSM("city.osm.pbf", Driving())    # OSM('city.osm.pbf', Driving(), unread)

    Nodes are labelled with their OSM ids, so a route can be traced back to the
    map it came from. Reading is deferred until the layer is first used, but the
    path is checked immediately — a typo should not survive until the first
    query.

    Args:
        path: An `.osm.pbf` extract, or `.osm`/`.xml` for small hand-made files.
        profile: Who is travelling. Defaults to :func:`Walking`.
    """

    cost_model = "scalar"

    def __init__(self, path: "str | os.PathLike[str]", profile: Optional[Profile] = None):
        self.path = os.fspath(path)
        self.profile = profile if profile is not None else Walking()
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"no OSM extract at {self.path!r}")
        self._network: "Optional[_routelab.OsmNetwork]" = None
        self._coordinates: "Optional[Mapping[int, Tuple[float, float]]]" = None
        self._windows: "Optional[Mapping[int, List[Tuple[int, int]]]]" = None

    @property
    def cost_per_distance(self) -> float:
        """Seconds per metre at the profile's top speed — the fastest this layer
        can move, which is what a distance bound has to assume."""
        return 1.0 / self.profile.max_speed

    @property
    def network(self) -> "_routelab.OsmNetwork":
        """The parsed network, read on first use."""
        if self._network is None:
            self._network = _routelab.load_osm(
                self.path,
                list(self.profile.speeds.items()),
                respect_oneway=self.profile.respect_oneway,
                use_maxspeed=self.profile.use_maxspeed,
                access_keys=list(self.profile.access_keys),
            )
        return self._network

    def load(self) -> "OSM":
        """Read the extract now. See :meth:`~routelab.EdgeSource.load`."""
        self.network
        return self

    def edges(self) -> Iterator[LabelledEdge]:
        network = self.network
        node_ids = network.node_ids
        for tail, head, seconds in network.edges():
            yield node_ids[tail], node_ids[head], seconds

    def positions(self) -> Mapping[Hashable, "Tuple[float, float]"]:
        """OSM node ids to local metres.

        Projected in the loader rather than left as latitude and longitude,
        conservatively enough that projected distance never exceeds the ground
        it covers — which is what keeps :class:`~routelab.Euclidean` admissible.
        """
        network = self.network
        xs, ys = network.projected()
        return dict(zip(network.node_ids, zip(xs, ys)))

    def coordinates(self) -> "Mapping[int, Tuple[float, float]]":
        """OSM node ids to `(lat, lon)`, for drawing rather than routing.

        Built once and kept: it is a dict entry per node, which is a tenth of a
        second at half a million of them, and anything drawing a map asks for it
        repeatedly.
        """
        if self._coordinates is None:
            network = self.network
            lats, lons = network.coordinates()
            self._coordinates = dict(zip(network.node_ids, zip(lats, lons)))
        return self._coordinates

    def windows(self, index: int) -> "Optional[List[Tuple[int, int]]]":
        """When this layer's ``index``-th edge may be travelled, if not always.

        Weekly-clock ``(start, end)`` seconds since Monday 00:00. ``None`` is the
        overwhelmingly common answer — a couple of hundred edges out of six
        hundred thousand carry a schedule — and it means no restriction. An
        empty list would mean never, which the reader drops rather than emits.

        The hook mirrors :meth:`geometry`: indices are this layer's own, and
        :class:`~routelab.CompiledEnvironment` does the translation.
        """
        if self._windows is None:
            self._windows = dict(self.network.windows())
        return self._windows.get(index)

    @property
    def unreadable_schedules(self) -> int:
        """Conditional tags the reader could not understand, and so ignored.

        Reported rather than hidden: `opening_hours` is a large specification
        and this reads a corner of it, so a schedule quietly dropped should be a
        number someone can look at.
        """
        return self.network.unreadable_schedules

    def geometry(self, index: int) -> "List[Tuple[float, float]]":
        """The `(lat, lon)` points along the edge this layer emitted at ``index``.

        Kept in Rust until asked for: a city has hundreds of thousands of edges
        and a drawn route has a hundred.
        """
        return self.network.geometry(index)

    def nearest(
        self, lat: float, lon: float, within: Optional[Iterable[int]] = None
    ) -> int:
        """The OSM node id closest to a coordinate — how you say "start here".

        Args:
            within: Only consider these node ids. Extracts contain stubs that
                connect to nothing under a given profile — a driveway reachable
                only on foot, a fragment cut off at the extract's edge — and
                snapping to one produces a query with no answer for reasons that
                have nothing to do with the routing. Pass the nodes a search
                actually reached to snap somewhere useful.
        """
        index = self.network.nearest(lat, lon, None if within is None else list(within))
        if index is None:
            raise ValueError(
                f"{self!r} has no nodes to be near {lat}, {lon}"
                if within is None
                else f"none of the given nodes are in {self!r}"
            )
        return self.network.node_ids[index]

    @property
    def bounds(self) -> "Tuple[float, float, float, float]":
        """`(min_lat, min_lon, max_lat, max_lon)` of the routable network."""
        return self.network.bounds

    def __repr__(self) -> str:
        state = "unread" if self._network is None else f"{self.network.num_edges} edges"
        return f"OSM({os.path.basename(self.path)!r}, {self.profile!r}, {state})"
