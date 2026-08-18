"""What an algorithm explored, in a form something can draw.

A route is the answer; the search space is the work. Between them sits most of
what distinguishes one algorithm from another — two planners can return the
identical journey while looking at wildly different parts of the map, and only
the second thing tells you why one took four milliseconds and the other twelve.

Every planner can hand over its search space, and what it hands over depends on
how it searches. Dijkstra and A* grow a **shortest-path tree**: every settled
node remembers the edge it arrived by, and those edges form a tree rooted at the
sources. A contraction hierarchy grows two, from either end, and reports where
they met. A round-based transit search keeps no tree at all — a label per stop
per round — and reports which round first reached each stop; a connection scan
keeps a label per stop and reports how far its sweep through the day got. Algorithms further
along the roadmap will report other shapes again — a multicriteria search a
frontier of incomparable labels. :class:`SearchSpace` is the common promise:
whatever it is, it is made of branches, each a named tuple with a place on the
ground, it has a heaviest one, and you can draw it.

Drawn plainly, a hundred thousand identical lines say very little, so every kind
of space gives each branch a ``share`` of its own heaviest — which is what lets
a renderer pick a width without knowing what is being measured. A tree weights
branches by everything hanging off them and renders like a river network; a
hierarchy weights them by how much road each one leapt over.
"""

from __future__ import annotations

import heapq
from typing import Any, Hashable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "Arrival",
    "Branch",
    "Leap",
    "MeetingTrees",
    "Reach",
    "Rounds",
    "Scan",
    "SearchSpace",
    "Segment",
    "Segments",
    "ShortestPathTree",
]

#: A point on the ground, `(latitude, longitude)`.
Point = Tuple[float, float]


def _feature(shape: "Sequence[Point]", properties: dict, kind: str = "LineString") -> dict:
    """One GeoJSON feature, with whatever a space wants to say about it.

    A ``LineString`` from a run of points, or a ``Point`` from one — a stop is
    a place, not a road, and a space made of stops draws those.
    """
    # GeoJSON is (longitude, latitude); everything else here is the other way
    # round, which is the classic way to plot a city into the ocean.
    coordinates: Any = [[lon, lat] for lat, lon in shape]
    if kind == "Point":
        coordinates = coordinates[0]
    return {
        "type": "Feature",
        "geometry": {"type": kind, "coordinates": coordinates},
        "properties": properties,
    }


def _heaviest(items: list, weight, limit: Optional[int]) -> list:
    """The heaviest ``limit`` items, or all of them in their own order.

    `nlargest`, not sort-then-slice: a limit worth setting is far smaller than
    the collection it is cutting down. Below the limit nothing is sorted at all,
    because a feature collection is unordered and ordering tens of thousands of
    branches nobody is going to drop is work for nothing.
    """
    if limit is None or limit >= len(items):
        return items
    return heapq.nlargest(limit, items, key=weight)


class SearchSpace:
    """The part of the problem an algorithm actually looked at."""

    #: What shape this space is, for a renderer that handles more than one.
    kind = "unknown"

    def __len__(self) -> int:
        """How many branches it has."""
        raise NotImplementedError

    @property
    def peak(self) -> int:
        """The heaviest branch — what ``share`` is measured against.

        What "heaviest" counts is the space's own business, and differs: subtree
        travel time for a tree, original edges leapt for a hierarchy. What holds
        across kinds is that dividing by it gives a number between zero and one.
        """
        raise NotImplementedError

    def geojson(self, *, limit: Optional[int] = None) -> dict:
        """The space as a GeoJSON ``FeatureCollection``.

        GeoJSON because it is the one format every map tool already reads —
        Leaflet, QGIS, geojson.io — so a search can be looked at without writing
        a renderer first. Every feature carries a ``share`` of :attr:`peak`,
        whatever the kind, so one width rule draws all of them.

        Implementations may add keyword options of their own; what they cannot
        do is quietly accept ones they do not understand.

        Args:
            limit: Keep only the heaviest this many branches. A city-wide search
                is hundreds of thousands of them and a map cannot show that; the
                heaviest are the ones that carry the shape.
        """
        raise NotImplementedError


class Branch(NamedTuple):
    """One edge of a tree, and the weight of everything beyond it."""

    tail: Hashable
    head: Hashable
    #: The graph edge, for asking the environment about its shape.
    edge: int
    #: The subtree total: see :class:`ShortestPathTree`.
    magnitude: int


class Leap(NamedTuple):
    """One branch of a hierarchy search: which half it belongs to, and how much
    road it stood for."""

    #: ``"forward"`` from the source, ``"backward"`` from the target.
    direction: str
    tail: Hashable
    head: Hashable
    #: The contraction rank of the branch's higher end.
    level: int
    #: The original edges this branch leapt over, in order.
    edges: List[int]


class Reach(NamedTuple):
    """One stop a round-based search reached, and when it first did."""

    stop: Hashable
    #: The round that first reached it: 0 for the origins and what they can
    #: walk to, `k` for a stop first reached with `k` trips.
    round: int
    #: When :attr:`round` got there, on the service-day clock — not the best
    #: arrival the whole search settled on, which a later round may improve
    #: without ever discovering the stop. Both halves of the pair come from
    #: the same round, so a reach describes one journey rather than two; the
    #: best arrival is the planner's answer, and lives on the result.
    arrives: int


class Arrival(NamedTuple):
    """One stop a connection scan labelled, and when."""

    stop: Hashable
    #: The earliest arrival, on the service-day clock.
    arrives: int


class Segment(NamedTuple):
    """One trip segment a trip-based sweep scanned: a vehicle, from the stop
    it was boarded at to the last stop whose transfers were read."""

    #: The trip, as the layer numbers it.
    trip: int
    #: The stops covered, boarded first.
    stops: Tuple[Hashable, ...]
    #: The number of changes it was reached with: 0 for a trip boarded at
    #: the origin or a walk from it.
    round: int


class ShortestPathTree(SearchSpace):
    """The tree a Dijkstra-family search grew, weighted by subtree totals.

    Each branch carries the total of everything beyond it, so the magnitudes
    grow toward the root: the trunk is where the entire search passed, the twigs
    are where it stopped. ``magnitude="weight"`` accumulates travel time beyond
    each branch; ``"nodes"`` counts settled nodes instead.

    Only settled nodes take part. A node the search touched but never settled
    has no final cost, and no place in a picture of what the search concluded.
    """

    kind = "shortest-path-tree"

    def __init__(self, compiled, result, magnitude: str = "weight"):
        self._compiled = compiled
        self.magnitude = magnitude
        # Kept as parallel arrays rather than objects: a city-wide search is
        # hundreds of thousands of branches, and most callers want to filter
        # before they want to iterate.
        tree = result.tree(compiled.graph, magnitude)
        self._tails = tree.tails
        self._heads = tree.heads
        self._edges = tree.edges
        self._magnitudes = tree.magnitudes
        self._peak = tree.peak

    def __len__(self) -> int:
        return len(self._edges)

    @property
    def peak(self) -> int:
        return self._peak

    def branches(self, *, min_magnitude: int = 0) -> Iterator[Branch]:
        """Every branch, optionally only the ones carrying at least so much.

        Filtering is how a big tree stays drawable: dropping the capillaries
        removes most of the branches and little of the picture.
        """
        for tail, head, edge, magnitude in zip(
            self._tails, self._heads, self._edges, self._magnitudes
        ):
            if magnitude >= min_magnitude:
                yield Branch(
                    self._compiled.label(tail), self._compiled.label(head), edge, magnitude
                )

    def geometry(self, branch: Branch) -> "Optional[List[Point]]":
        """The `(lat, lon)` shape of a branch, if its layer keeps one."""
        return self._compiled.geometry(branch.edge)

    def geojson(self, *, min_magnitude: int = 0, limit: Optional[int] = None) -> dict:
        """The tree as GeoJSON, one ``LineString`` per branch.

        Args:
            min_magnitude: Drop branches carrying less than this.
            limit: See :meth:`SearchSpace.geojson`.
        """
        # Straight down the arrays rather than through `branches()`: that
        # resolves both endpoint labels per branch, and a feature needs neither.
        selected = _heaviest(
            [
                (edge, magnitude)
                for edge, magnitude in zip(self._edges, self._magnitudes)
                if magnitude >= min_magnitude
            ],
            lambda branch: branch[1],
            limit,
        )

        peak = self.peak or 1
        features = []
        for edge, magnitude in selected:
            shape = self._compiled.geometry(edge)
            if shape is None:
                continue  # a layer without geometry has nothing to draw
            features.append(
                _feature(shape, {"magnitude": magnitude, "share": magnitude / peak})
            )
        return {"type": "FeatureCollection", "features": features}

    def __repr__(self) -> str:
        return (
            f"ShortestPathTree({len(self)} branches, "
            f"magnitude={self.magnitude!r}, peak={self.peak})"
        )


class MeetingTrees(SearchSpace):
    """Two searches climbing a hierarchy from opposite ends, and where they met.

    What a contraction hierarchy explores is not one tree: it is a small search
    up from the source, a small search up from the target, and a node above both
    where they join. Neither half is interesting alone — the shape is the pair.

    Every branch is reported with its **unpacked** geometry, so what gets drawn
    is real road rather than a straight line through the buildings a shortcut
    leapt over. The leaping shows up instead in two properties: ``span``, how
    many original edges the branch stood for, and ``level``, the contraction
    rank of its higher end. A branch with a span of three hundred is the search
    crossing a city in one step, which is the whole trick.
    """

    kind = "meeting-trees"

    #: What a branch's `direction` property says.
    DIRECTIONS = ("forward", "backward")

    def __init__(self, compiled, search):
        self._compiled = compiled
        self._branches = search.branches()
        self.meeting = search.meeting

    def __len__(self) -> int:
        return len(self._branches)

    @property
    def peak(self) -> int:
        """The longest branch, in original edges — the biggest single leap."""
        return max((len(edges) for _, _, _, _, edges in self._branches), default=0)

    def branches(self, *, min_span: int = 0) -> "Iterator[Leap]":
        """Every branch as a :class:`Leap`, in labels."""
        for direction, tail, head, level, edges in self._branches:
            if len(edges) >= min_span:
                yield Leap(
                    self.DIRECTIONS[direction],
                    self._compiled.label(tail),
                    self._compiled.label(head),
                    level,
                    edges,
                )

    def geometry(self, leap: "Leap") -> "Optional[List[Point]]":
        """The unpacked shape of one leap: real road, edge by edge, or ``None``
        where a layer along it has none."""
        return self._shape(leap.edges)

    def _shape(self, edges: "Sequence[int]") -> "Optional[List[Point]]":
        shape: "List[Point]" = []
        for edge in edges:
            piece = self._compiled.geometry(edge)
            if piece is None:
                return None
            # Each edge repeats the previous edge's last point.
            shape.extend(piece[1:] if shape else piece)
        return shape if len(shape) >= 2 else None

    def geojson(self, *, min_span: int = 0, limit: Optional[int] = None) -> dict:
        """Both halves as GeoJSON, one ``LineString`` per branch.

        Args:
            min_span: Drop branches standing for fewer than this many edges.
            limit: See :meth:`SearchSpace.geojson`.
        """
        selected = _heaviest(
            [
                (direction, level, edges)
                for direction, _, _, level, edges in self._branches
                if len(edges) >= min_span
            ],
            lambda branch: len(branch[2]),
            limit,
        )

        peak = self.peak or 1
        features = []
        for direction, level, edges in selected:
            shape = self._shape(edges)
            if shape is None:
                continue
            features.append(
                _feature(
                    shape,
                    {
                        "direction": self.DIRECTIONS[direction],
                        "span": len(edges),
                        "level": level,
                        "share": len(edges) / peak,
                    },
                )
            )
        return {"type": "FeatureCollection", "features": features}

    def __repr__(self) -> str:
        return f"MeetingTrees({len(self)} branches, longest span={self.peak})"


class Rounds(SearchSpace):
    """Every stop a round-based search reached, by the round that first got there.

    Not a tree: RAPTOR keeps a label per stop per round and never a parent graph
    the way Dijkstra does, so what it can honestly report is where each round's
    frontier lay. Drawn as points — a stop is a place, not a road — coloured by
    round, which is the picture in the paper: the origin's neighbourhood, then
    everything one bus away, then two.

    A stop's arrival here is the one its own round achieved, not the best the
    search ended up with — see :class:`Reach`. That is what makes this a
    frontier rather than a table of answers: the planner's answer for a stop
    is :meth:`~routelab.kernels.Planner.journey`, and may be a later round's.
    """

    kind = "rounds"

    def __init__(self, compiled, result):
        self._compiled = compiled
        self._reached = result.reached()
        self._points = compiled.coordinates()

    def __len__(self) -> int:
        return len(self._reached)

    @property
    def peak(self) -> int:
        """The last round that reached a stop no earlier round had.

        A round number rather than a count of rounds — it is what a renderer
        divides a stop's round by — and lower than the search's own round
        count whenever the final rounds only improved stops already found.
        """
        return max((round for _, round, _ in self._reached), default=0)

    def branches(self, *, min_round: int = 0) -> "Iterator[Reach]":
        """Every stop reached as a :class:`Reach`, in labels."""
        for stop, round, arrives in self._reached:
            if round >= min_round:
                yield Reach(self._compiled.label(stop), round, arrives)

    def geometry(self, reach: "Reach") -> "Optional[List[Point]]":
        """Where the stop is, as a one-point shape, or ``None`` if no layer
        places it."""
        point = self._points.get(reach.stop)
        return None if point is None else [point]

    def geojson(self, *, min_round: int = 0, limit: Optional[int] = None) -> dict:
        """The stops as GeoJSON ``Point`` features, each with the ``round``
        that reached it and when it ``arrives``.

        ``peak`` rides on the collection rather than a ``share`` on every
        feature: a round is a small integer and the last one is the same
        number for all of them, so a renderer divides once instead of reading
        a float per stop.

        Straight down the arrays rather than through :meth:`branches`, for the
        reason :meth:`ShortestPathTree.geojson` gives: a city's search is
        thousands of stops and a feature needs neither a `Reach` nor a label
        resolved twice.

        Args:
            min_round: Drop stops first reached before this round.
            limit: Keep only this many, the latest-reached first — the
                frontier, which is the part a crowded map can still show.
        """
        label, points = self._compiled.label, self._points
        selected = _heaviest(
            [reach for reach in self._reached if reach[1] >= min_round],
            lambda reach: reach[1],
            limit,
        )
        features = []
        for stop, round, arrives in selected:
            point = points.get(label(stop))
            if point is None:
                continue
            features.append(
                _feature([point], {"round": round, "arrives": arrives}, kind="Point")
            )
        return {"type": "FeatureCollection", "features": features, "peak": self.peak}

    def __repr__(self) -> str:
        # "out to round k" rather than "over k rounds": `peak` is a round
        # number, so reading it as a count names neither the rounds the search
        # ran nor the rounds that found something — and it reads as "1 rounds"
        # when the answer is round 1.
        return f"Rounds({len(self):,} stops, out to round {self.peak})"


class Scan(SearchSpace):
    """Every stop a connection scan labelled, by when it was reached.

    Not a tree and not rounds: CSA sweeps one array of connections in
    departure order and a stop's label is the moment the sweep first got
    there, so what it can honestly report is the sweep — the stops it
    labelled, each stamped with its arrival. Drawn as points, coloured by how
    long after departure each was reached: the origin's neighbourhood first
    and the far side of the network last, which is the picture of a scan
    running toward its target and stopping when it passes it.
    """

    kind = "scan"

    def __init__(self, compiled, result):
        self._compiled = compiled
        self._reached = result.reached()
        self._departing = result.departing
        self._points = compiled.coordinates()
        self._peak = max(
            (arrives - self._departing for _, arrives in self._reached), default=0
        )

    def __len__(self) -> int:
        return len(self._reached)

    @property
    def departing(self) -> int:
        """When the query left: what ``after`` counts from."""
        return self._departing

    @property
    def peak(self) -> int:
        """The scan's horizon: seconds from departure to the last arrival it
        labelled, which is what a stop's ``after`` is a share of. Held from
        construction, the way :class:`ShortestPathTree` holds its own."""
        return self._peak

    def branches(self, *, min_after: int = 0) -> "Iterator[Arrival]":
        """Every stop reached as an :class:`Arrival`, in labels."""
        for stop, arrives in self._reached:
            if arrives - self._departing >= min_after:
                yield Arrival(self._compiled.label(stop), arrives)

    def geometry(self, arrival: "Arrival") -> "Optional[List[Point]]":
        """Where the stop is, as a one-point shape, or ``None`` if no layer
        places it."""
        point = self._points.get(arrival.stop)
        return None if point is None else [point]

    def geojson(self, *, min_after: int = 0, limit: Optional[int] = None) -> dict:
        """The stops as GeoJSON ``Point`` features, each with when it
        ``arrives`` and how many seconds ``after`` departure that is.

        ``peak`` rides on the collection under the name every kind uses, the
        way :class:`Rounds` ships its last round: a renderer divides once
        rather than reading a float per stop, and does it the same way here.

        Args:
            min_after: Drop stops reached sooner than this after departure.
            limit: Keep only this many, the latest-reached first — the edge of
                the sweep, which is the part a crowded map can still show.
        """
        label, points, departing = self._compiled.label, self._points, self._departing
        selected = _heaviest(
            [reach for reach in self._reached if reach[1] - departing >= min_after],
            lambda reach: reach[1],
            limit,
        )
        features = []
        for stop, arrives in selected:
            point = points.get(label(stop))
            if point is None:
                continue
            features.append(
                _feature([point], {"arrives": arrives, "after": arrives - departing}, kind="Point")
            )
        return {"type": "FeatureCollection", "features": features, "peak": self._peak}

    def __repr__(self) -> str:
        return f"Scan({len(self):,} stops within {self._peak // 60} min)"


class Segments(SearchSpace):
    """Every trip segment a trip-based sweep scanned, by the round it was
    reached in.

    Not stops: Witt's search labels trips, and what it can honestly report is
    the vehicles it looked at — each from the stop it was boarded at to the
    last stop whose transfers were followed. Drawn as lines along the stops,
    coloured by round the way :class:`Rounds` colours its stops: the trips a
    rider could board first, then everything one change away, then two.
    """

    kind = "segments"

    def __init__(self, compiled, result):
        self._compiled = compiled
        self._reached = result.reached()
        self._points = compiled.coordinates()

    def __len__(self) -> int:
        return len(self._reached)

    @property
    def peak(self) -> int:
        """The last round any segment was reached in — a round number rather
        than a count of rounds, as :attr:`Rounds.peak` is."""
        return max((round for round, _, _ in self._reached), default=0)

    def branches(self, *, min_round: int = 0) -> "Iterator[Segment]":
        """Every segment scanned as a :class:`Segment`, in labels."""
        label = self._compiled.label
        for round, trip, stops in self._reached:
            if round >= min_round:
                yield Segment(trip, tuple(label(stop) for stop in stops), round)

    def geometry(self, segment: "Segment") -> "Optional[List[Point]]":
        """The segment as a polyline through its stops, or ``None`` if no
        layer places every one of them."""
        points = [self._points.get(stop) for stop in segment.stops]
        if any(point is None for point in points):
            return None
        return points  # type: ignore[return-value]

    def geojson(self, *, min_round: int = 0, limit: Optional[int] = None) -> dict:
        """The segments as GeoJSON ``LineString`` features, each with the
        ``round`` it was reached in and its ``trip``.

        ``peak`` rides on the collection, as it does for :class:`Rounds`, so a
        renderer divides once rather than reading a float per segment.

        Args:
            min_round: Drop segments reached before this round.
            limit: Keep only this many, the latest-reached first — the edge of
                the sweep, which is the part a crowded map can still show.
        """
        label, points = self._compiled.label, self._points
        selected = _heaviest(
            [reach for reach in self._reached if reach[0] >= min_round],
            lambda reach: reach[0],
            limit,
        )
        features = []
        for round, trip, stops in selected:
            shape = [points.get(label(stop)) for stop in stops]
            if any(point is None for point in shape):
                continue
            features.append(_feature(shape, {"round": round, "trip": trip}))
        return {"type": "FeatureCollection", "features": features, "peak": self.peak}

    def __repr__(self) -> str:
        # A round number, not a count — see :meth:`Rounds.__repr__`.
        return f"Segments({len(self):,} trip segments, out to round {self.peak})"
