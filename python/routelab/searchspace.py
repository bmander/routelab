"""What an algorithm explored, in a form something can draw.

A route is the answer; the search space is the work. Between them sits most of
what distinguishes one algorithm from another — two planners can return the
identical journey while looking at wildly different parts of the map, and only
the second thing tells you why one took four milliseconds and the other twelve.

Every planner can hand over its search space, and what it hands over depends on
how it searches. Dijkstra and A* grow a **shortest-path tree**: every settled
node remembers the edge it arrived by, and those edges form a tree rooted at the
sources. A contraction hierarchy grows two, from either end, and reports where
they met. Algorithms further along the roadmap explore differently again — a
schedule-based search produces a decision graph, a multicriteria one a frontier
of incomparable labels — and will report those instead. :class:`SearchSpace` is
the common promise: whatever it is, it is made of branches, it has a heaviest
one, and you can draw it.

Drawn plainly, a hundred thousand identical lines say very little, so every kind
of space gives each branch a ``share`` of its own heaviest — which is what lets
a renderer pick a width without knowing what is being measured. A tree weights
branches by everything hanging off them and renders like a river network; a
hierarchy weights them by how much road each one leapt over.
"""

from __future__ import annotations

import heapq
from typing import Hashable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

__all__ = ["Branch", "MeetingTrees", "SearchSpace", "ShortestPathTree"]

#: A point on the ground, `(latitude, longitude)`.
Point = Tuple[float, float]


def _feature(shape: "Sequence[Point]", properties: dict) -> dict:
    """One GeoJSON ``LineString``, with whatever a space wants to say about it."""
    return {
        "type": "Feature",
        # GeoJSON is (longitude, latitude); everything else here is the other
        # way round, which is the classic way to plot a city into the ocean.
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in shape],
        },
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

    def branches(self, *, min_span: int = 0) -> Iterator[tuple]:
        """Every branch as ``(direction, tail, head, level, edges)``, in labels."""
        for direction, tail, head, level, edges in self._branches:
            if len(edges) >= min_span:
                yield (
                    self.DIRECTIONS[direction],
                    self._compiled.label(tail),
                    self._compiled.label(head),
                    level,
                    edges,
                )

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
            shape: "List[Point]" = []
            for edge in edges:
                piece = self._compiled.geometry(edge)
                if piece is None:
                    break
                # Each edge repeats the previous edge's last point.
                shape.extend(piece[1:] if shape else piece)
            if len(shape) < 2:
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
