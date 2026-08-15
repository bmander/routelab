"""What an algorithm explored, in a form something can draw.

A route is the answer; the search space is the work. Between them sits most of
what distinguishes one algorithm from another — two planners can return the
identical journey while looking at wildly different parts of the map, and only
the second thing tells you why one took four milliseconds and the other twelve.

Every planner can hand over its search space, and what it hands over depends on
how it searches. Dijkstra and A* grow a **shortest-path tree**: every settled
node remembers the edge it arrived by, and those edges form a tree rooted at the
sources. Algorithms further along the roadmap explore differently — a
schedule-based search produces a decision graph, a multicriteria one a frontier
of incomparable labels — and will report those instead. :class:`SearchSpace` is
the common promise: whatever it is, you can draw it.

Drawn plainly, a tree of a hundred thousand identical lines says very little.
:class:`ShortestPathTree` therefore weights each branch by everything hanging off
it, which makes it render like a river network — thick trunks where the whole
search flowed, thinning to capillaries at the frontier.
"""

from __future__ import annotations

import heapq
from typing import Hashable, Iterator, List, NamedTuple, Optional

__all__ = ["Branch", "SearchSpace", "ShortestPathTree"]


class SearchSpace:
    """The part of the problem an algorithm actually looked at."""

    #: What shape this space is, for a renderer that handles more than one.
    kind = "unknown"

    def geojson(self) -> dict:
        """The space as a GeoJSON ``FeatureCollection``.

        GeoJSON because it is the one format every map tool already reads —
        Leaflet, QGIS, geojson.io — so a search can be looked at without writing
        a renderer first. Implementations may add keyword options of their own;
        what they cannot do is quietly accept ones they do not understand.
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
        """The largest magnitude — what a renderer scales its widths against."""
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

    def geometry(self, branch: Branch) -> "Optional[List[tuple]]":
        """The `(lat, lon)` shape of a branch, if its layer keeps one."""
        return self._compiled.geometry(branch.edge)

    def geojson(self, *, min_magnitude: int = 0, limit: Optional[int] = None) -> dict:
        """The tree as GeoJSON, one ``LineString`` per branch.

        Each feature carries its ``magnitude`` and a ``share`` of the peak, which
        is what a renderer needs to pick a width without knowing the units.

        Features come back heaviest-first when ``limit`` is given, and in tree
        order otherwise — a collection is unordered, and sorting tens of
        thousands of branches nobody is going to drop is work for nothing.

        Args:
            min_magnitude: Drop branches carrying less than this.
            limit: Keep only the heaviest this many branches. A city-wide search
                is hundreds of thousands of them and a map cannot show that; the
                heaviest are the ones that carry the shape.
        """
        # Straight down the arrays rather than through `branches()`: that
        # resolves both endpoint labels per branch, and a feature needs neither.
        selected = [
            (edge, magnitude)
            for edge, magnitude in zip(self._edges, self._magnitudes)
            if magnitude >= min_magnitude
        ]
        if limit is not None and limit < len(selected):
            # `nlargest`, not sort-then-slice: a limit worth setting is far
            # smaller than the tree it is cutting down.
            selected = heapq.nlargest(limit, selected, key=lambda branch: branch[1])

        peak = self.peak or 1
        features = []
        for edge, magnitude in selected:
            shape = self._compiled.geometry(edge)
            if shape is None:
                continue  # a layer without geometry has nothing to draw
            features.append(
                {
                    "type": "Feature",
                    # GeoJSON is (longitude, latitude); everything else here is
                    # the other way round, which is the classic way to plot a
                    # city into the ocean.
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for lat, lon in shape],
                    },
                    "properties": {"magnitude": magnitude, "share": magnitude / peak},
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def __repr__(self) -> str:
        return (
            f"ShortestPathTree({len(self)} branches, "
            f"magnitude={self.magnitude!r}, peak={self.peak})"
        )
