"""Shared test fixtures: random instances and an independent shortest-path oracle."""

from __future__ import annotations

import math
import random
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pytest

import routelab as rl


@pytest.fixture
def diamond() -> rl.Graph:
    """0->1->3 costs 3, 0->2->3 costs 30, and node 4 is isolated."""
    return rl.Graph.from_edges(
        [(0, 1, 1), (0, 2, 10), (1, 3, 2), (2, 3, 20)], num_nodes=5
    )


def random_graph(
    seed: int,
    *,
    num_nodes: int = 40,
    density: float = 2.5,
    max_weight: int = 20,
) -> Tuple[rl.Graph, List[Tuple[int, int, int]]]:
    """A random directed graph, and the edge list it was built from.

    Deliberately messy: zero weights, self-loops, parallel edges, and usually a
    few unreachable nodes. Those are the cases where two implementations that
    agree on a clean instance start to disagree.
    """
    rng = random.Random(seed)
    edges = [
        (
            rng.randrange(num_nodes),
            rng.randrange(num_nodes),
            rng.randrange(max_weight + 1),
        )
        for _ in range(int(num_nodes * density))
    ]
    return rl.Graph.from_edges(edges, num_nodes=num_nodes), edges


class GeometricGraph(NamedTuple):
    """A graph whose edges are priced against the distance they cover."""

    graph: rl.Graph
    xs: List[float]
    ys: List[float]
    cost_per_distance: float

    def heuristic(self) -> "rl.SearchResult":
        """The kernel heuristic this instance is built to make admissible."""
        return rl._routelab.Heuristic.euclidean(self.xs, self.ys, self.cost_per_distance)

    def distance(self, a: int, b: int) -> float:
        return math.dist((self.xs[a], self.ys[a]), (self.xs[b], self.ys[b]))


def random_geometric_graph(
    seed: int,
    *,
    num_nodes: int = 40,
    density: float = 3.0,
    cost_per_distance: float = 2.0,
    extent: float = 1000.0,
) -> GeometricGraph:
    """Random points, with every edge costing at least the ground it covers.

    That floor is what makes ``cost_per_distance`` a true lower bound and the
    Euclidean heuristic admissible — by construction, so a test failure means the
    search is wrong rather than the instance being unfair. Weights are inflated by
    a random detour factor, because a heuristic that is exactly tight is a
    suspiciously easy case.
    """
    rng = random.Random(seed)
    xs = [rng.uniform(0.0, extent) for _ in range(num_nodes)]
    ys = [rng.uniform(0.0, extent) for _ in range(num_nodes)]

    edges = []
    for _ in range(int(num_nodes * density)):
        tail = rng.randrange(num_nodes)
        head = rng.randrange(num_nodes)
        straight = math.dist((xs[tail], ys[tail]), (xs[head], ys[head]))
        weight = math.ceil(straight * cost_per_distance) + rng.randrange(0, 100)
        edges.append((tail, head, weight))

    graph = rl.Graph.from_edges(edges, num_nodes=num_nodes)
    return GeometricGraph(graph, xs, ys, cost_per_distance)


def grid_environment(
    side: int, *, spacing: float = 100.0, diagonal: bool = True
) -> rl.Environment:
    """A `side` x `side` grid as a labelled environment, with coordinates.

    Every edge costs exactly the distance it covers, so the Euclidean bound is
    tight along a straight line. `diagonal` decides how much that is worth: with
    diagonal moves a straight-line path exists and the bound is nearly exact;
    without them, movement is L1 while the estimate is L2, so the bound is off by
    up to sqrt(2) everywhere and guidance buys almost nothing.
    """
    cost_per_distance = 1.0
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if diagonal else [])
    edges = []
    positions = {}
    for row in range(side):
        for col in range(side):
            positions[(row, col)] = (col * spacing, row * spacing)
            for row_step, col_step in steps:
                neighbor = (row + row_step, col + col_step)
                if 0 <= neighbor[0] < side and 0 <= neighbor[1] < side:
                    weight = math.ceil(math.hypot(row_step, col_step) * spacing)
                    edges.append(((row, col), neighbor, weight))
                    edges.append((neighbor, (row, col), weight))

    return rl.Environment(
        rl.ScalarEdges(edges, cost_per_distance=cost_per_distance),
        rl.Positions(positions),
    )


def bellman_ford(
    graph: rl.Graph, sources: Sequence[Tuple[int, int]]
) -> "List[Optional[int]]":
    """Shortest-path costs by relaxing every edge until nothing improves.

    An oracle by way of a different algorithm, not a second copy of the same one:
    no priority queue, no settle order, no assumption that a node is done when it
    is first popped. If Dijkstra and this disagree, one of them is wrong.
    """
    costs: Dict[int, int] = {}
    for node, cost in sources:
        if node not in costs or cost < costs[node]:
            costs[node] = cost

    all_edges = graph.edges()
    for _ in range(graph.num_nodes):
        changed = False
        for tail, head, weight in all_edges:
            if tail in costs and costs[tail] + weight < costs.get(head, float("inf")):
                costs[head] = costs[tail] + weight
                changed = True
        if not changed:
            break

    return [costs.get(node) for node in range(graph.num_nodes)]
