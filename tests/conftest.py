"""Shared test fixtures: random instances and an independent shortest-path oracle."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

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
