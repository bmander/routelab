"""Differential tests: the kernel against the reference, and both against an oracle.

Three layers, weakest claim to strongest:

1. The Rust kernel and the pure-Python reference agree on the whole result —
   costs, parents, and settle order — over messy random graphs.
2. Both agree with Bellman-Ford, which is a different algorithm rather than a
   second copy of the same one.
3. Every path a search returns is walked on the graph and has to land on the
   right node at the reported cost.

Layer 1 alone would only prove the two implementations share a bug.
"""

from __future__ import annotations

import random

import pytest

import routelab as rl

from conftest import bellman_ford, random_graph

SEEDS = range(12)


def assert_same_result(fast, reference, num_nodes):
    """Compare every field of a kernel result against the reference's."""
    assert fast.costs == reference.costs
    assert fast.order == reference.order
    for node in range(num_nodes):
        assert fast.cost(node) == reference.cost(node), f"cost at {node}"
        assert fast.parent(node) == reference.parent(node), f"parent of {node}"
        assert fast.parent_edge(node) == reference.parent_edge(node), f"edge to {node}"
        assert fast.path(node) == reference.path(node), f"path to {node}"
        assert fast.edge_path(node) == reference.edge_path(node), f"edge path to {node}"


def assert_paths_are_walkable(graph, result, sources, *, unit_cost=False):
    """A path is only correct if walking it arrives where it claims, at its cost.

    `unit_cost` switches the cost the path is checked against from summed edge
    weights to hop count, which is what BFS reports.
    """
    source_costs = dict(sources)
    for node in range(graph.num_nodes):
        edge_path = result.edge_path(node)
        if edge_path is None:
            continue
        path = result.path(node)
        start = path[0]
        end, walked = graph.walk(start, edge_path)
        assert end == node
        assert start in source_costs, f"path to {node} starts at non-source {start}"
        traversed = len(edge_path) if unit_cost else walked
        assert source_costs[start] + traversed == result.cost(node)


@pytest.mark.parametrize("seed", SEEDS)
def test_dijkstra_matches_reference(seed):
    graph, _ = random_graph(seed)
    rng = random.Random(seed)
    sources = [
        (rng.randrange(graph.num_nodes), rng.randrange(10))
        for _ in range(rng.randint(1, 3))
    ]

    result = rl.dijkstra(graph, sources)
    assert_same_result(result, rl.reference.dijkstra(graph, sources), graph.num_nodes)
    assert_paths_are_walkable(graph, result, sources)


@pytest.mark.parametrize("seed", SEEDS)
def test_dijkstra_matches_bellman_ford(seed):
    graph, _ = random_graph(seed)
    sources = [(0, 0), (graph.num_nodes // 2, 3)]
    assert rl.dijkstra(graph, sources).costs == bellman_ford(graph, sources)


@pytest.mark.parametrize("seed", SEEDS)
def test_bounded_search_matches_the_unbounded_one(seed):
    """Under `max_cost`, settled nodes keep their true distances; the rest vanish."""
    graph, _ = random_graph(seed)
    full = rl.dijkstra(graph, 0)
    bound = 15
    bounded = rl.dijkstra(graph, 0, max_cost=bound)

    expected = [
        cost if cost is not None and cost <= bound else None for cost in full.costs
    ]
    assert bounded.costs == expected
    assert_same_result(
        bounded, rl.reference.dijkstra(graph, 0, max_cost=bound), graph.num_nodes
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_early_exit_does_not_change_the_answer(seed):
    """Stopping at a target must not change what was already settled."""
    graph, _ = random_graph(seed)
    full = rl.dijkstra(graph, 0)
    reachable = [node for node in full.order if full.cost(node) is not None]
    target = reachable[len(reachable) // 2]

    stopped = rl.dijkstra(graph, 0, targets=[target])
    assert stopped.cost(target) == full.cost(target)
    assert stopped.order == full.order[: full.order.index(target) + 1]
    for node in stopped.order:
        assert stopped.cost(node) == full.cost(node)
    assert_same_result(
        stopped, rl.reference.dijkstra(graph, 0, targets=[target]), graph.num_nodes
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_matches_reference(seed):
    graph, _ = random_graph(seed)
    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_nodes) for _ in range(rng.randint(1, 3))]

    result = rl.bfs(graph, sources)
    assert_same_result(result, rl.reference.bfs(graph, sources), graph.num_nodes)
    assert_paths_are_walkable(
        graph, result, [(node, 0) for node in sources], unit_cost=True
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_matches_dijkstra_on_unit_weights(seed):
    _, edges = random_graph(seed)
    unit = rl.Graph.from_edges([(tail, head, 1) for tail, head, _ in edges])
    assert rl.bfs(unit, 0).costs == rl.dijkstra(unit, 0).costs


@pytest.mark.parametrize("seed", SEEDS)
def test_bounded_bfs_matches_reference(seed):
    graph, _ = random_graph(seed)
    assert_same_result(
        rl.bfs(graph, 0, max_depth=2),
        rl.reference.bfs(graph, 0, max_depth=2),
        graph.num_nodes,
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_dense_and_sparse_instances_agree(seed):
    for density in (0.5, 8.0):
        graph, _ = random_graph(seed, num_nodes=25, density=density)
        assert_same_result(
            rl.dijkstra(graph, 0), rl.reference.dijkstra(graph, 0), graph.num_nodes
        )
