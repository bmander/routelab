"""Behaviour of the searches themselves: what they find and when they stop."""

from __future__ import annotations

import pytest

import routelab as rl


def test_dijkstra_takes_the_cheaper_route(diamond):
    result = rl.dijkstra(diamond, 0)
    assert result.cost(3) == 3
    assert result.path(3) == [0, 1, 3]
    assert diamond.walk(0, result.edge_path(3)) == (3, 3)


def test_a_source_is_its_own_root(diamond):
    result = rl.dijkstra(diamond, 0)
    assert result.cost(0) == 0
    assert result.path(0) == [0]
    assert result.edge_path(0) == []
    assert result.parent(0) is None
    assert result.parent_edge(0) is None


def test_unreached_nodes_report_nothing(diamond):
    result = rl.dijkstra(diamond, 0)
    assert result.cost(4) is None
    assert result.path(4) is None
    assert result.edge_path(4) is None
    assert result.costs == [0, 1, 10, 3, None]


def test_initial_costs_move_the_frontier(diamond):
    # Node 2 free beats node 1 at a 25-unit penalty, flipping which route wins.
    result = rl.dijkstra(diamond, {1: 25, 2: 0})
    assert result.cost(3) == 20
    assert result.path(3) == [2, 3]


def test_sources_accept_every_spelling(diamond):
    expected = rl.dijkstra(diamond, [(0, 0), (2, 5)]).costs
    assert rl.dijkstra(diamond, {0: 0, 2: 5}).costs == expected
    assert rl.dijkstra(diamond, [(0, 0), (2, 5)]).costs == expected
    assert rl.dijkstra(diamond, 0).costs == rl.dijkstra(diamond, [0]).costs


def test_duplicate_sources_keep_the_cheapest(diamond):
    assert rl.dijkstra(diamond, [(1, 7), (1, 2)]).cost(1) == 2


def test_max_cost_bounds_the_search(diamond):
    result = rl.dijkstra(diamond, 0, max_cost=2)
    assert result.cost(1) == 1
    assert result.cost(3) is None
    assert result.order == [0, 1]


def test_targets_stop_the_search_early(diamond):
    result = rl.dijkstra(diamond, 0, targets=1)
    assert result.cost(1) == 1
    assert result.order == [0, 1]


def test_unreachable_target_runs_to_exhaustion(diamond):
    result = rl.dijkstra(diamond, 0, targets=[4])
    assert result.cost(4) is None
    assert result.order == [0, 1, 3, 2]  # by cost: 0, 1, 3, 10


def test_settle_order_is_by_cost_then_node_id():
    # Two nodes reachable at the same cost settle in node-id order, so results
    # are reproducible across runs and implementations.
    graph = rl.Graph.from_edges([(0, 2, 5), (0, 1, 5)], num_nodes=3)
    assert rl.dijkstra(graph, 0).order == [0, 1, 2]


def test_zero_weights_and_cycles_terminate():
    graph = rl.Graph.from_edges([(0, 0, 0), (0, 1, 0), (1, 2, 0), (2, 1, 0)])
    assert rl.dijkstra(graph, 0).costs == [0, 0, 0]


def test_dijkstra_rejects_unknown_sources(diamond):
    with pytest.raises(ValueError, match="out of range"):
        rl.dijkstra(diamond, 99)


def test_bfs_counts_hops_not_weights():
    graph = rl.Graph.from_edges([(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 100)])
    result = rl.bfs(graph, 0)
    assert result.cost(3) == 1
    assert result.path(3) == [0, 3]


def test_bfs_max_depth_bounds_the_frontier():
    graph = rl.Graph.from_edges([(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    result = rl.bfs(graph, 0, max_depth=1)
    assert result.costs == [0, 1, None, None]


def test_bfs_starts_every_source_at_zero():
    graph = rl.Graph.from_edges([(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    result = rl.bfs(graph, [0, 2])
    assert result.costs == [0, 1, 0, 1]


def test_bfs_matches_dijkstra_on_unit_weights():
    graph = rl.Graph.from_edges([(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)])
    assert rl.bfs(graph, 0).costs == rl.dijkstra(graph, 0).costs
