"""Graph construction and the CSR bookkeeping callers depend on."""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import random_graph


def test_from_edges_infers_node_count():
    graph = rl.Graph.from_edges([(0, 1, 5), (1, 2, 5)])
    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert len(graph) == 3


def test_num_nodes_keeps_trailing_isolated_nodes():
    graph = rl.Graph.from_edges([(0, 1, 5)], num_nodes=10)
    assert graph.num_nodes == 10
    assert graph.out_degree(9) == 0


def test_undirected_inserts_both_directions():
    graph = rl.Graph.from_edges([(0, 1, 5)], directed=False)
    assert graph.num_edges == 2
    assert graph.neighbors(0) == [(1, 5, 0)]
    assert graph.neighbors(1) == [(0, 5, 1)]


def test_from_adjacency_matches_from_edges():
    adjacency = {0: [(1, 5), (2, 7)], 2: [(1, 1)]}
    by_adjacency = rl.Graph.from_adjacency(adjacency)
    by_edges = rl.Graph.from_edges([(0, 1, 5), (0, 2, 7), (2, 1, 1)])
    assert by_adjacency.edges() == by_edges.edges()


def test_input_index_recovers_caller_ordering():
    # CSR groups by tail, so edge ids are not input positions; input_index is how
    # per-edge attributes stay attached.
    edges = [(2, 0, 1), (0, 1, 2), (2, 1, 3)]
    graph = rl.Graph.from_edges(edges, num_nodes=3)
    assert [graph.edge(e) for e in range(graph.num_edges)] == [
        (0, 1, 2),
        (2, 0, 1),
        (2, 1, 3),
    ]
    assert [edges[graph.input_index(e)] for e in range(graph.num_edges)] == [
        graph.edge(e) for e in range(graph.num_edges)
    ]


def test_parallel_edges_and_self_loops_are_kept():
    graph = rl.Graph.from_edges([(0, 0, 1), (0, 1, 2), (0, 1, 3)], num_nodes=2)
    assert graph.out_degree(0) == 3
    assert graph.neighbors(0) == [(0, 1, 0), (1, 2, 1), (1, 3, 2)]


def test_walk_reports_where_a_path_lands():
    graph = rl.Graph.from_edges([(0, 1, 5), (1, 2, 7)])
    assert graph.walk(0, [0, 1]) == (2, 12)
    assert graph.walk(0, []) == (0, 0)
    with pytest.raises(ValueError, match="do not form a walk"):
        graph.walk(1, [0])


@pytest.mark.parametrize("seed", range(5))
def test_neighbors_agree_with_the_edge_list(seed):
    graph, edges = random_graph(seed, num_nodes=20)
    from_neighbors = sorted(
        (tail, head, weight)
        for tail in range(graph.num_nodes)
        for head, weight, _edge in graph.neighbors(tail)
    )
    assert from_neighbors == sorted(edges)


def test_rejects_out_of_range_nodes():
    with pytest.raises(ValueError, match="out of range"):
        rl.Graph.from_edges([(0, 5, 1)], num_nodes=2)


def test_rejects_reserved_infinite_weight():
    with pytest.raises(ValueError, match="unreachable"):
        rl.Graph.from_edges([(0, 1, 2**32 - 1)])


def test_rejects_negative_weights():
    with pytest.raises((ValueError, OverflowError, TypeError)):
        rl.Graph.from_edges([(0, 1, -1)])


def test_indexing_errors_are_index_errors(diamond):
    with pytest.raises(IndexError):
        diamond.neighbors(99)
    with pytest.raises(IndexError):
        diamond.edge(99)
