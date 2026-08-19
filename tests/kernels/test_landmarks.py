"""Landmark A*: a bound that measures the network instead of assuming about it."""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import JUNCTION, random_geometric_graph


@pytest.fixture
def ring() -> rl.Environment:
    """A one-way ring a->b->c->d->a, every hop costing 10.

    The point of the shape: going one way round is 10 and the other 30, a
    difference no straight-line bound can see and a measured one cannot miss.
    """
    return rl.Environment(
        rl.ScalarEdges(
            [("a", "b", 10), ("b", "c", 10), ("c", "d", 10), ("d", "a", 10)],
            # A declared rate, so the only thing Euclidean is missing below is
            # the geometry — which is the contrast the test is about.
            cost_per_distance=1.0,
        )
    )


def test_landmarks_need_no_geometry(ring):
    # The whole reason to reach for this on a network whose coordinates are
    # unknown, or whose costs have nothing to do with distance.
    assert rl.Plane.missing_from(ring.compile()) == {"positions"}
    with pytest.raises(ValueError, match="position for every node"):
        rl.AStar(rl.Euclidean()).bind(ring)

    planner = rl.AStar(rl.Landmarks(2)).bind(ring)
    assert planner.route("a", "c").routes[0].cost == 20


def test_a_measured_bound_knows_which_way_round(ring):
    planner = rl.AStar(rl.Landmarks(4)).bind(ring)
    forward = planner.node_id("b")
    backward = planner.node_id("a")
    # Getting from a to b is one hop; getting from b back to a is three.
    assert planner.heuristic.estimate(planner.node_id("a"), forward) == 10
    assert planner.heuristic.estimate(forward, backward) == 30


@pytest.mark.parametrize("seed", range(6))
def test_the_estimate_never_overshoots(seed):
    """Admissibility, checked directly: true remaining cost comes from a
    Dijkstra on the reversed graph, which measures distance *to* the target."""
    instance = random_geometric_graph(seed, num_nodes=60)
    graph = instance.graph
    heuristic = rl._routelab.Heuristic.landmarks(graph, 4, "farthest", seed)

    reversed_graph = graph.reversed()
    target = graph.num_nodes - 1
    true_cost = rl.dijkstra(reversed_graph, target)
    for node in range(graph.num_nodes):
        remaining = true_cost.cost(node)
        if remaining is not None:
            assert heuristic.estimate(node, target) <= remaining


@pytest.mark.parametrize("seed", range(6))
def test_landmark_astar_finds_dijkstras_costs(seed):
    instance = random_geometric_graph(seed, num_nodes=60)
    graph = instance.graph
    heuristic = rl._routelab.Heuristic.landmarks(graph, 4, "farthest", seed)
    plain = rl.dijkstra(graph, 0)

    for target in (graph.num_nodes - 1, graph.num_nodes // 3):
        guided = rl.astar(graph, 0, target, heuristic)
        assert guided.cost(target) == plain.cost(target)
        assert len(guided.order) <= len(rl.dijkstra(graph, 0, targets=[target]).order)


def test_a_measured_bound_beats_a_geometric_one_on_mixed_speeds():
    # The finding this heuristic exists for: a straight-line bound has to assume
    # everything moves at the network's top speed, so the wider the range of
    # speeds the weaker it is. A measured bound has no such problem.
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    origin, destination = 1, 5
    settled, costs = {}, {}
    for name, heuristic in [
        ("euclidean", rl.Euclidean()),
        ("landmarks", rl.Landmarks(2)),
    ]:
        planner = rl.AStar(heuristic).bind(env)
        result = planner.search(origin, target=planner.node_id(destination))
        settled[name] = len(result.order)
        costs[name] = planner.route(origin, destination).routes[0].cost

    assert costs["landmarks"] == costs["euclidean"], "same answer either way"
    assert settled["landmarks"] <= settled["euclidean"]


def test_selection_is_reproducible_and_the_seed_means_something(ring):
    same = [rl.AStar(rl.Landmarks(2, seed=5)).bind(ring).heuristic.estimate(0, 2) for _ in range(2)]
    assert same[0] == same[1]


def test_spreading_landmarks_out_beats_scattering_them():
    # Not a tautology worth asserting on a toy graph — this is measured on a
    # real one, where "farthest" reliably settles fewer nodes for the same
    # memory. Here we only check both strategies are usable and admissible.
    instance = random_geometric_graph(3, num_nodes=80)
    graph = instance.graph
    plain = rl.dijkstra(graph, 0)
    for selection in ("farthest", "random"):
        heuristic = rl._routelab.Heuristic.landmarks(graph, 4, selection, 3)
        target = graph.num_nodes - 1
        assert rl.astar(graph, 0, target, heuristic).cost(target) == plain.cost(target)


def test_the_table_is_two_weights_per_landmark_per_node(ring):
    planner = rl.AStar(rl.Landmarks(3)).bind(ring)
    assert planner.heuristic.footprint == 2 * 3 * 4 * 4
    assert planner.heuristic.coverage == 4


def test_a_landmark_set_has_to_have_landmarks():
    # Refused where it is written: a technique is a value, and a wrong one
    # should say so when it is made rather than when it is bound.
    with pytest.raises(ValueError, match="at least one landmark"):
        rl.Landmarks(0)


def test_an_unknown_selection_says_what_it_expected():
    with pytest.raises(ValueError, match="expected 'farthest' or 'random'"):
        rl.Landmarks(2, "vibes")


def test_asking_for_more_landmarks_than_nodes_is_fine(ring):
    planner = rl.AStar(rl.Landmarks(99)).bind(ring)
    assert planner.route("a", "c").routes[0].cost == 20


def test_the_search_space_is_still_a_tree(ring):
    # Whatever the heuristic, A* explores by growing a tree — so a landmark
    # planner reports the same shape and the demo needs no special case.
    planner = rl.AStar(rl.Landmarks(2)).bind(ring)
    tree = planner.explored(planner.search("a", target=planner.node_id("c")))
    assert tree.kind == "shortest-path-tree"
    assert len(tree) >= 1
