"""Planners: an algorithm bound to an environment, answering questions."""

from __future__ import annotations

import pytest

import routelab as rl


@pytest.fixture
def env() -> rl.Environment:
    """'a' reaches 'c' either directly (30) or via 'b' (16), and 'd' not at all."""
    return rl.Environment(
        rl.ScalarEdges(("a", "b", 1), ("b", "c", 15), ("a", "c", 30), ("d", "a", 1))
    )


def test_hello_world(env):
    journey = rl.Dijkstra(env).route("a", "c")
    assert journey.cost == 16
    assert journey.nodes == ["a", "b", "c"]
    assert repr(journey) == "Journey('a' → 'b' → 'c', cost=16)"


def test_legs_carry_weights_and_provenance(env):
    journey = rl.Dijkstra(env).route("a", "c")
    assert [(leg.tail, leg.head, leg.weight) for leg in journey.legs] == [
        ("a", "b", 1),
        ("b", "c", 15),
    ]
    assert all(leg.source is env.sources[0] for leg in journey.legs)
    assert sum(leg.weight for leg in journey.legs) == journey.cost


def test_routing_to_yourself_costs_nothing(env):
    journey = rl.Dijkstra(env).route("a", "a")
    assert journey.cost == 0
    assert journey.legs == ()
    assert journey.nodes == ["a"]


def test_unreachable_destination_returns_none(env):
    assert rl.Dijkstra(env).route("c", "d") is None


def test_options_reach_the_search(env):
    planner = rl.Dijkstra(env)
    assert planner.route("a", "c", max_cost=16).cost == 16
    assert planner.route("a", "c", max_cost=15) is None


def test_several_origins_with_access_costs(env):
    # The shape a multimodal query starts in: each entry point already costs a walk.
    planner = rl.Dijkstra(env)
    assert planner.route({"a": 0, "b": 10}, "c").cost == 16
    assert planner.route({"a": 0, "b": 0}, "c").cost == 15
    # The journey reports the origin that actually won, not the first one named.
    assert planner.route({"a": 100, "b": 0}, "c").origin == "b"


def test_bare_labels_are_a_single_origin(env):
    planner = rl.Dijkstra(env)
    assert planner.route("a", "c").cost == planner.route(["a"], "c").cost


def test_a_tuple_is_one_label_and_a_list_is_several():
    # Labels may be tuples, so iterability cannot decide how many origins there
    # are. A list means several; a tuple is a single label.
    env = rl.Environment(rl.ScalarEdges((("stop", 1), ("stop", 2), 60)))
    planner = rl.Dijkstra(env)
    assert planner.route(("stop", 1), ("stop", 2)).cost == 60
    assert planner.route([("stop", 1)], ("stop", 2)).cost == 60


def test_bfs_counts_hops_not_costs(env):
    assert rl.BFS(env).route("a", "c").cost == 1  # the direct 30-cost edge
    assert rl.Dijkstra(env).route("a", "c").cost == 16


def test_bfs_rejects_priced_origins(env):
    with pytest.raises(ValueError, match="cannot carry an initial cost"):
        rl.BFS(env).route({"a": 60}, "c")


def test_a_planner_refuses_cost_models_it_cannot_route_over():
    class Timetable(rl.EdgeSource):
        cost_model = "timetable"

        def edges(self):
            return [("a", "b", 1)]

    env = rl.Environment(Timetable())
    with pytest.raises(TypeError, match="cannot route over timetable layers"):
        rl.Dijkstra(env)


def test_route_helper_takes_a_class_or_a_name(env):
    assert rl.route(rl.Dijkstra, env, "a", "c").cost == 16
    assert rl.route("dijkstra", env, "a", "c").cost == 16
    assert rl.route("bfs", env, "a", "c").cost == 1
    with pytest.raises(KeyError, match="unknown planner"):
        rl.route("raptor", env, "a", "c")


def test_a_planner_holds_the_world_it_was_built_with(env):
    planner = rl.Dijkstra(env)
    env.register(rl.ScalarEdges(("c", "e", 1)))
    assert planner.route("a", "c").cost == 16
    with pytest.raises(KeyError):
        planner.route("a", "e")
    assert rl.Dijkstra(env).route("a", "e").cost == 17


def test_search_is_the_escape_hatch_to_the_kernel(env):
    planner = rl.Dijkstra(env)
    result = planner.search("a")
    assert isinstance(result, rl.SearchResult)
    assert result.cost(planner.node_id("c")) == 16
    assert planner.label(result.path(planner.node_id("c"))[0]) == "a"


def test_planners_agree_with_the_kernel_they_wrap(env):
    planner = rl.Dijkstra(env)
    graph = env.compile().graph
    direct = rl.dijkstra(graph, planner.node_id("a"))
    assert direct.cost(planner.node_id("c")) == planner.route("a", "c").cost
