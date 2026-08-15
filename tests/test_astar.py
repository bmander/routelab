"""A* through the planner API: what it needs, what it refuses, what it saves."""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import grid_environment


@pytest.fixture
def corridor() -> rl.Environment:
    """A line 'a'-'b'-'c'-'d' plus a spur off 'a' pointing the other way."""
    return rl.Environment(
        rl.ScalarEdges(
            [("a", "b", 10), ("b", "c", 10), ("c", "d", 10), ("a", "spur", 10)],
            cost_per_distance=10.0,
        ),
        rl.Positions(
            {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (3, 0), "spur": (-5, 0)}
        ),
    )


def test_hello_world(corridor):
    journey = rl.AStar(corridor, rl.Euclidean()).route("a", "d")
    assert journey.cost == 30
    assert journey.nodes == ["a", "b", "c", "d"]


def test_guidance_skips_what_leads_away(corridor):
    guided = rl.AStar(corridor, rl.Euclidean())
    plain = rl.Dijkstra(corridor)
    target = [guided.node_id("d")]

    guided_order = guided.search("a", targets=target).order
    plain_order = plain.search("a", targets=target).order
    assert guided.label(guided_order[-1]) == "d"
    assert "spur" not in [guided.label(node) for node in guided_order]
    assert "spur" in [plain.label(node) for node in plain_order]
    assert len(guided_order) < len(plain_order)


def test_an_environment_says_what_it_provides(corridor):
    assert corridor.compile().provides == frozenset(
        {"positions", "cost_per_distance"}
    )
    bare = rl.Environment(rl.ScalarEdges(("a", "b", 1))).compile()
    assert bare.provides == frozenset()


def test_a_heuristic_says_what_it_needs_before_you_build_it(corridor):
    # The question worth asking with a shelf full of techniques: which of these
    # can this dataset support? Answered without preprocessing anything.
    compiled = corridor.compile()
    assert rl.Euclidean.missing_from(compiled) == frozenset()
    assert rl.Landmarks.missing_from(compiled) == frozenset()

    bare = rl.Environment(rl.ScalarEdges(("a", "b", 1))).compile()
    assert rl.Euclidean.missing_from(bare) == {"positions", "cost_per_distance"}
    assert rl.Landmarks.missing_from(bare) == frozenset(), "measures the graph itself"
    assert rl.Zero.missing_from(bare) == frozenset()


@pytest.mark.parametrize("heuristic", [rl.Zero, rl.Euclidean, rl.Landmarks])
def test_declared_requirements_agree_with_what_binding_actually_does(corridor, heuristic):
    """The invariant that keeps the declaration honest: if nothing is missing,
    binding works; if something is, it does not."""
    for environment in (corridor, rl.Environment(rl.ScalarEdges(("a", "b", 1)))):
        compiled = environment.compile()
        if heuristic.missing_from(compiled):
            with pytest.raises(ValueError):
                heuristic().bind(compiled)
        else:
            assert heuristic().bind(compiled) is not None


def test_the_heuristic_is_required(corridor):
    with pytest.raises(TypeError):
        rl.AStar(corridor)  # type: ignore[call-arg]


def test_zero_is_available_but_must_be_asked_for(corridor):
    # A* that quietly became Dijkstra is the one failure a benchmark can't see,
    # so the degenerate heuristic is spelled out rather than defaulted to.
    assert rl.AStar(corridor, rl.Zero()).route("a", "d").cost == 30


def test_euclidean_needs_positions():
    env = rl.Environment(rl.ScalarEdges(("a", "b", 1), cost_per_distance=1.0))
    with pytest.raises(ValueError, match="position for every node"):
        rl.AStar(env, rl.Euclidean())


def test_euclidean_names_the_nodes_it_cannot_place():
    env = rl.Environment(
        rl.ScalarEdges([("a", "b", 1), ("b", "c", 1)], cost_per_distance=1.0),
        rl.Positions({"a": (0, 0)}),
    )
    with pytest.raises(ValueError, match=r"'b'.*'c'|'c'.*'b'"):
        rl.AStar(env, rl.Euclidean())


def test_euclidean_needs_a_rate_from_every_edge_layer():
    # One layer that won't say how fast it is could be arbitrarily fast, so no
    # distance-based bound is safe — it disables the heuristic rather than being
    # assumed slow.
    env = rl.Environment(
        rl.ScalarEdges(("a", "b", 1), cost_per_distance=1.0),
        rl.ScalarEdges(("b", "c", 1)),
        rl.Positions({"a": (0, 0), "b": (1, 0), "c": (2, 0)}),
    )
    with pytest.raises(ValueError, match="cost_per_distance"):
        rl.AStar(env, rl.Euclidean())


def test_the_fastest_layer_sets_the_bound():
    walking = rl.ScalarEdges(("a", "b", 300), cost_per_distance=0.71)
    transit = rl.ScalarEdges(("b", "c", 120), cost_per_distance=0.04)
    env = rl.Environment(walking, transit, rl.Positions({"a": (0, 0), "b": (1, 0), "c": (2, 0)}))
    assert env.compile().cost_per_distance == 0.04


def test_an_explicit_rate_overrides_the_layers(corridor):
    weaker = rl.AStar(corridor, rl.Euclidean(cost_per_distance=0.0))
    assert weaker.route("a", "d").cost == 30
    # Priced at zero it estimates nothing, so it degenerates to Dijkstra.
    target = [weaker.node_id("d")]
    assert weaker.search("a", targets=target).order == rl.Dijkstra(corridor).search(
        "a", targets=target
    ).order


def test_a_star_searches_toward_exactly_one_target(corridor):
    planner = rl.AStar(corridor, rl.Euclidean())
    with pytest.raises(ValueError, match="single target, and got no target"):
        planner.search("a")
    with pytest.raises(ValueError, match="got 2 targets"):
        planner.search("a", targets=[planner.node_id("c"), planner.node_id("d")])


def test_max_cost_bounds_the_real_cost_not_the_estimate(corridor):
    planner = rl.AStar(corridor, rl.Euclidean())
    assert planner.route("a", "c", max_cost=20).cost == 20
    assert planner.route("a", "d", max_cost=20) is None


def test_unreachable_destination_returns_none(corridor):
    assert rl.AStar(corridor, rl.Euclidean()).route("d", "a") is None


def test_several_origins_with_access_costs(corridor):
    planner = rl.AStar(corridor, rl.Euclidean())
    assert planner.route({"a": 0, "c": 5}, "d").cost == 15


def test_the_heuristic_is_bound_once_in_preprocess(corridor):
    planner = rl.AStar(corridor, rl.Euclidean())
    assert planner.heuristic.coverage == corridor.compile().graph.num_nodes
    assert planner.heuristic is planner.heuristic, "bound once, reused per query"
    assert repr(planner) == "AStar(Environment(2 layers), Euclidean())"


def test_journeys_are_walkable(corridor):
    journey = rl.AStar(corridor, rl.Euclidean()).route("a", "d")
    compiled = corridor.compile()
    edge_ids = [
        edge_id
        for edge_id in range(compiled.graph.num_edges)
        for tail, head, _ in [compiled.graph.edge(edge_id)]
        if (compiled.label(tail), compiled.label(head))
        in [(leg.tail, leg.head) for leg in journey.legs]
    ]
    assert compiled.graph.walk(compiled.node_id("a"), edge_ids) == (
        compiled.node_id("d"),
        journey.cost,
    )


def settled(env, planner, origin, destination):
    """(nodes A*/Dijkstra settled, cost found) for one corner-to-corner query."""
    target = [planner.node_id(destination)]
    result = planner.search(origin, targets=target)
    return len(result.order), result.cost(target[0])


def test_on_a_grid_it_settles_far_fewer_nodes():
    env = grid_environment(25)
    guided_count, guided_cost = settled(env, rl.AStar(env, rl.Euclidean()), (0, 0), (24, 24))
    plain_count, plain_cost = settled(env, rl.Dijkstra(env), (0, 0), (24, 24))

    assert guided_cost == plain_cost, "same answer"
    assert guided_count < plain_count / 10, "for a tenth of the work"


def test_a_heuristic_only_helps_when_it_is_tight():
    # The counterpoint, and the reason a lab measures instead of assuming: on a
    # 4-connected grid you must move in L1 while the estimate measures L2, so the
    # bound is loose by up to sqrt(2) everywhere and A* settles nearly everything
    # Dijkstra does. Same code, same heuristic, no win.
    env = grid_environment(15, diagonal=False)
    guided_count, guided_cost = settled(env, rl.AStar(env, rl.Euclidean()), (0, 0), (14, 14))
    plain_count, plain_cost = settled(env, rl.Dijkstra(env), (0, 0), (14, 14))

    assert guided_cost == plain_cost
    assert guided_count > plain_count * 0.9
