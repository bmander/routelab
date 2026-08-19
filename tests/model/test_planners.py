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
    journey = rl.Dijkstra().bind(env).route("a", "c").routes[0]
    assert journey.cost == 16
    assert journey.nodes == ["a", "b", "c"]
    assert repr(journey) == "Journey('a' → 'b' → 'c', cost=16)"


def test_legs_carry_weights_and_provenance(env):
    journey = rl.Dijkstra().bind(env).route("a", "c").routes[0]
    assert [(leg.tail, leg.head, leg.weight) for leg in journey.legs] == [
        ("a", "b", 1),
        ("b", "c", 15),
    ]
    assert all(leg.source is env.sources[0] for leg in journey.legs)
    assert sum(leg.weight for leg in journey.legs) == journey.cost


def test_routing_to_yourself_costs_nothing(env):
    journey = rl.Dijkstra().bind(env).route("a", "a").routes[0]
    assert journey.cost == 0
    assert journey.legs == ()
    assert journey.nodes == ["a"]


def test_unreachable_destination_returns_none(env):
    assert rl.Dijkstra().bind(env).route("c", "d").routes == []


def test_options_reach_the_search(env):
    planner = rl.Dijkstra().bind(env)
    assert planner.route("a", "c", max_cost=16).routes[0].cost == 16
    assert planner.route("a", "c", max_cost=15).routes == []


def test_several_origins_with_access_costs(env):
    # The shape a multimodal query starts in: each entry point already costs a walk.
    planner = rl.Dijkstra().bind(env)
    assert planner.route({"a": 0, "b": 10}, "c").routes[0].cost == 16
    assert planner.route({"a": 0, "b": 0}, "c").routes[0].cost == 15
    # The journey reports the origin that actually won, not the first one named.
    assert planner.route({"a": 100, "b": 0}, "c").routes[0].origin == "b"


def test_bare_labels_are_a_single_origin(env):
    planner = rl.Dijkstra().bind(env)
    assert planner.route("a", "c").routes[0].cost == planner.route(["a"], "c").routes[0].cost


def test_a_tuple_is_one_label_and_a_list_is_several():
    # Labels may be tuples, so iterability cannot decide how many origins there
    # are. A list means several; a tuple is a single label.
    env = rl.Environment(rl.ScalarEdges((("stop", 1), ("stop", 2), 60)))
    planner = rl.Dijkstra().bind(env)
    assert planner.route(("stop", 1), ("stop", 2)).routes[0].cost == 60
    assert planner.route([("stop", 1)], ("stop", 2)).routes[0].cost == 60


def test_bfs_counts_hops_not_costs(env):
    assert rl.BFS().bind(env).route("a", "c").routes[0].cost == 1  # the direct 30-cost edge
    assert rl.Dijkstra().bind(env).route("a", "c").routes[0].cost == 16


def test_bfs_rejects_priced_origins(env):
    with pytest.raises(ValueError, match="cannot carry an initial cost"):
        rl.BFS().bind(env).route({"a": 60}, "c").routes[0]


def test_a_planner_refuses_cost_models_it_cannot_route_over():
    class Timetable(rl.EdgeSource):
        cost_model = "timetable"

        def edges(self):
            return [("a", "b", 1)]

    env = rl.Environment(Timetable())
    with pytest.raises(TypeError, match="cannot route over timetable layers"):
        rl.Dijkstra().bind(env)


def test_route_helper_binds_a_technique_and_asks_it_once(env):
    assert rl.route(rl.Dijkstra(), env, "a", "c").routes[0].cost == 16
    assert rl.route(rl.BFS(), env, "a", "c").routes[0].cost == 1


def test_a_technique_is_a_value_you_can_keep(env):
    # The whole point of configuring apart from binding: a technique can be
    # written down, named, put in a list, and pointed at more than one dataset.
    technique = rl.AStar(rl.Landmarks(2))
    assert repr(technique) == "AStar(Landmarks(2, selection='farthest'))"

    other = rl.Environment(rl.ScalarEdges(("a", "b", 5), ("b", "c", 5)))
    assert technique.bind(env).route("a", "c").routes[0].cost == 16
    assert technique.bind(other).route("a", "c").routes[0].cost == 10
    assert not hasattr(technique, "environment"), "a technique holds no data at all"


def test_binding_gives_each_environment_its_own_planner(env):
    technique = rl.Dijkstra()
    first, second = technique.bind(env), technique.bind(env)
    assert first is not second
    assert first.compiled is second.compiled, "the same world, compiled once"


def test_an_unbound_technique_has_no_verbs_to_call(env):
    # The contract, as types rather than as a sentence at the bottom of a
    # stack trace: a technique is configuration, and asking one a question is
    # not a call that exists.
    technique = rl.Dijkstra()
    for verb in ("route", "search", "explored", "node_id"):
        assert not hasattr(technique, verb), verb
    assert hasattr(technique.bind(env), "route")


def test_a_technique_reports_what_a_dataset_cannot_give_it(env):
    # Asked before binding, so a study can skip what will not work without
    # paying the preprocessing to find out. What a heuristic needs is pinned in
    # test_astar.py; what matters here is that A* passes the question along.
    bare = env.compile()
    assert rl.Dijkstra().missing_from(bare) == frozenset()
    for heuristic in (rl.Euclidean(), rl.Landmarks(4), rl.Zero()):
        assert rl.AStar(heuristic).missing_from(bare) == type(heuristic).missing_from(bare)
    assert rl.AStar(rl.Euclidean()).missing_from(bare), "and Euclidean needs something"


def test_the_feasibility_gate_covers_everything_binding_checks(env):
    """`missing_from` is only useful if it is complete: a study that skips on
    it must never hit an error it could have been warned about.

    Cost models are the half that is invisible today, because every planner
    accepts every cost model that compiles. A planner that accepts nothing
    stands in for the schedule-based ones that will not accept everything.
    """

    class Fussy(rl.Dijkstra):
        accepts = frozenset()

    compiled = env.compile()
    assert Fussy().missing_from(compiled) == {"scalar"}
    with pytest.raises(TypeError, match="cannot route over scalar layers"):
        Fussy().bind(env)

    # And A* reports its own needs on top of the ones every planner has.
    class FussyAStar(rl.AStar):
        accepts = frozenset()

    assert FussyAStar(rl.Euclidean()).missing_from(compiled) == {
        "scalar",
        "positions",
        "cost_per_distance",
    }


def test_a_study_is_a_list_of_techniques(env):
    """The shape this exists for: write down the comparison, filter it to what
    the data supports, then spend anything."""
    study = {
        "dijkstra": rl.Dijkstra(),
        "astar": rl.AStar(rl.Euclidean()),
        "alt-2": rl.AStar(rl.Landmarks(2)),
    }
    compiled = env.compile()
    feasible = {
        name: technique
        for name, technique in study.items()
        if not technique.missing_from(compiled)
    }
    assert set(feasible) == {"dijkstra", "alt-2"}, "no coordinates here"

    costs = {name: t.bind(env).route("a", "c").routes[0].cost for name, t in feasible.items()}
    assert set(costs.values()) == {16}, "same answer, whatever the technique"


def test_a_planner_holds_the_world_it_was_built_with(env):
    planner = rl.Dijkstra().bind(env)
    env.register(rl.ScalarEdges(("c", "e", 1)))
    assert planner.route("a", "c").routes[0].cost == 16
    with pytest.raises(KeyError):
        planner.route("a", "e").routes[0]
    assert rl.Dijkstra().bind(env).route("a", "e").routes[0].cost == 17


def test_search_is_the_escape_hatch_to_the_kernel(env):
    planner = rl.Dijkstra().bind(env)
    result = planner.search("a")
    assert isinstance(result, rl.SearchResult)
    assert result.cost(planner.node_id("c")) == 16
    assert planner.label(result.path(planner.node_id("c"))[0]) == "a"


def test_planners_agree_with_the_kernel_they_wrap(env):
    planner = rl.Dijkstra().bind(env)
    graph = env.compile().graph
    direct = rl.dijkstra(graph, planner.node_id("a"))
    assert direct.cost(planner.node_id("c")) == planner.route("a", "c").routes[0].cost


# --- one protocol, every technique ---------------------------------------------


EVERY_TECHNIQUE = [
    rl.Dijkstra(),
    rl.BFS(),
    rl.AStar(rl.Zero()),
    rl.ContractionHierarchy(),
    rl.TimeDependentDijkstra(),
    rl.TimeDependent(),
    rl.TimeExpanded(),
    rl.RAPTOR(),
    rl.CSA(),
    rl.TripBased(),
    rl.PTL(),
    rl.ULTRA(rl.RAPTOR()),
    rl.LabelConstrained(),
    rl.UCCH(),
]


def test_every_technique_is_one_of_these():
    # The list above is written by hand, because these are configurations and
    # some take arguments — so it is checked against what the package exports
    # rather than trusted. A technique that landed without a line here would
    # otherwise be tested by nothing below.
    shelf = {
        value
        for value in vars(rl.kernels).values()
        if isinstance(value, type)
        and issubclass(value, rl.kernels.Technique)
        and value not in (rl.kernels.Technique, rl.kernels.TimetableTechnique)
    }
    assert {type(technique) for technique in EVERY_TECHNIQUE} == shelf


@pytest.mark.parametrize("technique", EVERY_TECHNIQUE, ids=repr)
def test_every_technique_is_a_configuration_and_nothing_more(technique):
    # What a technique knows is which cost models it can route over, which is
    # what `missing_from` answers with. What a query takes is the planner's,
    # on its own signature, so there is no second place for it to be declared.
    assert technique.accepts
    assert not hasattr(technique, "route")


def test_an_option_a_technique_does_not_take_is_refused_by_name(env):
    # No registry, and nothing to keep in step with the shelf: a knob a
    # technique does not have is a keyword its `route` never declared, and
    # Python's own sentence names the method the caller called.
    planner = rl.Dijkstra().bind(env)
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_depth'"):
        planner.route("a", "c", max_depth=1).routes[0]
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_transfers'"):
        planner.route("a", "c", max_transfers=1).routes[0]
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonsense'"):
        planner.route("a", "c", nonsense=True).routes[0]
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_cost'"):
        rl.BFS().bind(env).route("a", "c", max_depth=1, max_cost=5).routes[0]


def test_a_departure_time_on_an_unscheduled_network_is_not_an_argument(env):
    # `Dijkstra` routes the always-open network, which is a question worth
    # asking; a departure time is not part of it, and never was.
    with pytest.raises(TypeError, match="unexpected keyword argument 'departing'"):
        rl.Dijkstra().bind(env).route("a", "c", departing=0).routes[0]


def test_the_package_exports_the_whole_onramp():
    for name in ("RAPTOR", "CSA", "TripBased", "Technique", "TimetablePlanner",
                 "RAPTORPlanner", "DijkstraPlanner", "Rounds", "Reach", "Scan", "Arrival",
                 "Segments", "Segment", "Leap", "EdgeResult",
                 "service_seconds", "kernels"):
        assert hasattr(rl, name), name


@pytest.mark.parametrize(
    "technique, aim",
    [
        (rl.Dijkstra(), "targets"),
        (rl.BFS(), "targets"),
        (rl.AStar(rl.Zero()), "target"),
        (rl.ContractionHierarchy(), "target"),
    ],
    ids=lambda value: value if isinstance(value, str) else repr(value),
)
def test_route_is_the_journey_a_search_holds(env, technique, aim):
    # The identity every technique with a cost table keeps: asking for the
    # journey is asking for the search and reading it back. A goal-directed
    # search takes the one target it aims at; the rest stop at a set of them.
    planner = technique.bind(env)
    node = planner.node_id("c")
    result = planner.search("a", **{aim: node if aim == "target" else [node]})
    assert planner.journey(result, "c") == planner.route("a", "c").routes[0]
    assert planner.journey(result, "d") is None
