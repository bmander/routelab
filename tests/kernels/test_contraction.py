"""Contraction hierarchies through the planner API.

The claim is exactness, so most of these compare against Dijkstra. The rest are
about the thing CH forced on the library: a technique that searches a graph the
environment has never seen still has to answer in the environment's own terms.
"""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import JUNCTION, grid_environment, random_geometric_graph


@pytest.fixture
def town() -> rl.Environment:
    """A small grid with coordinates — somewhere a hierarchy has work to do."""
    return grid_environment(9)


def test_hello_world(town):
    planner = rl.ContractionHierarchy().bind(town)
    journey = planner.route((0, 0), (8, 8)).routes[0]
    assert journey.cost == rl.Dijkstra().bind(town).route((0, 0), (8, 8)).routes[0].cost


@pytest.mark.parametrize("seed", range(5))
def test_it_answers_exactly_what_dijkstra_answers(seed):
    """The whole claim. A hierarchy that is usually right is not a hierarchy."""
    instance = random_geometric_graph(seed, num_nodes=70)
    env = rl.Environment(
        rl.ScalarEdges([(t, h, w) for t, h, w in instance.graph.edges()])
    )
    hierarchy = rl.ContractionHierarchy().bind(env)
    plain = rl.Dijkstra().bind(env)

    labels = list(env.compile().labels)
    for source in labels[:12]:
        truth = plain.search(source)
        for target in labels[:12]:
            found = hierarchy.route(source, target).routes
            expected = truth.cost(plain.node_id(target))
            # An unreachable pair is a Pareto set with nothing in it.
            assert (found[0].cost if found else None) == expected, f"{source} -> {target}"


def test_the_path_it_reports_is_a_real_one(town):
    # A hierarchy searches over shortcuts, so its answer has to be unpacked back
    # into edges the environment recognises. Walking them is the proof.
    compiled = town.compile()
    journey = rl.ContractionHierarchy().bind(town).route((0, 0), (8, 8)).routes[0]

    edges = [leg.edge for leg in journey.legs]
    start = compiled.node_id(journey.origin)
    assert compiled.graph.walk(start, edges) == (
        compiled.node_id((8, 8)),
        journey.cost,
    )


def test_legs_come_back_with_geometry_and_provenance():
    # The point of unpacking: everything downstream works unchanged, including
    # the parts that only know about the environment's own layers.
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    compiled = env.compile()
    journey = rl.ContractionHierarchy().bind(env).route(1, 5).routes[0]

    assert journey.cost == rl.Dijkstra().bind(env).route(1, 5).routes[0].cost
    assert all(leg.source is env.sources[0] for leg in journey.legs)
    assert all(compiled.geometry(leg.edge) for leg in journey.legs)


def test_the_advantage_grows_with_the_graph():
    """The claim is sublinear growth, not a constant factor.

    A uniform grid is close to the worst case for a hierarchy — there are no
    arterials, so nothing is genuinely more important than anything else, and
    corner to corner the saving is a factor of two on 81 nodes. What holds even
    here is the shape: as the grid grows, Dijkstra's work grows with it and the
    hierarchy's grows much more slowly. On a real road network, where importance
    is real, that gap is four orders of magnitude.
    """

    def ratio(side: int) -> float:
        env = grid_environment(side)
        plain = rl.Dijkstra().bind(env)
        target = [plain.node_id((side - 1, side - 1))]
        settled = rl.ContractionHierarchy().bind(env).search((0, 0), targets=target)
        return plain.search((0, 0), targets=target).settled / settled.settled

    assert ratio(9) < ratio(20) < ratio(40)


def test_ordering_changes_the_hierarchy_and_not_the_answers(town):
    careful = rl.ContractionHierarchy(rl.EdgeDifference()).bind(town)
    arbitrary = rl.ContractionHierarchy(rl.RandomOrder(seed=3)).bind(town)

    assert careful.route((0, 0), (8, 8)).routes[0].cost == arbitrary.route((0, 0), (8, 8)).routes[0].cost
    assert careful.hierarchy.num_shortcuts < arbitrary.hierarchy.num_shortcuts


def test_a_hierarchy_needs_nothing_from_the_environment():
    # Like landmarks and unlike Euclidean: it measures the graph itself. The
    # ordering is asked too, so an ordering that wanted coordinates could say so.
    bare = rl.Environment(rl.ScalarEdges([("a", "b", 1)])).compile()
    assert rl.ContractionHierarchy().missing_from(bare) == frozenset()
    assert rl.EdgeDifference().missing_from(bare) == frozenset()


def test_what_it_costs_is_reportable_before_and_after_binding(town):
    technique = rl.ContractionHierarchy()
    with pytest.raises(ValueError, match="is a technique, not a planner"):
        technique.footprint
    assert technique.bind(town).footprint > 0


def test_its_result_is_one_a_journey_can_be_built_from(town):
    # Not the same class as Dijkstra's — two searches met in the middle — but
    # the same contract, which is what lets everything downstream not care.
    planner = rl.ContractionHierarchy().bind(town)
    result = planner.search((0, 0), targets=[planner.node_id((8, 8))])
    assert isinstance(result, rl.Result)
    assert isinstance(rl.Dijkstra().bind(town).search((0, 0)), rl.Result)


def test_the_same_query_twice_gives_the_same_path(town):
    """Deterministic, even where it may differ from Dijkstra's choice.

    A hierarchy climbs from both ends, so it cannot honour the `(cost, node)`
    tie-break a one-directional search settles by, and among equally cheap
    routes it may pick a different one. What it cannot do is pick differently
    from run to run — that would make any diff against it meaningless.
    """
    first = rl.ContractionHierarchy().bind(town).route((0, 0), (8, 8)).routes[0]
    second = rl.ContractionHierarchy().bind(town).route((0, 0), (8, 8)).routes[0]
    assert first.nodes == second.nodes

    # And however it broke the tie, the cost is Dijkstra's exactly.
    assert first.cost == rl.Dijkstra().bind(town).route((0, 0), (8, 8)).routes[0].cost


def test_it_refuses_options_that_mean_nothing_to_it(town):
    # `magnitude` describes how to weight a tree's branches. There is no tree.
    planner = rl.ContractionHierarchy().bind(town)
    result = planner.search((0, 0), targets=[planner.node_id((8, 8))])
    with pytest.raises(ValueError, match="meeting trees has no magnitude"):
        planner.explored(result, magnitude="nodes")


def test_it_searches_toward_exactly_one_target(town):
    planner = rl.ContractionHierarchy().bind(town)
    with pytest.raises(ValueError, match="single target, and got no target"):
        planner.search((0, 0))
    with pytest.raises(ValueError, match="got 2 targets"):
        planner.search((0, 0), targets=[1, 2])


def test_it_refuses_bounds_it_cannot_honour(town):
    # max_cost over a contracted graph would cut paths that are still cheap in
    # the original, so it is refused rather than quietly answered wrong.
    planner = rl.ContractionHierarchy().bind(town)
    with pytest.raises(ValueError, match="takes no max_cost; a cost bound belongs to"):
        planner.route((0, 0), (8, 8), max_cost=100).routes[0]


def test_unreachable_destinations_return_nothing():
    env = rl.Environment(rl.ScalarEdges([("a", "b", 1), ("c", "d", 1)]))
    assert rl.ContractionHierarchy().bind(env).route("a", "d").routes == []


def test_several_origins_with_access_costs(town):
    hierarchy = rl.ContractionHierarchy().bind(town)
    plain = rl.Dijkstra().bind(town)
    origins = {(0, 0): 0, (8, 0): 500}
    assert hierarchy.route(origins, (8, 8)).routes[0].cost == plain.route(origins, (8, 8)).routes[0].cost


def test_the_search_space_is_two_trees_that_met(town):
    planner = rl.ContractionHierarchy().bind(town)
    result = planner.search((0, 0), targets=[planner.node_id((8, 8))])
    space = planner.explored(result)

    assert space.kind == "meeting-trees"
    assert isinstance(space, rl.SearchSpace)
    assert space.meeting is not None
    directions = {leap.direction for leap in space.branches()}
    assert directions == {"forward", "backward"}


def test_the_search_space_draws_unpacked_road():
    # On road data, where the branches have shapes to draw. A grid built from
    # `ScalarEdges` has positions but no geometry, so it renders to nothing.
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    planner = rl.ContractionHierarchy().bind(env)
    result = planner.search(1, targets=[planner.node_id(5)])
    collection = planner.explored(result).geojson()

    assert collection["features"], "OSM ways carry geometry"
    for feature in collection["features"]:
        properties = feature["properties"]
        assert properties["direction"] in {"forward", "backward"}
        assert properties["span"] >= 1
        # A branch standing for n edges draws at least n+1 points, because it is
        # real road rather than a straight line across whatever it leapt.
        assert len(feature["geometry"]["coordinates"]) >= properties["span"] + 1


def test_shortcuts_show_up_as_branches_that_span_many_edges(town):
    planner = rl.ContractionHierarchy().bind(town)
    result = planner.search((0, 0), targets=[planner.node_id((8, 8))])
    space = planner.explored(result)
    assert space.peak > 1, "a hierarchy that never leaps is not a hierarchy"
    assert list(space.branches(min_span=space.peak)), "the longest branch is reportable"
