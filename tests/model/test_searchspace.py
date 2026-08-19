"""The search space a planner reports: what it explored, ready to draw."""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import JUNCTION


@pytest.fixture
def forked() -> rl.Environment:
    """A trunk a->b forking at b into c and into d->e, every edge costing 10."""
    return rl.Environment(
        rl.ScalarEdges([("a", "b", 10), ("b", "c", 10), ("b", "d", 10), ("d", "e", 10)])
    )


def magnitudes(tree) -> dict:
    return {(branch.tail, branch.head): branch.magnitude for branch in tree.branches()}


def test_a_branch_carries_everything_beyond_it(forked):
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"))

    assert magnitudes(tree) == {
        ("a", "b"): 40,  # its own 10, plus the 30 beyond
        ("b", "c"): 10,
        ("b", "d"): 20,
        ("d", "e"): 10,
    }
    assert tree.peak == 40
    assert len(tree) == 4


def test_counting_nodes_is_the_other_measure(forked):
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"), magnitude="nodes")
    assert magnitudes(tree) == {
        ("a", "b"): 4,
        ("b", "c"): 1,
        ("b", "d"): 2,
        ("d", "e"): 1,
    }


def test_the_trunk_carries_the_whole_search(forked):
    # What makes the picture readable: everything drains through the root, so
    # the branches nearest it are the heaviest.
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"))
    trunk = next(branch for branch in tree.branches() if branch.tail == "a")
    assert trunk.magnitude == tree.peak


def test_the_root_is_not_a_branch(forked):
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"))
    assert "a" not in [branch.head for branch in tree.branches()]


def test_capillaries_can_be_filtered_out(forked):
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"))
    heavy = list(tree.branches(min_magnitude=20))
    assert {branch.head for branch in heavy} == {"b", "d"}


def test_an_unknown_measure_says_what_it_expected(forked):
    planner = rl.Dijkstra().bind(forked)
    with pytest.raises(ValueError, match="expected 'nodes' or 'weight'"):
        planner.explored(planner.search("a"), magnitude="vibes")


def test_every_planner_reports_a_space(forked):
    for planner in (rl.Dijkstra().bind(forked), rl.BFS().bind(forked), rl.AStar(rl.Zero()).bind(forked)):
        result = (
            planner.search("a", target=planner.node_id("e"))
            if isinstance(planner, rl.AStarPlanner)
            else planner.search("a")
        )
        space = planner.explored(result)
        assert space.kind == "shortest-path-tree"
        assert isinstance(space, rl.SearchSpace)


def test_guidance_shows_up_as_a_smaller_tree():
    # The point of reporting the search space: the same journey, visibly less
    # work, without having to take the node counts on faith.
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    target = rl.Dijkstra().bind(env).node_id(5)
    guided = rl.AStar(rl.Euclidean()).bind(env)
    plain = rl.Dijkstra().bind(env)

    guided_tree = guided.explored(guided.search(1, target=target))
    plain_tree = plain.explored(plain.search(1, targets=[target]))
    assert len(guided_tree) <= len(plain_tree)


def test_geojson_is_a_feature_collection_in_longitude_latitude_order():
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    planner = rl.Dijkstra().bind(env)
    tree = planner.explored(planner.search(1))
    collection = tree.geojson()

    assert collection["type"] == "FeatureCollection"
    assert collection["features"], "the fixture has geometry to draw"
    for feature in collection["features"]:
        assert feature["geometry"]["type"] == "LineString"
        for lon, lat in feature["geometry"]["coordinates"]:
            # The fixture sits near (0, 0.00x): longitude first, or the whole
            # map lands in the ocean.
            assert 0.0 <= lon <= 0.01 and 0.0 <= lat <= 0.01
        assert 0 < feature["properties"]["share"] <= 1


def test_geojson_keeps_only_the_heaviest_when_limited():
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    planner = rl.Dijkstra().bind(env)
    tree = planner.explored(planner.search(1))

    limited = tree.geojson(limit=2)
    assert len(limited["features"]) == 2
    magnitudes = [f["properties"]["magnitude"] for f in limited["features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[0] == tree.peak


def test_a_layer_without_geometry_draws_nothing(forked):
    # ScalarEdges knows its topology and not its shape, so there is a tree but
    # nothing to put on a map.
    planner = rl.Dijkstra().bind(forked)
    tree = planner.explored(planner.search("a"))
    assert len(tree) == 4
    assert tree.geojson()["features"] == []
