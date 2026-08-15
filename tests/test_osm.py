"""The OSM layer, against the same hand-written extracts the Rust crate uses.

One fixture, two languages: the crate checks parsing and splitting, this checks
what an environment makes of the result.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import routelab as rl

FIXTURES = Path(__file__).resolve().parent.parent / "crates/routelab-osm/tests/data"
JUNCTION = FIXTURES / "junction.osm"


def haversine(a, b):
    """Ground distance in metres, on the same polar-radius sphere the loader uses."""
    radius = 6_356_752.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    inner = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(inner))


def test_nodes_are_labelled_with_their_osm_ids():
    layer = rl.OSM(JUNCTION, rl.Driving())
    labels = {tail for tail, _, _ in layer.edges()}
    assert labels <= {1, 2, 3, 4}, "a route can be traced back to the map"


def test_reading_is_deferred_until_the_layer_is_used():
    layer = rl.OSM(JUNCTION, rl.Driving())
    assert "unread" in repr(layer)
    list(layer.edges())
    assert "edges" in repr(layer)


def test_a_missing_extract_is_caught_at_construction():
    with pytest.raises(FileNotFoundError, match="no OSM extract"):
        rl.OSM(FIXTURES / "nowhere.osm.pbf")


def test_cost_per_distance_comes_from_the_profile():
    layer = rl.OSM(JUNCTION, rl.Driving())
    assert layer.cost_per_distance == pytest.approx(1 / rl.Driving().max_speed)
    # And it reaches the environment, which is what makes Euclidean usable.
    env = rl.Environment(layer)
    assert env.compile().cost_per_distance == layer.cost_per_distance


def test_profiles_disagree_about_one_way_streets():
    driving = {(tail, head) for tail, head, _ in rl.OSM(JUNCTION, rl.Driving()).edges()}
    walking = {(tail, head) for tail, head, _ in rl.OSM(JUNCTION, rl.Walking()).edges()}

    assert (1, 2) in driving and (2, 1) not in driving, "Main St is one-way"
    assert (1, 2) in walking and (2, 1) in walking, "not on foot it isn't"


def test_a_profile_decides_what_counts_as_a_road():
    walking = rl.OSM(JUNCTION, rl.Walking())
    driving = rl.OSM(JUNCTION, rl.Driving())
    reached = lambda layer: {node for tail, head, _ in layer.edges() for node in (tail, head)}

    assert 5 in reached(walking), "the footway is a road on foot"
    assert 5 not in reached(driving), "and not in a car"


def test_walking_and_driving_are_different_environments():
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    planner = rl.Dijkstra(env)
    assert planner.route(3, 1) is not None, "on foot you can walk back down Main St"

    driving = rl.Dijkstra(rl.Environment(rl.OSM(JUNCTION, rl.Driving())))
    assert driving.route(3, 1) is None, "by car there is no way back"


def test_the_projection_never_overstates_the_ground():
    # The property the whole heuristic rests on: if projected distance could
    # exceed real distance, Euclidean would overestimate and A* would quietly
    # stop returning cheapest paths.
    layer = rl.OSM(JUNCTION, rl.Walking())
    positions = layer.positions()
    coordinates = layer.coordinates()
    assert positions.keys() == coordinates.keys()

    nodes = list(coordinates)
    for i, first in enumerate(nodes):
        for second in nodes[i + 1 :]:
            planar = math.dist(positions[first], positions[second])
            ground = haversine(coordinates[first], coordinates[second])
            assert planar <= ground + 1e-6, f"{first} to {second}"


def test_nearest_finds_the_node_you_pointed_at():
    layer = rl.OSM(JUNCTION, rl.Driving())
    assert layer.nearest(0.0001, 0.00195) == 3


def test_nearest_can_be_confined_to_nodes_that_are_any_use():
    # Real extracts are full of stubs that connect to nothing under a given
    # profile. Snapping to one produces a query with no answer for reasons that
    # have nothing to do with routing, so a caller can say which nodes count.
    layer = rl.OSM(JUNCTION, rl.Driving())
    assert layer.nearest(0.0001, 0.00195, within=[1, 2]) == 2
    with pytest.raises(ValueError, match="none of the given nodes"):
        layer.nearest(0.0, 0.0, within=[])


def test_a_leg_can_find_its_own_shape():
    layer = rl.OSM(JUNCTION, rl.Walking())
    env = rl.Environment(layer)
    compiled = env.compile()
    journey = rl.AStar(env, rl.Euclidean()).route(1, 3)

    shapes = [compiled.geometry(leg.edge) for leg in journey.legs]
    assert all(shape for shape in shapes), "every leg knows where it runs"
    # The shapes join up: each leg's last point is the next leg's first.
    for before, after in zip(shapes, shapes[1:]):
        assert before[-1] == after[0]


def test_a_layer_without_geometry_simply_has_none():
    env = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert env.compile().geometry(0) is None


def test_astar_and_dijkstra_agree_on_a_real_extract():
    env = rl.Environment(rl.OSM(JUNCTION, rl.Walking()))
    guided = rl.AStar(env, rl.Euclidean()).route(1, 5)
    plain = rl.Dijkstra(env).route(1, 5)
    assert guided.cost == plain.cost
    assert guided.nodes == plain.nodes
