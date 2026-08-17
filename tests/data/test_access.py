"""Access: the step from a stop onto the pavement outside it.

Without it a feed and a street network registered together are two graphs
sharing one numbering and no edges, and no journey can begin on foot and end
on a bus. What is checked here is that the join is made where it should be,
refused where it should not, and counted where it cannot be.
"""

from __future__ import annotations

import pytest

import routelab as rl


class Stops(rl.ScalarEdges):
    """A stand-in feed: two stops with coordinates and nothing else."""

    cost_model = "timetable"

    def __init__(self, places):
        super().__init__([])
        self._places = places

    def coordinates(self):
        return self._places


class Streets(rl.ScalarEdges):
    """A stand-in street network: corners, and the nearest one to a point."""

    def __init__(self, places, edges=()):
        super().__init__(list(edges))
        self._places = places

    def coordinates(self):
        return self._places

    def nearest(self, lat, lon, within=None):
        best = min(
            self._places.items(),
            key=lambda item: (item[1][0] - lat) ** 2 + (item[1][1] - lon) ** 2,
        )
        return best[0]


def test_a_stop_joins_the_corner_nearest_it():
    stops = Stops({"stop": (47.60, -122.33)})
    streets = Streets({"near": (47.6001, -122.33), "far": (47.70, -122.33)})
    access = rl.Access(stops, streets, within=400)
    edges = list(access.edges())
    # Both ways, because a rider walks to the bus and away from it.
    assert {(tail, head) for tail, head, _ in edges} == {("stop", "near"), ("near", "stop")}
    assert all(weight > 0 for _, _, weight in edges)
    assert access.stranded == 0


def test_a_stop_the_map_does_not_reach_is_counted_not_invented():
    # Eleven kilometres from the only corner there is: no rider walks that, so
    # the stop is left unattached and said so, rather than joined to somewhere
    # implausible and quietly making journeys nobody could take.
    stops = Stops({"stop": (47.60, -122.33)})
    streets = Streets({"miles_off": (47.70, -122.33)})
    access = rl.Access(stops, streets, within=400)
    assert list(access.edges()) == []
    assert access.stranded == 1


def test_it_says_what_it_needs_of_the_layers_it_takes():
    stops = Stops({"stop": (47.60, -122.33)})
    streets = Streets({"corner": (47.60, -122.33)})
    with pytest.raises(TypeError, match="has no coordinates"):
        rl.Access(rl.ScalarEdges([]), streets)
    with pytest.raises(TypeError, match="has no nearest"):
        rl.Access(stops, rl.ScalarEdges([]))
    with pytest.raises(ValueError, match="within must be a positive distance"):
        rl.Access(stops, streets, within=0)
    with pytest.raises(ValueError, match="speed must be positive"):
        rl.Access(stops, streets, speed=-1)


def test_the_join_is_an_ordinary_scalar_layer():
    # Which is the whole design: nothing about the environment had to change
    # to carry it, and a walk along it is a leg like any other.
    stops = Stops({"stop": (47.60, -122.33)})
    streets = Streets({"corner": (47.6001, -122.33)}, edges=[])
    access = rl.Access(stops, streets)
    assert access.cost_model == "scalar"
    assert access.cost_per_distance == pytest.approx(1 / 1.4)
    assert access.positions() == {}
    shape = access.geometry(0)
    assert shape == [(47.60, -122.33), (47.6001, -122.33)]
    assert "joined" in repr(access)


def test_a_feed_and_a_street_network_become_one_graph(feed):
    # The point of it, on the tiny fixture: three stops and a street each of
    # them stands beside, in one environment with edges between the two.
    places = {stop: point for stop, point in feed.coordinates().items()}
    streets = Streets(
        {f"corner-{stop}": (lat + 0.00001, lon) for stop, (lat, lon) in places.items()},
        edges=[],
    )
    access = rl.Access(feed, streets, within=400)
    env = rl.Environment(streets, feed, access)
    compiled = env.compile()
    assert compiled.cost_models == frozenset({"scalar", "timetable"})
    # Every stop reaches its corner and back.
    for stop in places:
        assert compiled.node_id(f"corner-{stop}") >= 0
        assert compiled.node_id(stop) >= 0
    assert len(access) == 2 * len(places)
