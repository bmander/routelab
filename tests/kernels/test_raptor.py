"""RAPTOR: rounds over routes, and the answers only it can give.

Delling, Pajor & Werneck (2012). The three-way agreement with the Pyrga models
lives in test_timetable.py and test_footpaths.py, where RAPTOR is one of
``MODELS``; what is here is the rest of its surface — the Pareto front, the cap
on changes, the one-to-all search and the rounds it reports — on the tiny
fixture, where every number can be checked by eye: A→C at 08:00 is 08:30 with
no change or 08:20 with one.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def planner(env) -> rl.RAPTOR:
    return rl.RAPTOR().bind(env)


def test_hello_world(planner):
    journey = planner.route("A", "C", departing=time(8, 0))
    assert journey.arrives == 8 * 3600 + 20 * 60
    assert journey.transfers == 1
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    assert repr(rl.RAPTOR()) == "RAPTOR()"


def test_the_frontier_is_one_journey_per_number_of_changes(planner):
    front = planner.frontier("A", "C", departing=time(8, 0))
    assert [(j.transfers, j.arrives) for j in front] == [
        (0, 8 * 3600 + 30 * 60),
        (1, 8 * 3600 + 20 * 60),
    ]
    # Fewest changes first, arrivals strictly decreasing: nothing dominated.
    assert front[-1].arrives == planner.route("A", "C", departing=time(8, 0)).arrives
    assert planner.frontier("A", "C", departing=time(12, 0)) == []


def test_max_transfers_caps_the_changes(planner):
    assert planner.route("A", "C", departing=time(8, 0), max_transfers=0).arrives == 8 * 3600 + 30 * 60
    assert planner.route("A", "C", departing=time(8, 0), max_transfers=1).arrives == 8 * 3600 + 20 * 60
    with pytest.raises(ValueError, match="cannot be -1"):
        planner.route("A", "C", departing=time(8, 0), max_transfers=-1)


def test_a_search_is_one_to_all(planner):
    result = planner.search("A", departing=time(8, 0))
    assert result.cost(planner.node_id("B")) == 8 * 3600 + 10 * 60
    assert result.cost(planner.node_id("C")) == 8 * 3600 + 20 * 60
    assert result.settled == 3
    assert result.rounds == 2
    assert result.scanned >= 2
    assert planner.searches == ("stops", 3)
    # A kept search answers for any destination without searching again — and
    # answers exactly as route() would.
    assert planner.journey(result, "C") == planner.route("A", "C", departing=time(8, 0))
    assert planner.journey(result, "B").arrives == 8 * 3600 + 10 * 60


def test_the_rounds_are_a_search_space(planner):
    result = planner.search("A", departing=time(8, 0))
    space = planner.explored(result)
    assert isinstance(space, rl.Rounds)
    assert space.kind == "rounds"
    assert {(reach.stop, reach.round) for reach in space.branches()} == {("A", 0), ("B", 1), ("C", 1)}
    assert space.peak == 1
    assert len(space) == 3
    drawn = space.geojson()
    features = drawn["features"]
    assert len(features) == 3
    assert {f["geometry"]["type"] for f in features} == {"Point"}
    assert {f["properties"]["round"] for f in features} == {0, 1}
    assert drawn["peak"] == 1, "the collection carries what a round is a share of"
    assert repr(space) == "Rounds(3 stops over 1 rounds)"
    with pytest.raises(ValueError, match="rounds has no magnitude"):
        planner.explored(result, magnitude="weight")


def test_several_origins_each_carry_a_head_start(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.RAPTOR().bind(env)
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0))
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)
    result = planner.search({"A": 0, "C": 300}, departing=time(8, 0))
    assert planner.journey(result, "B").cost == 360, "cost elapsed from the query's departure"


def test_footprint_counts_the_index_on_top_of_the_timetable(env, planner):
    assert planner.footprint > rl.TimeDependent().bind(env).footprint
    assert planner.num_routes == 3
    assert planner.num_trips == 3


def test_it_needs_a_timetable_and_a_departure(planner):
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.RAPTOR().missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a departure time"):
        planner.route("A", "C")
