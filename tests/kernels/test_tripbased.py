"""Trip-based routing: trips, and the transfers between them, computed once.

Witt (2015). The five-way agreement with the other timetable techniques
lives in test_timetable.py and test_footpaths.py, where TripBased is one of
``MODELS``; what is here is the rest of its surface — the transfer set and
what reduction does to it, the goal-directed search and the segments it
reports, the front, the profile, and the refusals — on the tiny fixture,
where every number can be checked by eye: A→C at 08:00 is 08:20 with one
change, and the change is the one transfer in the set.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def planner(env) -> rl.TripBased:
    return rl.TripBased().bind(env)


def test_hello_world(planner):
    journey = planner.route("A", "C", departing=time(8, 0))
    assert journey.arrives == 8 * 3600 + 20 * 60
    assert journey.transfers == 1
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    assert repr(rl.TripBased()) == "TripBased()"
    assert repr(rl.TripBased(reduce=False)) == "TripBased(reduce=False)"


def test_bind_computes_the_transfer_set(planner):
    # WEEKDAY1 reaches B at 08:10 and WEEKDAY2 leaves it at 08:15: one
    # transfer, and reduction keeps it because it is the one that reaches C
    # sooner. Nothing else in the fixture can change onto anything.
    assert planner.num_lines == 3
    assert planner.num_trips == 3
    assert planner.num_initial_transfers == 1
    assert planner.num_transfers == 1
    assert planner.searches == ("trips", 3)


def test_the_search_is_goal_directed(planner):
    result = planner.search("A", targets=[planner.node_id("C")], departing=time(8, 0))
    assert result.cost(planner.node_id("C")) == 8 * 3600 + 20 * 60
    assert result.cost(planner.node_id("B")) is None, "the query is point-to-point"
    assert result.settled == 3, "every trip in the fixture was reached"
    assert result.scanned == 3, "each was scanned once"
    assert result.rounds == 2
    # A kept search answers for its target exactly as route() would.
    assert planner.journey(result, "C") == planner.route("A", "C", departing=time(8, 0))
    assert planner.journey(result, "B") is None
    with pytest.raises(ValueError, match="searches toward a single target, and got no target"):
        planner.search("A", departing=time(8, 0))


def test_the_front_is_one_journey_per_number_of_changes(planner):
    front = planner.frontier("A", "C", departing=time(8, 0))
    assert [(j.transfers, j.arrives) for j in front] == [
        (0, 8 * 3600 + 30 * 60),
        (1, 8 * 3600 + 20 * 60),
    ]
    assert planner.route("A", "C", departing=time(8, 0)) == front[-1]
    capped = planner.route("A", "C", departing=time(8, 0), max_transfers=0)
    assert (capped.arrives, capped.transfers) == (front[0].arrives, 0)
    assert capped.settled < front[0].settled, "a capped sweep never reaches the third trip"
    result = planner.search("A", targets=[planner.node_id("C")], departing=time(8, 0))
    assert planner.journeys(result, "C") == front
    with pytest.raises(ValueError, match="cannot be -1"):
        planner.route("A", "C", departing=time(8, 0), max_transfers=-1)


def test_the_segments_are_a_search_space(planner):
    result = planner.search("A", targets=[planner.node_id("C")], departing=time(8, 0))
    space = planner.explored(result)
    assert isinstance(space, rl.Segments)
    assert space.kind == "segments"
    assert len(space) == 3
    assert space.peak == 1
    # Boarded at A: WEEKDAY1 the whole way and NIGHT1 to B, round 0; and
    # WEEKDAY2 from B, one change away.
    assert {(s.stops, s.round) for s in space.branches()} == {
        (("A", "B", "C"), 0),
        (("A", "B"), 0),
        (("B", "C"), 1),
    }
    assert [s.round for s in space.branches(min_round=1)] == [1]
    drawn = space.geojson()
    assert len(drawn["features"]) == 3
    assert {f["geometry"]["type"] for f in drawn["features"]} == {"LineString"}
    assert {f["properties"]["round"] for f in drawn["features"]} == {0, 1}
    assert drawn["peak"] == 1
    assert len(space.geojson(min_round=1)["features"]) == 1
    assert repr(space) == "Segments(3 trip segments, out to round 1)"
    with pytest.raises(ValueError, match="trip segments has no magnitude"):
        planner.explored(result, magnitude="weight")


def test_several_origins_each_carry_a_head_start(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.TripBased().bind(env)
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0))
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)
    result = planner.search({"A": 0, "C": 300}, targets=[planner.node_id("B")], departing=time(8, 0))
    assert planner.journey(result, "B").cost == 360, "cost elapsed from the query's departure"


def test_a_profile_is_one_journey_per_departure_worth_taking(planner):
    # A→B is served at 08:00 and at 23:50, and nothing in between beats
    # waiting for the night bus, so those are the two steps of the profile.
    front = planner.profile("A", "B", departing=time(0, 0), until=25 * 3600)
    assert [(j.departs, j.arrives, j.transfers) for j in front] == [
        (8 * 3600, 8 * 3600 + 10 * 60, 0),
        (23 * 3600 + 50 * 60, 24 * 3600 + 10 * 60, 0),
    ]
    assert all(j.origin == "A" and j.destination == "B" for j in front)
    assert front[1].cost == 20 * 60, "cost is elapsed from that journey's own departure"
    # A→C: the 08:00 departure, staying aboard for 08:30 or changing at B for
    # 08:20 — Pareto over three criteria keeps both, fewest changes first.
    front = planner.profile("A", "C", departing=time(6, 0), until=time(12, 0))
    assert [(j.departs, j.arrives, j.transfers) for j in front] == [
        (8 * 3600, 8 * 3600 + 30 * 60, 0),
        (8 * 3600, 8 * 3600 + 20 * 60, 1),
    ]
    # The window trims: nothing leaves A after 09:00 before the night bus.
    assert planner.profile("A", "C", departing=time(9, 0), until=time(12, 0)) == []
    # And every step agrees with asking route() at that moment with that
    # many changes allowed.
    for journey in planner.profile("A", "C", departing=time(0, 0), until=25 * 3600):
        again = planner.route("A", "C", departing=journey.departs, max_transfers=journey.transfers)
        assert again.arrives == journey.arrives


def test_a_walk_beats_the_steps_it_dominates(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("A", "B", 60)))
    planner = rl.TripBased().bind(env)
    assert planner.profile("A", "B", departing=time(0, 0), until=25 * 3600) == []
    assert planner.route("A", "B", departing=time(12, 0)).arrives == 12 * 3600 + 60


def test_the_profile_refuses_a_missing_or_backwards_window(planner):
    with pytest.raises(ValueError, match="needs a departure window"):
        planner.profile("A", "C", departing=time(8, 0))
    with pytest.raises(ValueError, match="cannot close before it opens"):
        planner.profile("A", "C", departing=time(9, 0), until=time(8, 0))
    with pytest.raises(ValueError, match=r"takes no until; .*profile\(\) on CSA\(\), PTL\(\) or TripBased\(\)"):
        planner.route("A", "C", departing=time(8, 0), until=time(9, 0))


def test_reduction_is_a_policy_and_never_an_answer(env, planner):
    unreduced = rl.TripBased(reduce=False).bind(env)
    assert unreduced.num_transfers == unreduced.num_initial_transfers
    assert unreduced.num_transfers >= planner.num_transfers
    for hour in (7, 8, 12, 23):
        for origin in "ABC":
            for destination in "ABC":
                a = planner.route(origin, destination, departing=time(hour, 0))
                b = unreduced.route(origin, destination, departing=time(hour, 0))
                assert (a is None) == (b is None)
                assert a is None or a.arrives == b.arrives


def test_footprint_counts_the_transfers_on_top_of_the_timetable(env, planner):
    assert planner.footprint > rl.TimeDependent().bind(env).footprint
    # Binding is the expensive half, and it counts trips through both phases;
    # a watcher left holding the counter sees the second phase finished.
    progress = rl._routelab.Progress()
    rl.TripBased().bind(env, progress)
    assert progress.read() == ("reducing transfers", 3, 3)
    progress = rl._routelab.Progress()
    rl.TripBased(reduce=False).bind(env, progress)
    assert progress.read() == ("computing transfers", 3, 3)


def test_it_needs_a_timetable_and_a_departure(planner):
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.TripBased().missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a departure time"):
        planner.route("A", "C")
    with pytest.raises(ValueError, match="takes no max_cost; a cost bound belongs to"):
        planner.route("A", "C", departing=time(8, 0), max_cost=10)
