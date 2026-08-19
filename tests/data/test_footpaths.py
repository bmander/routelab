"""Footpaths: walks between stops, and what the timetable techniques do with them.

The paper's foot-edges. A real feed does not say that two stops across a
street are one place, so without walks a timetable technique cannot cross the
street; with them, a scalar edge beside the timetable is a link a rider takes
at any time. Both models must still agree, and a journey must be able to say
which stretches were walked.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl
#: Every timetable technique. Written out, and checked against the shelf in
#: test_timetable.py, because a technique that nobody added here would
#: silently not be tested — which is the one thing a list like this must not
#: allow.
MODELS = [
    rl.TimeDependent,
    rl.TimeExpanded,
    rl.RAPTOR,
    rl.CSA,
    rl.TripBased,
    rl.PTL,
    rl.ULTRA,
    rl.LabelConstrained,
    rl.UCCH,
]


# --- The layer -----------------------------------------------------------------


def test_footpaths_join_stops_within_reach(feed):
    # A, B and C sit 1.1 km apart in a line, so 1,200 m joins neighbours both
    # ways and leaves the ends unjoined.
    walks = rl.Footpaths(feed, within=1200)
    edges = list(walks.edges())
    assert {(tail, head) for tail, head, _ in edges} == {
        ("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")
    }
    weight = dict(((tail, head), w) for tail, head, w in edges)[("A", "B")]
    assert 700 < weight < 900, "about 1,112 m at 1.4 m/s"
    assert len(walks) == 4
    assert walks.geometry(0) == [feed.coordinates()[edges[0][0]], feed.coordinates()[edges[0][1]]]
    assert walks.cost_model == "scalar"
    assert walks.cost_per_distance == pytest.approx(1 / 1.4)
    assert "4 walks" in repr(walks)


def test_footpaths_out_of_reach_are_no_footpaths(feed):
    assert len(rl.Footpaths(feed, within=100)) == 0


def test_footpaths_refuse_a_layer_with_no_coordinates_and_bad_knobs(feed):
    with pytest.raises(TypeError, match="coordinates"):
        rl.Footpaths(rl.ScalarEdges(("a", "b", 1)))
    with pytest.raises(ValueError, match="within"):
        rl.Footpaths(feed, within=0)
    with pytest.raises(ValueError, match="speed"):
        rl.Footpaths(feed, speed=-1)


# --- The derivation --------------------------------------------------------------


def test_walks_gather_the_scalar_edges_between_stops_and_close_them(feed):
    plain = rl.Environment(feed).compile()
    assert len(rl.Walks().bind(plain)) == 0, "no scalar layer, no walks — and no refusal"
    assert rl.Walks.missing_from(plain) == frozenset()

    joined = rl.Environment(feed, rl.ScalarEdges([("A", "B", 30), ("B", "C", 30)])).compile()
    footpaths = rl.Walks().bind(joined)
    a, b, c = (joined.node_id(label) for label in "ABC")
    assert footpaths.duration(a, b) == 30
    assert footpaths.duration(a, c) == 60, "closed under composition"
    assert footpaths.duration(c, a) is None, "and only in the directions given"
    assert len(footpaths) == 3


def test_walks_take_only_edges_between_stops_the_timetable_serves(feed):
    # A scalar edge to somewhere no vehicle stops — an access link, a street —
    # is not a foot-edge, and is left alone rather than swept in and closed.
    env = rl.Environment(feed, rl.ScalarEdges([("A", "B", 30), ("B", "home", 30), ("home", "A", 30)]))
    footpaths = rl.Walks().bind(env.compile())
    assert len(footpaths) == 1, "A → B only"


# --- The techniques ---------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_a_walk_beats_waiting_for_the_next_bus(model, feed):
    # WEEKDAY1 reaches B at 08:10; the fixture's good connection leaves B at
    # 08:15 and reaches C at 08:20. A one-minute walk from B to C is better,
    # and both models must take it.
    walk = rl.ScalarEdges(("B", "C", 60))
    env = rl.Environment(feed, walk)
    journey = model().bind(env).route("A", "C", departing=time(8, 0)).routes[0]
    assert journey.arrives == 8 * 3600 + 11 * 60
    assert [(leg.tail, leg.head, leg.trip is None) for leg in journey.legs] == [
        ("A", "B", False),
        ("B", "C", True),
    ]
    assert journey.legs[1].source is walk, "the walk leg knows the layer it came from"
    assert journey.legs[1].weight == 60
    assert journey.transfers == 0, "one bus and a walk is not a change of vehicle"
    assert journey.waiting == 0
    assert journey.cost == 11 * 60


@pytest.mark.parametrize("model", MODELS)
def test_a_walk_can_start_or_be_the_whole_journey(model, feed):
    # Nothing runs at noon, but A → B → C is a chain of two footpaths. Both
    # models must find it on foot alone, and the journey must come back as the
    # two edges the walk was made of — the kernel closed them into one.
    env = rl.Environment(feed, rl.ScalarEdges([("A", "B", 30), ("B", "C", 30)]))
    journey = model().bind(env).route("A", "C", departing=time(12, 0)).routes[0]
    assert journey.arrives == 12 * 3600 + 60
    assert [(leg.tail, leg.head) for leg in journey.legs] == [("A", "B"), ("B", "C")]
    assert all(leg.trip is None for leg in journey.legs)
    assert journey.legs[0].departs == 12 * 3600 and journey.legs[1].arrives == 12 * 3600 + 60
    assert journey.transfers == 0

    # And a walk that leads to a better departure: nothing leaves C, but B is
    # a short walk away and WEEKDAY2's 08:15 to... nowhere useful. Use A: from
    # C at 07:59, walk to A by 08:00 and ride WEEKDAY1 to B by 08:10.
    env = rl.Environment(feed, rl.ScalarEdges(("C", "A", 60)))
    journey = model().bind(env).route("C", "B", departing=time(7, 59)).routes[0]
    assert journey.arrives == 8 * 3600 + 10 * 60
    assert [leg.trip is None for leg in journey.legs] == [True, False]


def test_the_two_models_agree_with_footpaths_between_real_stops(feed):
    # The paper's thesis, on the layer a real feed would get: footpaths from
    # coordinates, closed by the kernel, and every query answered alike.
    env = rl.Environment(feed, rl.Footpaths(feed, within=1200))
    planners = [model().bind(env) for model in MODELS]
    for origin in "ABC":
        for destination in "ABC":
            for hour in (7, 8, 12, 23):
                answers = [
                    planner.route(origin, destination, departing=time(hour, 0)).routes[0]
                    for planner in planners
                ]
                arrivals = [None if journey is None else journey.arrives for journey in answers]
                assert len(set(arrivals)) == 1, (origin, destination, hour, arrivals)


def test_footpaths_on_the_board_style_environment_repr(feed):
    env = rl.Environment(feed, rl.Footpaths(feed, within=1200))
    assert repr(env) == "Environment(2 layers)"
    planner = rl.TimeExpanded().bind(env)
    assert planner.footprint > 0


def test_a_change_made_on_foot_is_still_a_change():
    # Ride, walk, ride is one change of vehicle, not zero because the two
    # rides are not adjacent legs — and ride, walk, walk, ride is still one.
    walk = rl.ScalarEdges(("b", "c", 30))
    def leg(tail, head, trip):
        return rl.Leg(tail=tail, head=head, weight=60, source=walk, edge=0, position=0,
                      departs=0, arrives=60, trip=trip)
    journey = rl.Journey(origin="a", destination="d", cost=240, legs=(
        leg("a", "b", 7), leg("b", "c", None), leg("c", "d", 9)))
    assert journey.transfers == 1
    journey = rl.Journey(origin="a", destination="e", cost=300, legs=(
        leg("a", "b", 7), leg("b", "c", None), leg("c", "d", None), leg("d", "e", 7)))
    assert journey.transfers == 0, "back on the same trip after a walk is not a change"
