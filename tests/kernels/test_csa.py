"""CSA: one array of connections, scanned once — and read backwards for a profile.

Dibbelt, Pajor, Strasser & Wagner (2013). The four-way agreement with the
other timetable techniques lives in test_timetable.py and test_footpaths.py,
where CSA is one of ``MODELS``; what is here is the rest of its surface — the
one-to-all search and the scan it reports, the profile, and the refusals — on
the tiny fixture, where every number can be checked by eye: A→C at 08:00 is
08:20 with one change, and NIGHT1 leaves A at 23:50.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def planner(env) -> rl.CSA:
    return rl.CSA().bind(env)


def test_hello_world(planner):
    journey = planner.route("A", "C", departing=time(8, 0))
    assert journey.arrives == 8 * 3600 + 20 * 60
    assert journey.transfers == 1
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    assert repr(rl.CSA()) == "CSA()"


def test_a_search_is_one_to_all(planner):
    result = planner.search("A", departing=time(8, 0))
    assert result.cost(planner.node_id("B")) == 8 * 3600 + 10 * 60
    assert result.cost(planner.node_id("C")) == 8 * 3600 + 20 * 60
    assert result.settled == 3
    assert result.scanned == 4, "the whole day, from 08:00 on: every connection"
    assert planner.searches == ("stops", 3)
    # A kept search answers for any destination without searching again — and
    # answers exactly as route() would.
    assert planner.journey(result, "C") == planner.route("A", "C", departing=time(8, 0))
    assert planner.journey(result, "B").arrives == 8 * 3600 + 10 * 60
    # Toward a target the scan stops early: nothing after 08:10 is read.
    toward = planner.search("A", targets=[planner.node_id("B")], departing=time(8, 0))
    assert toward.scanned < result.scanned


def test_the_scan_is_a_search_space(planner):
    result = planner.search("A", departing=time(8, 0))
    space = planner.explored(result)
    assert isinstance(space, rl.Scan)
    assert space.kind == "scan"
    assert {(a.stop, a.arrives - space.departing) for a in space.branches()} == {
        ("A", 0), ("B", 600), ("C", 1200)
    }
    assert space.peak == 1200, "the horizon, under the name every kind reports"
    assert len(space) == 3
    drawn = space.geojson()
    features = drawn["features"]
    assert len(features) == 3
    assert {f["geometry"]["type"] for f in features} == {"Point"}
    assert {f["properties"]["after"] for f in features} == {0, 600, 1200}
    assert drawn["peak"] == 1200, "the collection carries what `after` is a share of"
    assert len(space.geojson(min_after=600)["features"]) == 2
    assert repr(space) == "Scan(3 stops within 20 min)"
    with pytest.raises(ValueError, match="a scan has no magnitude"):
        planner.explored(result, magnitude="weight")


def test_several_origins_each_carry_a_head_start(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.CSA().bind(env)
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0))
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)
    result = planner.search({"A": 0, "C": 300}, departing=time(8, 0))
    assert planner.journey(result, "B").cost == 360, "cost elapsed from the query's departure"


def test_a_profile_is_one_journey_per_departure_worth_taking(planner):
    # A→B is served at 08:00 and at 23:50, and nothing in between beats
    # waiting for the night bus, so those are the two steps of the profile.
    front = planner.profile("A", "B", departing=time(0, 0), until=25 * 3600)
    assert [(j.departs, j.arrives) for j in front] == [
        (8 * 3600, 8 * 3600 + 10 * 60),
        (23 * 3600 + 50 * 60, 24 * 3600 + 10 * 60),
    ]
    assert all(j.origin == "A" and j.destination == "B" for j in front)
    assert front[1].cost == 20 * 60, "cost is elapsed from that journey's own departure"
    # A→C: the 08:00 departure, changing at B; WEEKDAY2's own 08:15 leaves
    # from B and is no departure from A.
    front = planner.profile("A", "C", departing=time(6, 0), until=time(12, 0))
    assert [(j.departs, j.arrives, j.transfers) for j in front] == [(8 * 3600, 8 * 3600 + 20 * 60, 1)]
    # The window trims: nothing leaves A after 09:00 before the night bus.
    assert planner.profile("A", "C", departing=time(9, 0), until=time(12, 0)) == []
    # And every step agrees with asking route() at that moment.
    for journey in planner.profile("A", "B", departing=time(0, 0), until=25 * 3600):
        assert planner.route("A", "B", departing=journey.departs).arrives == journey.arrives


def test_a_profile_from_several_origins_is_merged_on_the_query_clock(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.CSA().bind(env)
    # Standing at A, and at C five minutes along: from C the walk to B leaves
    # any time, so it holds no step — a step is a departure you can miss.
    # A's 08:00 bus survives only because the walk from C is not a walk from
    # the query's origin set as a whole; the merge keeps whatever leaves an
    # origin and is not beaten by a later-leaving journey from any origin.
    front = planner.profile({"A": 0, "C": 300}, "B", departing=time(7, 0), until=time(9, 0))
    assert [(j.origin, j.departs, j.arrives) for j in front] == [("A", 8 * 3600, 8 * 3600 + 600)]


def test_a_walk_beats_the_steps_it_dominates(feed):
    # A→B on foot in a minute: no bus from A to B is worth a step in the
    # profile, and the earliest-arrival scan walks it too.
    env = rl.Environment(feed, rl.ScalarEdges(("A", "B", 60)))
    planner = rl.CSA().bind(env)
    assert planner.profile("A", "B", departing=time(0, 0), until=25 * 3600) == []
    assert planner.route("A", "B", departing=time(12, 0)).arrives == 12 * 3600 + 60


def test_the_profile_refuses_a_missing_or_backwards_window(planner):
    with pytest.raises(ValueError, match="needs a departure window"):
        planner.profile("A", "C", departing=time(8, 0))
    with pytest.raises(ValueError, match="cannot close before it opens"):
        planner.profile("A", "C", departing=time(9, 0), until=time(8, 0))
    with pytest.raises(ValueError, match="takes no until"):
        planner.route("A", "C", departing=time(8, 0), until=time(9, 0))


def test_footprint_counts_the_array_on_top_of_the_timetable(env, planner):
    assert planner.footprint > rl.TimeDependent().bind(env).footprint
    assert planner.num_connections == 4
    assert planner.num_trips == 3


def test_it_needs_a_timetable_and_a_departure(planner):
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.CSA().missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a departure time"):
        planner.route("A", "C")
    with pytest.raises(ValueError, match="takes no max_transfers; a cap on changes belongs to RAPTOR"):
        planner.route("A", "C", departing=time(8, 0), max_transfers=1)
