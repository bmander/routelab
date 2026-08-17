"""PTL: hub labels over the events, and then no search at all.

Delling, Dibbelt, Pajor & Werneck (2015). The five-way agreement with the
other timetable techniques lives in test_timetable.py and test_footpaths.py,
where PTL is one of ``MODELS``; what is here is the rest of its surface — the
labeling it builds and reports, the profile checked against CSA's, and the
refusals — on the tiny fixture, where every number can be checked by eye:
A→C at 08:00 is 08:20 with one change, and NIGHT1 leaves A at 23:50.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def planner(env) -> rl.PTL:
    return rl.PTL().bind(env)


def test_hello_world(planner):
    journey = planner.route("A", "C", departing=time(8, 0))
    assert journey.arrives == 8 * 3600 + 20 * 60
    assert journey.transfers == 1
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    assert journey.settled > 0, "label entries read: the paper's measure of work"
    assert repr(rl.PTL()) == "PTL()"


def test_the_labeling_is_the_thing_it_reports(planner):
    # Tiny feed: A→B 08:00–08:10, B→C 08:12–08:20 (and 08:15–08:20 for the
    # change), A→B 23:50–00:10 — the events are the distinct (stop, time)s.
    assert planner.num_events == 8
    assert planner.num_hubs >= 2 * planner.num_events, "every event is at least its own hub"
    assert planner.hubs_per_label >= 1.0
    assert planner.searches == ("hubs", planner.num_hubs)


def test_it_keeps_no_table(planner):
    # Like the two Pyrga models: a journey, and nothing to draw.
    with pytest.raises(NotImplementedError, match="answers with a journey rather than a cost"):
        planner.search("A", departing=time(8, 0))
    with pytest.raises(NotImplementedError, match=r"nothing to draw\..*CSA\(\), RAPTOR\(\) or TripBased\(\)"):
        planner.explored(None)


def test_several_origins_each_carry_a_head_start(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.PTL().bind(env)
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0))
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)


def test_a_profile_is_one_journey_per_departure_worth_taking(planner):
    front = planner.profile("A", "B", departing=time(0, 0), until=25 * 3600)
    assert [(j.departs, j.arrives) for j in front] == [
        (8 * 3600, 8 * 3600 + 10 * 60),
        (23 * 3600 + 50 * 60, 24 * 3600 + 10 * 60),
    ]
    assert all(j.origin == "A" and j.destination == "B" for j in front)
    assert front[1].cost == 20 * 60
    front = planner.profile("A", "C", departing=time(6, 0), until=time(12, 0))
    assert [(j.departs, j.arrives, j.transfers) for j in front] == [(8 * 3600, 8 * 3600 + 20 * 60, 1)]
    assert planner.profile("A", "C", departing=time(9, 0), until=time(12, 0)) == []
    for journey in planner.profile("A", "B", departing=time(0, 0), until=25 * 3600):
        assert planner.route("A", "B", departing=journey.departs).arrives == journey.arrives


def test_the_profile_agrees_with_csa(env, feed):
    # Two techniques answering the profile from opposite ends — a scan of
    # the connections, a sweep of the labels — must list the same steps.
    walked = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60), ("B", "A", 90)))
    for world in (env, walked):
        labels, scan = rl.PTL().bind(world), rl.CSA().bind(world)
        for origin in ("A", "B", "C"):
            for destination in ("A", "B", "C"):
                if origin == destination:
                    continue
                by_labels = labels.profile(origin, destination, departing=time(0, 0), until=30 * 3600)
                by_scan = scan.profile(origin, destination, departing=time(0, 0), until=30 * 3600)
                assert [(j.departs, j.arrives) for j in by_labels] == [
                    (j.departs, j.arrives) for j in by_scan
                ], f"{origin} -> {destination}"


def test_a_profile_from_several_origins_is_merged_on_the_query_clock(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = rl.PTL().bind(env)
    front = planner.profile({"A": 0, "C": 300}, "B", departing=time(7, 0), until=time(9, 0))
    assert [(j.origin, j.departs, j.arrives) for j in front] == [("A", 8 * 3600, 8 * 3600 + 600)]


def test_a_walk_beats_the_steps_it_dominates(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("A", "B", 60)))
    planner = rl.PTL().bind(env)
    assert planner.profile("A", "B", departing=time(0, 0), until=25 * 3600) == []
    assert planner.route("A", "B", departing=time(12, 0)).arrives == 12 * 3600 + 60


def test_the_profile_refuses_a_missing_or_backwards_window(planner):
    with pytest.raises(ValueError, match="needs a departure window"):
        planner.profile("A", "C", departing=time(8, 0))
    with pytest.raises(ValueError, match="cannot close before it opens"):
        planner.profile("A", "C", departing=time(9, 0), until=time(8, 0))
    with pytest.raises(ValueError, match=r"takes no until; .*profile\(\) on CSA\(\), PTL\(\) or TripBased\(\)"):
        planner.route("A", "C", departing=time(8, 0), until=time(9, 0))


def test_footprint_counts_the_labels_on_top_of_the_timetable(env, planner):
    assert planner.footprint > rl.TimeDependent().bind(env).footprint


def test_binding_reports_its_progress(env):
    progress = rl._routelab.Progress()
    rl.PTL().bind(env, progress=progress)
    assert progress.phase == "merging"
    assert progress.fraction == 1.0


def test_it_needs_a_timetable_and_a_departure(planner):
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.PTL().missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a departure time"):
        planner.route("A", "C")
    with pytest.raises(ValueError, match="takes no max_transfers; a cap on changes belongs to RAPTOR"):
        planner.route("A", "C", departing=time(8, 0), max_transfers=1)
