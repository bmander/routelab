"""Time-dependent routing through the planner API.

The fixture is a gate with a way round it, in the shape of the Ballard Locks:
a short path shut outside 07:00–21:00, and a long one always open. Which one a
search takes says plainly whether it read the schedule.
"""

from __future__ import annotations

import datetime

import pytest

import routelab as rl

from conftest import FIXTURES

CONDITIONAL = FIXTURES / "conditional.osm"


@pytest.fixture
def gated() -> rl.Environment:
    return rl.Environment(rl.OSM(CONDITIONAL, rl.Walking()))


def at(hour: int, minute: int = 0) -> datetime.time:
    return datetime.time(hour, minute)


def test_hello_world(gated):
    planner = rl.TimeDependentDijkstra().bind(gated)
    assert planner.route(1, 3, departing=at(12)).nodes == [1, 2, 3]


def test_the_gate_is_shut_outside_its_hours(gated):
    planner = rl.TimeDependentDijkstra().bind(gated)
    assert planner.route(1, 3, departing=at(12)).nodes == [1, 2, 3], "through"
    assert planner.route(1, 3, departing=at(3)).nodes == [1, 3], "round"


def test_plain_dijkstra_ignores_the_schedule_and_says_so_by_its_name(gated):
    # Not a bug: `Dijkstra` is the always-open network, which is a question
    # worth asking. You get the clock by asking for the technique that has one.
    plain = rl.Dijkstra().bind(gated).route(1, 3)
    assert plain.nodes == [1, 2, 3]
    assert plain.cost == rl.TimeDependentDijkstra().bind(gated).route(
        1, 3, departing=at(12)
    ).cost


def test_waiting_beats_detouring_only_once_it_is_cheaper(gated):
    """The whole argument for costing the wait rather than deciding by policy.

    Nothing here chooses between waiting and going round; the clock does.
    """
    planner = rl.TimeDependentDijkstra().bind(gated)
    detour = planner.route(1, 3, departing=at(3)).cost

    early = planner.route(1, 3, departing=at(6, 40))
    assert early.nodes == [1, 3], "an hour's wait loses to the way round"

    late = planner.route(1, 3, departing=at(6, 58))
    assert late.nodes == [1, 2, 3], "two minutes' wait beats it"
    assert late.cost < detour


def test_forbidding_waiting_never_helps_and_sometimes_hurts(gated):
    patient = rl.TimeDependentDijkstra().bind(gated)
    hurried = rl.TimeDependentDijkstra(waiting="forbidden").bind(gated)
    for hour, minute in [(12, 0), (6, 58), (3, 0), (20, 59), (21, 1)]:
        when = at(hour, minute)
        assert patient.route(1, 3, departing=when).cost <= hurried.route(
            1, 3, departing=when
        ).cost


def test_an_unknown_waiting_policy_says_what_it_expected():
    with pytest.raises(ValueError, match="expected 'unrestricted' or 'forbidden'"):
        rl.TimeDependentDijkstra(waiting="whenever")


def test_a_departure_time_has_no_default(gated):
    planner = rl.TimeDependentDijkstra().bind(gated)
    with pytest.raises(ValueError, match="needs a departure time"):
        planner.route(1, 3)


def test_a_dataset_with_no_schedule_says_so_before_you_bind():
    # The existing capability mechanism, doing its job: `requires` against
    # `provides`, answered without building anything.
    bare = rl.Environment(rl.ScalarEdges([("a", "b", 1)]))
    assert rl.TimeDependentDijkstra().missing_from(bare.compile()) == {"schedule"}
    assert rl.Dijkstra().missing_from(bare.compile()) == frozenset()


def test_an_environment_with_schedules_provides_one(gated):
    compiled = gated.compile()
    assert "schedule" in compiled.provides
    assert rl.TimeDependentDijkstra().missing_from(compiled) == frozenset()
    assert len(compiled.calendar) > 0


def test_the_day_of_the_week_matters(gated):
    # The gate is daily, so it should not. The reversible lane is not, and the
    # clock has to carry a weekday for that to be expressible at all.
    planner = rl.TimeDependentDijkstra().bind(gated)
    monday = datetime.datetime(2026, 8, 10, 12, 0)  # a Monday
    sunday = datetime.datetime(2026, 8, 16, 12, 0)
    assert planner.route(1, 3, departing=monday).cost == planner.route(
        1, 3, departing=sunday
    ).cost


def test_a_journey_still_reports_geometry_and_provenance(gated):
    journey = rl.TimeDependentDijkstra().bind(gated).route(1, 3, departing=at(12))
    assert all(leg.geometry for leg in journey)
    assert all(leg.source is gated.sources[0] for leg in journey)
    assert journey.geometry[0] and journey.geometry[-1]


def test_the_search_space_is_an_ordinary_tree(gated):
    planner = rl.TimeDependentDijkstra().bind(gated)
    result = planner.search(1, targets=[planner.node_id(3)], departing=at(12))
    space = planner.explored(result)
    assert space.kind == "shortest-path-tree"
    # The work is reported by the result, not by the drawing of it.
    assert result.settled > 0 and len(space) > 0


def test_an_overnight_closure_is_the_other_tag_form(gated):
    # `no @ (23:00-05:00)`: open by default, shut at night — the complement of
    # the gate's `yes @`, and the one that would break if the default were
    # read the same way for both.
    compiled = gated.compile()
    trail = [
        edge
        for edge in range(compiled.graph.num_edges)
        if compiled.label(compiled.graph.edge(edge)[0]) == 7
    ]
    assert trail, "the fixture has an overnight trail"
    for edge in trail:
        assert compiled.calendar.is_open(edge, 12 * 3600), "open at noon"
        assert not compiled.calendar.is_open(edge, 2 * 3600), "shut at 2am"


def test_a_walking_profile_reads_foot_tags_and_a_driving_one_does_not():
    # The gate is `access:conditional`, which speaks to everyone; the point here
    # is that which keys a profile reads is what decides whose schedule binds.
    assert rl.Walking().access_keys == ("foot", "access")
    assert rl.Driving().access_keys == ("motor_vehicle", "vehicle", "access")


def test_seconds_since_monday_accepts_what_a_caller_would_reach_for():
    assert rl.weekly_seconds(datetime.time(7, 30)) == 27000
    assert rl.weekly_seconds(datetime.datetime(2026, 8, 14, 7, 30)) == 4 * 86400 + 27000
    assert rl.weekly_seconds(27000) == 27000
    assert rl.weekly_seconds(rl.WEEK + 60) == 60, "the clock wraps"
    with pytest.raises(TypeError):
        rl.weekly_seconds("07:30")


def test_a_journey_separates_waiting_from_walking(gated):
    planner = rl.TimeDependentDijkstra().bind(gated)
    moving = planner.route(1, 3, departing=at(12))
    assert moving.waiting == 0
    assert moving.moving == moving.cost

    # Arriving two minutes before the gate opens: the walk is unchanged and the
    # wait is the whole of the difference. Reporting one total instead makes a
    # short walk after a long wait look like a long walk.
    waited = planner.route(1, 3, departing=at(6, 58))
    assert waited.moving == moving.cost, "the walking is the same walking"
    assert waited.waiting == waited.cost - moving.cost > 0


def test_waiting_is_zero_for_a_search_with_no_clock(gated):
    journey = rl.Dijkstra().bind(gated).route(1, 3)
    assert journey.waiting == 0 and journey.moving == journey.cost


def test_a_hop_count_is_not_a_wait(gated):
    # BFS costs hops while its legs carry seconds, so the difference is
    # meaningless rather than a delay — and is reported as no delay.
    journey = rl.BFS().bind(gated).route(1, 3)
    assert journey.waiting == 0
