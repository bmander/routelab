"""ULTRA: the walks worth taking, worked out once, so a radius is not needed.

Baum, Buchhold, Sauer, Wagner & Zündorf (2019). The family-wide agreement runs
in test_timetable.py, where ULTRA is one of ``MODELS`` like any other timetable
technique; what is here is the rest of its surface — that it wraps a technique
rather than replacing one, that it answers exactly as the closure it stands in
for, and the refusals.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def walkable(feed):
    """The tiny feed with walks between its stops, which is a transfer graph."""
    return rl.Environment(
        feed, rl.ScalarEdges(("A", "B", 60), ("B", "A", 60), ("B", "C", 60), ("C", "B", 60))
    )


def test_hello_world(walkable):
    planner = rl.ULTRA(rl.RAPTOR()).bind(walkable)
    journey = planner.route("A", "C", departing=time(8, 0))
    assert journey.arrives == 8 * 3600 + 120, "two minutes of walking beats waiting"
    assert repr(rl.ULTRA(rl.RAPTOR())) == "ULTRA(RAPTOR())"
    assert repr(rl.ULTRA(rl.CSA())) == "ULTRA(CSA())"


def test_it_wraps_a_technique_that_keeps_a_table(walkable):
    # The query has to read every stop's arrival to know where to get off, so
    # a technique that answers only with a journey cannot be wrapped — and is
    # refused by name, from the shelf rather than from a list.
    with pytest.raises(TypeError, match=r"keeps none.*Wrap CSA\(\), RAPTOR\(\) or TripBased\(\)"):
        rl.ULTRA(rl.TimeDependent())
    with pytest.raises(TypeError, match="keeps none"):
        rl.ULTRA(rl.PTL())
    assert isinstance(rl.ULTRA().technique, rl.kernels.TimetablePlanner)


def test_the_wrapped_technique_has_the_last_word_on_the_knobs(walkable):
    # ULTRA advertises what any of its techniques takes; only the one
    # underneath knows which, so it is asked.
    capped = rl.ULTRA(rl.RAPTOR()).bind(walkable)
    assert capped.route("A", "C", departing=time(8, 0), max_transfers=0) is not None
    with pytest.raises(ValueError, match="CSA takes no max_transfers"):
        rl.ULTRA(rl.CSA()).bind(walkable).route(
            "A", "C", departing=time(8, 0), max_transfers=1
        )


def test_a_transfer_graph_of_islands_is_the_plain_model(env):
    # No scalar layer, so nothing walks: not a refusal, the same plain model
    # an environment without footpaths is for every other technique.
    planner = rl.ULTRA(rl.RAPTOR()).bind(env)
    assert rl.ULTRA(rl.RAPTOR()).missing_from(env.compile()) == frozenset()
    assert planner.num_shortcuts == 0
    assert planner.route("A", "C", departing=time(8, 0)).arrives == 8 * 3600 + 20 * 60


def test_it_needs_a_timetable_and_a_departure(walkable):
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.ULTRA(rl.RAPTOR()).missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a departure time"):
        rl.ULTRA(rl.RAPTOR()).bind(walkable).route("A", "C")


def test_a_search_is_the_wrapped_technique_s_and_says_so(walkable):
    planner = rl.ULTRA(rl.RAPTOR()).bind(walkable)
    with pytest.raises(NotImplementedError, match="brackets its technique's search"):
        planner.search("A", departing=time(8, 0))


def test_every_leg_is_an_edge_of_this_environment(walkable):
    # A shortcut is a path, not an edge, so a journey that walked one has to
    # be told hop by hop — which is what makes provenance and geometry work.
    journey = rl.ULTRA(rl.RAPTOR()).bind(walkable).route("A", "C", departing=time(8, 0))
    assert [leg.tail for leg in journey.legs] == ["A", "B"]
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    for leg in journey.legs:
        assert leg.source is not None
        assert leg.arrives - leg.departs == leg.weight
    assert journey.legs[-1].arrives == journey.arrives


def test_several_origins_each_carry_a_head_start(feed):
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60), ("B", "C", 60)))
    planner = rl.ULTRA(rl.RAPTOR()).bind(env)
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0))
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)
