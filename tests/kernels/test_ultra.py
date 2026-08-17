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


class Streets(rl.ScalarEdges):
    """A stand-in street network: corners, the ways between them, and the
    nearest corner to a point — the three things :class:`~routelab.Access` and
    a click both ask of a map."""

    def __init__(self, places, edges):
        super().__init__(list(edges))
        self._places = places

    def coordinates(self):
        return self._places

    def nearest(self, lat, lon, within=None):
        return min(
            self._places.items(),
            key=lambda item: (item[1][0] - lat) ** 2 + (item[1][1] - lon) ** 2,
        )[0]


@pytest.fixture
def multimodal(feed):
    """Streets beside each stop, joined to it — the multimodal environment.

    A corner outside every stop, a pavement running the length of them, and an
    Access layer saying which corner is which stop's. Nothing here is a
    stop-to-stop footpath: every walk is along a street, which is the whole
    difference.
    """
    corners = {
        "corner-A": (47.6000, -122.3301),
        "corner-B": (47.6100, -122.3301),
        "corner-C": (47.6200, -122.3301),
        "doorstep": (47.5999, -122.3302),
    }
    pavement = [
        ("corner-A", "corner-B", 600), ("corner-B", "corner-A", 600),
        ("corner-B", "corner-C", 600), ("corner-C", "corner-B", 600),
        ("doorstep", "corner-A", 30), ("corner-A", "doorstep", 30),
    ]
    streets = Streets(corners, pavement)
    return rl.Environment(streets, feed, rl.Access(feed, streets, within=400))


def test_a_journey_can_start_on_a_doorstep(multimodal):
    # The multimodal query: begin where the caller stood rather than at a
    # stop, walk to one, ride, and walk off the other end. None of those three
    # walks is a shortcut — the ends are searched when the query is asked.
    planner = rl.ULTRA(rl.RAPTOR()).bind(multimodal)
    journey = planner.route("doorstep", "corner-C", departing=time(8, 0))
    assert journey is not None
    assert journey.origin == "doorstep"
    assert journey.destination == "corner-C"
    # It rode something, rather than walking the whole way.
    assert any(leg.trip is not None for leg in journey.legs)
    # And the legs join up, doorstep to corner, every one an edge of this
    # environment rather than a straight line somebody invented.
    chain = [journey.legs[0].tail] + [leg.head for leg in journey.legs]
    assert chain[0] == "doorstep" and chain[-1] == "corner-C"
    for before, after in zip(journey.legs, journey.legs[1:]):
        assert before.head == after.tail
    assert journey.legs[-1].arrives == journey.arrives


def test_walking_the_streets_beats_waiting_when_it_should(multimodal):
    # Not every multimodal answer rides: from one corner to the next, the
    # pavement is ten minutes and the bus does not come sooner.
    planner = rl.ULTRA(rl.RAPTOR()).bind(multimodal)
    journey = planner.route("doorstep", "corner-A", departing=time(12, 0))
    assert journey is not None
    assert all(leg.trip is None for leg in journey.legs), "nothing runs at noon"
    assert journey.cost == 30


def test_the_streets_are_contracted_away_before_the_shortcuts(multimodal):
    # Core-CH is what makes the transfer graph searchable: the vertices a
    # vehicle never calls at go, and the stops stay.
    planner = rl.ULTRA(rl.RAPTOR()).bind(multimodal)
    transfers = planner.transfers
    assert transfers.contraction is not None, "a street network should be contracted"
    assert transfers.core.num_edges <= transfers.graph.num_edges
    for stop in ("A", "B", "C"):
        assert transfers.contraction.is_core(planner.node_id(stop))
