"""LCSPP: the journey whose sequence of modes you allowed, and no preprocessing.

Barrett, Jacob & Marathe (2000) in the multimodal form of Dibbelt, Pajor &
Wagner (ALENEX 2012) §2.2. The family-wide agreement runs in test_timetable.py,
where this is one of ``MODELS`` like any other timetable technique; what is here
is the rest of its surface — that the language decides, that the automaton is
read off the environment when nobody writes one, and that it answers a
multimodal query without building anything first.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


class Streets(rl.ScalarEdges):
    """A stand-in street network: corners, the ways between them, and the
    nearest corner to a point."""

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
def streets():
    corners = {
        "corner-A": (47.6000, -122.3301),
        "corner-B": (47.6100, -122.3301),
        "corner-C": (47.6200, -122.3301),
        "doorstep": (47.5999, -122.3302),
    }
    # Half an hour a block, so that riding is plainly worth it — at ten minutes
    # the pavement ties the fixture's twenty-minute ride and the comparisons
    # below stop comparing anything. The queries then leave at 07:55, early
    # enough to reach the stop before the 08:00 leaves and late enough that
    # walking cannot get there first.
    pavement = [
        ("corner-A", "corner-B", 1800), ("corner-B", "corner-A", 1800),
        ("corner-B", "corner-C", 1800), ("corner-C", "corner-B", 1800),
        ("doorstep", "corner-A", 30), ("corner-A", "doorstep", 30),
    ]
    return Streets(corners, pavement)


@pytest.fixture
def multimodal(feed, streets):
    """Streets, a timetable, and the link arcs between them."""
    return rl.Environment(streets, feed, rl.Access(feed, streets, within=400))


def test_hello_world(multimodal):
    planner = rl.LabelConstrained().bind(multimodal)
    journey = planner.route("doorstep", "corner-C", departing=time(7, 55)).routes[0]
    assert journey is not None
    assert journey.origin == "doorstep"
    assert journey.destination == "corner-C"
    assert any(leg.trip is not None for leg in journey.legs), "it rode something"
    # The legs join up, doorstep to corner, each one an arc of this environment.
    for before, after in zip(journey.legs, journey.legs[1:]):
        assert before.head == after.tail
    assert journey.legs[0].tail == "doorstep"
    assert journey.legs[-1].head == "corner-C"
    assert journey.legs[-1].arrives == journey.arrives


def test_nothing_is_precomputed(multimodal):
    # The whole reason it is here. Every other timetable technique closes its
    # walks under composition at bind; this one reads the arcs in the search, so
    # the closure is empty and there is nothing else built.
    planner = rl.LabelConstrained().bind(multimodal)
    assert len(planner.footpaths) == 0
    # What it does hold is a byte per arc and an automaton, which is not a
    # preprocessed structure so much as a reading of the one already there.
    assert planner.network.num_arcs == multimodal.compile().graph.num_edges
    assert planner.network.footprint == planner.network.num_arcs


def test_the_language_is_what_decides(multimodal):
    # Two languages, one environment, two answers: riding is worth something,
    # and refusing to ride costs exactly that. Without this the test above
    # cannot tell an automaton that was applied from one that was ignored.
    riding = rl.LabelConstrained().bind(multimodal)
    afoot = rl.LabelConstrained(
        rl.Modes(states={"foot": ["foot"]}, link="link", start=["foot"], end=["foot"])
    ).bind(multimodal)
    quick = riding.route("doorstep", "corner-C", departing=time(7, 55)).routes[0]
    slow = afoot.route("doorstep", "corner-C", departing=time(7, 55)).routes[0]
    assert slow is not None and quick is not None
    assert quick.arrives < slow.arrives
    assert all(leg.trip is None for leg in slow.legs), "on foot, so nothing was ridden"
    # Walking it is the pavement end to end: 30 to the corner and 1800 twice.
    assert slow.cost == 30 + 1800 + 1800


def test_the_automaton_is_read_off_the_environment(multimodal, env):
    # Where a link layer joins two networks it is the paper's Figure 1(a): a
    # walking state and a riding state, joined by the link.
    joined = rl.LabelConstrained().bind(multimodal)
    assert joined.automaton.num_states == 2
    # Where there is none, the stops *are* the walking network's nodes, so one
    # state stands for both modes and a rider changes vehicles standing still —
    # which is what every other timetable technique here assumes.
    alone = rl.LabelConstrained().bind(env)
    assert alone.automaton.num_states == 1
    assert alone.route("A", "C", departing=time(8, 0)).routes != []


def test_it_agrees_with_ultra_on_the_same_network(multimodal):
    # The load-bearing test, and the reason both are on the shelf. Two papers,
    # two models of unlimited walking — one that precomputes the walks worth
    # taking, one that precomputes nothing — must answer the same question the
    # same way.
    lcspp = rl.LabelConstrained().bind(multimodal)
    ultra = rl.ULTRA(rl.RAPTOR()).bind(multimodal)
    ridden = 0
    for origin in ("doorstep", "corner-A", "corner-B"):
        for destination in ("corner-B", "corner-C", "doorstep"):
            for at in (time(7, 30), time(7, 55), time(12, 0)):
                mine = lcspp.route(origin, destination, departing=at).routes[0]
                theirs = ultra.route(origin, destination, departing=at).routes[0]
                assert (mine is None) == (theirs is None), f"{origin}->{destination} at {at}"
                if mine is not None:
                    assert mine.arrives == theirs.arrives, f"{origin}->{destination} at {at}"
                    ridden += any(leg.trip is not None for leg in mine.legs)
    # Agreeing that nothing is reachable would agree about nothing.
    assert ridden >= 5, f"only {ridden} of these journeys rode anything"


def test_a_language_naming_a_mode_the_network_lacks_says_so(multimodal):
    with pytest.raises(ValueError, match="travels by 'ferry', which no layer here emits"):
        rl.LabelConstrained(
            rl.Modes(states={"sail": ["ferry"]}, start=["sail"], end=["sail"])
        ).bind(multimodal)
    with pytest.raises(ValueError, match="no state named 'nowhere'"):
        rl.LabelConstrained(
            rl.Modes(states={"foot": ["foot"]}, start=["nowhere"], end=["foot"])
        ).bind(multimodal)


def test_it_needs_a_departure(multimodal):
    with pytest.raises(TypeError, match="required keyword-only argument: 'departing'"):
        rl.LabelConstrained().bind(multimodal).route("doorstep", "corner-C").routes[0]


def test_it_says_what_it_searches(multimodal):
    # A product vertex is a stop and a state, which is what the search is over —
    # so the number it reports is not the stop count every other model gives.
    planner = rl.LabelConstrained().bind(multimodal)
    kind, size = planner.searches
    assert kind == "states"
    assert size == len(multimodal.compile()) * 2


# --- UCCH, which is the same answers with the walking contracted first --------


def test_ucch_answers_as_the_search_it_accelerates(multimodal):
    # The load-bearing test, and the only claim UCCH makes: it is a speedup for
    # LabelConstrained, so where they differ the hierarchy has lost something.
    plain = rl.LabelConstrained().bind(multimodal)
    quick = rl.UCCH().bind(multimodal)
    ridden = 0
    for origin in ("doorstep", "corner-A", "corner-B", "A"):
        for destination in ("corner-B", "corner-C", "doorstep", "C"):
            for at in (time(7, 30), time(7, 55), time(12, 0)):
                mine = quick.route(origin, destination, departing=at).routes[0]
                theirs = plain.route(origin, destination, departing=at).routes[0]
                assert (mine is None) == (theirs is None), f"{origin}->{destination} at {at}"
                if mine is not None:
                    assert mine.arrives == theirs.arrives, f"{origin}->{destination} at {at}"
                    ridden += any(leg.trip is not None for leg in mine.legs)
    assert ridden >= 5, f"only {ridden} of these journeys rode anything"


def test_ucch_contracts_the_walking_and_keeps_the_rest(multimodal):
    # Every stop and every corner a stop is linked to survives, whatever its
    # importance — that rule is what makes it UCCH rather than a hierarchy that
    # bakes the language in. What is left over is the doorstep.
    planner = rl.UCCH().bind(multimodal)
    compiled = multimodal.compile()
    assert planner.num_core < len(compiled), "nothing was contracted"
    for label in ("A", "B", "C", "corner-A", "corner-B", "corner-C"):
        assert planner.hierarchy.is_core(planner.node_id(label)), f"{label} was contracted"
    kind, size = planner.searches
    assert kind == "states"
    assert size == planner.num_core * 2


def test_ucch_tells_a_shortcut_as_the_arcs_it_stands_for(multimodal):
    # A contracted walk is a path, not an arc, so the journey has to be told hop
    # by hop or a caller cannot draw it. Every leg joins the one before it.
    journey = rl.UCCH().bind(multimodal).route("doorstep", "corner-C", departing=time(7, 55)).routes[0]
    assert journey is not None
    for before, after in zip(journey.legs, journey.legs[1:]):
        assert before.head == after.tail
    assert journey.legs[0].tail == "doorstep"
    assert journey.legs[-1].head == "corner-C"
    assert journey.legs[-1].arrives == journey.arrives
    assert any(leg.trip is not None for leg in journey.legs), "it rode something"


def test_ucch_on_a_network_with_nothing_to_contract_is_the_plain_model(env):
    # No streets and nothing joining them, so there are no transfer nodes and no
    # core distinct from the network. That is the plain model rather than a
    # refusal, for the reason ULTRA's Transfers gives, and the answers are the
    # ones LabelConstrained gives.
    assert rl.UCCH().missing_from(env.compile()) == frozenset()
    quick = rl.UCCH().bind(env)
    plain = rl.LabelConstrained().bind(env)
    for at in (time(8, 0), time(12, 0)):
        mine = quick.route("A", "C", departing=at).routes
        theirs = plain.route("A", "C", departing=at).routes
        assert bool(mine) == bool(theirs)
        if mine:
            assert mine[0].arrives == theirs[0].arrives
