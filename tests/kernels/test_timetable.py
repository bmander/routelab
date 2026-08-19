"""Timetable routing, end to end: a GTFS feed becomes a layer becomes a journey.

Pyrga, Schulz, Wagner & Zaroliagis (2007) give two models of the same timetable,
and the load-bearing claim is that they agree. The kernels check that against
random instances and a brute-force oracle in Rust; what is checked here is the
veneer — that a feed reaches both models with its stops, times and trips intact,
and that the answers come back as journeys someone could act on.
"""

from __future__ import annotations

from datetime import date, time

import pytest

import routelab as rl
from routelab import GTFS

from conftest import TINY_GTFS

#: Every timetable technique. Every behavioural test runs against each,
#: because "these two agree" is the paper's thesis and a test that only asked
#: one would not notice the day they stopped. Written out because a technique
#: is a configuration and some take arguments, and checked against the shelf
#: in test_planners.py so a list somebody forgot to extend fails there.
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

#: The ones that answer with an itinerary and nothing else, so a cost table
#: and a search space are things they refuse: Pyrga's two; PTL, whose labels
#: answer a pair of stops rather than filling a table; and LabelConstrained,
#: whose table is over the product of stops and automaton states, which is not
#: a cost per stop and would be a lie told in the right shape.
ITINERARY_ONLY = [rl.TimeDependent, rl.TimeExpanded, rl.PTL, rl.LabelConstrained, rl.UCCH]


def test_the_feed_reads_its_service_day(feed: GTFS):
    # WEEKDAY1's two connections plus WEEKDAY2's one plus NIGHT1's one. The
    # weekend trip is not running, which is the date filter doing its job.
    assert feed.feed.num_connections == 4
    assert set(feed.names().values()) == {"Ash Street", "Birch Street", "Cedar Street"}
    assert feed.unimplemented == 0


def test_a_feed_compiles_to_stops_and_departures(env: rl.Environment):
    compiled = env.compile()
    assert len(compiled) == 3
    # A→B is served by two trips and B→C by two, so four connections collapse
    # onto two edges.
    assert compiled.graph.num_edges == 2
    assert rl.Departures().bind(compiled).num_connections == 4
    assert rl.Departures.missing_from(compiled) == frozenset()


def test_dijkstra_refuses_a_timetable(env: rl.Environment):
    # The refusal that makes the cost model worth declaring: A→B→C weighted by
    # its shortest rides is a real path and a wrong answer.
    with pytest.raises(TypeError, match="cannot route over timetable"):
        rl.Dijkstra().bind(env)


def test_a_timetable_technique_refuses_a_network_with_no_departures():
    plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    assert rl.TimeDependent().missing_from(plain.compile()) == frozenset({"timetable"})
    with pytest.raises(ValueError, match="needs a timetable"):
        rl.TimeDependent().bind(plain).route("a", "b", departing=0).routes[0]


@pytest.mark.parametrize("model", MODELS)
def test_changing_beats_staying_aboard(model, env: rl.Environment):
    """The fixture's whole point: leave A at 08:00 and reach C at 08:20.

    Staying on WEEKDAY1 arrives at 08:30. A model that only ever rides the trip
    it boarded, or that charges for a change it should not, gets that instead.
    """
    journey = model().bind(env).route("A", "C", departing=time(8, 0)).routes[0]
    assert journey.arrives == 8 * 3600 + 20 * 60
    assert journey.departs == 8 * 3600
    assert [leg.head for leg in journey.legs] == ["B", "C"]
    assert journey.transfers == 1
    # 20 minutes elapsed, of which 5 are spent standing at B between 08:10 and
    # 08:15 — which a static search could only ever report as a residual.
    assert journey.cost == 20 * 60
    assert journey.waiting == 5 * 60
    assert journey.moving == 15 * 60


@pytest.mark.parametrize("model", MODELS)
def test_a_missed_last_departure_is_no_journey(model, env: rl.Environment):
    assert model().bind(env).route("A", "C", departing=time(12, 0)).routes == []


@pytest.mark.parametrize("model", MODELS)
def test_arriving_where_you_started_costs_nothing(model, env: rl.Environment):
    journey = model().bind(env).route("A", "A", departing=time(8, 0)).routes[0]
    assert journey.cost == 0
    assert journey.legs == ()
    assert journey.transfers == 0


@pytest.mark.parametrize("model", MODELS)
def test_a_service_day_does_not_wrap_at_midnight(model, env: rl.Environment):
    """NIGHT1 leaves at 23:50 and arrives at 24:10, and that is one ride.

    Folding the arrival back to 00:10 would put it before its own departure,
    which is why :func:`routelab.clock.service_seconds` does not wrap the way
    the weekly clock does.
    """
    journey = model().bind(env).route("A", "B", departing=time(23, 0)).routes[0]
    assert journey.departs == 23 * 3600 + 50 * 60
    assert journey.arrives == 24 * 3600 + 10 * 60
    assert journey.cost == 70 * 60


@pytest.mark.parametrize("model", MODELS)
def test_a_leg_names_the_trip_it_rides(model, env: rl.Environment, feed: GTFS):
    journey = model().bind(env).route("A", "C", departing=time(8, 0)).routes[0]
    trips = feed.trips()
    assert [trips[leg.trip][0] for leg in journey.legs] == ["WEEKDAY1", "WEEKDAY2"]
    assert [trips[leg.trip][1] for leg in journey.legs] == ["R1", "R1"]


@pytest.mark.parametrize("model", MODELS)
def test_a_leg_traces_back_to_the_layer_that_supplied_it(
    model, env: rl.Environment, feed: GTFS
):
    journey = model().bind(env).route("A", "C", departing=time(8, 0)).routes[0]
    assert all(leg.source is feed for leg in journey.legs)
    # No shapes.txt yet, so a transit leg is a straight line between stops and
    # says so rather than guessing.
    assert all(leg.geometry is None for leg in journey.legs)


@pytest.mark.parametrize("model", MODELS)
def test_a_departure_time_is_required(model, env: rl.Environment):
    # No default, for the reason Zero() has to be asked for out loud: a
    # timetable query without a time is a different question, not this one
    # with a fallback. Python's own sentence, from the signature.
    with pytest.raises(TypeError, match="required keyword-only argument: 'departing'"):
        model().bind(env).route("A", "C").routes[0]


@pytest.mark.parametrize("model", MODELS)
def test_several_origins_each_carry_a_head_start(model, feed):
    """The multimodal query, the same way every planner takes it: several
    origins, each already this many seconds along when the query departs."""
    env = rl.Environment(feed, rl.ScalarEdges(("C", "B", 60)))
    planner = model().bind(env)
    # Standing at A and at C at 08:00: walking C -> B lands at 08:01, well
    # before WEEKDAY1 reaches B at 08:10.
    journey = planner.route({"A": 0, "C": 0}, "B", departing=time(8, 0)).routes[0]
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 60, 60)
    # A head start of 15 minutes at C is a walk landing at 08:16; A's bus wins.
    journey = planner.route({"A": 0, "C": 900}, "B", departing=time(8, 0)).routes[0]
    assert (journey.origin, journey.arrives, journey.cost) == ("A", 8 * 3600 + 600, 600)
    # And a head start of 5 minutes at C: the walk lands at 08:06, cost counted
    # from the query's departure, head start included.
    journey = planner.route({"A": 0, "C": 300}, "B", departing=time(8, 0)).routes[0]
    assert (journey.origin, journey.arrives, journey.cost) == ("C", 8 * 3600 + 360, 360)
    # A list of origins is several origins with no head start.
    assert planner.route(["A", "C"], "B", departing=time(8, 0)).routes[0].origin == "C"


@pytest.mark.parametrize("model", ITINERARY_ONLY)
def test_a_model_without_a_table_answers_with_a_journey_and_nothing_else(model, env: rl.Environment):
    # No cost table and no search space, and neither is a refusal a caller has
    # to run into: there is no `search` on the planner at all, so a type
    # checker says so before anything runs.
    planner = model().bind(env)
    assert not hasattr(planner, "search")
    answer = planner.route("A", "C", departing=time(8, 0))
    assert answer.raw is None
    with pytest.raises(NotImplementedError, match="keeps no search space"):
        answer.searchspace()


@pytest.mark.parametrize("model", MODELS)
def test_an_option_a_model_cannot_honour_is_not_on_its_signature(model, env: rl.Environment):
    # A knob a technique does not have is a keyword its `route` never
    # declared, so Python refuses it by name — no registry, and nothing that
    # can go stale as the shelf grows.
    planner = model().bind(env)
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_cost'"):
        planner.route("A", "C", departing=time(8, 0), max_cost=10).routes[0]
    if model not in (rl.RAPTOR, rl.TripBased, rl.ULTRA):
        with pytest.raises(TypeError, match="unexpected keyword argument 'max_transfers'"):
            planner.route("A", "C", departing=time(8, 0), max_transfers=1).routes[0]
    with pytest.raises(TypeError, match="unexpected keyword argument 'until'"):
        planner.route("A", "C", departing=time(8, 0), until=time(9, 0)).routes[0]


def test_the_expanded_model_pays_its_size_at_bind_time(env: rl.Environment):
    """Four connections, so eight events — the model's cost, stated up front.

    Nothing coincides in this fixture, so nothing merges. The dedup that would
    merge them matters at city scale, where two buses leaving one stop together
    must be one node or nobody could wait between them.
    """
    planner = rl.TimeExpanded().bind(env)
    assert planner.num_events == 8
    # Every model holds the timetable and the walks; the expanded one holds
    # its event graph on top, and says so.
    dependent = rl.TimeDependent().bind(env)
    assert dependent.footprint > 0
    assert planner.footprint > dependent.footprint
    assert (dependent.searches, planner.searches) == (("stops", 3), ("events", 8))


def test_a_date_nothing_runs_on_is_an_empty_day_not_an_error():
    """2026-09-04 is a Friday the fixture cancels, and adds weekend service to.

    Distinguishing "you asked for a holiday" from "the file is broken" is worth
    a test: the difference is a feed with no weekday connections rather than an
    exception.
    """
    feed = GTFS(TINY_GTFS, date(2026, 9, 4))
    assert feed.feed.num_connections == 1  # WEEKEND1 alone, by exception
    journey = rl.TimeDependent().bind(rl.Environment(feed)).route(
        "A", "B", departing=time(8, 0)
    ).routes[0]
    assert journey.arrives == 9 * 3600 + 20 * 60
