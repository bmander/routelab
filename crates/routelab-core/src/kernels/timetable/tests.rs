//! The paper's claim is that two very different graphs answer the same
//! question, so that is what is tested: the models against each other, and both
//! against an oracle that enumerates itineraries the obvious way.

use super::{earliest_arrival, TimeExpanded};
use crate::kernels::raptor::Raptor;
use crate::model::graph::{NodeId, UNREACHABLE};
use crate::model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Time, Timetable, Transfer, Walk,
};
use crate::util::rng::Rng;

/// Stops 0 -> 1 -> 2, with a fast trip and a slow one.
///
///   trip 1: leaves 0 at 08:00, reaches 1 at 08:10, leaves 08:12, reaches 2 at 08:30
///   trip 2: leaves 0 at 08:05, reaches 1 at 08:20  (slower, and a dead end)
///   trip 3: leaves 1 at 08:15, reaches 2 at 08:20  (the good connection)
/// A connection, spelled shortly enough to read a timetable off the page.
fn c(trip: u32, from: NodeId, to: NodeId, departs: Time, arrives: Time) -> Connection {
    Connection {
        trip,
        from,
        to,
        departs,
        arrives,
    }
}

fn town() -> Timetable {
    Timetable::new(
        3,
        [
            c(1, 0, 1, 28_800, 29_400),
            c(1, 1, 2, 29_520, 30_600),
            c(2, 0, 1, 29_100, 30_000),
            c(3, 1, 2, 29_700, 30_000),
        ],
    )
}

/// Every itinerary, the obvious way — an oracle by exhaustion rather than a
/// second copy of either model.
fn best_by_brute_force(
    timetable: &Timetable,
    footpaths: &Footpaths,
    here: NodeId,
    now: Time,
    to: NodeId,
    left: usize,
) -> Option<Time> {
    if here == to {
        return Some(now);
    }
    if left == 0 {
        return None;
    }
    let by_vehicle = timetable
        .connections()
        .iter()
        .filter(|c| c.from == here && c.departs >= now)
        .filter_map(|c| best_by_brute_force(timetable, footpaths, c.to, c.arrives, to, left - 1))
        .min();
    let on_foot = footpaths
        .from(here)
        .filter_map(|(next, walk)| {
            best_by_brute_force(timetable, footpaths, next, now + walk, to, left - 1)
        })
        .min();
    [by_vehicle, on_foot].into_iter().flatten().min()
}

fn random_timetable(seed: u64, stops: u32, trips: u32) -> Timetable {
    let mut rng = Rng::new(seed);
    let mut connections = Vec::new();
    for trip in 0..trips {
        let mut stop = rng.below(u64::from(stops)) as NodeId;
        let mut now = rng.below(3600) as Time;
        // A trip is a short chain of hops, which is what a bus route is.
        for _ in 0..(1 + rng.below(4)) {
            let next = rng.below(u64::from(stops)) as NodeId;
            if next == stop {
                continue;
            }
            let departs = now + rng.below(600) as Time;
            let arrives = departs + 60 + rng.below(600) as Time;
            connections.push(c(trip, stop, next, departs, arrives));
            stop = next;
            now = arrives;
        }
    }
    Timetable::new(stops as usize, connections)
}

/// A scatter of short walks between random pairs of stops, both ways.
fn random_footpaths(seed: u64, stops: u32, count: u32) -> Footpaths {
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut links = Vec::new();
    for _ in 0..count {
        let a = rng.below(u64::from(stops)) as NodeId;
        let b = rng.below(u64::from(stops)) as NodeId;
        let walk = 30 + rng.below(300) as Time;
        links.push((a, b, walk));
        links.push((b, a, walk));
    }
    Footpaths::new(stops as usize, links)
}

fn expanded(timetable: &Timetable) -> TimeExpanded {
    TimeExpanded::build(timetable, Transfer::instant(), &Footpaths::none())
}

fn expanded_with(timetable: &Timetable, footpaths: &Footpaths) -> TimeExpanded {
    TimeExpanded::build(timetable, Transfer::instant(), footpaths)
}

fn raptor(timetable: &Timetable) -> Raptor {
    Raptor::build(timetable, Transfer::instant(), &Footpaths::none())
}

fn raptor_with(timetable: &Timetable, footpaths: &Footpaths) -> Raptor {
    Raptor::build(timetable, Transfer::instant(), footpaths)
}

// --- The timetable itself ---------------------------------------------------

#[test]
fn connections_that_go_back_in_time_are_dropped() {
    assert_eq!(
        Timetable::new(2, [c(0, 0, 1, 100, 50)]).num_connections(),
        0
    );
}

#[test]
fn connections_off_the_end_of_the_stop_list_are_dropped() {
    assert_eq!(
        Timetable::new(2, [c(0, 0, 7, 100, 200)]).num_connections(),
        0
    );
}

#[test]
fn edges_collect_the_connections_that_run_along_them() {
    let table = town();
    let from_zero: Vec<_> = table.edges_from(0).collect();
    assert_eq!(from_zero.len(), 1, "0 -> 1 is one edge");
    let (to, connections, _first) = from_zero[0];
    assert_eq!(to, 1);
    assert_eq!(connections.len(), 2, "two trips run it");
    assert!(
        connections[0].departs <= connections[1].departs,
        "in departure order, which is what a search reads"
    );
}

// --- The two models ---------------------------------------------------------

#[test]
fn both_models_find_the_connection_worth_making() {
    // Leaving 0 at 08:00 you take trip 1, and at stop 1 you could stay aboard
    // until 08:30 or step onto trip 3 and be there at 08:20. The second is
    // better and every model should say so.
    let table = town();
    let by_events = expanded(&table)
        .earliest_arrival(&[(0, 28_800)], 2)
        .unwrap();
    let by_stops = earliest_arrival(
        &table,
        &[(0, 28_800)],
        2,
        Transfer::instant(),
        &Footpaths::none(),
    )
    .unwrap();

    assert_eq!(by_events.arrives, 30_000, "08:20");
    assert_eq!(by_stops.arrives, by_events.arrives);
    assert_eq!(by_events.transfers(), 1, "trip 1, then trip 3");
}

#[test]
fn leaving_later_cannot_arrive_earlier() {
    let table = town();
    let expanded = expanded(&table);
    let mut previous = 0;
    for minute in 0..60 {
        let at = 28_800 + minute * 60;
        if let Some(trip) = expanded.earliest_arrival(&[(0, at)], 2) {
            assert!(trip.arrives >= previous, "departing {at}");
            previous = trip.arrives;
        }
    }
}

#[test]
fn a_stop_you_cannot_reach_has_no_itinerary() {
    let table = town();
    // Nothing runs to stop 0.
    assert!(expanded(&table).earliest_arrival(&[(2, 0)], 0).is_none());
    assert!(earliest_arrival(
        &table,
        &[(2, 0)],
        0,
        Transfer::instant(),
        &Footpaths::none()
    )
    .is_none());
}

#[test]
fn arriving_after_the_last_departure_finds_nothing() {
    let table = town();
    assert!(expanded(&table)
        .earliest_arrival(&[(0, 60_000)], 2)
        .is_none());
    assert!(earliest_arrival(
        &table,
        &[(0, 60_000)],
        2,
        Transfer::instant(),
        &Footpaths::none()
    )
    .is_none());
}

#[test]
fn the_two_models_agree_on_everything() {
    // The paper's thesis, and this module's load-bearing test. Neither model is
    // the reference: each is the other's.
    for seed in 0..12 {
        let table = random_timetable(seed, 12, 25);
        let expanded = expanded(&table);
        let rounds = raptor(&table);
        for from in 0..12u32 {
            for to in 0..12u32 {
                for at in [0, 900, 1800, 3600] {
                    let by_events = expanded.earliest_arrival(&[(from, at)], to);
                    let by_stops = earliest_arrival(
                        &table,
                        &[(from, at)],
                        to,
                        Transfer::instant(),
                        &Footpaths::none(),
                    );
                    let by_rounds = rounds.earliest_arrival(from, at, to);
                    assert_eq!(
                        by_events.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: {from} -> {to} at {at}"
                    );
                    assert_eq!(
                        by_rounds.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: RAPTOR, {from} -> {to} at {at}"
                    );
                }
            }
        }
    }
}

#[test]
fn both_models_agree_with_brute_force() {
    // An oracle by a different method, not a third copy of the same one.
    for seed in 0..6 {
        let table = random_timetable(seed, 6, 10);
        let expanded = expanded(&table);
        let rounds = raptor(&table);
        for from in 0..6u32 {
            for to in 0..6u32 {
                let truth = best_by_brute_force(&table, &Footpaths::none(), from, 0, to, 6);
                assert_eq!(
                    expanded
                        .earliest_arrival(&[(from, 0)], to)
                        .map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-expanded, {from} -> {to}"
                );
                assert_eq!(
                    earliest_arrival(
                        &table,
                        &[(from, 0)],
                        to,
                        Transfer::instant(),
                        &Footpaths::none()
                    )
                    .map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-dependent, {from} -> {to}"
                );
                assert_eq!(
                    rounds.earliest_arrival(from, 0, to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: RAPTOR, {from} -> {to}"
                );
            }
        }
    }
}

#[test]
fn every_itinerary_returned_is_one_you_could_actually_ride() {
    // The falsifiability check: the rides must join up, in order, in time.
    for seed in 0..12 {
        let table = random_timetable(seed, 10, 20);
        let expanded = expanded(&table);
        let rounds = raptor(&table);
        for from in 0..10u32 {
            for to in 0..10u32 {
                for query in [
                    expanded.earliest_arrival(&[(from, 0)], to),
                    earliest_arrival(
                        &table,
                        &[(from, 0)],
                        to,
                        Transfer::instant(),
                        &Footpaths::none(),
                    ),
                    rounds.earliest_arrival(from, 0, to),
                ] {
                    let Some(itinerary) = query else { continue };
                    assert!(
                        itinerary.is_valid(&[(from, 0)], Transfer::instant(), &Footpaths::none()),
                        "seed {seed}: {from} -> {to} returned {itinerary:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_expanded_graph_is_bigger_than_the_network_and_that_is_the_trade() {
    // Worth pinning, because it is the paper's central comparison: the same
    // timetable is a handful of stops or a great many events.
    let table = random_timetable(1, 12, 40);
    let expanded = expanded(&table);
    assert!(expanded.num_events() > table.num_stops() * 4);
}

#[test]
fn you_are_already_where_you_already_are() {
    // Both models must answer "now", and the expanded one has to be told: its
    // answers are arrival events, and standing still is not an event.
    let table = town();
    let expanded = expanded(&table);
    let rounds = raptor(&table);
    for at in [0, 28_800, 60_000] {
        for answer in [
            expanded.earliest_arrival(&[(1, at)], 1),
            earliest_arrival(
                &table,
                &[(1, at)],
                1,
                Transfer::instant(),
                &Footpaths::none(),
            ),
            rounds.earliest_arrival(1, at, 1),
        ] {
            let answer = answer.expect("standing still is an answer");
            assert_eq!(answer.arrives, at);
            assert!(answer.legs.is_empty());
            assert!(answer.is_valid(&[(1, at)], Transfer::instant(), &Footpaths::none()));
        }
    }
    // Several sources: standing at the target already beats riding to it,
    // unless another source rides in sooner.
    let both = earliest_arrival(
        &table,
        &[(0, 28_800), (2, 40_000)],
        2,
        Transfer::instant(),
        &Footpaths::none(),
    )
    .unwrap();
    assert_eq!(both.arrives, 30_000, "riding from 0 arrives before 40_000");
    let both = expanded
        .earliest_arrival(&[(0, 28_800), (2, 29_000)], 2)
        .unwrap();
    assert_eq!(
        (both.arrives, both.legs.len()),
        (29_000, 0),
        "already there first"
    );
}

#[test]
fn stopping_at_the_first_target_finds_the_same_answer_as_exhausting() {
    // The expanded model asks for *any* arrival at the target rather than all
    // of them, which is only right because costs are absolute times and the
    // search settles in cost order. If that reasoning were wrong, it would show
    // up as an early exit on a later arrival — so check it against the model
    // that has no early exit to get wrong.
    for seed in 0..12 {
        let table = random_timetable(seed, 12, 25);
        let expanded = expanded(&table);
        for from in 0..12u32 {
            for to in 0..12u32 {
                assert_eq!(
                    expanded
                        .earliest_arrival(&[(from, 0)], to)
                        .map(|i| i.arrives),
                    earliest_arrival(
                        &table,
                        &[(from, 0)],
                        to,
                        Transfer::instant(),
                        &Footpaths::none()
                    )
                    .map(|i| i.arrives),
                    "seed {seed}: {from} -> {to}"
                );
            }
        }
    }
}

// --- Footpaths --------------------------------------------------------------

#[test]
fn footpaths_are_closed_under_composition() {
    // A -> B and B -> C make A -> C, at the sum; a shorter direct walk wins.
    let paths = Footpaths::new(4, [(0, 1, 60), (1, 2, 60), (0, 2, 200), (2, 3, 10)]);
    assert_eq!(
        paths.duration(0, 2),
        Some(120),
        "the two hops beat the direct 200"
    );
    assert_eq!(paths.duration(0, 3), Some(130));
    assert_eq!(paths.duration(1, 3), Some(70));
    assert_eq!(paths.duration(3, 0), None, "nothing goes the other way");
    assert_eq!(paths.duration(0, 0), None, "no walk to where you stand");
    assert_eq!(paths.len(), 6);
    // And a closed walk still knows the given links it chains.
    assert_eq!(paths.hops(0, 3), vec![(0, 1, 60), (1, 2, 60), (2, 3, 10)]);
    assert_eq!(paths.hops(0, 1), vec![(0, 1, 60)]);
    assert!(paths.is_given(0, 1) && !paths.is_given(0, 2));
    let told = paths.expand(Walk {
        from: 0,
        to: 3,
        departs: 100,
        arrives: 230,
    });
    assert_eq!(told.len(), 3);
    assert_eq!((told[0].departs, told[0].arrives), (100, 160));
    assert_eq!((told[2].from, told[2].to, told[2].arrives), (2, 3, 230));
}

#[test]
fn a_walk_reaches_a_stop_no_vehicle_does() {
    // Stop 3 is served by nothing, but it is a short walk from stop 2, and
    // every model must find it that way — and stop 0 to stop 3 by foot alone
    // when that is quicker than waiting for a bus.
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(2, 3, 120), (3, 2, 120), (0, 3, 300), (3, 0, 300)]);
    let by_events = expanded_with(&table, &paths);

    let ride_then_walk = by_events.earliest_arrival(&[(0, 28_800)], 3).unwrap();
    assert_eq!(
        ride_then_walk.arrives,
        28_800 + 300,
        "walking there beats any bus"
    );
    assert_eq!(ride_then_walk.legs.len(), 1);
    assert!(matches!(ride_then_walk.legs[0], Leg::Walk(_)));
    let by_stops =
        earliest_arrival(&table, &[(0, 28_800)], 3, Transfer::instant(), &paths).unwrap();
    assert_eq!(by_stops.arrives, ride_then_walk.arrives);

    // From stop 1 the only way to 3 is to ride to 2 and walk.
    let by_events = by_events.earliest_arrival(&[(1, 29_600)], 3).unwrap();
    let by_stops =
        earliest_arrival(&table, &[(1, 29_600)], 3, Transfer::instant(), &paths).unwrap();
    assert_eq!(
        by_events.arrives,
        30_000 + 120,
        "trip 3 to stop 2, then the walk"
    );
    assert_eq!(by_stops.arrives, by_events.arrives);
    assert_eq!(by_events.legs.len(), 2);
    assert!(matches!(by_events.legs[1], Leg::Walk(_)));
    assert_eq!(
        by_events.transfers(),
        0,
        "one bus and a walk is not a change"
    );
    for itinerary in [&by_events, &by_stops] {
        assert!(itinerary.is_valid(&[(1, 29_600)], Transfer::instant(), &paths));
    }
}

#[test]
fn a_walk_from_the_origin_can_reach_a_better_departure() {
    // Nothing leaves stop 3, but stop 0 is a walk away and trip 1 leaves it.
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(3, 0, 60), (0, 3, 60)]);
    let by_events = expanded_with(&table, &paths)
        .earliest_arrival(&[(3, 28_700)], 2)
        .unwrap();
    let by_stops =
        earliest_arrival(&table, &[(3, 28_700)], 2, Transfer::instant(), &paths).unwrap();
    assert_eq!(
        by_events.arrives, 30_000,
        "walk to 0 by 08:00, ride as before"
    );
    assert_eq!(by_stops.arrives, by_events.arrives);
    assert!(matches!(by_events.legs[0], Leg::Walk(_)));
    assert!(matches!(by_stops.legs[0], Leg::Walk(_)));
    assert!(by_events.is_valid(&[(3, 28_700)], Transfer::instant(), &paths));
    assert!(by_stops.is_valid(&[(3, 28_700)], Transfer::instant(), &paths));
}

#[test]
fn the_two_models_agree_with_footpaths_too() {
    // The load-bearing test again, now with walks: one model chains them one
    // hop at a time in the search, the other takes them in a single hop from
    // a closed set, and they must still never disagree.
    for seed in 0..12 {
        let table = random_timetable(seed, 12, 25);
        let paths = random_footpaths(seed, 12, 8);
        let expanded = expanded_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        for from in 0..12u32 {
            for to in 0..12u32 {
                for at in [0, 900, 1800, 3600] {
                    let by_events = expanded.earliest_arrival(&[(from, at)], to);
                    let by_stops =
                        earliest_arrival(&table, &[(from, at)], to, Transfer::instant(), &paths);
                    let by_rounds = rounds.earliest_arrival(from, at, to);
                    assert_eq!(
                        by_events.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: {from} -> {to} at {at}"
                    );
                    assert_eq!(
                        by_rounds.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: RAPTOR, {from} -> {to} at {at}"
                    );
                    for itinerary in [by_events, by_stops, by_rounds].into_iter().flatten() {
                        assert!(
                            itinerary.is_valid(&[(from, at)], Transfer::instant(), &paths),
                            "seed {seed}: {from} -> {to} at {at} returned {itinerary:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn both_models_agree_with_brute_force_when_walking() {
    for seed in 0..6 {
        let table = random_timetable(seed, 6, 10);
        let paths = random_footpaths(seed, 6, 3);
        let expanded = expanded_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                let truth = best_by_brute_force(&table, &paths, from, 0, to, 6);
                assert_eq!(
                    expanded
                        .earliest_arrival(&[(from, 0)], to)
                        .map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-expanded, {from} -> {to}"
                );
                assert_eq!(
                    earliest_arrival(&table, &[(from, 0)], to, Transfer::instant(), &paths)
                        .map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-dependent, {from} -> {to}"
                );
                assert_eq!(
                    rounds.earliest_arrival(from, 0, to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: RAPTOR, {from} -> {to}"
                );
            }
        }
    }
}

#[test]
fn the_two_models_agree_with_dense_footpaths_over_many_seeds() {
    // Denser walks and more seeds than the test above, because the expanded
    // model's finish-on-foot rule is the kind of thing that is right on
    // twelve seeds and wrong on the thirteenth.
    for seed in 0..60 {
        let table = random_timetable(seed, 10, 20);
        let paths = random_footpaths(seed, 10, 12);
        let expanded = expanded_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        for from in 0..10u32 {
            for to in 0..10u32 {
                for at in [0, 600, 1500, 2700] {
                    let by_events = expanded.earliest_arrival(&[(from, at)], to);
                    let by_stops =
                        earliest_arrival(&table, &[(from, at)], to, Transfer::instant(), &paths);
                    let by_rounds = rounds.earliest_arrival(from, at, to);
                    assert_eq!(
                        by_events.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: {from} -> {to} at {at}"
                    );
                    assert_eq!(
                        by_rounds.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: RAPTOR, {from} -> {to} at {at}"
                    );
                    for itinerary in [by_events, by_stops, by_rounds].into_iter().flatten() {
                        assert!(
                            itinerary.is_valid(&[(from, at)], Transfer::instant(), &paths),
                            "seed {seed}: {from} -> {to} at {at} returned {itinerary:?}"
                        );
                    }
                }
            }
        }
    }
}

// --- RAPTOR -----------------------------------------------------------------

/// Every itinerary with at most `trips_left` boardings and `hops_left` legs,
/// the obvious way: a boarding is consumed when the trip changes, a walk puts
/// you off whatever you were on.
#[allow(clippy::too_many_arguments)]
fn best_by_brute_force_with_trips(
    timetable: &Timetable,
    footpaths: &Footpaths,
    here: NodeId,
    now: Time,
    aboard: Option<u32>,
    to: NodeId,
    hops_left: usize,
    trips_left: usize,
) -> Option<Time> {
    if here == to {
        return Some(now);
    }
    if hops_left == 0 {
        return None;
    }
    let by_vehicle = timetable
        .connections()
        .iter()
        .filter(|c| c.from == here && c.departs >= now)
        .filter_map(|c| {
            let boarding = aboard != Some(c.trip);
            if boarding && trips_left == 0 {
                return None;
            }
            best_by_brute_force_with_trips(
                timetable,
                footpaths,
                c.to,
                c.arrives,
                Some(c.trip),
                to,
                hops_left - 1,
                trips_left - usize::from(boarding),
            )
        })
        .min();
    let on_foot = footpaths
        .from(here)
        .filter_map(|(next, walk)| {
            best_by_brute_force_with_trips(
                timetable,
                footpaths,
                next,
                now + walk,
                None,
                to,
                hops_left - 1,
                trips_left,
            )
        })
        .min();
    [by_vehicle, on_foot].into_iter().flatten().min()
}

/// How many times an itinerary got onto a vehicle it was not already on.
fn boardings(itinerary: &Itinerary) -> usize {
    let mut aboard: Option<u32> = None;
    let mut count = 0;
    for leg in &itinerary.legs {
        match leg {
            Leg::Ride(ride) => {
                if aboard != Some(ride.trip) {
                    count += 1;
                    aboard = Some(ride.trip);
                }
            }
            Leg::Walk(_) => aboard = None,
        }
    }
    count
}

#[test]
fn raptor_reads_town_as_three_routes() {
    // One feed's line, three stop sequences: 0-1-2, 0-1 and 1-2 are three
    // routes in the paper's sense, each with one trip.
    let table = town();
    let rounds = raptor(&table);
    assert_eq!(rounds.num_routes(), 3);
    assert_eq!(rounds.num_trips(), 3);
    assert_eq!(rounds.num_connections(), 4);
    assert_eq!(rounds.num_stops(), 3);
    assert!(rounds.footprint() > 0);
}

#[test]
fn changing_at_b_is_a_second_round() {
    // Leaving 0 at 08:00: staying on trip 1 reaches 2 at 08:30 with no
    // change; stepping onto trip 3 at stop 1 reaches it at 08:20 with one.
    // Both are worth having, and RAPTOR says so — one per round.
    let table = town();
    let rounds = raptor(&table);
    let front = rounds.pareto(0, 28_800, 2);
    assert_eq!(front.len(), 2);
    assert_eq!((front[0].arrives, boardings(&front[0])), (30_600, 1));
    assert_eq!((front[1].arrives, boardings(&front[1])), (30_000, 2));
    assert_eq!(front[1].transfers(), 1);
    assert_eq!(
        front[1].legs.len(),
        2,
        "one leg per connection: 0->1 on trip 1, 1->2 on trip 3"
    );
    let best = rounds.earliest_arrival(0, 28_800, 2).unwrap();
    assert_eq!(best.arrives, 30_000);
    for itinerary in front {
        assert!(itinerary.is_valid(&[(0, 28_800)], Transfer::instant(), &Footpaths::none()));
    }
}

#[test]
fn a_walk_only_journey_is_round_zero() {
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(0, 3, 300), (3, 0, 300)]);
    let rounds = raptor_with(&table, &paths);
    let front = rounds.pareto(0, 28_800, 3);
    assert_eq!(front.len(), 1);
    assert_eq!(front[0].arrives, 28_800 + 300);
    assert!(front[0].rides().next().is_none());
    let search = rounds.search(&[(0, 28_800)], None, None, None);
    assert_eq!(
        search.round_reached(3),
        Some(0),
        "reached before any vehicle"
    );
    assert_eq!(search.round_reached(2), Some(1), "one ride away");
}

#[test]
fn overtaking_trips_are_split_into_two_routes() {
    // Trip 1 leaves 0 first but trip 2 overtakes it: same stops, two routes,
    // and a query catches the one that lands first.
    let table = Timetable::new(
        3,
        [
            c(1, 0, 1, 100, 200),
            c(1, 1, 2, 200, 400),
            c(2, 0, 1, 110, 190),
            c(2, 1, 2, 190, 300),
        ],
    );
    let rounds = raptor(&table);
    assert_eq!(rounds.num_routes(), 2);
    let best = rounds.earliest_arrival(0, 100, 2).unwrap();
    assert_eq!(best.arrives, 300);
    assert_eq!(
        best.arrives,
        earliest_arrival(
            &table,
            &[(0, 100)],
            2,
            Transfer::instant(),
            &Footpaths::none()
        )
        .unwrap()
        .arrives
    );
}

#[test]
fn a_trip_that_revisits_a_stop_can_be_boarded_at_the_later_visit() {
    // Trip 7 goes 0 -> 1 -> 0 -> 2. From 0 at 250 the first visit has gone,
    // and the second is the ride to 2.
    let table = Timetable::new(
        3,
        [
            c(7, 0, 1, 100, 200),
            c(7, 1, 0, 220, 300),
            c(7, 0, 2, 320, 400),
        ],
    );
    let rounds = raptor(&table);
    assert_eq!(rounds.num_routes(), 1);
    for (from, at, legs) in [(0, 250, 1), (0, 50, 3), (1, 210, 2)] {
        let itinerary = rounds.earliest_arrival(from, at, 2).unwrap();
        assert_eq!(itinerary.arrives, 400, "{from} at {at}");
        assert_eq!(itinerary.legs.len(), legs, "{from} at {at}");
        assert!(itinerary.is_valid(&[(from, at)], Transfer::instant(), &Footpaths::none()));
        assert_eq!(
            itinerary.arrives,
            earliest_arrival(
                &table,
                &[(from, at)],
                2,
                Transfer::instant(),
                &Footpaths::none()
            )
            .unwrap()
            .arrives
        );
    }
}

#[test]
fn a_broken_trip_chain_is_two_trips() {
    // One trip id whose hops do not join up: a rider cannot ride 0 -> 3.
    let table = Timetable::new(4, [c(1, 0, 1, 100, 200), c(1, 2, 3, 300, 400)]);
    let rounds = raptor(&table);
    assert_eq!(rounds.num_trips(), 2);
    assert!(rounds.earliest_arrival(0, 0, 3).is_none());
}

#[test]
fn the_pareto_front_matches_a_trip_counting_oracle() {
    // For each cap on boardings, the oracle's best arrival; RAPTOR's front
    // must hold exactly the caps that improve on the one before, at exactly
    // those arrivals — which is also the proof that a round rides one trip.
    for seed in 0..8 {
        let table = random_timetable(seed, 6, 10);
        let paths = random_footpaths(seed, 6, 3);
        let rounds = raptor_with(&table, &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                for at in [0, 900] {
                    let front = rounds.pareto(from, at, to);
                    let mut expected = Vec::new();
                    let mut previous = None;
                    for trips in 0..=6usize {
                        let best = best_by_brute_force_with_trips(
                            &table, &paths, from, at, None, to, 8, trips,
                        );
                        if let Some(arrives) = best {
                            if arrives < previous.unwrap_or(UNREACHABLE) {
                                expected.push((trips, arrives));
                            }
                            previous = Some(arrives);
                        }
                    }
                    let got: Vec<(usize, Time)> =
                        front.iter().map(|i| (boardings(i), i.arrives)).collect();
                    assert_eq!(got, expected, "seed {seed}: {from} -> {to} at {at}");
                    for itinerary in &front {
                        assert!(
                            itinerary.is_valid(&[(from, at)], Transfer::instant(), &paths),
                            "seed {seed}: {from} -> {to} at {at} returned {itinerary:?}"
                        );
                    }
                    if let Some(last) = front.last() {
                        assert_eq!(
                            Some(last.arrives),
                            earliest_arrival(
                                &table,
                                &[(from, at)],
                                to,
                                Transfer::instant(),
                                &paths
                            )
                            .map(|i| i.arrives)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn max_rounds_caps_the_journeys() {
    let table = town();
    let rounds = raptor(&table);
    let capped = rounds.search(&[(0, 28_800)], Some(2), Some(1), None);
    let front = rounds.itineraries(&capped, 2);
    assert_eq!(front.len(), 1, "one round: the through ride only");
    assert_eq!(front[0].arrives, 30_600);
    let none = rounds.search(&[(0, 28_800)], Some(2), Some(0), None);
    assert!(
        rounds.itinerary(&none, 2).is_none(),
        "no rounds: nowhere to ride"
    );
}

#[test]
fn several_sources_each_carry_their_own_time() {
    let table = town();
    let rounds = raptor(&table);
    // From 0 at 08:00 the answer is 08:20; standing at 1 from 08:13 as well
    // (trip 3 leaves it at 08:15) it is still 08:20 — the pair answers as the
    // better single source does.
    let both = rounds.search(&[(0, 28_800), (1, 29_600)], Some(2), None, None);
    let one = rounds.search(&[(0, 28_800)], Some(2), None, None);
    assert_eq!(
        rounds.itinerary(&both, 2).unwrap().arrives,
        rounds.itinerary(&one, 2).unwrap().arrives
    );
    // And a source that is already ahead wins outright.
    let ahead = rounds.search(&[(0, 28_800), (1, 29_000)], Some(2), None, None);
    let itinerary = rounds.itinerary(&ahead, 2).unwrap();
    assert_eq!(itinerary.arrives, 30_000);
    assert_eq!(
        itinerary.legs.len(),
        1,
        "boarded trip 3 at stop 1 without riding from 0"
    );
    assert_eq!(itinerary.legs[0].from(), 1);
}

#[test]
fn a_search_without_a_target_labels_every_stop_the_time_dependent_model_reaches() {
    for seed in 0..6 {
        let table = random_timetable(seed, 10, 20);
        let paths = random_footpaths(seed, 10, 4);
        let rounds = raptor_with(&table, &paths);
        for from in 0..10u32 {
            let search = rounds.search(&[(from, 0)], None, None, None);
            for to in 0..10u32 {
                let truth = earliest_arrival(&table, &[(from, 0)], to, Transfer::instant(), &paths)
                    .map(|i| i.arrives);
                assert_eq!(search.cost(to), truth, "seed {seed}: {from} -> {to}");
                if let Some(arrives) = truth {
                    assert!(search.round_reached(to).is_some());
                    let itinerary = rounds.itinerary(&search, to).unwrap();
                    assert!(itinerary.is_valid(&[(from, 0)], Transfer::instant(), &paths));
                    assert_eq!(itinerary.arrives, arrives);
                }
            }
            assert_eq!(search.settled, search.reached().len());
            assert!(search.scanned > 0 || search.settled <= 1 + paths.len());
        }
    }
}

#[test]
fn the_three_models_agree_from_several_sources() {
    // A multimodal query starts at several stops, each already reached at its
    // own time. All three must take whichever wins, and say so the same way.
    for seed in 0..8 {
        let table = random_timetable(seed, 10, 20);
        let paths = random_footpaths(seed, 10, 4);
        let expanded = expanded_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        for a in 0..10u32 {
            let b = (a + 3) % 10;
            let sources = [(a, 0), (b, 400)];
            for to in 0..10u32 {
                let by_stops = earliest_arrival(&table, &sources, to, Transfer::instant(), &paths);
                let by_events = expanded.earliest_arrival(&sources, to);
                let by_rounds =
                    rounds.itinerary(&rounds.search(&sources, Some(to), None, None), to);
                let want = by_stops.as_ref().map(|i| i.arrives);
                assert_eq!(
                    by_events.as_ref().map(|i| i.arrives),
                    want,
                    "seed {seed}: {sources:?} -> {to}"
                );
                assert_eq!(
                    by_rounds.as_ref().map(|i| i.arrives),
                    want,
                    "seed {seed}: RAPTOR {sources:?} -> {to}"
                );
                for itinerary in [by_stops, by_events, by_rounds].into_iter().flatten() {
                    assert!(
                        itinerary.is_valid(&sources, Transfer::instant(), &paths),
                        "seed {seed}: {sources:?} -> {to}: {itinerary:?}"
                    );
                }
            }
        }
    }
}
