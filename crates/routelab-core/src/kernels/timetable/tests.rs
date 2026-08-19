//! The paper's claim is that two very different graphs answer the same
//! question, so that is what is tested: the models against each other, and both
//! against an oracle that enumerates itineraries the obvious way.

use super::{earliest_arrival, TimeDependentTechnique, TimeExpanded, TimeExpandedTechnique};
use crate::kernels::csa::{ConnectionScan, ConnectionScanTechnique, ScanQuery};
use crate::kernels::oracles::{
    best_by_brute_force, c, profile_by_brute_force, random_footpaths, random_timetable, town,
};
use crate::kernels::ptl::{PtlTechnique, PublicTransitLabeling};
use crate::kernels::raptor::{Raptor, RaptorQuery, RaptorTechnique};
use crate::kernels::tripbased::{TripBased, TripBasedQuery, TripBasedTechnique};
use crate::model::graph::{NodeId, UNREACHABLE};
use crate::model::technique::{EarliestArrival, Profiled, Technique, TransitNetwork};
use crate::model::timetable::{Footpaths, Itinerary, Leg, Time, Timetable, Transfer, TripId, Walk};
use crate::util::progress::Progress;

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

fn csa(timetable: &Timetable) -> ConnectionScan {
    ConnectionScan::build(timetable, Transfer::instant(), &Footpaths::none())
}

fn csa_with(timetable: &Timetable, footpaths: &Footpaths) -> ConnectionScan {
    ConnectionScan::build(timetable, Transfer::instant(), footpaths)
}

fn tripbased(timetable: &Timetable) -> TripBased {
    TripBased::build(timetable, Transfer::instant(), &Footpaths::none())
}

fn tripbased_with(timetable: &Timetable, footpaths: &Footpaths) -> TripBased {
    TripBased::build(timetable, Transfer::instant(), footpaths)
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
        let scan = csa(&table);
        let trips = tripbased(&table);
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
                    let by_rounds = rounds.earliest_arrival(&[(from, at)], to);
                    let by_scan = scan.earliest_arrival(&[(from, at)], to);
                    let by_trips = trips.earliest_arrival(&[(from, at)], to);
                    assert_eq!(
                        by_trips.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: trip-based, {from} -> {to} at {at}"
                    );
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
                    assert_eq!(
                        by_scan.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: CSA, {from} -> {to} at {at}"
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
        let scan = csa(&table);
        let trips = tripbased(&table);
        for from in 0..6u32 {
            for to in 0..6u32 {
                let truth = best_by_brute_force(&table, &Footpaths::none(), from, 0, to, 6);
                assert_eq!(
                    scan.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: CSA, {from} -> {to}"
                );
                assert_eq!(
                    trips.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: trip-based, {from} -> {to}"
                );
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
                    rounds.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: RAPTOR, {from} -> {to}"
                );
            }
        }
    }
}

/// Every vehicle an itinerary names is one the timetable actually runs.
///
/// [`Itinerary::is_valid`] cannot ask this — it is handed the footpaths and
/// the change rule, not the timetable — so a journey that rides the right bus
/// under the wrong name passes it, and passes the agreement tests too, since
/// those compare arrival times. Which bus you are on is most of the answer, so
/// it gets a check of its own.
fn rides_are_real(table: &Timetable, itinerary: &Itinerary) -> bool {
    itinerary.rides().all(|ride| table.holds(ride))
}

#[test]
fn every_vehicle_named_is_one_the_timetable_runs() {
    // The kernels number trips densely for themselves — a feed's trip whose
    // chain broke becomes several, and the layouts renumber what is left — so
    // every one of them has an internal index and a feed id in flight at once,
    // both of them plain integers. This is what says they never got swapped.
    for seed in 0..8 {
        let table = random_timetable(seed, 8, 16);
        let paths = random_footpaths(seed, 8, 3);
        let expanded = expanded_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        let scan = csa_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        let labels = PublicTransitLabeling::build(&table, Transfer::instant(), &paths);
        for from in 0..8u32 {
            for to in 0..8u32 {
                for at in [0, 900, 2400] {
                    for itinerary in [
                        expanded.earliest_arrival(&[(from, at)], to),
                        earliest_arrival(&table, &[(from, at)], to, Transfer::instant(), &paths),
                        rounds.earliest_arrival(&[(from, at)], to),
                        scan.earliest_arrival(&[(from, at)], to),
                        trips.earliest_arrival(&[(from, at)], to),
                        labels.earliest_arrival(&[(from, at)], to),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        assert!(
                            rides_are_real(&table, &itinerary),
                            "seed {seed}: {from} -> {to} at {at} rides a vehicle the \
                             timetable does not run: {itinerary:?}"
                        );
                    }
                    // And the fronts, which are read back off different
                    // pointers than the single best journey is.
                    for itinerary in rounds
                        .pareto(from, at, to)
                        .iter()
                        .chain(trips.pareto(from, at, to).iter())
                    {
                        assert!(
                            rides_are_real(&table, itinerary),
                            "seed {seed}: a front entry for {from} -> {to} at {at} \
                             rides a vehicle the timetable does not run"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_profile_names_vehicles_the_timetable_runs_too() {
    // The profiles read their journeys back off yet another set of pointers —
    // decisions for the scan, queue entries kept across runs for the sweep.
    for seed in 0..6 {
        let table = random_timetable(seed, 6, 12);
        let paths = random_footpaths(seed, 6, 3);
        let scan = csa_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        let labels = PublicTransitLabeling::build(&table, Transfer::instant(), &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                if from == to {
                    continue;
                }
                // PTL's profile rebuilds its journeys from the label pointers,
                // a third set again.
                for (_, itinerary) in labels.profile(from, to, 0, 12_000) {
                    assert!(
                        rides_are_real(&table, &itinerary),
                        "seed {seed}: a PTL profile entry for {from} -> {to} rides a \
                         vehicle the timetable does not run"
                    );
                }
                let profile = scan.profile(to, 0, Some(from));
                for (_, itinerary) in scan.journeys(&profile, from, 0, 12_000) {
                    assert!(
                        rides_are_real(&table, &itinerary),
                        "seed {seed}: CSA profile"
                    );
                }
                let profile = trips.profile(from, to, 0, 12_000);
                for (_, itinerary) in trips.journeys(&profile) {
                    assert!(
                        rides_are_real(&table, &itinerary),
                        "seed {seed}: trip-based profile"
                    );
                }
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
        let scan = csa(&table);
        let trips = tripbased(&table);
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
                    rounds.earliest_arrival(&[(from, 0)], to),
                    scan.earliest_arrival(&[(from, 0)], to),
                    trips.earliest_arrival(&[(from, 0)], to),
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
            rounds.earliest_arrival(&[(1, at)], 1),
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
        let scan = csa_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        for from in 0..12u32 {
            for to in 0..12u32 {
                for at in [0, 900, 1800, 3600] {
                    let by_events = expanded.earliest_arrival(&[(from, at)], to);
                    let by_stops =
                        earliest_arrival(&table, &[(from, at)], to, Transfer::instant(), &paths);
                    let by_rounds = rounds.earliest_arrival(&[(from, at)], to);
                    let by_scan = scan.earliest_arrival(&[(from, at)], to);
                    let by_trips = trips.earliest_arrival(&[(from, at)], to);
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
                    assert_eq!(
                        by_scan.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: CSA, {from} -> {to} at {at}"
                    );
                    assert_eq!(
                        by_trips.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: trip-based, {from} -> {to} at {at}"
                    );
                    for itinerary in [by_events, by_stops, by_rounds, by_scan, by_trips]
                        .into_iter()
                        .flatten()
                    {
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
        let scan = csa_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                let truth = best_by_brute_force(&table, &paths, from, 0, to, 6);
                assert_eq!(
                    trips.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: trip-based, {from} -> {to}"
                );
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
                    rounds.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: RAPTOR, {from} -> {to}"
                );
                assert_eq!(
                    scan.earliest_arrival(&[(from, 0)], to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: CSA, {from} -> {to}"
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
        let scan = csa_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        for from in 0..10u32 {
            for to in 0..10u32 {
                for at in [0, 600, 1500, 2700] {
                    let by_events = expanded.earliest_arrival(&[(from, at)], to);
                    let by_stops =
                        earliest_arrival(&table, &[(from, at)], to, Transfer::instant(), &paths);
                    let by_rounds = rounds.earliest_arrival(&[(from, at)], to);
                    let by_scan = scan.earliest_arrival(&[(from, at)], to);
                    let by_trips = trips.earliest_arrival(&[(from, at)], to);
                    assert_eq!(
                        by_trips.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: trip-based, {from} -> {to} at {at}"
                    );
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
                    assert_eq!(
                        by_scan.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: CSA, {from} -> {to} at {at}"
                    );
                    for itinerary in [by_events, by_stops, by_rounds, by_scan, by_trips]
                        .into_iter()
                        .flatten()
                    {
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
    aboard: Option<TripId>,
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
    let mut aboard: Option<TripId> = None;
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
    let best = rounds.earliest_arrival(&[(0, 28_800)], 2).unwrap();
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
    let search = rounds.search(&[(0, 28_800)], &RaptorQuery::default());
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
    let best = rounds.earliest_arrival(&[(0, 100)], 2).unwrap();
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
        let itinerary = rounds.earliest_arrival(&[(from, at)], 2).unwrap();
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
    assert!(rounds.earliest_arrival(&[(0, 0)], 3).is_none());
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
    let capped = rounds.search(
        &[(0, 28_800)],
        &RaptorQuery {
            target: Some(2),
            max_rounds: Some(1),
            departing: None,
        },
    );
    let front = rounds.itineraries(&capped, 2);
    assert_eq!(front.len(), 1, "one round: the through ride only");
    assert_eq!(front[0].arrives, 30_600);
    let none = rounds.search(
        &[(0, 28_800)],
        &RaptorQuery {
            target: Some(2),
            max_rounds: Some(0),
            departing: None,
        },
    );
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
    let both = rounds.search(&[(0, 28_800), (1, 29_600)], &RaptorQuery::to(2));
    let one = rounds.search(&[(0, 28_800)], &RaptorQuery::to(2));
    assert_eq!(
        rounds.itinerary(&both, 2).unwrap().arrives,
        rounds.itinerary(&one, 2).unwrap().arrives
    );
    // And a source that is already ahead wins outright.
    let ahead = rounds.search(&[(0, 28_800), (1, 29_000)], &RaptorQuery::to(2));
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
            let search = rounds.search(&[(from, 0)], &RaptorQuery::default());
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

/// A question put to every model — where you may start and when, and where
/// you are going — with the arrival the oracle says is best.
type Asked = (Vec<(NodeId, Time)>, NodeId, Option<Time>);

/// A pair of stops, with every departure worth taking between them.
type Window = (NodeId, NodeId, Vec<(Time, Time)>);

/// Every question the models are asked from several sources, with the answer
/// the time-dependent model gives — this file's oracle, and rideable itself.
///
/// Asked once per instance rather than once per model: it is the same
/// question whoever is answering, and enumerating it is far dearer than any
/// of the answers.
fn what_the_stops_say(table: &Timetable, paths: &Footpaths) -> Vec<Asked> {
    let mut asked = Vec::new();
    for a in 0..10u32 {
        let b = (a + 3) % 10;
        let sources = vec![(a, 0), (b, 400)];
        for to in 0..10u32 {
            let want = earliest_arrival(table, &sources, to, Transfer::instant(), paths);
            if let Some(itinerary) = &want {
                assert!(
                    itinerary.is_valid(&sources, Transfer::instant(), paths),
                    "the oracle's own answer: {sources:?} -> {to}: {itinerary:?}"
                );
            }
            asked.push((sources.clone(), to, want.map(|i| i.arrives)));
        }
    }
    asked
}

/// Hold one planner to those answers, and check every itinerary it returns
/// could actually be ridden.
fn agrees_with_stops<P: EarliestArrival>(
    name: &str,
    planner: &P,
    asked: &[Asked],
    paths: &Footpaths,
    seed: u64,
) {
    for (sources, to, want) in asked {
        let got = planner.earliest_arrival(sources, *to);
        assert_eq!(
            got.as_ref().map(|i| i.arrives),
            *want,
            "seed {seed}: {name} {sources:?} -> {to}"
        );
        if let Some(itinerary) = got {
            assert!(
                itinerary.is_valid(sources, Transfer::instant(), paths),
                "seed {seed}: {name} {sources:?} -> {to}: {itinerary:?}"
            );
        }
    }
}

#[test]
fn the_six_models_agree_from_several_sources() {
    // A multimodal query starts at several stops, each already reached at its
    // own time. Every model must take whichever wins, and say so the same way
    // — and now that they all answer through one trait, one loop asks them.
    for seed in 0..8 {
        let table = random_timetable(seed, 10, 20);
        let paths = random_footpaths(seed, 10, 4);
        let net = TransitNetwork {
            timetable: &table,
            transfer: Transfer::instant(),
            footpaths: &paths,
        };
        let progress = Progress::new();
        let asked = what_the_stops_say(&table, &paths);
        // The oracle bound through the trait. Not the tautology it looks:
        // what is under test is that binding wires this planner to the same
        // timetable, transfer and footpaths the free function was handed.
        let dependent = TimeDependentTechnique.bind(net, &progress).unwrap();
        agrees_with_stops("time-dependent", &dependent, &asked, &paths, seed);
        let expanded = TimeExpandedTechnique.bind(net, &progress).unwrap();
        agrees_with_stops("time-expanded", &expanded, &asked, &paths, seed);
        let rounds = RaptorTechnique.bind(net, &progress).unwrap();
        agrees_with_stops("RAPTOR", &rounds, &asked, &paths, seed);
        let scan = ConnectionScanTechnique.bind(net, &progress).unwrap();
        agrees_with_stops("CSA", &scan, &asked, &paths, seed);
        let trips = TripBasedTechnique::default().bind(net, &progress).unwrap();
        agrees_with_stops("trip-based", &trips, &asked, &paths, seed);
        let labels = PtlTechnique.bind(net, &progress).unwrap();
        agrees_with_stops("PTL", &labels, &asked, &paths, seed);
    }
}

// --- CSA --------------------------------------------------------------------

#[test]
fn csa_lays_town_out_as_one_array() {
    let table = town();
    let scan = csa(&table);
    assert_eq!(scan.num_connections(), 4);
    assert_eq!(scan.num_trips(), 3);
    assert_eq!(scan.num_stops(), 3);
    assert!(scan.footprint() > 0);
    // Leaving 0 at 08:00: trip 1 to stop 1, then trip 3 to stop 2 by 08:20 —
    // one change, and every leg is one connection.
    let best = scan.earliest_arrival(&[(0, 28_800)], 2).unwrap();
    assert_eq!(best.arrives, 30_000);
    assert_eq!(best.transfers(), 1);
    assert_eq!(best.legs.len(), 2);
    assert!(best.is_valid(&[(0, 28_800)], Transfer::instant(), &Footpaths::none()));
}

#[test]
fn the_stop_criterion_scans_less_than_the_whole_day() {
    let table = town();
    let scan = csa(&table);
    let toward = scan.search(&[(0, 28_800)], &ScanQuery::to(1));
    let everything = scan.search(&[(0, 28_800)], &ScanQuery::default());
    assert!(
        toward.scanned < everything.scanned,
        "{} < {}",
        toward.scanned,
        everything.scanned
    );
    assert_eq!(toward.cost(1), everything.cost(1));
    // One-to-all: every stop holds a label, and reads back as an itinerary.
    assert_eq!(everything.settled, 3);
    assert_eq!(everything.reached().len(), 3);
    assert_eq!(scan.itinerary(&everything, 2).unwrap().arrives, 30_000);
    assert_eq!(scan.path(&everything, 2), Some(vec![0, 1, 2]));
    // A start criterion: nothing before 08:05 is scanned when leaving then.
    let late = scan.search(&[(0, 29_100)], &ScanQuery::default());
    assert!(late.scanned < everything.scanned);
    // The departure an elapsed cost is measured from.
    assert_eq!(
        scan.search(
            &[(0, 100), (1, 50)],
            &ScanQuery {
                target: None,
                departing: Some(0)
            }
        )
        .departing,
        0
    );
    assert_eq!(
        scan.search(&[(0, 100), (1, 50)], &ScanQuery::default())
            .departing,
        50
    );
}

#[test]
fn a_broken_trip_chain_is_two_trips_to_the_scan_too() {
    // Trip 1's second hop leaves before its first arrives: no vehicle did
    // that, so the trip flag must not carry across the break.
    let table = Timetable::new(3, [c(1, 0, 1, 100, 200), c(1, 1, 2, 150, 250)]);
    let scan = csa(&table);
    assert_eq!(scan.num_trips(), 2);
    assert!(scan.earliest_arrival(&[(0, 0)], 2).is_none());
}

#[test]
fn a_walk_from_the_source_needs_no_connection_to_set_it_off() {
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(0, 3, 300), (3, 0, 300)]);
    let scan = csa_with(&table, &paths);
    let walked = scan.earliest_arrival(&[(0, 28_800)], 3).unwrap();
    assert_eq!(walked.arrives, 29_100);
    assert!(walked.rides().next().is_none());
    // And from 3, the walk to 0 catches trip 1.
    let ridden = scan.earliest_arrival(&[(3, 28_500)], 1).unwrap();
    assert_eq!(ridden.arrives, 29_400);
    assert!(ridden.is_valid(&[(3, 28_500)], Transfer::instant(), &paths));
}

#[test]
fn the_profile_matches_asking_at_every_second() {
    // A window past the last departure, so nothing leaves after it and the
    // oracle's "latest moment" is the profile's departure exactly.
    for seed in 0..6 {
        let table = random_timetable(seed, 8, 16);
        let paths = random_footpaths(seed, 8, 3);
        let scan = csa_with(&table, &paths);
        for to in 0..8u32 {
            let profile = scan.profile(to, 0, None);
            for from in 0..8u32 {
                if from == to {
                    continue;
                }
                let truth = profile_by_brute_force(&table, &paths, from, to, 0..=12_000);
                let pairs = scan.pairs(&profile, from, 0, 12_000);
                assert_eq!(pairs, truth, "seed {seed}: {from} -> {to}");
                assert_eq!(scan.walk(&profile, from), paths.duration(from, to));
                // Every pair is a journey you could ride, leaving when it says
                // and arriving when it says.
                for (dep, itinerary) in scan.journeys(&profile, from, 0, 12_000) {
                    assert!(
                        itinerary.is_valid(&[(from, dep)], Transfer::instant(), &paths),
                        "seed {seed}: {from} -> {to} at {dep}: {itinerary:?}"
                    );
                    assert_eq!(itinerary.legs.first().map(|leg| leg.departs()), Some(dep));
                    assert!(pairs.contains(&(dep, itinerary.arrives)));
                }
                // The one-to-one pruning changes what the other stops hold,
                // not what the source does.
                let pruned = scan.profile(to, 0, Some(from));
                assert_eq!(
                    scan.pairs(&pruned, from, 0, 12_000),
                    truth,
                    "seed {seed}: pruned {from} -> {to}"
                );
                assert!(pruned.settled <= profile.settled);
            }
        }
    }
}

/// Hold one planner's `departures` to the brute-force profile: the same
/// `(departs, arrives)` pairs, in the same order, each behind a journey you
/// could actually ride. A kernel that also tells journeys apart by changes
/// may list journeys a later departure dominates on time alone, because they
/// change less; projected onto (departs, arrives) its front is the oracle's.
fn departures_agree<P: Profiled>(
    name: &str,
    planner: &P,
    asked: &[Window],
    paths: &Footpaths,
    seed: u64,
) {
    for &(from, to, ref truth) in asked {
        let departures = planner.departures(from, to, 0, 12_000);
        // Project onto (departs, arrives) and drop what a later departure
        // with an equal or earlier arrival dominates.
        let mut pairs: Vec<(Time, Time)> = Vec::new();
        for (dep, itinerary) in departures.iter().rev() {
            if pairs
                .last()
                .is_none_or(|&(_, best)| itinerary.arrives < best)
            {
                pairs.push((*dep, itinerary.arrives));
            }
        }
        pairs.reverse();
        assert_eq!(&pairs, truth, "seed {seed}: {name} {from} -> {to}");
        for (dep, itinerary) in departures {
            assert!(
                itinerary.is_valid(&[(from, dep)], Transfer::instant(), paths),
                "seed {seed}: {name} {from} -> {to} at {dep}: {itinerary:?}"
            );
        }
    }
}

#[test]
fn every_profiled_technique_agrees_with_the_oracle() {
    for seed in 0..6 {
        let table = random_timetable(seed, 6, 12);
        let paths = random_footpaths(seed, 6, 3);
        let net = TransitNetwork {
            timetable: &table,
            transfer: Transfer::instant(),
            footpaths: &paths,
        };
        let progress = Progress::new();
        // The oracle asks the time-dependent model at every second of the
        // window, which costs far more than any profile it checks — so it is
        // enumerated once per instance rather than once per technique.
        let asked: Vec<Window> = (0..6u32)
            .flat_map(|from| (0..6u32).map(move |to| (from, to)))
            .filter(|(from, to)| from != to)
            .map(|(from, to)| {
                (
                    from,
                    to,
                    profile_by_brute_force(&table, &paths, from, to, 0..=12_000),
                )
            })
            .collect();
        let scan = ConnectionScanTechnique.bind(net, &progress).unwrap();
        departures_agree("CSA", &scan, &asked, &paths, seed);
        let trips = TripBasedTechnique::default().bind(net, &progress).unwrap();
        departures_agree("trip-based", &trips, &asked, &paths, seed);
        let labels = PtlTechnique.bind(net, &progress).unwrap();
        departures_agree("PTL", &labels, &asked, &paths, seed);
    }
}

#[test]
fn a_window_keeps_the_pairs_that_leave_inside_it() {
    let table = random_timetable(3, 8, 16);
    let paths = random_footpaths(3, 8, 3);
    let scan = csa_with(&table, &paths);
    let whole = scan.profile(5, 0, None);
    let all = scan.pairs(&whole, 2, 0, 12_000);
    let some: Vec<_> = all
        .iter()
        .copied()
        .filter(|&(dep, _)| (900..=3_000).contains(&dep))
        .collect();
    assert_eq!(scan.pairs(&whole, 2, 900, 3_000), some);
    // Scanning only from the window's start finds the same pairs inside it:
    // connections before it cannot be in a journey leaving after it.
    let trimmed = scan.profile(5, 900, None);
    assert!(trimmed.scanned <= whole.scanned);
    assert_eq!(scan.pairs(&trimmed, 2, 900, 3_000), some);
}

#[test]
fn a_profile_of_town_is_one_departure_worth_taking() {
    let table = town();
    let scan = csa(&table);
    let profile = scan.profile(2, 0, Some(0));
    // Leave 0 at 08:00 on trip 1, change to trip 3, arrive 08:20. Trip 2 at
    // 08:05 reaches 1 at 08:20, too late for anything: not a pair.
    assert_eq!(scan.pairs(&profile, 0, 0, 86_400), vec![(28_800, 30_000)]);
    let (dep, itinerary) = scan.journeys(&profile, 0, 0, 86_400).remove(0);
    assert_eq!(
        (dep, itinerary.arrives, itinerary.transfers()),
        (28_800, 30_000, 1)
    );
    // From 1, two departures worth having: 08:12 aboard trip 1 arriving
    // 08:30, and 08:15 on trip 3 arriving 08:20 — the earlier is dominated.
    assert_eq!(
        scan.pairs(&scan.profile(2, 0, Some(1)), 1, 0, 86_400),
        vec![(29_700, 30_000)]
    );
}

// --- Trip-based -------------------------------------------------------------

#[test]
fn town_keeps_the_one_transfer_worth_making() {
    // Trip 1 reaches stop 1 at 08:10; from there trip 3 (08:15) is a transfer
    // and trip 2's arrival at 1 (08:20) offers nothing to change onto beyond
    // it. Trip 2 at stop 1 could change onto trip 1 (leaves 08:12? no — it
    // arrives 08:20, too late) or trip 3 (left 08:15): nothing. So one.
    let table = town();
    let trips = tripbased(&table);
    assert_eq!(trips.num_lines(), 3);
    assert_eq!(trips.num_trips(), 3);
    assert_eq!(trips.num_transfers(), 1);
    assert_eq!(trips.num_initial_transfers(), 1);
    let front = trips.pareto(0, 28_800, 2);
    assert_eq!(
        front
            .iter()
            .map(|i| (i.arrives, i.transfers()))
            .collect::<Vec<_>>(),
        vec![(30_600, 0), (30_000, 1)]
    );
    assert!(front[1].is_valid(&[(0, 28_800)], Transfer::instant(), &Footpaths::none()));
    let search = trips.search(&[(0, 28_800)], &TripBasedQuery::to(2));
    assert_eq!(search.cost(2), Some(30_000));
    assert_eq!(search.cost(1), None, "the query is point-to-point");
    assert_eq!(
        search.settled, 3,
        "trips 1 and 2 boarded at 0, trip 3 by the change"
    );
    assert_eq!(search.rounds(), 2);
    assert_eq!(trips.path(&search, 2), Some(vec![0, 1, 2]));
}

#[test]
fn a_u_turn_is_not_a_transfer() {
    // Trip 1 runs 0 -> 1 -> 2 and trip 2 runs 2 -> 1 -> 0 behind it. Both
    // changes a rider could make — off trip 1 at 1 onto trip 2 (which goes
    // straight back to 0), or off at 2 onto trip 2 (straight back to 1) — ride
    // back to a stop the rider was just standing at, in time to have changed
    // there instead. Algorithm 2 drops both, and the answers lose nothing.
    let table = Timetable::new(
        3,
        [
            c(1, 0, 1, 100, 200),
            c(1, 1, 2, 210, 300),
            c(2, 2, 1, 320, 400),
            c(2, 1, 0, 410, 500),
        ],
    );
    let trips = tripbased(&table);
    assert_eq!(trips.num_initial_transfers(), 0);
    assert_eq!(trips.earliest_arrival(&[(0, 100)], 2).unwrap().arrives, 300);
    let back = trips.pareto(1, 150, 0);
    assert_eq!(
        back.iter()
            .map(|i| (i.arrives, i.transfers()))
            .collect::<Vec<_>>(),
        vec![(500, 0)]
    );
    // And a change that is not a U-turn is kept: a third trip 1 -> 3.
    let table = Timetable::new(
        4,
        [
            c(1, 0, 1, 100, 200),
            c(1, 1, 2, 210, 300),
            c(2, 2, 1, 320, 400),
            c(2, 1, 0, 410, 500),
            c(3, 1, 3, 450, 600),
        ],
    );
    let trips = tripbased(&table);
    assert_eq!(
        trips.num_initial_transfers(),
        2,
        "onto trip 3 from either trip at 1"
    );
    assert_eq!(trips.num_transfers(), 2);
    assert_eq!(
        trips.earliest_arrival(&[(2, 310)], 3).unwrap().transfers(),
        1
    );
}

#[test]
fn reduction_changes_the_transfer_count_and_never_an_answer() {
    // The paper's claim for Algorithm 3: the transfers it drops are never
    // needed for a Pareto-optimal journey. Held on random instances with
    // walks, front against front.
    let mut fewer = 0;
    for seed in 0..10 {
        let table = random_timetable(seed, 8, 16);
        let paths = random_footpaths(seed, 8, 4);
        let reduced = tripbased_with(&table, &paths);
        let unreduced = TripBased::build_reporting(
            &table,
            Transfer::instant(),
            &paths,
            false,
            &crate::util::progress::Progress::new(),
        );
        assert_eq!(unreduced.num_transfers(), unreduced.num_initial_transfers());
        assert!(reduced.num_transfers() <= unreduced.num_transfers());
        fewer += unreduced.num_transfers() - reduced.num_transfers();
        for from in 0..8u32 {
            for to in 0..8u32 {
                for at in [0, 900, 2000] {
                    let a: Vec<_> = reduced
                        .pareto(from, at, to)
                        .iter()
                        .map(|i| (i.arrives, i.transfers()))
                        .collect();
                    let b: Vec<_> = unreduced
                        .pareto(from, at, to)
                        .iter()
                        .map(|i| (i.arrives, i.transfers()))
                        .collect();
                    assert_eq!(a, b, "seed {seed}: {from} -> {to} at {at}");
                }
            }
        }
    }
    assert!(fewer > 0, "reduction removed nothing on any seed");
}

#[test]
fn the_trip_based_front_is_raptors_and_the_oracles() {
    // Two kernels that are Pareto by construction over the same two criteria
    // must hold the same front — and the front must be the trip-counting
    // oracle's, exactly as RAPTOR's is held to it.
    for seed in 0..8 {
        let table = random_timetable(seed, 6, 10);
        let paths = random_footpaths(seed, 6, 3);
        let rounds = raptor_with(&table, &paths);
        let trips = tripbased_with(&table, &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                for at in [0, 900] {
                    let front = trips.pareto(from, at, to);
                    let by_rounds = rounds.pareto(from, at, to);
                    assert_eq!(
                        front
                            .iter()
                            .map(|i| (boardings(i), i.arrives))
                            .collect::<Vec<_>>(),
                        by_rounds
                            .iter()
                            .map(|i| (boardings(i), i.arrives))
                            .collect::<Vec<_>>(),
                        "seed {seed}: {from} -> {to} at {at}"
                    );
                    let mut expected = Vec::new();
                    let mut previous = None;
                    for boarded in 0..=6usize {
                        let best = best_by_brute_force_with_trips(
                            &table, &paths, from, at, None, to, 8, boarded,
                        );
                        if let Some(arrives) = best {
                            if arrives < previous.unwrap_or(UNREACHABLE) {
                                expected.push((boarded, arrives));
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
                }
            }
        }
    }
}

#[test]
fn max_transfers_caps_the_trip_based_front() {
    let table = town();
    let trips = tripbased(&table);
    let capped = trips.search(
        &[(0, 28_800)],
        &TripBasedQuery {
            target: 2,
            max_transfers: Some(0),
            departing: None,
        },
    );
    let front = trips.itineraries(&capped, 2);
    assert_eq!(front.len(), 1, "no changes: the through ride only");
    assert_eq!(front[0].arrives, 30_600);
    let one = trips.search(
        &[(0, 28_800)],
        &TripBasedQuery {
            target: 2,
            max_transfers: Some(1),
            departing: None,
        },
    );
    assert_eq!(trips.itinerary(&one, 2).unwrap().arrives, 30_000);
    assert!(one.scanned >= capped.scanned);
}

#[test]
fn a_trip_that_revisits_a_stop_is_transferred_onto_at_the_right_visit() {
    // Trip 1 loops 0 -> 1 -> 0 -> 2; trip 2 runs 3 -> 0 arriving 250, in time
    // for trip 1's second call at 0 (leaves 300) but not its first (leaves
    // 100). The transfer must name the second visit.
    let table = Timetable::new(
        4,
        [
            c(1, 0, 1, 100, 150),
            c(1, 1, 0, 160, 250),
            c(1, 0, 2, 300, 400),
            c(2, 3, 0, 200, 250),
        ],
    );
    let trips = tripbased(&table);
    let itinerary = trips.earliest_arrival(&[(3, 0)], 2).unwrap();
    assert_eq!(itinerary.arrives, 400);
    assert_eq!(itinerary.transfers(), 1);
    assert_eq!(itinerary.stops(3), vec![3, 0, 2]);
    assert!(itinerary.is_valid(&[(3, 0)], Transfer::instant(), &Footpaths::none()));
}

#[test]
fn several_sources_reach_the_trip_based_target_as_the_best_one_does() {
    let table = town();
    let trips = tripbased(&table);
    let both = trips.search(&[(0, 28_800), (1, 29_600)], &TripBasedQuery::to(2));
    assert_eq!(trips.itinerary(&both, 2).unwrap().arrives, 30_000);
    let ahead = trips.search(&[(0, 28_800), (1, 29_000)], &TripBasedQuery::to(2));
    let itinerary = trips.itinerary(&ahead, 2).unwrap();
    assert_eq!(itinerary.arrives, 30_000);
    assert_eq!(
        itinerary.legs.len(),
        1,
        "boarded trip 3 at stop 1 without riding from 0"
    );
    assert!(itinerary.is_valid(
        &[(0, 28_800), (1, 29_000)],
        Transfer::instant(),
        &Footpaths::none()
    ));
    // Segments scanned are the search space: each names its round and stops.
    let reached = ahead.reached(&trips);
    assert!(!reached.is_empty());
    assert!(reached
        .iter()
        .all(|(round, _, stops)| *round == 0 && stops.len() >= 2));
}

/// The profile the other kernel's way: RAPTOR's front at every second of the
/// window, kept as the Pareto set over (latest departure, arrival, changes)
/// — leaving out what the walk beats. RAPTOR's front is itself held to the
/// trip-counting oracle above, and asking it every second is what a profile
/// saves.
fn profile_by_asking_raptor(
    rounds: &Raptor,
    paths: &Footpaths,
    from: NodeId,
    to: NodeId,
    window: std::ops::RangeInclusive<Time>,
) -> Vec<(Time, Time, usize)> {
    let walk = paths.duration(from, to);
    let mut all: Vec<(Time, Time, usize)> = Vec::new();
    for t in window {
        for itinerary in rounds.pareto(from, t, to) {
            if itinerary.legs.is_empty() || boardings(&itinerary) == 0 {
                continue;
            }
            if walk.is_some_and(|d| itinerary.arrives >= t + d) {
                continue;
            }
            all.push((t, itinerary.arrives, boardings(&itinerary) - 1));
        }
    }
    // Pareto: later departure, earlier arrival, fewer changes; ties in every
    // criterion are one entry.
    all.sort_unstable();
    all.dedup();
    let mut kept: Vec<(Time, Time, usize)> = Vec::new();
    for &(dep, arr, n) in &all {
        let dominated = all
            .iter()
            .any(|&(d, a, m)| d >= dep && a <= arr && m <= n && (d, a, m) != (dep, arr, n));
        if !dominated {
            kept.push((dep, arr, n));
        }
    }
    // Earliest departure first and, within one, fewest changes — the order a
    // front is read in.
    kept.sort_by_key(|&(dep, _, n)| (dep, n));
    kept
}

#[test]
fn the_trip_based_profile_matches_asking_at_every_second() {
    for seed in 0..6 {
        let table = random_timetable(seed, 6, 10);
        let paths = random_footpaths(seed, 6, 3);
        let trips = tripbased_with(&table, &paths);
        let rounds = raptor_with(&table, &paths);
        for from in 0..6u32 {
            for to in 0..6u32 {
                if from == to {
                    continue;
                }
                let truth = profile_by_asking_raptor(&rounds, &paths, from, to, 0..=6_000);
                let profile = trips.profile(from, to, 0, 6_000);
                assert_eq!(
                    trips.triples(&profile),
                    truth,
                    "seed {seed}: {from} -> {to}"
                );
                assert_eq!(trips.walk(&profile), paths.duration(from, to));
                for (dep, itinerary) in trips.journeys(&profile) {
                    assert!(
                        itinerary.is_valid(&[(from, dep)], Transfer::instant(), &paths),
                        "seed {seed}: {from} -> {to} at {dep}: {itinerary:?}"
                    );
                    assert_eq!(itinerary.legs.first().map(|leg| leg.departs()), Some(dep));
                    assert!(truth.contains(&(dep, itinerary.arrives, itinerary.transfers())));
                }
            }
        }
    }
}

#[test]
fn a_profile_of_town_by_trips_is_one_departure_worth_taking() {
    let table = town();
    let trips = tripbased(&table);
    let profile = trips.profile(0, 2, 0, 86_400);
    // Leave 0 at 08:00 on trip 1: change to trip 3 and arrive 08:20 with one
    // change, or stay aboard and arrive 08:30 with none — both worth having.
    assert_eq!(
        trips.triples(&profile),
        vec![(28_800, 30_600, 0), (28_800, 30_000, 1)]
    );
    assert_eq!(profile.runs, 2, "trip 1 at 08:00 and trip 2 at 08:05");
    // From 1: 08:12 aboard trip 1 arriving 08:30, dominated by 08:15 on trip
    // 3 arriving 08:20.
    assert_eq!(
        trips.triples(&trips.profile(1, 2, 0, 86_400)),
        vec![(29_700, 30_000, 0)]
    );
    // A window trims: nothing leaves 0 for 2 after 08:05 — and a window that
    // closes before 08:00 holds nothing either, though a rider standing at 0
    // from 07:00 would catch trip 1: a journey is dated by when it leaves.
    assert!(trips.profile(0, 2, 29_200, 86_400).is_empty());
    assert!(trips.profile(0, 2, 25_200, 28_799).is_empty());
}
