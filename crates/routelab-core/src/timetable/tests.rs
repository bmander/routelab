//! The paper's claim is that two very different graphs answer the same
//! question, so that is what is tested: the models against each other, and both
//! against an oracle that enumerates itineraries the obvious way.

use super::*;
use crate::rng::Rng;

/// Stops 0 -> 1 -> 2, with a fast trip and a slow one.
///
///   trip 1: leaves 0 at 08:00, reaches 1 at 08:10, leaves 08:12, reaches 2 at 08:30
///   trip 2: leaves 0 at 08:05, reaches 1 at 08:20  (slower, and a dead end)
///   trip 3: leaves 1 at 08:15, reaches 2 at 08:20  (the good connection)
fn town() -> Timetable {
    let c = |trip, from, to, departs, arrives| Connection {
        trip,
        from,
        to,
        departs,
        arrives,
    };
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
    from: NodeId,
    at: Time,
    to: NodeId,
    depth: usize,
) -> Option<Time> {
    fn walk(
        timetable: &Timetable,
        here: NodeId,
        now: Time,
        to: NodeId,
        left: usize,
        best: &mut Option<Time>,
    ) {
        if here == to {
            *best = Some(best.map_or(now, |b: Time| b.min(now)));
            return;
        }
        if left == 0 {
            return;
        }
        for connection in timetable.connections() {
            if connection.from == here && connection.departs >= now {
                walk(
                    timetable,
                    connection.to,
                    connection.arrives,
                    to,
                    left - 1,
                    best,
                );
            }
        }
    }
    let mut best = None;
    walk(timetable, from, at, to, depth, &mut best);
    best
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
            connections.push(Connection {
                trip,
                from: stop,
                to: next,
                departs,
                arrives,
            });
            stop = next;
            now = arrives;
        }
    }
    Timetable::new(stops as usize, connections)
}

fn expanded(timetable: &Timetable) -> TimeExpanded {
    TimeExpanded::build(timetable, Transfer::instant())
}

// --- The timetable itself ---------------------------------------------------

#[test]
fn connections_that_go_back_in_time_are_dropped() {
    let table = Timetable::new(
        2,
        [Connection {
            trip: 0,
            from: 0,
            to: 1,
            departs: 100,
            arrives: 50,
        }],
    );
    assert_eq!(table.num_connections(), 0);
}

#[test]
fn connections_off_the_end_of_the_stop_list_are_dropped() {
    let table = Timetable::new(
        2,
        [Connection {
            trip: 0,
            from: 0,
            to: 7,
            departs: 100,
            arrives: 200,
        }],
    );
    assert_eq!(table.num_connections(), 0);
}

#[test]
fn edges_collect_the_connections_that_run_along_them() {
    let table = town();
    let from_zero: Vec<_> = table.edges_from(0).collect();
    assert_eq!(from_zero.len(), 1, "0 -> 1 is one edge");
    let (to, connections) = from_zero[0];
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
    // better and both models should say so.
    let table = town();
    let by_events = expanded(&table).earliest_arrival(0, 28_800, 2).unwrap();
    let by_stops = time_dependent_query(&table, 0, 28_800, 2, Transfer::instant()).unwrap();

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
        if let Some(trip) = expanded.earliest_arrival(0, at, 2) {
            assert!(trip.arrives >= previous, "departing {at}");
            previous = trip.arrives;
        }
    }
}

#[test]
fn a_stop_you_cannot_reach_has_no_itinerary() {
    let table = town();
    // Nothing runs to stop 0.
    assert!(expanded(&table).earliest_arrival(2, 0, 0).is_none());
    assert!(time_dependent_query(&table, 2, 0, 0, Transfer::instant()).is_none());
}

#[test]
fn arriving_after_the_last_departure_finds_nothing() {
    let table = town();
    assert!(expanded(&table).earliest_arrival(0, 60_000, 2).is_none());
    assert!(time_dependent_query(&table, 0, 60_000, 2, Transfer::instant()).is_none());
}

#[test]
fn the_two_models_agree_on_everything() {
    // The paper's thesis, and this module's load-bearing test. Neither model is
    // the reference: each is the other's.
    for seed in 0..12 {
        let table = random_timetable(seed, 12, 25);
        let expanded = expanded(&table);
        for from in 0..12u32 {
            for to in 0..12u32 {
                for at in [0, 900, 1800, 3600] {
                    let by_events = expanded.earliest_arrival(from, at, to);
                    let by_stops = time_dependent_query(&table, from, at, to, Transfer::instant());
                    assert_eq!(
                        by_events.as_ref().map(|i| i.arrives),
                        by_stops.as_ref().map(|i| i.arrives),
                        "seed {seed}: {from} -> {to} at {at}"
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
        for from in 0..6u32 {
            for to in 0..6u32 {
                let truth = best_by_brute_force(&table, from, 0, to, 6);
                assert_eq!(
                    expanded.earliest_arrival(from, 0, to).map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-expanded, {from} -> {to}"
                );
                assert_eq!(
                    time_dependent_query(&table, from, 0, to, Transfer::instant())
                        .map(|i| i.arrives),
                    truth,
                    "seed {seed}: time-dependent, {from} -> {to}"
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
        for from in 0..10u32 {
            for to in 0..10u32 {
                for query in [
                    expanded.earliest_arrival(from, 0, to),
                    time_dependent_query(&table, from, 0, to, Transfer::instant()),
                ] {
                    let Some(itinerary) = query else { continue };
                    assert!(
                        itinerary.is_valid(from, 0, Transfer::instant()),
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
    assert_eq!(table.num_stops(), 12);
}

#[test]
fn you_are_already_where_you_already_are() {
    // Both models must answer "now", and the expanded one has to be told: its
    // answers are arrival events, and standing still is not an event.
    let table = town();
    for at in [0, 28_800, 60_000] {
        assert_eq!(
            expanded(&table).earliest_arrival(1, at, 1),
            Some(Itinerary {
                arrives: at,
                rides: Vec::new()
            })
        );
        assert_eq!(
            time_dependent_query(&table, 1, at, 1, Transfer::instant()),
            Some(Itinerary {
                arrives: at,
                rides: Vec::new()
            })
        );
    }
}

#[test]
fn a_transfer_time_is_refused_rather_than_approximated() {
    // Both models take the parameter; neither pretends to honour it yet. The
    // reason differs and is in each module's docs.
    let table = town();
    assert!(
        std::panic::catch_unwind(|| { TimeExpanded::build(&table, Transfer::of(120)) }).is_err()
    );
    assert!(std::panic::catch_unwind(|| {
        time_dependent_query(&table, 0, 0, 2, Transfer::of(120))
    })
    .is_err());
}
