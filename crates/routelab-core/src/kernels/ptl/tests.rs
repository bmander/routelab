//! The labels are checked against searches of the graph they label — the
//! cover property, exhaustively on small graphs — and the queries against the
//! four timetable techniques that answer the same question, plus the oracles
//! those are checked against.

use super::events::EventGraph;
use super::labels::{meet, Labels};
use super::PublicTransitLabeling;
use crate::kernels::csa::ConnectionScan;
use crate::kernels::raptor::Raptor;
use crate::kernels::timetable::tests::{
    best_by_brute_force, c, profile_by_brute_force, random_footpaths, random_timetable, town,
};
use crate::kernels::timetable::{earliest_arrival, TimeExpanded};
use crate::model::graph::NodeId;
use crate::model::timetable::{Footpaths, Timetable, Transfer};
use crate::util::progress::Progress;

fn ptl(timetable: &Timetable) -> PublicTransitLabeling {
    PublicTransitLabeling::build(timetable, Transfer::instant(), &Footpaths::none())
}

fn ptl_with(timetable: &Timetable, footpaths: &Footpaths) -> PublicTransitLabeling {
    PublicTransitLabeling::build(timetable, Transfer::instant(), footpaths)
}

/// Every event reachable from `from`, by a plain search of the arcs.
fn reachable_by_search(graph: &EventGraph, from: u32) -> Vec<bool> {
    let mut seen = vec![false; graph.num_events()];
    let mut stack = vec![from];
    seen[from as usize] = true;
    while let Some(u) = stack.pop() {
        for (w, _) in graph.out(u) {
            if !seen[w as usize] {
                seen[w as usize] = true;
                stack.push(w);
            }
        }
    }
    seen
}

// --- The graph and the labels -----------------------------------------------

#[test]
fn town_becomes_its_events() {
    // town: departures at 0 (08:00, 08:05), 1 (08:12, 08:15); arrivals at 1
    // (08:10, 08:20), 2 (08:20, 08:30) — eight distinct (stop, time) pairs.
    let table = town();
    let graph = EventGraph::build(&table, &Footpaths::none());
    assert_eq!(graph.num_events(), 8);
    // Ids are chronological.
    for pair in (0..graph.num_events() as u32)
        .collect::<Vec<_>>()
        .windows(2)
    {
        assert!(graph.time(pair[0]) <= graph.time(pair[1]));
    }
    // 4 connection arcs, and waiting arcs: stop 0 has 2 events (1 arc), stop
    // 1 has 4 (3 arcs), stop 2 has 2 (1 arc).
    assert_eq!(graph.num_arcs(), 4 + 1 + 3 + 1);
    assert_eq!(graph.events_at(1).len(), 4);
    assert_eq!(
        graph.first_event_at(0, 28_900).map(|e| graph.time(e)),
        Some(29_100)
    );
    assert_eq!(graph.first_event_at(0, 29_101), None);
    assert_eq!(graph.event_at(1, 29_400).map(|e| graph.stop(e)), Some(1));
    assert_eq!(graph.event_at(1, 29_401), None);
}

#[test]
fn labels_cover_exactly_what_a_search_reaches() {
    // The cover property, exhaustively: for every pair of events, the labels
    // meet if and only if a search of the arcs gets there.
    for seed in 0..6 {
        let table = random_timetable(seed, 8, 16);
        let paths = random_footpaths(seed, 8, 3);
        let graph = EventGraph::build(&table, &paths);
        let labels = Labels::build(&graph, &Progress::new());
        for u in 0..graph.num_events() as u32 {
            let truth = reachable_by_search(&graph, u);
            for w in 0..graph.num_events() as u32 {
                let (hub, _) = meet(labels.forward(u).0, labels.backward(w).0);
                assert_eq!(hub.is_some(), truth[w as usize], "seed {seed}: {u} -> {w}");
                if let Some(hub) = hub {
                    // And the pointers walk there: forward to the hub, back
                    // from it, along real arcs.
                    let mut here = u;
                    while here != hub {
                        let next = labels.next_toward(here, hub).unwrap();
                        assert!(graph.arc_kind(here, next).is_some());
                        here = next;
                    }
                    let mut here = w;
                    while here != hub {
                        let previous = labels.previous_from(here, hub).unwrap();
                        assert!(graph.arc_kind(previous, here).is_some());
                        here = previous;
                    }
                }
            }
        }
    }
}

#[test]
fn a_hub_labels_itself_once() {
    let table = town();
    let graph = EventGraph::build(&table, &Footpaths::none());
    let labels = Labels::build(&graph, &Progress::new());
    for e in 0..graph.num_events() as u32 {
        assert!(labels.forward(e).0.contains(&e));
        assert!(labels.backward(e).0.contains(&e));
        assert!(labels.forward(e).0.windows(2).all(|p| p[0] < p[1]));
    }
    assert!(labels.total_hubs() >= 2 * graph.num_events());
}

// --- Earliest arrival -------------------------------------------------------

#[test]
fn ptl_answers_town() {
    let table = town();
    let labels = ptl(&table);
    assert_eq!(labels.num_stops(), 3);
    assert_eq!(labels.num_events(), 8);
    assert!(labels.footprint() > 0);
    assert!(labels.hubs_per_label() >= 1.0);
    // Leaving 0 at 08:00: trip 1 to stop 1, then trip 3 to stop 2 by 08:20.
    let best = labels.earliest_arrival(&[(0, 28_800)], 2).unwrap();
    assert_eq!(best.arrives, 30_000);
    assert_eq!(best.transfers(), 1);
    assert_eq!(best.legs.len(), 2);
    assert!(best.is_valid(&[(0, 28_800)], Transfer::instant(), &Footpaths::none()));
    assert!(best.settled > 0);
    // Leaving after the last departure finds nothing; a stop off the end is
    // not a stop.
    assert!(labels.earliest_arrival(&[(0, 29_200)], 2).is_none());
    assert!(labels.earliest_arrival(&[(0, 28_800)], 7).is_none());
    assert!(labels.earliest_arrival(&[(7, 28_800)], 2).is_none());
}

#[test]
fn you_are_already_where_you_already_are() {
    let table = town();
    let labels = ptl(&table);
    let here = labels.earliest_arrival(&[(1, 29_000)], 1).unwrap();
    assert_eq!(here.arrives, 29_000);
    assert!(here.legs.is_empty());
    assert!(here.is_valid(&[(1, 29_000)], Transfer::instant(), &Footpaths::none()));
}

#[test]
fn a_walk_from_the_source_needs_no_event_to_set_it_off() {
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(0, 3, 300), (3, 0, 300)]);
    let labels = ptl_with(&table, &paths);
    let walked = labels.earliest_arrival(&[(0, 28_800)], 3).unwrap();
    assert_eq!(walked.arrives, 29_100);
    assert!(walked.rides().next().is_none());
    // And from 3, the walk to 0 catches trip 1: the seed is stop 0's first
    // event after the walk lands, not stop 3's (which has none).
    let ridden = labels.earliest_arrival(&[(3, 28_500)], 1).unwrap();
    assert_eq!(ridden.arrives, 29_400);
    assert!(ridden.is_valid(&[(3, 28_500)], Transfer::instant(), &paths));
}

#[test]
fn a_journey_can_end_on_foot_before_the_next_event() {
    // Ride to 1, walk to 3: arrival is the arrival at 1 plus the walk, and
    // stop 3 has no event to land on at all.
    let table = Timetable::new(4, town().connections().iter().copied());
    let paths = Footpaths::new(4, [(1, 3, 120), (3, 1, 120)]);
    let labels = ptl_with(&table, &paths);
    let found = labels.earliest_arrival(&[(0, 28_800)], 3).unwrap();
    assert_eq!(found.arrives, 29_400 + 120);
    assert!(found.is_valid(&[(0, 28_800)], Transfer::instant(), &paths));
    let by_stops =
        earliest_arrival(&table, &[(0, 28_800)], 3, Transfer::instant(), &paths).unwrap();
    assert_eq!(
        (found.arrives, &found.legs),
        (by_stops.arrives, &by_stops.legs)
    );
}

#[test]
fn ptl_agrees_with_the_other_four_and_the_oracle() {
    for seed in 0..8 {
        let table = random_timetable(seed, 10, 20);
        for paths in [Footpaths::none(), random_footpaths(seed, 10, 4)] {
            let labels = ptl_with(&table, &paths);
            let expanded = TimeExpanded::build(&table, Transfer::instant(), &paths);
            let rounds = Raptor::build(&table, Transfer::instant(), &paths);
            let scan = ConnectionScan::build(&table, Transfer::instant(), &paths);
            for from in 0..10u32 {
                for to in 0..10u32 {
                    for at in [0u32, 900, 2_400] {
                        let sources = [(from, at)];
                        let by_labels = labels.earliest_arrival(&sources, to);
                        let want =
                            earliest_arrival(&table, &sources, to, Transfer::instant(), &paths)
                                .map(|i| i.arrives);
                        assert_eq!(
                            by_labels.as_ref().map(|i| i.arrives),
                            want,
                            "seed {seed}: {sources:?} -> {to}"
                        );
                        assert_eq!(
                            expanded.earliest_arrival(&sources, to).map(|i| i.arrives),
                            want
                        );
                        assert_eq!(
                            rounds
                                .itinerary(&rounds.search(&sources, Some(to), None, None), to)
                                .map(|i| i.arrives),
                            want
                        );
                        assert_eq!(
                            scan.itinerary(&scan.search(&sources, Some(to), None), to)
                                .map(|i| i.arrives),
                            want
                        );
                        if at == 0 {
                            assert_eq!(
                                want,
                                best_by_brute_force(&table, &paths, from, 0, to, 6),
                                "seed {seed}: oracle {from} -> {to}"
                            );
                        }
                        if let Some(itinerary) = by_labels {
                            assert!(
                                itinerary.is_valid(&sources, Transfer::instant(), &paths),
                                "seed {seed}: {sources:?} -> {to}: {itinerary:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn several_sources_each_carry_their_own_time() {
    for seed in 0..6 {
        let table = random_timetable(seed, 10, 20);
        let paths = random_footpaths(seed, 10, 4);
        let labels = ptl_with(&table, &paths);
        for a in 0..10u32 {
            let b = (a + 3) % 10;
            let sources = [(a, 0), (b, 400)];
            for to in 0..10u32 {
                let want = earliest_arrival(&table, &sources, to, Transfer::instant(), &paths)
                    .map(|i| i.arrives);
                let found = labels.earliest_arrival(&sources, to);
                assert_eq!(
                    found.as_ref().map(|i| i.arrives),
                    want,
                    "seed {seed}: {sources:?} -> {to}"
                );
                if let Some(itinerary) = found {
                    assert!(itinerary.is_valid(&sources, Transfer::instant(), &paths));
                }
            }
        }
    }
}

// --- Profile ----------------------------------------------------------------

#[test]
fn the_profile_matches_asking_at_every_second() {
    for seed in 0..6 {
        let table = random_timetable(seed, 8, 16);
        let paths = random_footpaths(seed, 8, 3);
        let labels = ptl_with(&table, &paths);
        for to in 0..8u32 {
            for from in 0..8u32 {
                if from == to {
                    continue;
                }
                let truth = profile_by_brute_force(&table, &paths, from, to, 0..=12_000);
                let journeys = labels.profile(from, to, 0, 12_000);
                let pairs: Vec<(u32, u32)> =
                    journeys.iter().map(|(dep, i)| (*dep, i.arrives)).collect();
                assert_eq!(pairs, truth, "seed {seed}: {from} -> {to}");
                for (dep, itinerary) in journeys {
                    assert!(
                        itinerary.is_valid(&[(from, dep)], Transfer::instant(), &paths),
                        "seed {seed}: {from} -> {to} at {dep}: {itinerary:?}"
                    );
                    assert_eq!(itinerary.legs.first().map(|leg| leg.departs()), Some(dep));
                }
            }
        }
    }
}

#[test]
fn a_window_keeps_the_pairs_that_leave_inside_it() {
    let table = random_timetable(3, 8, 16);
    let paths = random_footpaths(3, 8, 3);
    let labels = ptl_with(&table, &paths);
    let all: Vec<(u32, u32)> = labels
        .profile(2, 5, 0, 12_000)
        .iter()
        .map(|(dep, i)| (*dep, i.arrives))
        .collect();
    let some: Vec<_> = all
        .iter()
        .copied()
        .filter(|&(dep, _)| (900..=3_000).contains(&dep))
        .collect();
    let windowed: Vec<(u32, u32)> = labels
        .profile(2, 5, 900, 3_000)
        .iter()
        .map(|(dep, i)| (*dep, i.arrives))
        .collect();
    assert_eq!(windowed, some);
    assert!(labels.profile(2, 5, 3_000, 900).is_empty());
    assert!(labels.profile(2, 2, 0, 12_000).is_empty());
}

#[test]
fn a_profile_of_town_is_one_departure_worth_taking() {
    let table = town();
    let labels = ptl(&table);
    let journeys = labels.profile(0, 2, 0, 86_400);
    assert_eq!(journeys.len(), 1);
    let (dep, itinerary) = &journeys[0];
    assert_eq!(
        (*dep, itinerary.arrives, itinerary.transfers()),
        (28_800, 30_000, 1)
    );
    // From 1: 08:12 aboard trip 1 arriving 08:30 is dominated by 08:15 on
    // trip 3 arriving 08:20.
    let from_one: Vec<(u32, u32)> = labels
        .profile(1, 2, 0, 86_400)
        .iter()
        .map(|(dep, i)| (*dep, i.arrives))
        .collect();
    assert_eq!(from_one, vec![(29_700, 30_000)]);
}

// --- Progress ---------------------------------------------------------------

#[test]
fn building_reports_and_finishes_at_one() {
    let table = town();
    let progress = Progress::new();
    let watched = PublicTransitLabeling::build_reporting(
        &table,
        Transfer::instant(),
        &Footpaths::none(),
        &progress,
    );
    assert_eq!(progress.phase(), "merging");
    assert_eq!(progress.fraction(), Some(1.0));
    let quiet = ptl(&table);
    assert_eq!(quiet.num_hubs(), watched.num_hubs());
    assert_eq!(quiet.num_stop_hubs(), watched.num_stop_hubs());
}

#[test]
fn a_connection_that_reads_as_one_hop_is_one_leg() {
    // Two trips leaving together and landing together share one arc; the
    // leg still names a real connection.
    let table = Timetable::new(
        2,
        [
            c(1, 0, 1, 100, 200),
            c(2, 0, 1, 100, 200),
            c(3, 0, 1, 150, 300),
        ],
    );
    let labels = ptl(&table);
    let found = labels.earliest_arrival(&[(0, 0)], 1).unwrap();
    assert_eq!(found.arrives, 200);
    assert_eq!(found.legs.len(), 1);
    assert!(found.is_valid(&[(0, 0)], Transfer::instant(), &Footpaths::none()));
    let _ = NodeId::default();
}
