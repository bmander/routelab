//! What every technique is checked against.
//!
//! Fixtures and oracles over the [`Timetable`] and [`Footpaths`] the papers
//! all read, and the random [`Graph`] the graph searches share: a hand-written town whose answers can be counted off the page,
//! random instances, and two brute-force enumerations that are not a second
//! copy of anybody's algorithm. They live here rather than with one paper
//! because every paper needs them, and the contract asks each to be checked
//! against "something that cannot be wrong in the same direction" — a thing
//! no single kernel should own.
//!
//! Not under [`crate::model`]: [`profile_by_brute_force`] asks the
//! time-dependent model for an answer, and the model names nothing above it.

use crate::kernels::timetable::earliest_arrival;
use crate::model::graph::{Graph, NodeId};
use crate::model::timetable::{Connection, Footpaths, Time, Timetable, Transfer, TripId};
use crate::util::rng::Rng;

/// A random directed graph: `num_nodes * density` arcs between random
/// endpoints, each costing 1 to 50.
///
/// Dense enough that most pairs are connected and sparse enough to stay
/// quick, which is what a "does this agree with Dijkstra" loop wants.
pub(crate) fn random_graph(num_nodes: u32, density: f64, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let edges: Vec<_> = (0..(f64::from(num_nodes) * density) as usize)
        .map(|_| {
            (
                rng.below(u64::from(num_nodes)) as u32,
                rng.below(u64::from(num_nodes)) as u32,
                1 + rng.below(50) as u32,
            )
        })
        .collect();
    Graph::from_edges(num_nodes as usize, &edges).expect("real endpoints")
}

/// A connection, spelled shortly enough to read a timetable off the page.
pub(crate) fn c(trip: u32, from: NodeId, to: NodeId, departs: Time, arrives: Time) -> Connection {
    Connection {
        trip: TripId(trip),
        from,
        to,
        departs,
        arrives,
    }
}

/// Stops 0 -> 1 -> 2, with a fast trip and a slow one.
///
///   trip 1: leaves 0 at 08:00, reaches 1 at 08:10, leaves 08:12, reaches 2 at 08:30
///   trip 2: leaves 0 at 08:05, reaches 1 at 08:20  (slower, and a dead end)
///   trip 3: leaves 1 at 08:15, reaches 2 at 08:20  (the good connection)
pub(crate) fn town() -> Timetable {
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
pub(crate) fn best_by_brute_force(
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

pub(crate) fn random_timetable(seed: u64, stops: u32, trips: u32) -> Timetable {
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
pub(crate) fn random_footpaths(seed: u64, stops: u32, count: u32) -> Footpaths {
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

/// The profile the obvious way: ask the time-dependent model at every second
/// of the window and keep, for each arrival, the latest moment that still
/// makes it — leaving out what the direct walk would beat.
pub(crate) fn profile_by_brute_force(
    table: &Timetable,
    paths: &Footpaths,
    from: NodeId,
    to: NodeId,
    window: std::ops::RangeInclusive<Time>,
) -> Vec<(Time, Time)> {
    let walk = paths.duration(from, to);
    let mut latest: Vec<(Time, Time)> = Vec::new();
    for t in window {
        let Some(itinerary) = earliest_arrival(table, &[(from, t)], to, Transfer::instant(), paths)
        else {
            continue;
        };
        if walk.is_some_and(|d| itinerary.arrives >= t + d) {
            continue;
        }
        match latest.last_mut() {
            Some(last) if last.1 == itinerary.arrives => last.0 = t,
            _ => latest.push((t, itinerary.arrives)),
        }
    }
    latest
}
