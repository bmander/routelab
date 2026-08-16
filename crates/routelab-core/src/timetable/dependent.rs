//! The time-dependent model: a node per stop, and a search that reads the clock.
//!
//! Pyrga et al. §4. Where the time-expanded model spends nodes to make the
//! problem static, this one keeps the graph the size of the network — one node
//! per stop — and pays for it in the search. Relaxing an edge is no longer
//! reading a weight: it is asking "what is the next thing leaving here", which
//! is a binary search over that edge's departures.
//!
//! The label at a stop is the earliest time you can be standing there, and the
//! search settles stops in that order. That is ordinary Dijkstra, and it is
//! correct for the same reason the static case is: arrival is non-decreasing in
//! departure, so a stop reached earliest is reached best.
//!
//! ## Footpaths
//!
//! A footpath is the one kind of edge here that *is* a weight: from a stop I
//! am standing at, I can be at the other end after a fixed walk, whatever the
//! clock says. So it relaxes exactly as a static edge would, and because it is
//! relaxed from any settled stop, a walk can follow a walk — the search chains
//! footpaths as far as they go, which is what the time-expanded model must
//! also do for the two to keep agreeing.
//!
//! ## Why there is no minimum change time
//!
//! It is not expressible with one label per stop. Staying in your seat is not a
//! change and must not be charged, so whether the next boarding costs the change
//! time depends on *which vehicle you arrived on* — and that is not something a
//! stop label carries. Charging it always would penalise riding through;
//! charging it never is the simple model, which is what this is.
//!
//! The paper's answer (§4.2) is more nodes: one per route at each stop, with
//! transfer edges between them through the stop. That is a real construction and
//! it belongs in its own increment. Until it lands, [`Transfer`] offers only
//! [`Transfer::instant`], so the gap is a missing constructor rather than a
//! parameter that would be quietly ignored.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{NodeId, UNREACHABLE};

use super::{Footpaths, Itinerary, Leg, Time, Timetable, Transfer, Walk};

/// Earliest arrival at `to`, leaving `from` no earlier than `at`, riding
/// `timetable` and walking `footpaths`.
pub fn earliest_arrival(
    timetable: &Timetable,
    from: NodeId,
    at: Time,
    to: NodeId,
    _transfer: Transfer,
    footpaths: &Footpaths,
) -> Option<Itinerary> {
    let stops = timetable.num_stops();
    if from as usize >= stops || to as usize >= stops {
        return None;
    }
    if from == to {
        return Some(Itinerary {
            arrives: at,
            legs: Vec::new(),
            settled: 0,
        });
    }

    // The clock reading at each stop, and how we got there. Absolute times, so
    // the queue orders by arrival and the answer needs no conversion.
    let mut earliest = vec![UNREACHABLE; stops];
    let mut arrived_by: Vec<Option<Leg>> = vec![None; stops];
    earliest[from as usize] = at;

    let mut settled = 0;
    let mut queue: BinaryHeap<Reverse<(Time, NodeId)>> = BinaryHeap::new();
    queue.push(Reverse((at, from)));

    while let Some(Reverse((now, stop))) = queue.pop() {
        // Lazy deletion, as everywhere else in this crate.
        if now > earliest[stop as usize] {
            continue;
        }
        settled += 1;
        if stop == to {
            break;
        }
        for (next, connections, first) in timetable.edges_from(stop) {
            // The one line the static case does not have: not "what does this
            // edge cost" but "what is the next thing along it". The binary
            // search finds what you could board; the suffix minimum finds which
            // of those lands soonest, since a later express can overtake an
            // earlier local on the same pair of stops.
            let boarding = connections.partition_point(|c| c.departs < now);
            if boarding == connections.len() {
                continue;
            }
            let connection = timetable.soonest_from(first + boarding);
            if connection.arrives < earliest[next as usize] {
                earliest[next as usize] = connection.arrives;
                arrived_by[next as usize] = Some(Leg::Ride(connection));
                queue.push(Reverse((connection.arrives, next)));
            }
        }
        // The static case, for the edges that have a weight after all: a walk
        // starts the moment you are standing here.
        for (next, duration) in footpaths.from(stop) {
            let arrives = now.saturating_add(duration);
            if arrives < earliest[next as usize] {
                earliest[next as usize] = arrives;
                arrived_by[next as usize] = Some(Leg::Walk(Walk {
                    from: stop,
                    to: next,
                    departs: now,
                    arrives,
                }));
                queue.push(Reverse((arrives, next)));
            }
        }
    }

    let arrives = earliest[to as usize];
    if arrives == UNREACHABLE {
        return None;
    }

    // Walk the legs back to the origin. A walk the closure made in one is
    // told as the given links it chains, so the answer names real footpaths.
    let mut legs = Vec::new();
    let mut here = to;
    while here != from {
        let leg = arrived_by[here as usize]?;
        match leg {
            Leg::Walk(walk) => legs.extend(footpaths.expand(walk).into_iter().rev().map(Leg::Walk)),
            ride => legs.push(ride),
        }
        here = leg.from();
    }
    legs.reverse();
    Some(Itinerary {
        arrives,
        legs,
        settled,
    })
}
