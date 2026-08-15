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

use super::{Itinerary, Ride, Time, Timetable, Transfer};

/// Earliest arrival at `to`, leaving `from` no earlier than `at`.
pub fn earliest_arrival(
    timetable: &Timetable,
    from: NodeId,
    at: Time,
    to: NodeId,
    _transfer: Transfer,
) -> Option<Itinerary> {
    let stops = timetable.num_stops();
    if from as usize >= stops || to as usize >= stops {
        return None;
    }
    if from == to {
        return Some(Itinerary {
            arrives: at,
            rides: Vec::new(),
            settled: 0,
        });
    }

    // The clock reading at each stop, and how we got there. Absolute times, so
    // the queue orders by arrival and the answer needs no conversion.
    let mut earliest = vec![UNREACHABLE; stops];
    let mut arrived_by: Vec<Option<Ride>> = vec![None; stops];
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
                arrived_by[next as usize] = Some(connection);
                queue.push(Reverse((connection.arrives, next)));
            }
        }
    }

    let arrives = earliest[to as usize];
    if arrives == UNREACHABLE {
        return None;
    }

    // Walk the rides back to the origin.
    let mut rides = Vec::new();
    let mut here = to;
    while here != from {
        let ride = arrived_by[here as usize]?;
        rides.push(ride);
        here = ride.from;
    }
    rides.reverse();
    Some(Itinerary {
        arrives,
        rides,
        settled,
    })
}
