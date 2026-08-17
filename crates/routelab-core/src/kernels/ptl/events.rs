//! The paper's time-expanded graph: a vertex per event, and arcs that only
//! ever point forward in time.
//!
//! §3 of Delling, Dibbelt, Pajor & Werneck. Every departure and every arrival
//! is an event; events at the same stop and the same time are **merged** into
//! one vertex, which is what makes changing vehicles at an instant a matter
//! of standing still. Three kinds of arc: a *waiting arc* from each event to
//! the next at the same stop, a *connection arc* from a connection's departure
//! event to its arrival event, and a *foot arc* from every vertex at stop *a*
//! to the first event at stop *b* you could be standing at after walking
//! there. Every arc points to a later (or equal) time, so what comes out is a
//! DAG, and reachability in it is exactly "there is a journey".
//!
//! Vertices are numbered by `(time, stop)`. That is not tidiness: §4 of the
//! paper reassigns hub ids chronologically so that a label sorted by hub is
//! sorted by time of day, and a query can skip everything before it left.
//! Numbering the events that way to begin with makes the reassignment free.
//!
//! This is deliberately **not** [`crate::kernels::timetable::TimeExpanded`],
//! Pyrga et al.'s graph, which keeps departures and arrivals apart and hangs
//! foot edges only off arrivals. PTL states its own graph and this is that
//! one; the two share a name and an idea, and nothing else.

use crate::model::graph::NodeId;
use crate::model::timetable::{Footpaths, Time, Timetable};

/// What an arc stands for, packed beside its head: a connection's index in
/// the timetable, or one of these two.
pub(super) const WAIT: u32 = u32::MAX;
pub(super) const WALK: u32 = u32::MAX - 1;

/// One event vertex per unique (stop, time), and the arcs between them, both
/// ways round: the labeling walks the graph forward from a hub and backward
/// from it, and a query unpacks a path in either direction.
#[derive(Debug)]
pub(super) struct EventGraph {
    stops: usize,
    /// Event id to its stop and time. Ids are in `(time, stop)` order.
    event_stop: Vec<NodeId>,
    event_time: Vec<Time>,
    /// CSR: `stop_events[stop_first[p]..stop_first[p + 1]]` are stop `p`'s
    /// events in time order.
    stop_first: Vec<u32>,
    stop_events: Vec<u32>,
    /// CSR out-arcs, `(head, kind)`, and the same arcs indexed by head.
    out_first: Vec<u32>,
    out_head: Vec<u32>,
    out_kind: Vec<u32>,
    in_first: Vec<u32>,
    in_tail: Vec<u32>,
    in_kind: Vec<u32>,
}

impl EventGraph {
    /// Expand `timetable` into events and arcs, with `footpaths` between
    /// stops.
    pub(super) fn build(timetable: &Timetable, footpaths: &Footpaths) -> Self {
        let stops = timetable.num_stops();
        let connections = timetable.connections();

        // The vertex set: every distinct (stop, time), numbered by time.
        let mut keyed: Vec<(Time, NodeId)> = Vec::with_capacity(connections.len() * 2);
        for c in connections {
            keyed.push((c.departs, c.from));
            keyed.push((c.arrives, c.to));
        }
        keyed.sort_unstable();
        keyed.dedup();
        let event_time: Vec<Time> = keyed.iter().map(|&(t, _)| t).collect();
        let event_stop: Vec<NodeId> = keyed.iter().map(|&(_, s)| s).collect();
        let events = keyed.len();

        // Per stop, its events in time order — ids are already in time order,
        // so a counting sort by stop keeps them that way.
        let mut stop_first = vec![0u32; stops + 1];
        for &stop in &event_stop {
            stop_first[stop as usize + 1] += 1;
        }
        for stop in 0..stops {
            stop_first[stop + 1] += stop_first[stop];
        }
        let mut fill = stop_first.clone();
        let mut stop_events = vec![0u32; events];
        for (event, &stop) in event_stop.iter().enumerate() {
            let at = &mut fill[stop as usize];
            stop_events[*at as usize] = event as u32;
            *at += 1;
        }

        let at_stop = |stop: NodeId| -> &[u32] {
            &stop_events[stop_first[stop as usize] as usize..stop_first[stop as usize + 1] as usize]
        };
        let exactly = |stop: NodeId, time: Time| -> u32 {
            let list = at_stop(stop);
            list[list.partition_point(|&e| event_time[e as usize] < time)]
        };

        // The arcs, as (tail, head, kind). Parallel connection arcs between
        // the same two events collapse to the first: reachability does not
        // care which vehicle, and any connection whose times fit is a legal
        // leg.
        let mut arcs: Vec<(u32, u32, u32)> = Vec::new();
        for (index, c) in connections.iter().enumerate() {
            arcs.push((
                exactly(c.from, c.departs),
                exactly(c.to, c.arrives),
                index as u32,
            ));
        }
        for stop in 0..stops as NodeId {
            let list = at_stop(stop);
            for pair in list.windows(2) {
                arcs.push((pair[0], pair[1], WAIT));
            }
            for &event in list {
                let time = event_time[event as usize];
                for (next, duration) in footpaths.from(stop) {
                    let ready = time.saturating_add(duration);
                    let there = at_stop(next);
                    let index = there.partition_point(|&e| event_time[e as usize] < ready);
                    if let Some(&landing) = there.get(index) {
                        arcs.push((event, landing, WALK));
                    }
                }
            }
        }
        arcs.sort_unstable();
        arcs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let (out_first, out_head, out_kind) = csr(events, arcs.iter().copied());
        let mut reversed: Vec<(u32, u32, u32)> = arcs.iter().map(|&(t, h, k)| (h, t, k)).collect();
        reversed.sort_unstable();
        let (in_first, in_tail, in_kind) = csr(events, reversed.into_iter());

        EventGraph {
            stops,
            event_stop,
            event_time,
            stop_first,
            stop_events,
            out_first,
            out_head,
            out_kind,
            in_first,
            in_tail,
            in_kind,
        }
    }

    pub(super) fn num_stops(&self) -> usize {
        self.stops
    }

    pub(super) fn num_events(&self) -> usize {
        self.event_time.len()
    }

    pub(super) fn num_arcs(&self) -> usize {
        self.out_head.len()
    }

    pub(super) fn stop(&self, event: u32) -> NodeId {
        self.event_stop[event as usize]
    }

    pub(super) fn time(&self, event: u32) -> Time {
        self.event_time[event as usize]
    }

    /// Stop `p`'s events, in time order; empty for a stop nothing serves.
    pub(super) fn events_at(&self, stop: NodeId) -> &[u32] {
        let stop = stop as usize;
        if stop + 1 < self.stop_first.len() {
            &self.stop_events[self.stop_first[stop] as usize..self.stop_first[stop + 1] as usize]
        } else {
            &[]
        }
    }

    /// The first event at `stop` at or after `at`, if there is one.
    pub(super) fn first_event_at(&self, stop: NodeId, at: Time) -> Option<u32> {
        let list = self.events_at(stop);
        let index = list.partition_point(|&e| self.event_time[e as usize] < at);
        list.get(index).copied()
    }

    /// The event at exactly (`stop`, `at`), if there is one.
    pub(super) fn event_at(&self, stop: NodeId, at: Time) -> Option<u32> {
        self.first_event_at(stop, at)
            .filter(|&e| self.event_time[e as usize] == at)
    }

    /// Out-arcs of `event`, as `(head, kind)`.
    pub(super) fn out(&self, event: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (a, b) = (
            self.out_first[event as usize] as usize,
            self.out_first[event as usize + 1] as usize,
        );
        self.out_head[a..b]
            .iter()
            .copied()
            .zip(self.out_kind[a..b].iter().copied())
    }

    /// In-arcs of `event`, as `(tail, kind)`.
    pub(super) fn into(&self, event: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (a, b) = (
            self.in_first[event as usize] as usize,
            self.in_first[event as usize + 1] as usize,
        );
        self.in_tail[a..b]
            .iter()
            .copied()
            .zip(self.in_kind[a..b].iter().copied())
    }

    /// In-degree plus out-degree: the labeling's notion of importance.
    pub(super) fn degree(&self, event: u32) -> usize {
        let e = event as usize;
        (self.out_first[e + 1] - self.out_first[e] + self.in_first[e + 1] - self.in_first[e])
            as usize
    }

    /// The kind of the arc from `tail` to `head`, if there is one.
    pub(super) fn arc_kind(&self, tail: u32, head: u32) -> Option<u32> {
        self.out(tail)
            .find(|&(h, _)| h == head)
            .map(|(_, kind)| kind)
    }

    /// Bytes held.
    pub(super) fn footprint(&self) -> usize {
        std::mem::size_of::<u32>()
            * (self.event_stop.len()
                + self.event_time.len()
                + self.stop_first.len()
                + self.stop_events.len()
                + self.out_first.len()
                + self.out_head.len()
                + self.out_kind.len()
                + self.in_first.len()
                + self.in_tail.len()
                + self.in_kind.len())
    }
}

/// Pack `(tail, head, kind)` triples, sorted by tail, into CSR arrays.
fn csr(
    nodes: usize,
    arcs: impl Iterator<Item = (u32, u32, u32)>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut first = vec![0u32; nodes + 1];
    let mut heads = Vec::new();
    let mut kinds = Vec::new();
    for (tail, head, kind) in arcs {
        first[tail as usize + 1] += 1;
        heads.push(head);
        kinds.push(kind);
    }
    for node in 0..nodes {
        first[node + 1] += first[node];
    }
    (first, heads, kinds)
}
