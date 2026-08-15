//! Routing over a timetable: the same question asked of two different graphs.
//!
//! Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
//! Information in Public Transportation Systems* (ACM Journal of Experimental
//! Algorithmics 12, Article 2.4, 2007). A timetable is not a network with
//! weights on it — it is a set of **connections**, each one a vehicle leaving
//! one stop at one instant and reaching another at another. Turning that into
//! something a shortest-path algorithm can read is the whole problem, and the
//! paper gives two answers:
//!
//! - **Time-expanded** ([`TimeExpanded`]) — a node per *event*. Every departure
//!   and every arrival is its own node, connections and waiting are edges, and
//!   what comes out is an ordinary static graph that [`crate::dijkstra`] routes
//!   with no changes at all.
//! - **Time-dependent** ([`earliest_arrival`]) — a node per *stop*. Far fewer
//!   nodes, but each edge carries the connections running along it and
//!   traversing one means finding the next departure, so the search has to be
//!   written for it.
//!
//! Both answer with the same verb — `earliest_arrival` — because comparing them
//! is the point. The two must agree on every query; that is the paper's thesis
//! and it is also this module's main test, since neither model is the reference
//! implementation. Each is the other's.
//!
//! ## The clock is a line, not a cycle
//!
//! [`Time`] is seconds since the service day's midnight, and it does **not**
//! wrap. That is deliberately different from [`crate::timedep::Clock`], which is
//! weekly and cyclic because the restrictions it reads repeat weekly. A
//! timetable does not repeat: a service day is a line, and it routinely runs
//! past its own end — `25:30:00` is how a feed writes a bus that left before
//! midnight and arrives after it. Wrapping that into the previous Monday would
//! be silently wrong.
//!
//! The two are not re-exported at the crate root together, so that a reader
//! meets them as `timetable::Time` and `timedep::Clock` rather than as two
//! interchangeable `u32`s. The consequence is worth stating rather than
//! discovering: **walking and transit are on different clocks**, so one
//! environment cannot yet carry both. That is the multimodal problem, and it is
//! a later increment — the conversion needs a service-day-to-calendar anchor,
//! which is exactly the thing that must not be implicit.

mod dependent;
mod expanded;

#[cfg(test)]
mod tests;

pub use dependent::earliest_arrival;
pub use expanded::TimeExpanded;

use crate::graph::NodeId;

/// A moment, in seconds since the service day's midnight.
///
/// May exceed a day. See the module docstring for why it does not wrap.
pub type Time = u32;

/// One vehicle, leaving one stop and reaching the next.
///
/// The atom of a timetable. A trip serving five stops is four connections, not
/// one thing with five parts, because what a search needs to ask is always
/// "from here, at this time, what can I board".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Connection {
    /// Which vehicle run this belongs to. Two connections sharing a trip can be
    /// ridden through without changing.
    pub trip: u32,
    pub from: NodeId,
    pub to: NodeId,
    pub departs: Time,
    pub arrives: Time,
}

/// One boarded vehicle, in an answer.
///
/// An itinerary is a list of these rather than of edges, because "which bus, at
/// what time" is the answer to a transit query and an edge id is not. It is the
/// same five facts a [`Connection`] carries — the name is the difference, and
/// the name is worth having at a call site.
pub type Ride = Connection;

/// How long changing vehicles takes at a stop.
///
/// **Only [`Transfer::instant`] exists**, which is the paper's *simple* model:
/// changing vehicles takes no time. Its *realistic* model charges a minimum
/// change time, and that is not a parameter this can honour yet — see
/// [`earliest_arrival`] for why one label per stop cannot express it. Rather
/// than offer a constructor whose every product would be rejected, the type
/// makes the unsupported case unrepresentable, the way [`crate::timedep::Waiting`]
/// is a closed set of the policies that actually exist.
///
/// The parameter stays in both signatures so that the realistic model, when it
/// lands, is a new constructor rather than a new argument everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Transfer {
    minimum: Time,
}

impl Transfer {
    /// The paper's simple model: changing vehicles takes no time.
    pub fn instant() -> Self {
        Transfer { minimum: 0 }
    }

    /// Seconds needed to change from one vehicle to another.
    pub fn minimum(&self) -> Time {
        self.minimum
    }
}

/// A day's connections, indexed for both models to read.
///
/// Stored once as a two-level index: for each stop the **stop pairs** leaving
/// it, and for each pair the connections along it in departure order. The
/// time-dependent search wants exactly that shape — pick an edge, then find the
/// next departure on it. The time-expanded builder wants only to walk
/// everything once, which any order allows.
#[derive(Debug, Clone, Default)]
pub struct Timetable {
    /// Sorted by `(from, to, departs)`, so an edge's connections are contiguous
    /// and already in the order a search reads them.
    connections: Vec<Connection>,
    /// Offsets into `edge_start`, one per stop plus a tail.
    stop_edges: Vec<u32>,
    /// `[edge_start[e], edge_start[e + 1])` is edge `e`'s run of connections.
    /// The edge's destination is that run's first `to`, so it is not stored.
    edge_start: Vec<u32>,
    /// For each connection, the index of the soonest-arriving connection at or
    /// after it on the same edge. A suffix minimum, so a relaxation is a binary
    /// search and then a lookup rather than a scan — see [`earliest_arrival`].
    best_from: Vec<u32>,
}

impl Timetable {
    /// Build from connections in any order.
    ///
    /// Connections whose stops fall outside `stops`, or which arrive before they
    /// depart, are dropped — a timetable that goes back in time is not something
    /// to route over.
    pub fn new(stops: usize, connections: impl IntoIterator<Item = Connection>) -> Self {
        let mut connections: Vec<Connection> = connections
            .into_iter()
            .filter(|c| {
                (c.from as usize) < stops && (c.to as usize) < stops && c.arrives >= c.departs
            })
            .collect();
        connections.sort_unstable_by_key(|c| (c.from, c.to, c.departs, c.arrives));

        // Cut the sorted run into edges, counting them per stop as we go.
        let mut stop_edges = vec![0u32; stops + 1];
        let mut edge_start = Vec::new();
        let mut start = 0usize;
        while start < connections.len() {
            let (from, to) = (connections[start].from, connections[start].to);
            let end = connections[start..]
                .iter()
                .position(|c| c.from != from || c.to != to)
                .map_or(connections.len(), |offset| start + offset);
            edge_start.push(start as u32);
            stop_edges[from as usize + 1] += 1;
            start = end;
        }
        edge_start.push(connections.len() as u32);
        for stop in 0..stops {
            stop_edges[stop + 1] += stop_edges[stop];
        }

        // Suffix minimum by arrival, within each edge. A later departure can
        // land sooner — an express overtaking a local on the same pair of stops
        // — so "the next one" is not always the best one.
        let mut best_from = vec![0u32; connections.len()];
        for edge in 0..edge_start.len().saturating_sub(1) {
            let (first, last) = (edge_start[edge] as usize, edge_start[edge + 1] as usize);
            let mut best = last.saturating_sub(1);
            for index in (first..last).rev() {
                if connections[index].arrives <= connections[best].arrives {
                    best = index;
                }
                best_from[index] = best as u32;
            }
        }

        Timetable {
            connections,
            stop_edges,
            edge_start,
            best_from,
        }
    }

    pub fn num_stops(&self) -> usize {
        self.stop_edges.len().saturating_sub(1)
    }

    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// How many stop-to-stop edges the connections collapse onto — the edge
    /// count of the time-dependent model's graph, whose node count is
    /// [`Timetable::num_stops`].
    pub fn num_edges(&self) -> usize {
        self.edge_start.len().saturating_sub(1)
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// The edges leaving `stop`, as `(to, connections, first_index)` — the index
    /// being where that run starts in [`Timetable::connections`], which is what
    /// makes the suffix minimum addressable.
    pub(super) fn edges_from(
        &self,
        stop: NodeId,
    ) -> impl Iterator<Item = (NodeId, &[Connection], usize)> + '_ {
        let stop = stop as usize;
        let range = if stop + 1 < self.stop_edges.len() {
            self.stop_edges[stop] as usize..self.stop_edges[stop + 1] as usize
        } else {
            0..0
        };
        range.map(move |edge| {
            let (first, last) = (
                self.edge_start[edge] as usize,
                self.edge_start[edge + 1] as usize,
            );
            let run = &self.connections[first..last];
            (run[0].to, run, first)
        })
    }

    /// The soonest-arriving connection at or after `index` on its own edge.
    pub(super) fn soonest_from(&self, index: usize) -> Connection {
        self.connections[self.best_from[index] as usize]
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.connections.len() * std::mem::size_of::<Connection>()
            + (self.stop_edges.len() + self.edge_start.len() + self.best_from.len())
                * std::mem::size_of::<u32>()
    }
}

/// When you get there, what you rode, and what it took to work out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Itinerary {
    /// Absolute arrival time at the target.
    pub arrives: Time,
    /// The connections ridden, in order.
    pub rides: Vec<Ride>,
    /// Nodes the search settled reaching this — stops for one model, events for
    /// the other. Carried on the answer because it is the number the two models
    /// are actually being compared on, and a caller that has the itinerary
    /// should not have to ask a second time for the work it cost.
    pub settled: usize,
}

impl Itinerary {
    /// How many times you changed vehicles — zero for a journey you never got
    /// off, and one less than the number of distinct trips ridden.
    pub fn transfers(&self) -> usize {
        self.rides
            .windows(2)
            .filter(|pair| pair[0].trip != pair[1].trip)
            .count()
    }

    /// Is this a legal itinerary under `transfer`?
    ///
    /// The falsifiability check, in the manner of [`crate::graph::Graph::walk`]:
    /// every ride must start where the last one ended, no earlier than it
    /// arrived, and leave enough time to change when the vehicle changes.
    pub fn is_valid(&self, from: NodeId, at: Time, transfer: Transfer) -> bool {
        let mut here = from;
        let mut now = at;
        let mut aboard: Option<u32> = None;
        for ride in &self.rides {
            if ride.from != here || ride.arrives < ride.departs {
                return false;
            }
            let ready = match aboard {
                // Getting on for the first time is not a change, and neither is
                // staying in your seat.
                None => now,
                Some(trip) if trip == ride.trip => now,
                _ => now.saturating_add(transfer.minimum()),
            };
            if ride.departs < ready {
                return false;
            }
            here = ride.to;
            now = ride.arrives;
            aboard = Some(ride.trip);
        }
        now == self.arrives
    }
}
