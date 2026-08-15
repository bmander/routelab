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
//! - **Time-expanded** ([`expanded`]) — a node per *event*. Every departure and
//!   every arrival is its own node, connections and waiting are edges, and what
//!   comes out is an ordinary static graph that [`crate::dijkstra`] routes with
//!   no changes at all.
//! - **Time-dependent** ([`dependent`]) — a node per *stop*. Far fewer nodes,
//!   but each edge carries the connections running along it and traversing one
//!   means finding the next departure, so the search has to be written for it.
//!
//! The two must agree on every query. That is the paper's thesis and it is also
//! this module's main test: neither model is the reference implementation,
//! because each is the other's.
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
//! The consequence is worth stating rather than discovering: **walking and
//! transit are on different clocks**, so one environment cannot yet carry both.
//! That is the multimodal problem, and it is a later increment.

mod dependent;
mod expanded;

#[cfg(test)]
mod tests;

pub use dependent::time_dependent_query;
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
    /// ridden through without changing, which is what a transfer time exempts.
    pub trip: u32,
    pub from: NodeId,
    pub to: NodeId,
    pub departs: Time,
    pub arrives: Time,
}

/// How long changing vehicles takes at a stop.
///
/// The paper's *simple* model assumes changing is instant and its *realistic*
/// model does not. That is one parameter rather than two implementations:
/// `Transfer::instant()` is the simple model, and having it be the degenerate
/// case of the realistic one means the simple model is tested for free rather
/// than maintained separately.
///
/// Staying aboard the same trip never costs it — that is not a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Transfer {
    pub minimum: Time,
}

impl Transfer {
    /// The paper's simple model: changing vehicles takes no time.
    pub fn instant() -> Self {
        Transfer { minimum: 0 }
    }

    /// The realistic model: `seconds` to change from one vehicle to another.
    pub fn of(seconds: Time) -> Self {
        Transfer { minimum: seconds }
    }
}

/// A day's connections, indexed for both models to read.
///
/// Stored once as a CSR over **stop pairs**: for each stop, the pairs leaving
/// it; for each pair, the connections along it in departure order. The
/// time-dependent search wants exactly that shape — an edge, then a binary
/// search within it. The time-expanded builder wants only to walk everything
/// once, which any order allows.
#[derive(Debug, Clone, Default)]
pub struct Timetable {
    stops: usize,
    /// Sorted by `(from, to, departs)`, so an edge's connections are contiguous
    /// and already in the order a search reads them.
    connections: Vec<Connection>,
    /// Offsets into `edge_to`/`edge_span`, one per stop plus a tail.
    stop_edges: Vec<u32>,
    edge_to: Vec<NodeId>,
    /// `[start, end)` into `connections` for each edge.
    edge_span: Vec<(u32, u32)>,
}

impl Timetable {
    /// Build from connections in any order.
    ///
    /// Connections whose stops fall outside `stops`, or which arrive before they
    /// depart, are dropped — a timetable that goes back in time is not something
    /// to route over, and a reader that produced one has a bug worth failing on
    /// rather than a schedule worth honouring.
    pub fn new(stops: usize, connections: impl IntoIterator<Item = Connection>) -> Self {
        let mut connections: Vec<Connection> = connections
            .into_iter()
            .filter(|c| {
                (c.from as usize) < stops && (c.to as usize) < stops && c.arrives >= c.departs
            })
            .collect();
        connections.sort_unstable_by_key(|c| (c.from, c.to, c.departs, c.arrives));

        // One pass to cut the sorted run into edges, and a second to index the
        // edges by their tail. Both are linear; the sort above is the cost.
        let mut stop_edges = vec![0u32; stops + 1];
        let mut edge_to = Vec::new();
        let mut edge_span = Vec::new();
        let mut start = 0usize;
        while start < connections.len() {
            let (from, to) = (connections[start].from, connections[start].to);
            let mut end = start + 1;
            while end < connections.len()
                && connections[end].from == from
                && connections[end].to == to
            {
                end += 1;
            }
            edge_to.push(to);
            edge_span.push((start as u32, end as u32));
            stop_edges[from as usize + 1] += 1;
            start = end;
        }
        for stop in 0..stops {
            stop_edges[stop + 1] += stop_edges[stop];
        }

        Timetable {
            stops,
            connections,
            stop_edges,
            edge_to,
            edge_span,
        }
    }

    pub fn num_stops(&self) -> usize {
        self.stops
    }

    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// How many stop-to-stop edges the connections collapse onto — the node
    /// count of the time-dependent model's graph is `num_stops`, and this is its
    /// edge count.
    pub fn num_edges(&self) -> usize {
        self.edge_to.len()
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// The edges leaving `stop`, as `(to, connections)` in no particular order.
    pub fn edges_from(&self, stop: NodeId) -> impl Iterator<Item = (NodeId, &[Connection])> + '_ {
        let range = match self.stop_edges.get(stop as usize + 1) {
            Some(&end) => self.stop_edges[stop as usize] as usize..end as usize,
            None => 0..0,
        };
        range.map(move |edge| {
            let (start, end) = self.edge_span[edge];
            (
                self.edge_to[edge],
                &self.connections[start as usize..end as usize],
            )
        })
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.connections.len() * std::mem::size_of::<Connection>()
            + self.stop_edges.len() * std::mem::size_of::<u32>()
            + self.edge_to.len() * std::mem::size_of::<NodeId>()
            + self.edge_span.len() * std::mem::size_of::<(u32, u32)>()
    }
}

/// One boarded vehicle, in an answer.
///
/// An itinerary is a list of these rather than of edges, because "which bus, at
/// what time" is the answer to a transit query — an edge id is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ride {
    pub trip: u32,
    pub from: NodeId,
    pub to: NodeId,
    pub departs: Time,
    pub arrives: Time,
}

impl From<Connection> for Ride {
    fn from(c: Connection) -> Self {
        Ride {
            trip: c.trip,
            from: c.from,
            to: c.to,
            departs: c.departs,
            arrives: c.arrives,
        }
    }
}

/// When you get there, and what you rode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Itinerary {
    /// Absolute arrival time at the target.
    pub arrives: Time,
    /// The connections ridden, in order.
    pub rides: Vec<Ride>,
}

impl Itinerary {
    /// How many times you changed vehicles — one less than the number of
    /// distinct trips ridden, and zero for a journey you never got off.
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
                _ => now.saturating_add(transfer.minimum),
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
