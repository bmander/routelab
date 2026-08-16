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
//! Both read the same optional third thing: [`Footpaths`], the paper's
//! *foot-edges* — a link between two stops a rider can take at any time for a
//! fixed duration, which is what lets a real feed's northbound and southbound
//! stops across a street be the same place for the purpose of changing buses.
//! Without them a city timetable is routable only between stops a single trip
//! chain connects, and a query dropped at the wrong side of the street has no
//! answer.
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

use crate::graph::{Graph, NodeId, Weight};
use crate::search::SearchOptions;

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

/// A stretch of an itinerary made on foot, between two stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Walk {
    pub from: NodeId,
    pub to: NodeId,
    pub departs: Time,
    pub arrives: Time,
}

/// One stretch of an itinerary: aboard something, or on foot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    Ride(Ride),
    Walk(Walk),
}

impl Leg {
    pub fn from(&self) -> NodeId {
        match self {
            Leg::Ride(ride) => ride.from,
            Leg::Walk(walk) => walk.from,
        }
    }

    pub fn to(&self) -> NodeId {
        match self {
            Leg::Ride(ride) => ride.to,
            Leg::Walk(walk) => walk.to,
        }
    }

    pub fn departs(&self) -> Time {
        match self {
            Leg::Ride(ride) => ride.departs,
            Leg::Walk(walk) => walk.departs,
        }
    }

    pub fn arrives(&self) -> Time {
        match self {
            Leg::Ride(ride) => ride.arrives,
            Leg::Walk(walk) => walk.arrives,
        }
    }

    /// The vehicle run, for a leg that rides one.
    pub fn trip(&self) -> Option<u32> {
        match self {
            Leg::Ride(ride) => Some(ride.trip),
            Leg::Walk(_) => None,
        }
    }
}

/// Stop-to-stop links a rider may take on foot at any time, for a fixed duration.
///
/// The paper's *foot-edges*. Kept apart from the [`Timetable`] because they are
/// not connections — nothing leaves, they are simply always open — and because
/// which of them exist is a modelling choice (how far will a rider walk?)
/// rather than a fact the feed states. Both models take them as a separate
/// argument, and [`Footpaths::none`] is the paper's plain model.
///
/// **Closed under composition.** If you can walk A→B and B→C, then A→C is a
/// footpath too, taking the shorter of the direct walk and the sum. That is
/// not a convenience: the time-dependent search chains walks on its own —
/// nothing stops it relaxing a footpath from a stop it walked to — while the
/// time-expanded graph must not, or a pair of opposite footpaths would spawn
/// events for ever. Closing the set is what lets one model take a walk one hop
/// at a time and the other take it in one, and still agree on every query.
/// It is also what the RAPTOR family requires of its footpaths, for the same
/// reason. The cost is that "stops within 200 m" quietly becomes "stops
/// reachable by 200 m hops"; on a real feed those components are small.
///
/// A closed link remembers the last stop it came through, so a walk taken in
/// one can still be told as the hops it was made of — see [`Footpaths::hops`].
/// An answer names the links that were given, never one the closure invented.
///
/// Stored CSR-style by origin stop, which is the shape both models read them
/// in: from a stop I am standing at, where can I walk and how long does it take.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Footpaths {
    /// Offsets into `links`, one per stop plus a tail.
    stop_links: Vec<u32>,
    /// `(to, duration, via)`, grouped by origin stop: `via` is the stop before
    /// `to` on the shortest chain from the origin, and the origin itself for a
    /// link that was given directly.
    links: Vec<(NodeId, Time, NodeId)>,
}

impl Footpaths {
    /// No footpaths at all: every stop is its own island, as in the paper's
    /// plain model.
    pub fn none() -> Self {
        Footpaths::default()
    }

    /// Build from `(from, to, duration)` triples in any order.
    ///
    /// Links whose stops fall outside `stops` are dropped. Both directions are
    /// the caller's business: a walk that only goes one way is unusual but
    /// not impossible (a one-way turnstile), so nothing is mirrored here.
    pub fn new(stops: usize, links: impl IntoIterator<Item = (NodeId, NodeId, Time)>) -> Self {
        let given: Vec<(NodeId, NodeId, Weight)> = links
            .into_iter()
            .filter(|(from, to, _)| {
                (*from as usize) < stops && (*to as usize) < stops && from != to
            })
            .collect();
        let mut links = Self::closure(stops, &given);
        links.sort_unstable();
        let mut stop_links = vec![0u32; stops + 1];
        for (from, _, _, _) in &links {
            stop_links[*from as usize + 1] += 1;
        }
        for stop in 0..stops {
            stop_links[stop + 1] += stop_links[stop];
        }
        Footpaths {
            stop_links,
            links: links
                .into_iter()
                .map(|(_, to, duration, via)| (to, duration, via))
                .collect(),
        }
    }

    /// Every walk reachable by chaining `given`, at its shortest duration, as
    /// `(from, to, duration, via)`.
    ///
    /// The links are a graph, so this is [`crate::dijkstra`] from each stop
    /// that has one, and `via` is the search's own parent pointer. The
    /// footpath graph is a scatter of small components — a street corner, a
    /// transit centre — so this is cheap; it is not, and need not be, an
    /// all-pairs pass over the network.
    fn closure(stops: usize, given: &[(NodeId, NodeId, Weight)]) -> Vec<(NodeId, NodeId, Time, NodeId)> {
        let graph = Graph::from_edges(stops, given).expect("links were filtered to the stop set");
        let mut closed = Vec::new();
        for origin in 0..stops as NodeId {
            if graph.out_edges(origin).next().is_none() {
                continue;
            }
            let Ok(reached) = crate::dijkstra::dijkstra(&graph, &[(origin, 0)], &SearchOptions::default())
            else {
                continue;
            };
            for &stop in &reached.order {
                if stop == origin {
                    continue;
                }
                let (Some(duration), Some(via)) = (reached.cost(stop), reached.parent(stop)) else {
                    continue;
                };
                closed.push((origin, stop, duration, via));
            }
        }
        closed
    }

    /// Build from **positions in a graph's input edge list**: each named edge
    /// becomes a footpath between its endpoints taking its weight.
    ///
    /// The join between an environment and the models, in one place, and by
    /// input position for the reason [`Timetable::from_input_connections`]
    /// gives — `Graph` permutes edges into CSR order, and whatever produced
    /// the edges holds input positions.
    pub fn from_input_edges(graph: &Graph, positions: impl IntoIterator<Item = u32>) -> Self {
        let to_edge = graph.edges_by_input();
        let links = positions
            .into_iter()
            .filter(|input| (*input as usize) < to_edge.len())
            .map(|input| graph.edge(to_edge[input as usize]));
        Footpaths::new(graph.num_nodes(), links)
    }

    /// Where you can walk from `stop`, as `(to, duration)`.
    pub fn from(&self, stop: NodeId) -> impl Iterator<Item = (NodeId, Time)> + '_ {
        self.links_from(stop).iter().map(|&(to, duration, _)| (to, duration))
    }

    fn links_from(&self, stop: NodeId) -> &[(NodeId, Time, NodeId)] {
        let stop = stop as usize;
        if stop + 1 < self.stop_links.len() {
            &self.links[self.stop_links[stop] as usize..self.stop_links[stop + 1] as usize]
        } else {
            &[]
        }
    }

    fn link(&self, from: NodeId, to: NodeId) -> Option<(Time, NodeId)> {
        self.links_from(from)
            .iter()
            .find(|(next, _, _)| *next == to)
            .map(|&(_, duration, via)| (duration, via))
    }

    /// How long the walk from `from` to `to` takes, if there is one.
    pub fn duration(&self, from: NodeId, to: NodeId) -> Option<Time> {
        self.link(from, to).map(|(duration, _)| duration)
    }

    /// Was the walk from `from` to `to` given directly, rather than made by
    /// the closure?
    pub fn is_given(&self, from: NodeId, to: NodeId) -> bool {
        self.link(from, to).is_some_and(|(_, via)| via == from)
    }

    /// The walk from `from` to `to` as the given links it chains, in order,
    /// each as `(from, to, duration)`. Empty if there is no such walk.
    pub fn hops(&self, from: NodeId, to: NodeId) -> Vec<(NodeId, NodeId, Time)> {
        let mut hops = Vec::new();
        let mut here = to;
        while here != from {
            let Some((walked, via)) = self.link(from, here) else {
                return Vec::new();
            };
            let before = if via == from { 0 } else { self.duration(from, via).unwrap_or(0) };
            hops.push((via, here, walked - before));
            here = via;
        }
        hops.reverse();
        hops
    }

    /// `walk`, told as the given links it chains: one [`Walk`] per hop, with
    /// the clock advanced along them. A walk that was given directly is one
    /// hop and comes back as it was.
    pub fn expand(&self, walk: Walk) -> Vec<Walk> {
        let mut clock = walk.departs;
        let hops = self.hops(walk.from, walk.to);
        if hops.is_empty() {
            return vec![walk];
        }
        hops.into_iter()
            .map(|(from, to, duration)| {
                let departs = clock;
                clock = clock.saturating_add(duration);
                Walk {
                    from,
                    to,
                    departs,
                    arrives: clock,
                }
            })
            .collect()
    }

    /// How many links there are.
    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.stop_links.len() * std::mem::size_of::<u32>()
            + self.links.len() * std::mem::size_of::<(NodeId, Time, NodeId)>()
    }
}

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

    /// Build from connections keyed by **position in a graph's input edge list**.
    ///
    /// The join between a feed and a graph, in one place. A timetable's stops
    /// are that graph's nodes, and which pair of stops a connection runs
    /// between comes from the edge it was filed under rather than from the
    /// caller — so there is no pair of parallel arrays to zip up wrongly.
    ///
    /// Keyed by input position and not by edge id for the reason
    /// [`crate::timedep::Calendar::from_input_windows`] is: `Graph` permutes its
    /// edges into CSR order, and everything that produces edges holds input
    /// positions. Getting it backwards would put every bus on the wrong street
    /// without failing.
    pub fn from_input_connections(
        graph: &Graph,
        entries: impl IntoIterator<Item = (u32, Vec<(u32, Time, Time)>)>,
    ) -> Self {
        let to_edge = graph.edges_by_input();
        let connections = entries
            .into_iter()
            .filter(|(input, _)| (*input as usize) < to_edge.len())
            .flat_map(|(input, times)| {
                let (from, to, _) = graph.edge(to_edge[input as usize]);
                times
                    .into_iter()
                    .map(move |(trip, departs, arrives)| Connection {
                        trip,
                        from,
                        to,
                        departs,
                        arrives,
                    })
            });
        Timetable::new(graph.num_nodes(), connections)
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

/// When you get there, what you rode and walked, and what it took to work out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Itinerary {
    /// Absolute arrival time at the target.
    pub arrives: Time,
    /// The stretches of the journey, in order: each a vehicle ridden or a
    /// footpath walked.
    pub legs: Vec<Leg>,
    /// Nodes the search settled reaching this — stops for one model, events for
    /// the other. Carried on the answer because it is the number the two models
    /// are actually being compared on, and a caller that has the itinerary
    /// should not have to ask a second time for the work it cost.
    pub settled: usize,
}

impl Itinerary {
    /// The vehicles ridden, in order, with the walks between them left out.
    pub fn rides(&self) -> impl Iterator<Item = &Ride> + '_ {
        self.legs.iter().filter_map(|leg| match leg {
            Leg::Ride(ride) => Some(ride),
            Leg::Walk(_) => None,
        })
    }

    /// The footpaths walked, in order.
    pub fn walks(&self) -> impl Iterator<Item = &Walk> + '_ {
        self.legs.iter().filter_map(|leg| match leg {
            Leg::Walk(walk) => Some(walk),
            Leg::Ride(_) => None,
        })
    }

    /// How many times you changed vehicles — zero for a journey you never got
    /// off, and one less than the number of distinct trips ridden. A walk
    /// between two vehicles is how a change is made, not a second one.
    pub fn transfers(&self) -> usize {
        let trips: Vec<u32> = self.rides().map(|ride| ride.trip).collect();
        trips.windows(2).filter(|pair| pair[0] != pair[1]).count()
    }

    /// Is this a legal itinerary under `transfer`, walking only `footpaths`?
    ///
    /// The falsifiability check, in the manner of [`crate::graph::Graph::walk`]:
    /// every leg must start where the last one ended, no earlier than it
    /// arrived; a ride must leave enough time to change when the vehicle
    /// changes; a walk must be a footpath that exists and take exactly as long
    /// as it says.
    pub fn is_valid(
        &self,
        from: NodeId,
        at: Time,
        transfer: Transfer,
        footpaths: &Footpaths,
    ) -> bool {
        let mut here = from;
        let mut now = at;
        let mut aboard: Option<u32> = None;
        for leg in &self.legs {
            if leg.from() != here || leg.arrives() < leg.departs() || leg.departs() < now {
                return false;
            }
            match leg {
                Leg::Ride(ride) => {
                    let ready = match aboard {
                        // Getting on for the first time is not a change, and
                        // neither is staying in your seat.
                        None => now,
                        Some(trip) if trip == ride.trip => now,
                        _ => now.saturating_add(transfer.minimum()),
                    };
                    if ride.departs < ready {
                        return false;
                    }
                    aboard = Some(ride.trip);
                }
                Leg::Walk(walk) => {
                    // A given link, not one the closure invented: an answer
                    // says which footpaths were walked, hop by hop.
                    if !footpaths.is_given(walk.from, walk.to)
                        || footpaths.duration(walk.from, walk.to) != Some(walk.arrives - walk.departs)
                    {
                        return false;
                    }
                    // Off the vehicle: the next boarding is a change whatever
                    // trip it is, but a change made on foot is already timed
                    // by the walk itself.
                    aboard = None;
                }
            }
            here = leg.to();
            now = leg.arrives();
        }
        now == self.arrives
    }
}
