//! The time-expanded model: a node per event, and then it is just a graph.
//!
//! Pyrga et al. §3. Every departure and every arrival becomes a node. Riding a
//! connection is an edge from its departure event to its arrival event; waiting
//! at a stop is an edge from one departure event there to the next. Nothing in
//! the resulting graph knows what time it is — the times are *in* the node set —
//! so [`crate::dijkstra`] routes it unchanged, and so would A*, or landmarks, or
//! a contraction hierarchy.
//!
//! That is the model's whole appeal and also its cost: the graph has a node per
//! event, which for a city's weekday is hundreds of thousands of them where the
//! time-dependent model has a few thousand stops.
//!
//! Costs in this graph are **absolute times**, not durations. Seeding the search
//! with the departure time rather than zero makes every settled cost the clock
//! reading at that event, which is what a timetable query actually asks for and
//! saves converting back at the end.

use std::collections::HashMap;

use crate::dijkstra::dijkstra;
use crate::graph::{Graph, NodeId, Weight};
use crate::search::SearchOptions;

use super::{Itinerary, Ride, Time, Timetable, Transfer};

/// What an event node stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Event {
    /// A vehicle leaving `stop` at `time`.
    Departure { stop: NodeId, time: Time },
    /// A vehicle reaching `stop` at `time`.
    Arrival { stop: NodeId, time: Time },
}

impl Event {
    fn time(&self) -> Time {
        match self {
            Event::Departure { time, .. } | Event::Arrival { time, .. } => *time,
        }
    }
}

/// Give `event` a node id, reusing one if this stop and time already has it.
///
/// Deduplicated so that two buses leaving together do not become two nodes
/// nobody can wait between.
fn intern(events: &mut Vec<Event>, index: &mut HashMap<Event, NodeId>, event: Event) -> NodeId {
    *index.entry(event).or_insert_with(|| {
        events.push(event);
        (events.len() - 1) as NodeId
    })
}

/// A timetable as a static graph of events.
#[derive(Debug)]
pub struct TimeExpanded {
    graph: Graph,
    events: Vec<Event>,
    /// Departure events at each stop, in time order — where a query enters.
    departures: Vec<Vec<NodeId>>,
    /// Arrival events at each stop, in time order — where a query finishes.
    arrivals: Vec<Vec<NodeId>>,
    /// The connections, in the order their edges were pushed. Connection edges
    /// come first, so an edge rides connection `i` exactly when its input index
    /// is `i` — no side table needed, only the CSR permutation undone.
    rides: Vec<Ride>,
}

impl TimeExpanded {
    /// Expand `timetable` into its event graph.
    pub fn build(timetable: &Timetable, _transfer: Transfer) -> Self {
        let stops = timetable.num_stops();
        let mut events: Vec<Event> = Vec::new();
        let mut index: HashMap<Event, NodeId> = HashMap::new();
        let mut edges: Vec<(NodeId, NodeId, Weight)> = Vec::new();

        // Connection edges first, and in order, which is what lets an edge find
        // its ride by input index alone.
        for connection in timetable.connections() {
            let leaves = intern(
                &mut events,
                &mut index,
                Event::Departure {
                    stop: connection.from,
                    time: connection.departs,
                },
            );
            let lands = intern(
                &mut events,
                &mut index,
                Event::Arrival {
                    stop: connection.to,
                    time: connection.arrives,
                },
            );
            edges.push((leaves, lands, connection.arrives - connection.departs));
        }

        // Per stop, order the events and chain them. Waiting runs forward in
        // time only, and an arrival joins the chain at the first departure at or
        // after it — which with an instant transfer is the same instant.
        let mut departures: Vec<Vec<NodeId>> = vec![Vec::new(); stops];
        let mut arrivals: Vec<Vec<NodeId>> = vec![Vec::new(); stops];
        for (node, event) in events.iter().enumerate() {
            let node = node as NodeId;
            match event {
                Event::Departure { stop, .. } => departures[*stop as usize].push(node),
                Event::Arrival { stop, .. } => arrivals[*stop as usize].push(node),
            }
        }
        for stop in 0..stops {
            departures[stop].sort_by_key(|&node| events[node as usize].time());
            arrivals[stop].sort_by_key(|&node| events[node as usize].time());

            // Waiting: each departure to the next.
            for pair in departures[stop].windows(2) {
                let (here, next) = (pair[0], pair[1]);
                let wait = events[next as usize].time() - events[here as usize].time();
                edges.push((here, next, wait));
            }
            // Alighting: each arrival to the first departure not before it. Both
            // lists are sorted, so this is a binary search rather than a scan —
            // a downtown stop has thousands of each.
            for &arrival in &arrivals[stop] {
                let ready = events[arrival as usize].time();
                let next =
                    departures[stop].partition_point(|&node| events[node as usize].time() < ready);
                if let Some(&boarding) = departures[stop].get(next) {
                    edges.push((arrival, boarding, events[boarding as usize].time() - ready));
                }
            }
        }

        let graph = Graph::from_edges(events.len(), &edges)
            .expect("event graph is built from its own node set");

        TimeExpanded {
            graph,
            events,
            departures,
            arrivals,
            rides: timetable.connections().to_vec(),
        }
    }

    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    /// Bytes held by this structure's own arrays. The graph reports its own.
    pub fn footprint(&self) -> usize {
        self.events.len() * std::mem::size_of::<Event>()
            + self.rides.len() * std::mem::size_of::<Ride>()
            + self.graph.footprint()
    }

    /// Earliest arrival at `to`, leaving `from` no earlier than `at`.
    pub fn earliest_arrival(&self, from: NodeId, at: Time, to: NodeId) -> Option<Itinerary> {
        let stops = self.departures.len();
        if from as usize >= stops || to as usize >= stops {
            return None;
        }
        if from == to {
            // Already there. Worth saying explicitly: this model's answers are
            // arrival *events*, and there is no event for standing still — left
            // to itself it would ride a loop back to where it started.
            return Some(Itinerary {
                arrives: at,
                rides: Vec::new(),
                settled: 0,
            });
        }
        // Enter at the first departure you could catch. Every later one is
        // reachable from it along the waiting chain, so one source is enough.
        let entry = self.departures[from as usize]
            .partition_point(|&node| self.events[node as usize].time() < at);
        let start = *self.departures[from as usize].get(entry)?;

        // Seeded with the clock rather than zero, so costs come out as times.
        // No target set: every arrival at `to` is a candidate, and a tracker
        // over eight hundred thousand nodes costs more than it saves when the
        // search has to reach the cheapest of them anyway.
        let sources = [(start, self.events[start as usize].time())];
        let result = dijkstra(&self.graph, &sources, &SearchOptions::default()).ok()?;

        let (event, arrives) = self.arrivals[to as usize]
            .iter()
            .filter_map(|&event| result.cost(event).map(|time| (event, time)))
            .min_by_key(|&(_, time)| time)?;

        let rides = result
            .edge_path(event)?
            .into_iter()
            .filter_map(|edge| {
                self.rides
                    .get(self.graph.input_index(edge) as usize)
                    .copied()
            })
            .collect();
        Some(Itinerary {
            arrives,
            rides,
            settled: result.order.len(),
        })
    }
}
