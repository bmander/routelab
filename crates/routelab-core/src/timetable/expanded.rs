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

use super::{Connection, Itinerary, Ride, Time, Timetable, Transfer};

/// What an event node stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// A vehicle leaving `stop` at `time`, on the connection at this index.
    Departure { stop: NodeId, time: Time },
    /// A vehicle reaching `stop` at `time`, off the connection at this index.
    Arrival { stop: NodeId, time: Time },
}

impl Event {
    fn stop(&self) -> NodeId {
        match self {
            Event::Departure { stop, .. } | Event::Arrival { stop, .. } => *stop,
        }
    }

    fn time(&self) -> Time {
        match self {
            Event::Departure { time, .. } | Event::Arrival { time, .. } => *time,
        }
    }
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
    /// The connection each graph edge rides, for edges that ride one.
    ridden: HashMap<u32, Connection>,
    stops: usize,
}

impl TimeExpanded {
    /// Expand `timetable` into its event graph.
    ///
    /// `transfer` must be [`Transfer::instant`] for now — see
    /// [`super::time_dependent_query`] for why the two models are only
    /// comparable there, and why a non-zero change time is refused rather than
    /// approximated.
    pub fn build(timetable: &Timetable, transfer: Transfer) -> Self {
        assert_eq!(
            transfer.minimum, 0,
            "a minimum change time needs the realistic model; see the module docs"
        );
        let stops = timetable.num_stops();

        // One departure and one arrival event per connection. Events are
        // deduplicated per (stop, time) so that two buses leaving together do
        // not become two nodes nobody can wait between.
        let mut events: Vec<Event> = Vec::new();
        let mut index: HashMap<(bool, NodeId, Time), NodeId> = HashMap::new();
        let mut intern = |event: Event, events: &mut Vec<Event>| -> NodeId {
            let key = (
                matches!(event, Event::Departure { .. }),
                event.stop(),
                event.time(),
            );
            *index.entry(key).or_insert_with(|| {
                events.push(event);
                (events.len() - 1) as NodeId
            })
        };

        let mut edges: Vec<(NodeId, NodeId, Weight)> = Vec::new();
        let mut ridden: HashMap<u32, Connection> = HashMap::new();
        for connection in timetable.connections() {
            let leaves = intern(
                Event::Departure {
                    stop: connection.from,
                    time: connection.departs,
                },
                &mut events,
            );
            let lands = intern(
                Event::Arrival {
                    stop: connection.to,
                    time: connection.arrives,
                },
                &mut events,
            );
            // The input position of this edge is where it will sit in `edges`;
            // CSR permutes, so the mapping is recovered through `input_index`
            // after the graph is built.
            ridden.insert(edges.len() as u32, *connection);
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
        let by_time = |a: &NodeId, b: &NodeId, events: &[Event]| {
            events[*a as usize].time().cmp(&events[*b as usize].time())
        };
        for stop in 0..stops {
            departures[stop].sort_by(|a, b| by_time(a, b, &events));
            arrivals[stop].sort_by(|a, b| by_time(a, b, &events));

            // Waiting: each departure to the next.
            for pair in departures[stop].windows(2) {
                let (here, next) = (pair[0], pair[1]);
                let wait = events[next as usize].time() - events[here as usize].time();
                edges.push((here, next, wait));
            }
            // Alighting: each arrival to the first departure not before it.
            for &arrival in &arrivals[stop] {
                let ready = events[arrival as usize].time();
                if let Some(&boarding) = departures[stop]
                    .iter()
                    .find(|&&d| events[d as usize].time() >= ready)
                {
                    edges.push((arrival, boarding, events[boarding as usize].time() - ready));
                }
            }
        }

        let graph = Graph::from_edges(events.len(), &edges)
            .expect("event graph is built from its own node set");
        // Re-key the ridden table from input position to CSR edge id, the same
        // trap `Calendar::from_input_windows` exists to avoid.
        let ridden = (0..graph.num_edges() as u32)
            .filter_map(|edge| {
                ridden
                    .get(&graph.input_index(edge))
                    .map(|connection| (edge, *connection))
            })
            .collect();

        TimeExpanded {
            graph,
            events,
            departures,
            arrivals,
            ridden,
            stops,
        }
    }

    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    pub fn footprint(&self) -> usize {
        self.events.len() * std::mem::size_of::<Event>()
            + self.graph.num_edges() * 12
            + self.ridden.len() * (4 + std::mem::size_of::<Connection>())
    }

    /// Earliest arrival at `to`, leaving `from` no earlier than `at`.
    pub fn earliest_arrival(&self, from: NodeId, at: Time, to: NodeId) -> Option<Itinerary> {
        if from as usize >= self.stops || to as usize >= self.stops {
            return None;
        }
        if from == to {
            // Already there. Worth saying explicitly: this model's answers are
            // arrival *events*, and there is no event for standing still — left
            // to itself it would ride a loop back to where it started.
            return Some(Itinerary {
                arrives: at,
                rides: Vec::new(),
            });
        }
        // Enter at the first departure you could catch. Every later one is
        // reachable from it along the waiting chain, so one source is enough.
        let start = *self.departures[from as usize]
            .iter()
            .find(|&&event| self.events[event as usize].time() >= at)?;

        // Seeded with the clock rather than zero, so costs come out as times.
        let sources = [(start, self.events[start as usize].time())];
        let targets: Vec<NodeId> = self.arrivals[to as usize].clone();
        let result = dijkstra(
            &self.graph,
            &sources,
            &SearchOptions::default().with_targets(targets.iter().copied()),
        )
        .ok()?;

        let (event, arrives) = targets
            .iter()
            .filter_map(|&event| result.cost(event).map(|time| (event, time)))
            .min_by_key(|&(_, time)| time)?;

        let rides = result
            .edge_path(event)?
            .into_iter()
            .filter_map(|edge| self.ridden.get(&edge).map(|c| Ride::from(*c)))
            .collect();
        Some(Itinerary { arrives, rides })
    }
}
