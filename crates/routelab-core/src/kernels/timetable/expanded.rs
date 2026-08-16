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
//!
//! ## Footpaths
//!
//! A walk is an edge like any other here, but it has to land on an event, and
//! there is no event for "standing at B at *t + walk*". Making one — an
//! arrival at B per vehicle arrival at A per footpath — is the obvious
//! construction and it is ruinous: it multiplied a city's 838k events by ten.
//! So a walk edge goes where an alighting edge goes: from the arrival at A
//! straight to the **first departure at B** you could catch after walking,
//! weighted by the whole gap. Nothing new is interned; the walk itself is
//! remembered beside the edge so the answer can still say it was walked. One
//! hop is enough because [`Footpaths`] are closed under composition.
//!
//! A walk edge is told from an alighting edge by its endpoints — arrival at
//! one stop, departure at another — so nothing beyond the graph and the event
//! kinds is needed to read an answer back.
//!
//! Two places a walk cannot be an edge in this graph, so the query handles
//! them itself. The **origin** is a stop and a time rather than an event, so
//! the search is seeded with the first departure at every stop the origin
//! can walk to as well as its own. And a journey may **end** on foot — ride
//! to A, walk to the target — which is an arrival at A plus a duration, not
//! an arrival at the target. So the targets are the arrivals at the target
//! *and* at every stop a footpath reaches it from. If the first of those the
//! search settles is at the target itself, no walk can beat it (every walk
//! candidate settles later and adds time); if it is at a neighbour, the true
//! answer may still be a little later in the search than that first hit, and
//! one bounded re-run up to *t + walk* finds it.

use std::collections::{HashMap, HashSet};

use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, NodeId, Weight};
use crate::model::search::{SearchOptions, SearchResult};

use super::{Footpaths, Itinerary, Leg, Ride, Time, Timetable, Transfer, Walk};

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

/// The first departure at `stop` at or after `at`, if there is one, given the
/// per-stop departure lists in time order.
fn first_departure(
    departures: &[Vec<NodeId>],
    events: &[Event],
    stop: NodeId,
    at: Time,
) -> Option<NodeId> {
    let list = &departures[stop as usize];
    let entry = list.partition_point(|&node| events[node as usize].time() < at);
    list.get(entry).copied()
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
    /// Kept for the query, which walks from the origin before there is any
    /// event to walk from, and to tell a walk as the hops it was made of.
    footpaths: Footpaths,
    /// The footpaths the other way round: for each stop, `(from, duration)`
    /// of every walk that reaches it — how a journey can end on foot.
    reached_by_foot: Vec<Vec<(NodeId, Time)>>,
}

impl TimeExpanded {
    /// Expand `timetable` into its event graph, with `footpaths` between stops.
    pub fn build(timetable: &Timetable, _transfer: Transfer, footpaths: &Footpaths) -> Self {
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

        // Per stop, order the events. That is the whole event set: nothing
        // below interns anything.
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
        }

        // Then the edges that pass time at a stop or between stops. Waiting
        // runs forward in time only; an arrival joins the chain at the first
        // departure at or after it — at the same stop with an instant
        // transfer, at another after the walk there. See the module docstring
        // for why a walk is not an event of its own.
        let mut reached_by_foot: Vec<Vec<(NodeId, Time)>> = vec![Vec::new(); stops];
        for (stop, arriving) in arrivals.iter().enumerate() {
            let stop = stop as NodeId;
            for (next, duration) in footpaths.from(stop) {
                reached_by_foot[next as usize].push((stop, duration));
            }
            for pair in departures[stop as usize].windows(2) {
                let (here, next) = (pair[0], pair[1]);
                let wait = events[next as usize].time() - events[here as usize].time();
                edges.push((here, next, wait));
            }
            for &arrival in arriving {
                let time = events[arrival as usize].time();
                let onward = std::iter::once((stop, 0)).chain(footpaths.from(stop));
                for (next, duration) in onward {
                    let ready = time.saturating_add(duration);
                    if let Some(boarding) = first_departure(&departures, &events, next, ready) {
                        edges.push((arrival, boarding, events[boarding as usize].time() - time));
                    }
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
            footpaths: footpaths.clone(),
            reached_by_foot,
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
            + self.footpaths.footprint()
            + self
                .reached_by_foot
                .iter()
                .map(|links| links.len() * std::mem::size_of::<(NodeId, Time)>())
                .sum::<usize>()
            + self.graph.footprint()
    }

    /// Earliest arrival at `to` from `sources` — each a stop and the time you
    /// are standing there.
    pub fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        let stops = self.departures.len();
        if to as usize >= stops {
            return None;
        }

        // Enter at the first departure you could catch — at each source, and
        // at every stop it could walk to first. Every later departure at a
        // stop is reachable from its first along the waiting chain, so one
        // seed per stop is enough. Each seed remembers the walk that reached
        // it, if one did, because that walk is not an edge in the graph.
        // Standing at the target already, or walking straight to it, is an
        // answer too, and one with no arrival event to find — this model's
        // answers are arrival *events*, and there is no event for standing
        // still — so it is kept aside and compared at the end.
        let mut seeds: Vec<(NodeId, Time)> = Vec::new();
        let mut seeded: HashSet<NodeId> = HashSet::new();
        let mut walked_to: HashMap<NodeId, Walk> = HashMap::new();
        let mut standing: Option<(Time, Option<Walk>)> = None;
        let mut consider_standing = |arrives: Time, walk: Option<Walk>| {
            if standing.is_none_or(|(known, _)| arrives < known) {
                standing = Some((arrives, walk));
            }
        };
        let mut seed = |stop: NodeId,
                        ready: Time,
                        walk: Option<Walk>,
                        seeds: &mut Vec<(NodeId, Time)>,
                        walked_to: &mut HashMap<NodeId, Walk>| {
            let Some(start) = first_departure(&self.departures, &self.events, stop, ready) else {
                return;
            };
            // One seed per departure event — its time is the event's, so two
            // sources that reach the same departure seed it identically, and
            // the first walk to claim it keeps it.
            if seeded.insert(start) {
                seeds.push((start, self.events[start as usize].time()));
                if let Some(walk) = walk {
                    walked_to.insert(start, walk);
                }
            }
        };
        for &(from, at) in sources {
            if from as usize >= stops {
                continue;
            }
            if from == to {
                consider_standing(at, None);
            }
            seed(from, at, None, &mut seeds, &mut walked_to);
            for (next, duration) in self.footpaths.from(from) {
                let walk = Walk {
                    from,
                    to: next,
                    departs: at,
                    arrives: at.saturating_add(duration),
                };
                if next == to {
                    consider_standing(walk.arrives, Some(walk));
                } else {
                    seed(next, walk.arrives, Some(walk), &mut seeds, &mut walked_to);
                }
            }
        }
        // Standing at the target, or walking straight to it, wins outright
        // when it happens no later than the earliest source: nothing ridden
        // can leave before its own departure, let alone arrive sooner. Worth
        // checking before the search rather than after, because "route from a
        // stop to itself" would otherwise settle the whole event graph.
        let ready = |arrives: Time| sources.iter().all(|&(_, at)| arrives <= at);
        let standing_wins = |arrives: Time, ridden: Option<&Found>| {
            ridden.is_none_or(|found| arrives <= found.arrives)
        };
        if let Some((arrives, walk)) = standing {
            if ready(arrives) {
                return Some(self.standing_still(arrives, walk));
            }
        }

        let ridden = self.ride(&seeds, to);
        if let Some((arrives, walk)) = standing {
            if standing_wins(arrives, ridden.as_ref()) {
                return Some(self.standing_still(arrives, walk));
            }
        }
        let Found {
            result,
            settled,
            event,
            walk_in,
            arrives,
        } = ridden?;

        // Read the legs back off the path: rides by input index, walks by
        // their endpoints, and the walks at either end that were never edges.
        let path = result.edge_path(event)?;
        let mut legs = Vec::new();
        let seed = path
            .first()
            .map(|&edge| self.graph.edge(edge).0)
            .unwrap_or(event);
        if let Some(&walk) = walked_to.get(&seed) {
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        for edge in path {
            let (tail, head, _) = self.graph.edge(edge);
            match (self.events[tail as usize], self.events[head as usize]) {
                (Event::Departure { .. }, Event::Arrival { .. }) => {
                    let ride = self.rides[self.graph.input_index(edge) as usize];
                    legs.push(Leg::Ride(ride));
                }
                (Event::Arrival { stop: from, time }, Event::Departure { stop: to, .. })
                    if from != to =>
                {
                    let duration = self.footpaths.duration(from, to).unwrap_or(0);
                    let walk = Walk {
                        from,
                        to,
                        departs: time,
                        arrives: time.saturating_add(duration),
                    };
                    legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
                }
                // Waiting, or alighting at the same stop: time passing, not a leg.
                _ => {}
            }
        }
        if let Some(walk) = walk_in {
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        Some(Itinerary {
            arrives,
            legs,
            settled,
        })
    }

    /// Already there, or a walk away: an answer with no event to find.
    fn standing_still(&self, arrives: Time, walk: Option<Walk>) -> Itinerary {
        Itinerary {
            arrives,
            legs: walk
                .map(|walk| {
                    self.footpaths
                        .expand(walk)
                        .into_iter()
                        .map(Leg::Walk)
                        .collect()
                })
                .unwrap_or_default(),
            settled: 0,
        }
    }

    /// The best finish reachable by riding from `seeds`: the arrival event it
    /// ends at, the walk from there to `to` if it ends on foot, and when.
    ///
    /// Any arrival at `to` will do, and because the search settles in cost
    /// order the first one settled is the earliest — asking for *all* of them
    /// runs the search to exhaustion: 608,611 events settled on a Seattle
    /// weekday against 18,956 for the first. So the targets are the arrivals
    /// at `to` and at every stop a walk reaches it from. If the first target
    /// settled is at `to` itself, nothing can beat it: every walk-in
    /// candidate settles later and adds time. If it is at a neighbour, the
    /// answer is at most that arrival plus its walk, and may be anything
    /// settled up to it — so run once more, bounded there, and take the best.
    /// Only settled costs are trusted; the early exit leaves the rest of the
    /// frontier tentative.
    fn ride(&self, seeds: &[(NodeId, Time)], to: NodeId) -> Option<Found> {
        if seeds.is_empty() {
            return None;
        }
        let by_foot = &self.reached_by_foot[to as usize];
        let finishes = self.arrivals[to as usize].iter().copied().chain(
            by_foot
                .iter()
                .flat_map(|&(stop, _)| self.arrivals[stop as usize].iter().copied()),
        );
        let first = dijkstra(
            &self.graph,
            seeds,
            &SearchOptions::default().with_any_target(finishes),
        )
        .ok()?;
        let hit = *first.order.last()?;
        let bound = match self.events[hit as usize] {
            Event::Arrival { stop, time } if stop != to => by_foot
                .iter()
                .find(|&&(from, _)| from == stop)
                .map(|&(_, duration)| time.saturating_add(duration)),
            _ => None,
        };
        let (result, settled) = match bound {
            None => {
                let settled = first.order.len();
                (first, settled)
            }
            Some(bound) => {
                let again = dijkstra(
                    &self.graph,
                    seeds,
                    &SearchOptions::default().with_max_cost(bound),
                )
                .ok()?;
                let settled = first.order.len() + again.order.len();
                (again, settled)
            }
        };

        let found = &result;
        let direct = self.arrivals[to as usize]
            .iter()
            .filter_map(|&event| found.cost(event).map(|time| (event, None, time)));
        let walked = by_foot.iter().flat_map(|&(stop, duration)| {
            self.arrivals[stop as usize]
                .iter()
                .filter_map(move |&event| {
                    found.cost(event).map(|time| {
                        let walk = Walk {
                            from: stop,
                            to,
                            departs: time,
                            arrives: time.saturating_add(duration),
                        };
                        (event, Some(walk), walk.arrives)
                    })
                })
        });
        let (event, walk_in, arrives) = direct.chain(walked).min_by_key(|&(_, _, time)| time)?;
        Some(Found {
            result,
            settled,
            event,
            walk_in,
            arrives,
        })
    }
}

/// What [`TimeExpanded::ride`] found: the search, and the finish read off it.
struct Found {
    result: SearchResult,
    settled: usize,
    event: NodeId,
    walk_in: Option<Walk>,
    arrives: Time,
}
