//! PTL: hub labels over the events, and then no search at all.
//!
//! Delling, Dibbelt, Pajor & Werneck, *Public Transit Labeling* (SEA 2015).
//! RAPTOR and CSA stopped building a graph; PTL builds the biggest one of all
//! — the time-expanded graph, a vertex per event — and then never searches
//! it. Because every arc points forward in time the graph is a DAG, "there is
//! a journey from this event to that one" is *reachability*, and reachability
//! in a DAG is what 2-hop labeling answers in a list intersection: give every
//! vertex a forward label of hubs it reaches and a backward label of hubs
//! that reach it, such that any reachable pair shares one. A query is then a
//! few label intersections, in microseconds, and the whole search has moved
//! into preprocessing.
//!
//! **Earliest arrival** (§3): the first event at the source at or after the
//! departure time is where the rider enters; the earliest event at the target
//! it can reach is the answer, and since reachability along a stop's events
//! is monotone (a later event is always reachable from an earlier one by
//! waiting) that is a **binary search** over the target's events, each probe
//! an intersection of the source event's forward label with the probed
//! event's backward label — *pruned* to events no earlier than the
//! departure. That is the configuration the paper's own comparison uses.
//!
//! **Profile** (§4): event labels cover every pair of events, most of them
//! dominated. The paper folds each stop's labels into **stop labels** — per
//! hub, the latest departure at the stop that reaches it and the earliest
//! arrival at the stop from it — and a profile is one coordinated sweep of
//! the source's forward and the target's backward stop label, every matching
//! hub offering a (leave at, arrive by) pair, kept if nothing dominates it.
//! Hubs are numbered by time of day, so the sweep starts where the window
//! opens.
//!
//! **Walks** at either end are the paper's §5 *location-to-location* case:
//! the neighbouring stops' labels swept alongside with the walk subtracted or
//! added — its *superlabels*, assembled during the query rather than stored.
//! Walks in the middle are arcs of the graph and need nothing.
//!
//! ## What is faithful, and what is not here
//!
//! The paper's time-expanded graph as it states it (merged events, waiting,
//! connection and foot arcs); reachability labels with the cover property;
//! the event-label earliest-arrival query with pruning and binary search; stop
//! labels and the stop-label profile sweep with chronological hub ids;
//! superlabels for walks at either end; and per-entry path pointers so a hub
//! match unpacks to a full journey — the paper's §5 says "known path unpacking
//! techniques" and this is that. The labels themselves are computed by
//! **pruned labeling** rather than RXL, which the paper uses as a black box;
//! see [`labels`] for the substitution and why it is a fair one. Not here:
//! the paper's §5 multicriteria labels (a second, weighted graph and distance
//! labels — 26–88× the preprocessing, its own increment), so `max_transfers`
//! is refused with a pointer at RAPTOR; and minimum transfer times, which no
//! kernel here honours yet. Changing vehicles is instantaneous, so PTL answers
//! the same question the other four timetable techniques do and is checked
//! against every one of them.
//!
//! What it costs is what the paper says it costs: minutes of preprocessing
//! and hundreds of megabytes of labels for a city, in exchange for queries
//! that touch a few hundred integers. That is the trade this kernel exists to
//! put a number on.

mod events;
mod labels;
mod query;
#[cfg(test)]
mod tests;

use crate::model::graph::NodeId;
use crate::model::technique::{
    BindError, EarliestArrival, Footprint, Profiled, Technique, TransitNetwork,
};
use crate::model::timetable::{Connection, Footpaths, Itinerary, Time, Timetable, Transfer};
use crate::util::progress::Progress;

use self::events::EventGraph;
use self::labels::{Labels, StopLabels};

/// Public transit labeling as a configuration: nothing to set. Binding it is
/// the labeling, which is the expensive half and reports progress.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PtlTechnique;

impl<'a> Technique<'a> for PtlTechnique {
    type Inputs = TransitNetwork<'a>;
    type Planner = PublicTransitLabeling;

    fn bind(
        &self,
        net: TransitNetwork<'a>,
        progress: &Progress,
    ) -> Result<PublicTransitLabeling, BindError> {
        Ok(PublicTransitLabeling::build_reporting(
            net.timetable,
            net.transfer,
            net.footpaths,
            progress,
        ))
    }
}

/// A timetable as its event graph and the labels over it. Built once, at
/// some expense, and read by every query.
#[derive(Debug)]
pub struct PublicTransitLabeling {
    events: EventGraph,
    labels: Labels,
    stop_labels: StopLabels,
    /// The connections in the timetable's order, which is what a connection
    /// arc's index names.
    connections: Vec<Connection>,
    footpaths: Footpaths,
    /// The same walks reversed: who can walk to a stop, for the target side
    /// of a query.
    incoming: Footpaths,
}

impl PublicTransitLabeling {
    /// Expand `timetable` into its event graph and label it, with
    /// `footpaths` between stops.
    ///
    /// The `transfer` is accepted for parity with the other timetable models
    /// and is only ever [`Transfer::instant`] today.
    pub fn build(timetable: &Timetable, transfer: Transfer, footpaths: &Footpaths) -> Self {
        Self::build_reporting(timetable, transfer, footpaths, &Progress::new())
    }

    /// [`PublicTransitLabeling::build`], counting into `progress`: hubs
    /// processed while `"labeling"`, then stops while `"merging"` the event
    /// labels into stop labels.
    pub fn build_reporting(
        timetable: &Timetable,
        _transfer: Transfer,
        footpaths: &Footpaths,
        progress: &Progress,
    ) -> Self {
        let events = EventGraph::build(timetable, footpaths);
        let labels = Labels::build(&events, progress);
        let stop_labels = StopLabels::build(&events, &labels, progress);
        PublicTransitLabeling {
            events,
            labels,
            stop_labels,
            connections: timetable.connections().to_vec(),
            footpaths: footpaths.clone(),
            incoming: footpaths.reversed(),
        }
    }

    pub fn num_stops(&self) -> usize {
        self.events.num_stops()
    }

    /// Vertices of the event graph: distinct (stop, time) pairs.
    pub fn num_events(&self) -> usize {
        self.events.num_events()
    }

    /// Arcs of the event graph: waiting, connection and foot arcs together.
    pub fn num_arcs(&self) -> usize {
        self.events.num_arcs()
    }

    /// Entries in the event labels, both directions — the size of the
    /// labeling, and the number a query's work is a share of.
    pub fn num_hubs(&self) -> usize {
        self.labels.total_hubs()
    }

    /// Entries in the stop labels, both directions.
    pub fn num_stop_hubs(&self) -> usize {
        self.stop_labels.total_hubs()
    }

    /// Average hubs per event label — the paper's "hubs per label".
    pub fn hubs_per_label(&self) -> f64 {
        let labels = 2 * self.num_events();
        if labels == 0 {
            0.0
        } else {
            self.num_hubs() as f64 / labels as f64
        }
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.events.footprint()
            + self.labels.footprint()
            + self.stop_labels.footprint()
            + self.connections.len() * std::mem::size_of::<Connection>()
            + self.footpaths.footprint()
            + self.incoming.footprint()
    }
}

impl Footprint for PublicTransitLabeling {
    fn footprint(&self) -> usize {
        PublicTransitLabeling::footprint(self)
    }

    fn searches(&self) -> (&'static str, usize) {
        ("hubs", self.num_hubs())
    }
}

impl EarliestArrival for PublicTransitLabeling {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        PublicTransitLabeling::earliest_arrival(self, sources, to)
    }
}

impl Profiled for PublicTransitLabeling {
    fn departures(
        &self,
        from: NodeId,
        to: NodeId,
        opens: Time,
        closes: Time,
    ) -> Vec<(Time, Itinerary)> {
        self.profile(from, to, opens, closes)
    }
}
