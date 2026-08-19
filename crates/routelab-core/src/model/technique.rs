//! What a technique promises, as traits.
//!
//! Every kernel in this crate answers some version of the same few questions —
//! how far, by when, along which edges, through which stops — and until now
//! each answered in its own shape: six kernels, six signatures for "the
//! earliest arrival from these stops at that one". The test that checked they
//! agreed had to spell all six out by hand. These traits are the shared shape,
//! written down once so that the compiler holds every kernel to it and a
//! caller can be generic over "anything that answers this".
//!
//! They are small on purpose. A kernel implements the ones it honestly meets
//! and no others: RAPTOR keeps a table and a Pareto front, so it is
//! [`Searches`] and [`Front`]; PTL answers with an itinerary and keeps nothing,
//! so it is [`EarliestArrival`] and not [`Searches`]. There is no everything-
//! trait, because there is no everything-kernel, and a method that half the
//! implementors would have to stub is a lie about the contract.
//!
//! Nothing here names a kernel — the kernels opt in with `impl`, the way a
//! heuristic does with [`Heuristic`](crate::model::heuristic::Heuristic). The
//! dependency still runs down.

use std::fmt;

use crate::model::graph::{EdgeId, GraphError, NodeId, Weight};
use crate::model::search::SearchError;
use crate::model::timetable::{Footpaths, Itinerary, Time, Timetable, Transfer};
use crate::util::progress::Progress;

/// The three things every schedule-based kernel is built from, stated once.
///
/// [`Transfer`] lives here and nowhere else. Four of the five transit kernels
/// take it and ignore it today; putting it on the bundle means they stop
/// pretending, and the one that will read it finds it where it belongs.
#[derive(Debug, Clone, Copy)]
pub struct TransitNetwork<'a> {
    pub timetable: &'a Timetable,
    pub transfer: Transfer,
    pub footpaths: &'a Footpaths,
}

/// Why a technique could not be bound to its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    /// The inputs held a graph the technique could not build over.
    Graph(GraphError),
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindError::Graph(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for BindError {}

impl From<GraphError> for BindError {
    fn from(err: GraphError) -> Self {
        BindError::Graph(err)
    }
}

/// A configured technique: what to build, not yet built.
///
/// `bind` is the one expensive step, and the only one that takes a
/// [`Progress`] — preprocessing is what takes long enough to watch. The plain
/// `build` constructors stay where they are and call through with a counter
/// nobody reads, per the convention in [`crate::util::progress`].
///
/// The lifetime is the inputs': a planner may borrow them (a search that keeps
/// nothing has nothing to own) or copy what it needs and let them go.
pub trait Technique<'a> {
    /// What this technique is built from — a graph, a [`TransitNetwork`], a
    /// graph and its calendar. Each family states its own.
    type Inputs;
    /// The bound, queryable result.
    type Planner;
    /// What binding can go wrong with. [`std::convert::Infallible`] for a
    /// technique that cannot fail — most of them, since laying a timetable
    /// out differently is not something a timetable can refuse. Contraction
    /// builds a graph, and a graph can be too large to build.
    type Error: std::error::Error;

    /// Do the preprocessing and hand back something that answers queries.
    fn bind(&self, inputs: Self::Inputs, progress: &Progress)
        -> Result<Self::Planner, Self::Error>;
}

/// What a bound planner holds, and what it counts a search over.
pub trait Footprint {
    /// Bytes of preprocessed data. Zero for a plain search, which is the
    /// honest number rather than a missing one.
    fn footprint(&self) -> usize;

    /// The unit a search is counted in and how many there are — `("stops",
    /// n)`, `("events", n)`, `("nodes", n)` — the denominator a settled count
    /// is a share of.
    fn searches(&self) -> (&'static str, usize);
}

/// The result contract every kept search meets: a cost per node and a count
/// of work done. Cost is a [`Weight`] or a [`Time`], both `u32`.
pub trait Distances {
    /// The cost to `node`, or `None` where this search has none to vouch for.
    ///
    /// `None` is not always "unreachable". A point-to-point search learns the
    /// true cost of exactly one node — the one it was aimed at — and reports
    /// `None` for every other node it touched, because an upward distance or
    /// a pruned label is not the answer and would be a lie told in the right
    /// shape. See [`MeetingSearch::cost`](crate::MeetingSearch::cost) and
    /// [`TripBasedSearch::cost`](crate::TripBasedSearch::cost).
    fn cost(&self, node: NodeId) -> Option<u32>;
    /// How many things the search settled.
    fn settled(&self) -> usize;
}

/// A planner that runs a query and keeps what it found.
///
/// The query is a per-kernel struct rather than a shared bag of options: a
/// knob the kernel does not have is a field it does not declare, and passing
/// it is a compile error rather than a value nobody reads.
pub trait Searches {
    /// How a start is written: `(node, cost)`, a bare node, `(stop, time)`.
    type Source: Copy;
    /// The knobs this kernel takes.
    type Query;
    /// What it hands back.
    type Search: Distances;
    /// What can go wrong. [`std::convert::Infallible`] for a kernel that
    /// silently ignores ids it does not know.
    type Error: std::error::Error;

    fn search(
        &self,
        sources: &[Self::Source],
        query: &Self::Query,
    ) -> Result<Self::Search, Self::Error>;
}

/// Reads a graph answer off a search, in the graph's own edges.
pub trait Unpacks: Searches {
    fn edge_path(&self, search: &Self::Search, to: NodeId) -> Option<Vec<EdgeId>>;
}

/// Reads a transit answer off a search.
pub trait Reads: Searches {
    fn itinerary(&self, search: &Self::Search, to: NodeId) -> Option<Itinerary>;
}

/// A front of incomparable answers, fewest changes first.
pub trait Front: Reads {
    fn itineraries(&self, search: &Self::Search, to: NodeId) -> Vec<Itinerary>;
}

/// What a search looked at, in a shape something can draw.
///
/// On the planner rather than the search because one kernel needs itself to
/// spell it out — a trip-based search reaches trips, and only the kernel knows
/// which stops those trips call at.
pub trait Explored: Searches {
    /// One thing reached: a stop and when, a stop and in which round, a
    /// trip and the stops along it.
    type Step;
    fn reached(&self, search: &Self::Search) -> Vec<Self::Step>;
}

/// The one-shot graph question: how far to there.
pub trait Distance {
    fn distance(
        &self,
        sources: &[(NodeId, Weight)],
        to: NodeId,
    ) -> Result<Option<Weight>, SearchError>;
}

/// The one-shot transit question every schedule-based kernel answers.
pub trait EarliestArrival {
    /// The earliest you can be at `to`, having stood at each source from the
    /// time given, or `None` if nothing gets you there.
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary>;
}

/// Every journey worth leaving on within a window, `(departs, itinerary)`,
/// earliest departure first and none dominated — in whatever criteria the
/// kernel tells journeys apart by, which is arrival for most and arrival
/// *and* changes for the trip-based query.
///
/// Named for what it returns rather than `profile`, because three kernels
/// already have a `profile` and each returns something of its own shape.
pub trait Profiled {
    fn departures(
        &self,
        from: NodeId,
        to: NodeId,
        opens: Time,
        closes: Time,
    ) -> Vec<(Time, Itinerary)>;
}
