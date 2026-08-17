//! Graph kernels for routing research.
//!
//! This crate is the Rust half of [routelab](https://github.com/bmander/routelab):
//! implementations of published routing algorithms, fast enough that their
//! constant factors mean something, behind an API stable enough that two
//! implementations of the same problem can be swapped and compared.
//!
//! It starts where the literature starts — a static, non-negatively weighted
//! directed graph and the two searches everything else is built on:
//!
//! ```
//! use routelab_core::{dijkstra, Graph, SearchOptions};
//!
//! let graph = Graph::from_edges(4, &[(0, 1, 60), (1, 3, 120), (0, 2, 90), (2, 3, 30)])?;
//! let result = dijkstra(&graph, &[(0, 0)], &SearchOptions::default())?;
//! assert_eq!(result.cost(3), Some(120));
//! assert_eq!(result.path(3), Some(vec![0, 2, 3]));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod kernels;
pub mod model;
pub mod util;

pub use kernels::astar::astar;
pub use kernels::bfs::bfs;
pub use kernels::contraction::{
    ContractionHierarchy, CoreHierarchy, Expansion, Half, MeetingSearch, Ordering, Policy,
};
pub use kernels::csa::{ConnectionScan, ScanProfile, ScanSearch};
pub use kernels::dijkstra::dijkstra;
pub use kernels::heuristics::{HeuristicError, StandardHeuristic};
pub use kernels::landmark::{Landmarks, Selection};
pub use kernels::lcspp::{label_constrained, Modes, Multimodal};
pub use kernels::ptl::PublicTransitLabeling;
pub use kernels::raptor::{Raptor, RaptorSearch};
pub use kernels::timedep::{
    time_dependent_dijkstra, Calendar, Clock, Departure, Waiting, Window, WEEK,
};
pub use kernels::timetable::TimeExpanded;
pub use kernels::tripbased::{TripBased, TripBasedProfile, TripBasedSearch};
pub use kernels::ultra::Ultra;
pub use model::graph::{EdgeId, Graph, GraphError, NodeId, Weight, NO_EDGE, NO_NODE, UNREACHABLE};
pub use model::heuristic::Heuristic;
pub use model::lines::{Lines, StopTime};
pub use model::search::{SearchError, SearchOptions, SearchResult};
pub use model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Ride, Timetable, Transfer, TripId, Walk,
};
pub use model::tree::{Magnitude, SearchTree};
pub use util::progress::Progress;
