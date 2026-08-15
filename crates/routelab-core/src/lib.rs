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

pub mod astar;
pub mod bfs;
pub mod contraction;
pub mod dijkstra;
pub mod graph;
pub mod heuristic;
pub mod landmark;
pub(crate) mod rng;
pub mod search;
pub mod timedep;
pub mod timetable;
pub mod tree;

pub use astar::astar;
pub use bfs::bfs;
pub use contraction::{ContractionHierarchy, Expansion, Half, MeetingSearch, Ordering, Policy};
pub use dijkstra::dijkstra;
pub use graph::{EdgeId, Graph, GraphError, NodeId, Weight, NO_EDGE, NO_NODE, UNREACHABLE};
pub use heuristic::{Heuristic, HeuristicError, StandardHeuristic};
pub use landmark::{Landmarks, Selection};
pub use search::{SearchError, SearchOptions, SearchResult};
pub use timedep::{time_dependent_dijkstra, Calendar, Clock, Departure, Waiting, Window, WEEK};
pub use timetable::{
    time_dependent_query, Connection, Itinerary, Ride, Time, TimeExpanded, Timetable, Transfer,
};
pub use tree::{Magnitude, SearchTree};
