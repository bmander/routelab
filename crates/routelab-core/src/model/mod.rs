//! What every kernel speaks.
//!
//! The graph they search, the options they take, the results they return, and
//! the timetable structures the schedule-based ones read. A type belongs here
//! when more than one paper reads it, which is the whole test — [`timetable`]
//! is here rather than beside one transit kernel precisely because all three
//! read it.
//!
//! These are descriptions, not decisions, with one exception worth naming
//! rather than hiding: building [`timetable::Footpaths`] closes the given links
//! under composition, and it does that by running
//! [`crate::kernels::dijkstra::dijkstra`] from each stop. So `model` depends on
//! one kernel, upward through the layering. It is a construction detail of a
//! shared structure rather than a routing choice, and the alternative — putting
//! a structure all three transit kernels read inside one of them — is worse.

pub mod graph;
pub mod heuristic;
pub mod search;
pub mod timetable;
pub mod tree;
