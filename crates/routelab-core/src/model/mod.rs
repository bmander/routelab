//! What every kernel speaks.
//!
//! The graph they search, the options they take, the results they return, and
//! the timetable structures the schedule-based ones read. A type belongs here
//! when more than one paper reads it, which is the whole test — [`timetable`]
//! is here rather than beside one transit kernel precisely because every one of
//! them reads it, and [`lines`] because two do.
//!
//! Nothing here decides anything, and nothing here names a kernel. Where a
//! structure needs work done to build it, that work lives with the techniques
//! and hands the finished thing back: closing footpaths under composition is a
//! search, so it is [`crate::kernels::footpaths`], while the container it fills
//! is [`timetable::Footpaths`]. The dependency runs one way, and it runs down.

pub mod graph;
pub mod heuristic;
pub mod lines;
pub mod search;
pub mod timetable;
pub mod tree;
