//! What every kernel speaks.
//!
//! The graph they search, the options they take, the results they return, and
//! the structures the schedule-based ones read. Nothing here decides anything;
//! these are the types that let two implementations of one problem be swapped
//! and compared.

pub mod graph;
pub mod heuristic;
pub mod search;
pub mod tree;
