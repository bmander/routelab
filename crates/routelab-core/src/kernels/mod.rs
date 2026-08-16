//! One entry per paper.
//!
//! Each module here is a published algorithm, implemented as its paper states
//! it. They share the vocabulary in [`crate::model`] — a graph, search options,
//! a result — and otherwise know nothing about each other. A new technique
//! arrives as a new module beside these, not as a generalisation of them.

pub mod astar;
pub mod bfs;
pub mod contraction;
pub mod dijkstra;
pub mod landmark;
pub mod raptor;
pub mod timedep;
pub mod timetable;
