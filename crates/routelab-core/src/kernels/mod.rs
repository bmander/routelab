//! The techniques, one module per paper.
//!
//! Each is a published algorithm implemented as its paper states it, reading
//! the vocabulary in [`crate::model`] — a graph, search options, a result.
//!
//! Two modules here are not papers but belong on this side of the line all the
//! same, because both name techniques and so cannot sit underneath them:
//! [`heuristics`] is the closed set of heuristics the library ships, ALT's
//! landmarks included, and [`footpaths`] is the search that closes foot-edges
//! under composition before any transit kernel reads them.
//!
//! A new technique arrives as a new module beside these, not as a
//! generalisation of them.

pub mod astar;
pub mod bfs;
pub mod contraction;
pub mod csa;
pub mod dijkstra;
pub mod footpaths;
pub mod heuristics;
pub mod landmark;
#[cfg(test)]
pub(crate) mod oracles;
pub mod ptl;
pub mod raptor;
pub mod timedep;
pub mod timetable;
pub mod tripbased;
