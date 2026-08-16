//! The estimate A* asks for, as an interface.
//!
//! A heuristic is the one thing A* needs that the graph cannot supply. The graph
//! knows edges and weights; it does not know that two nodes are 400 metres apart,
//! or that nothing in this network moves faster than 25 m/s. That knowledge comes
//! from outside — in routelab, from the layers of an
//! [`Environment`](../../routelab/environment.py) — and arrives here as a compact
//! object the search can query in constant time.

use crate::model::graph::{NodeId, Weight};

/// A lower bound on the cost of getting from `node` to `target`.
///
/// The bound must be **admissible**: never greater than the true remaining cost.
/// An overestimate makes A* faster and wrong — it will happily return a path that
/// is not the cheapest, with nothing in the result to say so.
///
/// It should also be **consistent** (`h(a) <= w(a, b) + h(b)` for every edge),
/// which is what lets the search settle each node once and never revisit it. Every
/// heuristic here is consistent; a custom one that is merely admissible needs
/// re-expansion that [`crate::astar`] does not do.
pub trait Heuristic {
    /// The estimated cost from `node` to `target`.
    fn estimate(&self, node: NodeId, target: NodeId) -> Weight;

    /// How many nodes this heuristic holds data for, if it is node-indexed.
    ///
    /// Lets a search reject a heuristic built for a different graph up front,
    /// rather than reading past the end of its tables mid-query.
    fn coverage(&self) -> Option<usize> {
        None
    }

    /// Bytes of precomputed table this heuristic holds.
    ///
    /// Zero for one that computes rather than remembers. What a caller weighs
    /// against how much the bound sharpens.
    fn footprint(&self) -> usize {
        0
    }
}
