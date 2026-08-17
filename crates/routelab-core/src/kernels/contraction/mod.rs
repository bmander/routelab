//! Contraction hierarchies: exact routing by rewriting the graph.
//!
//! Geisberger, Sanders, Schultes and Delling, *Exact Routing in Large Road
//! Networks Using Contraction Hierarchies*.
//!
//! Every other technique here leaves the graph alone — a heuristic changes the
//! order nodes come off a queue, a landmark table sits beside the network. This
//! one rewrites it. Nodes are contracted one at a time, least important first,
//! and wherever removing a node would have lengthened a shortest path a
//! **shortcut** is inserted to stand in for it. The result is the original graph
//! plus shortcuts, and a rank per node saying when it went.
//!
//! What that buys is a query that only ever climbs. Search forward from the
//! source and backward from the target, both following arcs to higher-ranked
//! nodes only, and the two halves meet above the trip. Neither half has to look
//! sideways at the thousands of residential streets between them, because the
//! shortcuts already carry that distance.
//!
//! Two things are worth knowing before reading the code. The answers are
//! **exact** — this is not an approximation, and the tests hold it to Dijkstra's
//! numbers on every instance. And the shortcuts are a *superset* of what is
//! strictly needed: the witness searches that decide whether a shortcut is
//! necessary are deliberately bounded, so an unnecessary one costs space and
//! query time but never accuracy.

mod build;
mod core;
mod query;

#[cfg(test)]
mod core_tests;
#[cfg(test)]
mod tests;

use crate::model::graph::{EdgeId, Graph, GraphError, NodeId, Weight};
use crate::model::search::SearchError;
use crate::util::progress::Progress;
use build::Builder;

pub use core::CoreHierarchy;
pub use query::{Half, MeetingSearch};

/// What an arc of the augmented graph stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expansion {
    /// An edge of the original graph, by its id there.
    Original(EdgeId),
    /// A shortcut over a contracted node, standing for two augmented arcs in
    /// order. Expanding recursively bottoms out in originals.
    Shortcut { first: u32, second: u32 },
}

/// How to decide which node to contract next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// The paper's practical recipe: prefer nodes whose contraction adds fewer
    /// shortcuts than it removes edges, and spread the work out by discouraging
    /// contraction next to already-contracted nodes.
    EdgeDifference { deleted_neighbours: bool },
    /// Contract in a fixed arbitrary order. The control: a hierarchy is only
    /// worth its preprocessing if it beats this, and on a road network it beats
    /// it by a very long way.
    Random { seed: u64 },
}

/// The knobs on preprocessing: which node next, and how hard to look for a
/// witness before giving up and inserting a shortcut.
#[derive(Debug, Clone, Copy)]
pub struct Ordering {
    pub policy: Policy,
    /// How many nodes a witness search may settle.
    pub max_settled: usize,
    /// How many hops a witness search may take.
    pub max_hops: usize,
}

impl Default for Ordering {
    fn default() -> Self {
        Ordering {
            policy: Policy::EdgeDifference {
                deleted_neighbours: true,
            },
            max_settled: 500,
            max_hops: 5,
        }
    }
}

impl Ordering {
    /// Whether a node's priority is worth recomputing when it comes off the
    /// queue. A fixed random order has nothing to recompute.
    fn recomputes(&self) -> bool {
        matches!(self.policy, Policy::EdgeDifference { .. })
    }
}

/// A contracted graph and the ranks that make it searchable.
#[derive(Debug, Clone)]
pub struct ContractionHierarchy {
    ranks: Vec<u32>,
    /// What each augmented arc stands for, indexed as the arcs were built.
    expansion: Vec<Expansion>,
    /// Arcs from lower rank to higher, for the search from the source.
    upward: Graph,
    /// Arcs from higher rank to lower, **reversed**, so the search from the
    /// target also only ever climbs.
    downward: Graph,
    up_augmented: Vec<u32>,
    down_augmented: Vec<u32>,
}

impl ContractionHierarchy {
    /// Contract a graph. This is the expensive step, paid once.
    pub fn build(graph: &Graph, ordering: Ordering) -> Result<Self, GraphError> {
        ContractionHierarchy::build_reporting(graph, ordering, &Progress::new())
    }

    /// Contract a graph, counting nodes retired into `progress`.
    ///
    /// Six seconds on a city, which is long enough that something watching
    /// wants to tell working from hung. See [`crate::util::progress`] for why this is
    /// a second entry point rather than a parameter on the first.
    pub fn build_reporting(
        graph: &Graph,
        ordering: Ordering,
        progress: &Progress,
    ) -> Result<Self, GraphError> {
        let mut builder = Builder::new(graph, ordering);
        let ranks = builder.contract_all(progress);
        let edges = builder.edges;

        // The second half, and not a small one: on a walking network this is a
        // third of the wall clock. Counted separately, because a fraction that
        // ignored it sat at 100% while it ran.
        progress.expect("assembling", edges.len() as u64);
        let mut upward = Vec::new();
        let mut downward = Vec::new();
        let mut up_augmented = Vec::new();
        let mut down_augmented = Vec::new();
        for (index, edge) in edges.iter().enumerate() {
            if ranks[edge.tail as usize] < ranks[edge.head as usize] {
                up_augmented.push(index as u32);
                upward.push((edge.tail, edge.head, edge.weight));
            } else {
                // Stored back to front: the backward search reads it as if the
                // arc pointed uphill, which is the only direction it walks.
                down_augmented.push(index as u32);
                downward.push((edge.head, edge.tail, edge.weight));
            }
            progress.step();
        }

        Ok(ContractionHierarchy {
            expansion: edges.iter().map(|edge| edge.expansion).collect(),
            upward: Graph::from_edges(graph.num_nodes(), &upward)?,
            downward: Graph::from_edges(graph.num_nodes(), &downward)?,
            up_augmented,
            down_augmented,
            ranks,
        })
    }

    /// The cheapest path from `sources` to `target`, and what it took to find.
    pub fn query(
        &self,
        sources: &[(NodeId, Weight)],
        target: NodeId,
    ) -> Result<MeetingSearch, SearchError> {
        query::query(self, sources, target)
    }

    pub fn num_nodes(&self) -> usize {
        self.ranks.len()
    }

    /// How many arcs the hierarchy holds, originals included.
    pub fn num_arcs(&self) -> usize {
        self.expansion.len()
    }

    /// How many of those are shortcuts — the number that says whether an
    /// ordering was any good.
    pub fn num_shortcuts(&self) -> usize {
        self.expansion
            .iter()
            .filter(|expansion| matches!(expansion, Expansion::Shortcut { .. }))
            .count()
    }

    /// Bytes held by the hierarchy's own arrays.
    pub fn footprint(&self) -> usize {
        self.ranks.len() * std::mem::size_of::<u32>()
            + self.expansion.len() * std::mem::size_of::<Expansion>()
            + (self.up_augmented.len() + self.down_augmented.len()) * std::mem::size_of::<u32>()
    }

    /// The rank a node was contracted at; higher means more important.
    pub fn rank(&self, node: NodeId) -> u32 {
        self.ranks[node as usize]
    }

    pub fn upward(&self) -> &Graph {
        &self.upward
    }

    pub fn downward(&self) -> &Graph {
        &self.downward
    }

    /// What an arc of one of the search graphs stands for.
    pub fn expansion_of(&self, augmented: u32) -> Expansion {
        self.expansion[augmented as usize]
    }

    /// The augmented arc behind an edge of the upward search graph.
    pub fn upward_edge(&self, edge: EdgeId) -> u32 {
        self.up_augmented[self.upward.input_index(edge) as usize]
    }

    /// The augmented arc behind an edge of the downward search graph.
    pub fn downward_edge(&self, edge: EdgeId) -> u32 {
        self.down_augmented[self.downward.input_index(edge) as usize]
    }
}
