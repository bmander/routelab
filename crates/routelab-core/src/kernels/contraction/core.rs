//! Core-CH: contracting a network down to the places that matter.
//!
//! The variant of contraction the multimodal literature runs on, named in
//! Baum, Buchhold, Sauer, Wagner & Zündorf's ULTRA paper as *Core-CH* and used
//! there and by UCCH and MCR before it. The hierarchy beside this one
//! contracts every vertex and answers a query by climbing; this one is asked a
//! different question. A timetable technique wants to walk between **stops**,
//! and a city's pavements are half a million vertices of which a few thousand
//! are stops. Contracting all the others leaves a graph over the stops alone,
//! whose arcs already carry the walking distance through everything that went
//! — the same shortest paths, over a graph small enough to search.
//!
//! ```text
//!   554,393 street vertices          6,313 stops
//!   1,480,122 arcs           ->      arcs that are whole walks
//! ```
//!
//! Two rules make it Core-CH rather than contraction with a list. A vertex in
//! the **core** is never contracted, whatever its priority — that is what
//! guarantees every stop survives. And contraction **stops early**, once the
//! vertices left standing average more than a given degree. That second rule
//! is not an optimisation but a necessity, and the paper says why: a core of
//! stops alone has arcs quadratic in the stops, "which slows down both the
//! precomputation and query algorithms to the point where they become
//! impractical". The last vertices to go are the most connected, and each
//! costs more shortcuts than the one before; stopping leaves some street
//! corners in the core and keeps it sparse.
//!
//! What comes out is a graph over the original numbering, with arcs only
//! between vertices that survived. A stop is numbered as it always was, so the
//! core is a transfer graph any kernel can read without a translation table.
//!
//! ## What is faithful, and what is not here
//!
//! The two rules, over the same contraction and the same bounded witness
//! searches the hierarchy uses — so the core's distances are exact, which is
//! this module's test. Not here: the **upward and downward searches** that
//! answer a query from a vertex the contraction retired. Every query this
//! serves starts and ends at a core vertex, which for a timetable technique
//! means a stop; routing from an arbitrary doorway needs those searches and is
//! its own increment.

use super::{Builder, Ordering};
use crate::model::graph::{Graph, GraphError, NodeId};
use crate::util::progress::Progress;

/// A network contracted down to its core, and the core itself.
#[derive(Debug, Clone)]
pub struct CoreHierarchy {
    core: Graph,
    standing: Vec<bool>,
    retired: usize,
}

impl CoreHierarchy {
    /// Contract `graph` around `keep`, stopping when the core averages more
    /// than `max_degree` arcs a vertex.
    ///
    /// `keep` is the vertices that must survive — the stops, for a transfer
    /// graph. Anything else may go, and what is left is [`CoreHierarchy::core`].
    pub fn build(
        graph: &Graph,
        keep: &[NodeId],
        ordering: Ordering,
        max_degree: f64,
    ) -> Result<Self, GraphError> {
        Self::build_reporting(graph, keep, ordering, max_degree, &Progress::new())
    }

    /// [`CoreHierarchy::build`], counting retired vertices into `progress`.
    pub fn build_reporting(
        graph: &Graph,
        keep: &[NodeId],
        ordering: Ordering,
        max_degree: f64,
        progress: &Progress,
    ) -> Result<Self, GraphError> {
        let mut kept = vec![false; graph.num_nodes()];
        for &node in keep {
            if (node as usize) < kept.len() {
                kept[node as usize] = true;
            }
        }
        let mut builder = Builder::new(graph, ordering);
        let retired = builder.contract_core(&kept, max_degree, progress);
        let edges = builder.core_edges();
        Ok(CoreHierarchy {
            core: Graph::from_edges(graph.num_nodes(), &edges)?,
            standing: builder.standing(),
            retired,
        })
    }

    /// The core: the vertices that survived and the arcs between them, in the
    /// original numbering, so a stop is the node it always was.
    pub fn core(&self) -> &Graph {
        &self.core
    }

    /// Did this vertex survive the contraction?
    pub fn is_core(&self, node: NodeId) -> bool {
        self.standing.get(node as usize).copied().unwrap_or(false)
    }

    /// How many vertices are in the core.
    pub fn num_core(&self) -> usize {
        self.standing.iter().filter(|standing| **standing).count()
    }

    /// How many were contracted away.
    pub fn num_retired(&self) -> usize {
        self.retired
    }

    /// Arcs in the core — what a search over it costs, against the network it
    /// stands in for.
    pub fn num_arcs(&self) -> usize {
        self.core.num_edges()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.core.footprint() + self.standing.len()
    }
}
