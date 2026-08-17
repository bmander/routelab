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

use super::{Builder, Ordering, UNRANKED};
use crate::model::graph::{Graph, GraphError, NodeId};
use crate::util::progress::Progress;

/// A network contracted down to its core, and the core itself.
#[derive(Debug, Clone)]
pub struct CoreHierarchy {
    core: Graph,
    /// Component arcs that climb, for a search from below into the core.
    upward: Graph,
    /// The same, reversed, for a search that ends below the core.
    downward: Graph,
    /// Rank per vertex, [`UNRANKED`] for one the contraction left standing —
    /// which is also what says whether a vertex is in the core, so there is no
    /// second table saying the same thing in another shape.
    ranks: Vec<u32>,
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
        let ranks = builder.contract_core(&kept, max_degree, progress);
        let retired = ranks.iter().filter(|rank| **rank != UNRANKED).count();
        let core = builder.core_edges();

        // The component's hierarchy, for a search that starts below the core and
        // has to climb into it. Arcs are split the way an ordinary contraction
        // hierarchy splits them — by which end ranks higher — except that an
        // arc between two core vertices goes into neither, because the core is
        // not searched by climbing. It is searched as a graph, and `core` is it.
        let mut upward = Vec::new();
        let mut downward = Vec::new();
        for edge in &builder.edges {
            let (tail, head) = (ranks[edge.tail as usize], ranks[edge.head as usize]);
            if tail == UNRANKED && head == UNRANKED {
                continue;
            }
            if tail < head {
                upward.push((edge.tail, edge.head, edge.weight));
            } else {
                // Stored back to front, as in [`ContractionHierarchy`]: the
                // search from the target reads it as if the arc pointed uphill,
                // which is the only direction it walks.
                downward.push((edge.head, edge.tail, edge.weight));
            }
        }

        Ok(CoreHierarchy {
            core: Graph::from_edges(graph.num_nodes(), &core)?,
            upward: Graph::from_edges(graph.num_nodes(), &upward)?,
            downward: Graph::from_edges(graph.num_nodes(), &downward)?,
            ranks,
            retired,
        })
    }

    /// The core: the vertices that survived and the arcs between them, in the
    /// original numbering, so a stop is the node it always was.
    pub fn core(&self) -> &Graph {
        &self.core
    }

    /// Where a vertex sits in the order it was contracted in, or
    /// [`UNRANKED`] if it never was — which is the same as being in the core.
    pub fn rank(&self, node: NodeId) -> u32 {
        self.ranks.get(node as usize).copied().unwrap_or(UNRANKED)
    }

    /// Component arcs from lower rank to higher: what a search from the source
    /// climbs to reach the core.
    pub fn upward(&self) -> &Graph {
        &self.upward
    }

    /// The same reversed, so a search from the target also only ever climbs.
    pub fn downward(&self) -> &Graph {
        &self.downward
    }

    /// Did this vertex survive the contraction?
    pub fn is_core(&self, node: NodeId) -> bool {
        self.rank(node) == UNRANKED
    }

    /// How many vertices are in the core.
    pub fn num_core(&self) -> usize {
        self.ranks.len() - self.retired
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
    /// Bytes held — the core, the two graphs a query climbs, and the ranks
    /// that tell them apart. All of it, because a footprint that leaves out the
    /// largest thing it holds is worse than none.
    pub fn footprint(&self) -> usize {
        self.core.footprint()
            + self.upward.footprint()
            + self.downward.footprint()
            + self.ranks.len() * std::mem::size_of::<u32>()
    }
}
