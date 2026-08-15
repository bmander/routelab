//! A compact, immutable directed graph in compressed-sparse-row (CSR) form.

use std::fmt;

/// Index of a node. Node ids are dense: `0..num_nodes`.
pub type NodeId = u32;
/// Index of an edge in CSR order. See [`Graph::from_edges`] for what that order is.
pub type EdgeId = u32;
/// Edge cost. Non-negative by construction; conventionally seconds.
pub type Weight = u32;

/// Cost of a node that was never reached.
pub const UNREACHABLE: Weight = Weight::MAX;
/// Sentinel for "no such node" (used for the parent of a search root).
pub const NO_NODE: NodeId = NodeId::MAX;
/// Sentinel for "no such edge".
pub const NO_EDGE: EdgeId = EdgeId::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// An edge referenced a node id outside `0..num_nodes`.
    NodeOutOfRange { node: NodeId, num_nodes: usize },
    /// More nodes or edges than the `u32` id space allows.
    TooLarge { what: &'static str, count: usize },
    /// A weight equal to [`UNREACHABLE`], which the search treats as infinity.
    InfiniteWeight { edge: usize },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NodeOutOfRange { node, num_nodes } => {
                write!(
                    f,
                    "node {node} is out of range for a graph with {num_nodes} nodes"
                )
            }
            GraphError::TooLarge { what, count } => {
                write!(f, "too many {what}: {count} exceeds the u32 id space")
            }
            GraphError::InfiniteWeight { edge } => write!(
                f,
                "edge {edge} has weight {UNREACHABLE}, which is reserved to mean 'unreachable'"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// An immutable directed graph with `u32` weights, stored as CSR.
///
/// Parallel edges and self-loops are allowed; nothing is deduplicated. Build one
/// with [`Graph::from_edges`], which is the only way to get a `Graph` — the
/// invariants (sorted offsets, in-range heads) hold for every instance.
#[derive(Debug, Clone)]
pub struct Graph {
    /// `offsets[u]..offsets[u + 1]` is the range of out-edges of `u`. Length `num_nodes + 1`.
    offsets: Vec<EdgeId>,
    /// Target node of each edge, in CSR order.
    heads: Vec<NodeId>,
    /// Weight of each edge, in CSR order.
    weights: Vec<Weight>,
    /// Position of each edge in the input edge list, in CSR order. Lets callers
    /// carry their own per-edge attributes (mode, trip id, ...) alongside the graph.
    input_indices: Vec<u32>,
}

impl Graph {
    /// Build a graph from an edge list of `(tail, head, weight)` triples.
    ///
    /// Edges are permuted into CSR order: sorted by tail, ties broken by position
    /// in `edges`. Use [`Graph::input_index`] to recover where an edge came from.
    pub fn from_edges(
        num_nodes: usize,
        edges: &[(NodeId, NodeId, Weight)],
    ) -> Result<Self, GraphError> {
        if num_nodes > NodeId::MAX as usize {
            return Err(GraphError::TooLarge {
                what: "nodes",
                count: num_nodes,
            });
        }
        if edges.len() > EdgeId::MAX as usize {
            return Err(GraphError::TooLarge {
                what: "edges",
                count: edges.len(),
            });
        }

        // Counting sort by tail: first count out-degrees, then prefix-sum them
        // into offsets, then place each edge at its tail's running cursor.
        let mut offsets = vec![0 as EdgeId; num_nodes + 1];
        for (i, &(tail, head, weight)) in edges.iter().enumerate() {
            for node in [tail, head] {
                if node as usize >= num_nodes {
                    return Err(GraphError::NodeOutOfRange { node, num_nodes });
                }
            }
            if weight == UNREACHABLE {
                return Err(GraphError::InfiniteWeight { edge: i });
            }
            offsets[tail as usize + 1] += 1;
        }
        for u in 0..num_nodes {
            offsets[u + 1] += offsets[u];
        }

        let mut heads = vec![0 as NodeId; edges.len()];
        let mut weights = vec![0 as Weight; edges.len()];
        let mut input_indices = vec![0u32; edges.len()];
        let mut cursor: Vec<EdgeId> = offsets[..num_nodes].to_vec();
        for (i, &(tail, head, weight)) in edges.iter().enumerate() {
            let slot = &mut cursor[tail as usize];
            heads[*slot as usize] = head;
            weights[*slot as usize] = weight;
            input_indices[*slot as usize] = i as u32;
            *slot += 1;
        }

        Ok(Graph {
            offsets,
            heads,
            weights,
            input_indices,
        })
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.heads.len()
    }

    /// The out-edge ids of `node`, in CSR order. Empty if `node` is out of range.
    #[inline]
    pub fn out_edges(&self, node: NodeId) -> std::ops::Range<EdgeId> {
        if (node as usize) < self.num_nodes() {
            self.offsets[node as usize]..self.offsets[node as usize + 1]
        } else {
            0..0
        }
    }

    #[inline]
    pub fn out_degree(&self, node: NodeId) -> usize {
        let range = self.out_edges(node);
        (range.end - range.start) as usize
    }

    #[inline]
    pub fn head(&self, edge: EdgeId) -> NodeId {
        self.heads[edge as usize]
    }

    #[inline]
    pub fn weight(&self, edge: EdgeId) -> Weight {
        self.weights[edge as usize]
    }

    /// The tail of `edge`, found by binary search over the offsets. `O(log n)`,
    /// so prefer iterating [`Graph::out_edges`] when you already know the tail.
    pub fn tail(&self, edge: EdgeId) -> NodeId {
        debug_assert!((edge as usize) < self.num_edges());
        // `partition_point` gives the first offset strictly greater than `edge`;
        // the node owning `edge` is one before that.
        (self.offsets.partition_point(|&off| off <= edge) - 1) as NodeId
    }

    /// Where `edge` sat in the edge list passed to [`Graph::from_edges`].
    #[inline]
    pub fn input_index(&self, edge: EdgeId) -> u32 {
        self.input_indices[edge as usize]
    }

    /// `(tail, head, weight)` of `edge`.
    pub fn edge(&self, edge: EdgeId) -> (NodeId, NodeId, Weight) {
        (self.tail(edge), self.head(edge), self.weight(edge))
    }

    /// Every edge as `(tail, head, weight)`, in CSR order.
    ///
    /// Walks the offsets once rather than asking each edge for its tail, which
    /// would binary-search per edge.
    pub fn iter_edges(&self) -> impl Iterator<Item = (NodeId, NodeId, Weight)> + '_ {
        (0..self.num_nodes() as NodeId).flat_map(move |tail| {
            self.out_edges(tail)
                .map(move |edge| (tail, self.head(edge), self.weight(edge)))
        })
    }

    /// Follow a sequence of edges from `start`, returning the node it ends at and
    /// the total weight. `None` if the edges do not form a walk from `start`.
    ///
    /// This is the independent checker that makes a returned path falsifiable: a
    /// path is only correct if walking it lands on the target at the reported cost.
    pub fn walk(&self, start: NodeId, edges: &[EdgeId]) -> Option<(NodeId, Weight)> {
        if start as usize >= self.num_nodes() {
            return None;
        }
        let mut node = start;
        let mut cost: Weight = 0;
        for &edge in edges {
            // The current node is already known, so membership in its out-edge
            // range settles both "is this a real edge" and "does it start here"
            // without the binary search `tail` would need.
            if !self.out_edges(node).contains(&edge) {
                return None;
            }
            cost = cost.checked_add(self.weight(edge))?;
            node = self.head(edge);
        }
        Some((node, cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond() -> Graph {
        // 0 -> 1 -> 3 (cost 3), 0 -> 2 -> 3 (cost 30)
        Graph::from_edges(4, &[(0, 1, 1), (0, 2, 10), (1, 3, 2), (2, 3, 20)]).unwrap()
    }

    #[test]
    fn csr_layout_groups_edges_by_tail() {
        let g = Graph::from_edges(3, &[(2, 0, 5), (0, 1, 7), (2, 1, 9)]).unwrap();
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.out_degree(0), 1);
        assert_eq!(g.out_degree(1), 0);
        assert_eq!(g.out_degree(2), 2);
        // Edge ids follow CSR order, but input_index recovers the caller's order.
        assert_eq!(g.edge(0), (0, 1, 7));
        assert_eq!(g.input_index(0), 1);
        assert_eq!(g.edge(1), (2, 0, 5));
        assert_eq!(g.edge(2), (2, 1, 9));
    }

    #[test]
    fn tail_is_consistent_with_out_edges() {
        let g = diamond();
        for node in 0..g.num_nodes() as NodeId {
            for edge in g.out_edges(node) {
                assert_eq!(g.tail(edge), node);
            }
        }
    }

    #[test]
    fn walk_validates_edge_sequences() {
        let g = diamond();
        let path: Vec<EdgeId> = g
            .out_edges(0)
            .filter(|&e| g.head(e) == 1)
            .chain(g.out_edges(1))
            .collect();
        assert_eq!(g.walk(0, &path), Some((3, 3)));
        assert_eq!(
            g.walk(1, &path),
            None,
            "walk must start at the first edge's tail"
        );
        assert_eq!(g.walk(0, &[NO_EDGE]), None);
    }

    #[test]
    fn rejects_malformed_edge_lists() {
        assert_eq!(
            Graph::from_edges(2, &[(0, 2, 1)]).unwrap_err(),
            GraphError::NodeOutOfRange {
                node: 2,
                num_nodes: 2
            }
        );
        assert_eq!(
            Graph::from_edges(2, &[(0, 1, UNREACHABLE)]).unwrap_err(),
            GraphError::InfiniteWeight { edge: 0 }
        );
    }

    #[test]
    fn allows_self_loops_and_parallel_edges() {
        let g = Graph::from_edges(2, &[(0, 0, 1), (0, 1, 2), (0, 1, 3)]).unwrap();
        assert_eq!(g.out_degree(0), 3);
    }
}
