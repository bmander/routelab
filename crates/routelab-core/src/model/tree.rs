//! The shape a search left behind, in a form something can draw.
//!
//! Dijkstra and A* both explore by growing a shortest-path tree: every settled
//! node knows the edge it was reached by, and those edges form a tree rooted at
//! the sources. That tree *is* the search space these algorithms explored, and
//! it is worth looking at — where the search went, where it wasted its time,
//! how differently two algorithms answer the same question.
//!
//! Drawn plainly, a tree of a hundred thousand equal-weight lines says very
//! little. What makes it legible is accumulation: give each branch the total of
//! everything hanging off it, and the tree renders like a river network — thick
//! trunks near the root where all the traffic passes, thinning to capillaries at
//! the frontier. The trunk is where a search committed; the capillaries are
//! where it was still guessing.

use crate::model::graph::{EdgeId, Graph, NodeId};
use crate::model::search::SearchResult;

/// What a branch's thickness should mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Magnitude {
    /// How many settled nodes lie beyond this branch, itself included. Reads as
    /// "how much of the search flowed through here".
    Nodes,
    /// The summed weight of every branch beyond this one, itself included. In a
    /// network priced in seconds, the travel time the branch leads to.
    Weight,
}

/// A shortest-path tree with each branch weighted by what lies beyond it.
///
/// One entry per settled node that has a parent — roots have no branch, since a
/// branch is an edge and a root was not reached by one.
#[derive(Debug, Clone, Default)]
pub struct SearchTree {
    pub tails: Vec<NodeId>,
    pub heads: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    /// Accumulated total of everything beyond each branch, itself included.
    pub magnitudes: Vec<u64>,
}

impl SearchTree {
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// The largest magnitude, which is what a renderer scales against.
    pub fn peak(&self) -> u64 {
        self.magnitudes.iter().copied().max().unwrap_or(0)
    }
}

impl SearchResult {
    /// The tree this search grew, with each branch carrying its subtree's total.
    ///
    /// Only settled nodes take part. A node the search touched but never settled
    /// has no final cost and no place in the tree — it was a guess the search
    /// had not yet confirmed or abandoned.
    pub fn tree(&self, graph: &Graph, magnitude: Magnitude) -> SearchTree {
        // Settle order is topological for the tree — a node is always settled
        // after the node it was reached from — so walking it backwards reaches
        // every child before its parent. By the time a node comes up, its whole
        // subtree has already poured into it; it adds its own contribution and
        // pours the lot into its parent.
        let mut totals = vec![0u64; self.costs.len()];
        for &node in self.order.iter().rev() {
            totals[node as usize] += match magnitude {
                Magnitude::Nodes => 1,
                Magnitude::Weight => self
                    .parent_edge(node)
                    .map_or(0, |edge| u64::from(graph.weight(edge))),
            };
            if let Some(parent) = self.parent(node) {
                totals[parent as usize] += totals[node as usize];
            }
        }

        let mut tree = SearchTree::default();
        for &node in &self.order {
            let (Some(parent), Some(edge)) = (self.parent(node), self.parent_edge(node)) else {
                continue; // a root: reached by nothing, so no branch to draw
            };
            tree.tails.push(parent);
            tree.heads.push(node);
            tree.edges.push(edge);
            tree.magnitudes.push(totals[node as usize]);
        }
        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dijkstra::dijkstra;
    use crate::model::search::SearchOptions;

    /// A trunk 0->1 that forks at 1 into 2 and 3, with 3 continuing to 4.
    /// Every edge costs 10, so magnitudes are easy to read off by hand.
    fn forked() -> Graph {
        Graph::from_edges(5, &[(0, 1, 10), (1, 2, 10), (1, 3, 10), (3, 4, 10)]).unwrap()
    }

    fn tree_of(magnitude: Magnitude) -> (Graph, SearchTree) {
        let graph = forked();
        let result = dijkstra(&graph, &[(0, 0)], &SearchOptions::default()).unwrap();
        let tree = result.tree(&graph, magnitude);
        (graph, tree)
    }

    fn magnitude_of(tree: &SearchTree, tail: NodeId, head: NodeId) -> u64 {
        let index = (0..tree.len())
            .find(|&i| tree.tails[i] == tail && tree.heads[i] == head)
            .unwrap_or_else(|| panic!("no branch {tail} -> {head}"));
        tree.magnitudes[index]
    }

    #[test]
    fn every_settled_node_but_the_root_is_a_branch() {
        let (_, tree) = tree_of(Magnitude::Nodes);
        assert_eq!(tree.len(), 4, "five nodes settled, one of them the root");
        assert!(!tree.tails.contains(&NodeId::MAX));
    }

    #[test]
    fn a_branch_counts_everything_beyond_it() {
        let (_, tree) = tree_of(Magnitude::Nodes);
        assert_eq!(magnitude_of(&tree, 0, 1), 4, "the trunk carries all four");
        assert_eq!(magnitude_of(&tree, 1, 3), 2, "3 and 4 beyond it");
        assert_eq!(magnitude_of(&tree, 1, 2), 1, "a leaf carries only itself");
        assert_eq!(magnitude_of(&tree, 3, 4), 1);
        assert_eq!(tree.peak(), 4);
    }

    #[test]
    fn weight_accumulates_the_travel_beyond_a_branch() {
        let (_, tree) = tree_of(Magnitude::Weight);
        assert_eq!(magnitude_of(&tree, 0, 1), 40, "its own 10, plus 30 beyond");
        assert_eq!(magnitude_of(&tree, 1, 3), 20);
        assert_eq!(magnitude_of(&tree, 1, 2), 10);
    }

    #[test]
    fn the_root_branches_carry_the_whole_search() {
        // Everything drains through the root, which is what makes the trunk
        // thickest and the picture readable.
        let (graph, tree) = tree_of(Magnitude::Weight);
        let result = dijkstra(&graph, &[(0, 0)], &SearchOptions::default()).unwrap();
        let total: u64 = result
            .order
            .iter()
            .filter_map(|&node| result.parent_edge(node))
            .map(|edge| u64::from(graph.weight(edge)))
            .sum();
        assert_eq!(magnitude_of(&tree, 0, 1), total);
    }

    #[test]
    fn an_unreached_graph_has_no_tree() {
        let graph = forked();
        let result = dijkstra(&graph, &[(4, 0)], &SearchOptions::default()).unwrap();
        assert!(
            result.tree(&graph, Magnitude::Nodes).is_empty(),
            "4 leads nowhere"
        );
    }

    #[test]
    fn several_roots_each_grow_their_own_trunk() {
        let graph = forked();
        let result = dijkstra(&graph, &[(0, 0), (3, 0)], &SearchOptions::default()).unwrap();
        let tree = result.tree(&graph, Magnitude::Nodes);
        // 3 is now a root, so 4 hangs off it rather than off the trunk at 1.
        assert_eq!(magnitude_of(&tree, 3, 4), 1);
        assert_eq!(
            magnitude_of(&tree, 0, 1),
            2,
            "1 and 2 only; 3 is its own root"
        );
    }
}
