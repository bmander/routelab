//! The shared shape of a one-to-all / one-to-many search: what it produced and
//! what it was asked for.

use crate::graph::{EdgeId, Graph, NodeId, Weight, NO_EDGE, NO_NODE, UNREACHABLE};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    /// A source node id that does not exist in the graph.
    SourceOutOfRange { node: NodeId, num_nodes: usize },
    /// A source whose initial cost is [`UNREACHABLE`], which would mean "start
    /// from a place you cannot be".
    InfiniteSourceCost { node: NodeId },
    /// A goal node id that does not exist in the graph. Distinct from a bad
    /// source: only a goal-directed search has one to get wrong.
    TargetOutOfRange { node: NodeId, num_nodes: usize },
    /// A heuristic holding data for a different graph than the one being searched.
    HeuristicCoverage {
        heuristic_nodes: usize,
        num_nodes: usize,
    },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::SourceOutOfRange { node, num_nodes } => {
                write!(
                    f,
                    "source node {node} is out of range for a graph with {num_nodes} nodes"
                )
            }
            SearchError::InfiniteSourceCost { node } => {
                write!(f, "source node {node} has infinite initial cost")
            }
            SearchError::TargetOutOfRange { node, num_nodes } => {
                write!(
                    f,
                    "target node {node} is out of range for a graph with {num_nodes} nodes"
                )
            }
            SearchError::HeuristicCoverage {
                heuristic_nodes,
                num_nodes,
            } => write!(
                f,
                "heuristic covers {heuristic_nodes} nodes but the graph has {num_nodes}"
            ),
        }
    }
}

impl std::error::Error for SearchError {}

/// Bounds on a search. The defaults (`SearchOptions::default()`) mean
/// "settle everything reachable".
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Stop once [`SearchOptions::reach`] many of these have been settled.
    /// `None` means run to exhaustion. Ids outside the graph are unreachable,
    /// so they suppress the early exit.
    pub targets: Option<Vec<NodeId>>,
    /// How much of `targets` the search is waiting for.
    pub reach: Reach,
    /// Do not settle nodes whose cost exceeds this (inclusive bound), where
    /// "cost" is whatever the search puts in [`SearchResult::costs`] — summed
    /// edge weights for Dijkstra, hop count for BFS.
    pub max_cost: Option<Weight>,
}

impl SearchOptions {
    /// Stop once **every** one of `targets` is settled.
    pub fn with_targets(mut self, targets: impl IntoIterator<Item = NodeId>) -> Self {
        self.targets = Some(targets.into_iter().collect());
        self.reach = Reach::All;
        self
    }

    /// Stop as soon as **any** one of `targets` is settled.
    ///
    /// For a search that settles in cost order — which is every search here —
    /// the first of a set to be settled is the cheapest of them. That makes
    /// this the right question whenever a caller wants "the nearest of these"
    /// rather than "all of these", and asking the other way round can be the
    /// difference between visiting a fifth of a graph and visiting all of it.
    pub fn with_any_target(mut self, targets: impl IntoIterator<Item = NodeId>) -> Self {
        self.targets = Some(targets.into_iter().collect());
        self.reach = Reach::Any;
        self
    }

    pub fn with_max_cost(mut self, max_cost: Weight) -> Self {
        self.max_cost = Some(max_cost);
        self
    }
}

/// The shortest-path tree produced by a search, plus the order it was built in.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Cost of the best path found to each node; [`UNREACHABLE`] if none.
    pub costs: Vec<Weight>,
    /// Predecessor of each node in the tree; [`NO_NODE`] for roots and unreached nodes.
    pub parent_nodes: Vec<NodeId>,
    /// Edge used to reach each node; [`NO_EDGE`] for roots and unreached nodes.
    pub parent_edges: Vec<EdgeId>,
    /// Nodes in the order they were settled. The search's trace, for teaching,
    /// visualization, and comparing implementations that agree on distances but
    /// not on tie-breaking.
    pub order: Vec<NodeId>,
}

impl SearchResult {
    pub(crate) fn new(num_nodes: usize) -> Self {
        SearchResult {
            costs: vec![UNREACHABLE; num_nodes],
            parent_nodes: vec![NO_NODE; num_nodes],
            parent_edges: vec![NO_EDGE; num_nodes],
            order: Vec::new(),
        }
    }

    /// Cost to `node`, or `None` if it was not reached.
    pub fn cost(&self, node: NodeId) -> Option<Weight> {
        match self.costs.get(node as usize) {
            Some(&UNREACHABLE) | None => None,
            Some(&cost) => Some(cost),
        }
    }

    /// The node `node` was reached from; `None` at a root or an unreached node.
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        match self.parent_nodes.get(node as usize) {
            Some(&NO_NODE) | None => None,
            Some(&parent) => Some(parent),
        }
    }

    /// The edge `node` was reached by; `None` at a root or an unreached node.
    pub fn parent_edge(&self, node: NodeId) -> Option<EdgeId> {
        match self.parent_edges.get(node as usize) {
            Some(&NO_EDGE) | None => None,
            Some(&edge) => Some(edge),
        }
    }

    /// Records `node` as settled and reports whether the search can stop —
    /// the bookkeeping every search does when it pops a node.
    pub(crate) fn settle(&mut self, node: NodeId, tracker: &mut Option<TargetTracker>) -> bool {
        self.order.push(node);
        match tracker {
            Some(tracker) => tracker.settle(node),
            None => false,
        }
    }

    /// The same bookkeeping for a search with exactly one goal, which needs no
    /// tracker to know when it is done.
    pub(crate) fn settle_toward(&mut self, node: NodeId, target: NodeId) -> bool {
        self.order.push(node);
        node == target
    }

    /// The nodes on the tree path to `node`, root first. `None` if unreached.
    pub fn path(&self, node: NodeId) -> Option<Vec<NodeId>> {
        self.cost(node)?;
        let mut path = vec![node];
        let mut current = node;
        while self.parent_nodes[current as usize] != NO_NODE {
            current = self.parent_nodes[current as usize];
            path.push(current);
        }
        path.reverse();
        Some(path)
    }

    /// The edges on the tree path to `node`, root first. `None` if unreached;
    /// empty if `node` is a root.
    pub fn edge_path(&self, node: NodeId) -> Option<Vec<EdgeId>> {
        self.cost(node)?;
        let mut edges = Vec::new();
        let mut current = node;
        while self.parent_edges[current as usize] != NO_EDGE {
            edges.push(self.parent_edges[current as usize]);
            current = self.parent_nodes[current as usize];
        }
        edges.reverse();
        Some(edges)
    }
}

/// Tracks which requested targets are still outstanding, so a search can stop
/// as soon as the last one is settled.
/// How many of a search's targets it is waiting for before it may stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reach {
    /// Every target. The conservative reading, and the default.
    #[default]
    All,
    /// The first one settled, which in a cost-ordered search is the cheapest.
    Any,
}

pub(crate) struct TargetTracker {
    is_target: Vec<bool>,
    remaining: usize,
}

impl TargetTracker {
    pub(crate) fn new(
        targets: &Option<Vec<NodeId>>,
        reach: Reach,
        num_nodes: usize,
    ) -> Option<Self> {
        let targets = targets.as_ref()?;
        let mut is_target = vec![false; num_nodes];
        let mut remaining = 0;
        for &node in targets {
            // Out-of-range targets can never be settled; counting them would mean
            // the search never exits early, which is the honest behaviour.
            match is_target.get_mut(node as usize) {
                Some(flag) if !*flag => {
                    *flag = true;
                    remaining += 1;
                }
                Some(_) => {}
                None => remaining += 1,
            }
        }
        Some(TargetTracker {
            is_target,
            remaining: match reach {
                Reach::All => remaining,
                Reach::Any => remaining.min(1),
            },
        })
    }

    /// Records that `node` was settled; returns true when nothing is left to find.
    #[inline]
    pub(crate) fn settle(&mut self, node: NodeId) -> bool {
        if self.is_target[node as usize] {
            self.is_target[node as usize] = false;
            self.remaining -= 1;
        }
        self.remaining == 0
    }

    #[inline]
    pub(crate) fn done(&self) -> bool {
        self.remaining == 0
    }
}

/// Rejects sources that do not name a node in `graph`, so callers hear about a
/// typo'd id instead of silently searching from nowhere.
pub(crate) fn check_nodes(
    graph: &Graph,
    nodes: impl IntoIterator<Item = NodeId>,
) -> Result<(), SearchError> {
    for node in nodes {
        if node as usize >= graph.num_nodes() {
            return Err(SearchError::SourceOutOfRange {
                node,
                num_nodes: graph.num_nodes(),
            });
        }
    }
    Ok(())
}

/// As [`check_nodes`], plus the cost check that only searches whose sources
/// carry an initial cost can fail.
pub(crate) fn check_sources(
    graph: &Graph,
    sources: &[(NodeId, Weight)],
) -> Result<(), SearchError> {
    check_nodes(graph, sources.iter().map(|&(node, _)| node))?;
    match sources.iter().find(|&&(_, cost)| cost == UNREACHABLE) {
        Some(&(node, _)) => Err(SearchError::InfiniteSourceCost { node }),
        None => Ok(()),
    }
}
