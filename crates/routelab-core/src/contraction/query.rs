//! The query: two searches climbing the hierarchy, meeting somewhere above.
//!
//! The halves are kept **sparse**, which is the whole reason this file does not
//! reuse [`crate::search::SearchResult`]. That type carries a cost, a parent and
//! an edge for every node in the graph, which is right for a search that might
//! settle all of them. A hierarchy query settles a couple of hundred nodes out
//! of a quarter of a million, so sizing its result to the map rather than to the
//! work spends more time clearing arrays than searching — a millisecond a query
//! on Seattle, against tens of microseconds once the result was made to fit what
//! the algorithm actually touches. Composing the shared type was the tidier
//! design and the measurement did not support it.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::ContractionHierarchy;
use crate::graph::{EdgeId, Graph, NodeId, Weight, UNREACHABLE};
use crate::search::SearchError;

/// How a node was reached by whichever half reached it.
#[derive(Debug, Clone, Copy)]
struct Reached {
    cost: Weight,
    parent: NodeId,
    edge: EdgeId,
}

/// One direction of the search: what it reached, and the order it settled.
#[derive(Debug, Clone, Default)]
pub struct Half {
    reached: HashMap<NodeId, Reached>,
    /// Nodes settled, in order — the trace every search here keeps.
    pub order: Vec<NodeId>,
}

impl Half {
    /// The cost of climbing to `node` from this half's own end.
    pub fn cost(&self, node: NodeId) -> Option<Weight> {
        self.reached.get(&node).map(|reached| reached.cost)
    }

    /// How this half arrived at `node`: the node before it and the arc between.
    ///
    /// One accessor rather than a `parent` and a `parent_edge`, because nothing
    /// wants one without the other and a sparse half pays a hash lookup for
    /// each. `None` at a root, which is stored as its own parent so a walk
    /// terminates without needing a sentinel.
    pub fn arrival(&self, node: NodeId) -> Option<(NodeId, EdgeId)> {
        self.reached
            .get(&node)
            .filter(|reached| reached.parent != node)
            .map(|reached| (reached.parent, reached.edge))
    }

    /// The arcs from this half's root out to `node`, root first.
    pub fn edge_path(&self, node: NodeId) -> Option<Vec<EdgeId>> {
        self.reached.get(&node)?;
        let mut edges = Vec::new();
        let mut current = node;
        while let Some((parent, edge)) = self.arrival(current) {
            edges.push(edge);
            current = parent;
        }
        edges.reverse();
        Some(edges)
    }

    /// Where this half's path to `node` started.
    pub fn root(&self, node: NodeId) -> Option<NodeId> {
        self.reached.get(&node)?;
        let mut current = node;
        while let Some((parent, _)) = self.arrival(current) {
            current = parent;
        }
        Some(current)
    }
}

/// What a contraction hierarchy query explored: a search from each end, and the
/// node where they met.
#[derive(Debug, Clone)]
pub struct MeetingSearch {
    /// Upward from the source.
    pub forward: Half,
    /// Upward from the target, over the reversed downward graph.
    pub backward: Half,
    /// Where the two halves joined, if they did.
    pub meeting: Option<NodeId>,
    /// The distance from source to target, if there is one.
    pub distance: Option<Weight>,
    /// The target the query was asked about.
    pub target: NodeId,
}

impl MeetingSearch {
    /// The cost to `node`, which only the target has an answer for.
    ///
    /// A bidirectional search learns the true distance to exactly one node: the
    /// one it was aiming at. Everything else it touched has an upward distance,
    /// which is not the same thing and would be a lie to report as one.
    pub fn cost(&self, node: NodeId) -> Option<Weight> {
        (node == self.target).then_some(self.distance).flatten()
    }

    /// How many nodes the two halves settled between them.
    pub fn settled(&self) -> usize {
        self.forward.order.len() + self.backward.order.len()
    }

    /// Where the cheapest path starts — with several sources, whichever one it
    /// turned out to be worth leaving from.
    pub fn origin(&self) -> Option<NodeId> {
        self.forward.root(self.meeting?)
    }
}

/// Search from both ends, upward, until neither half can improve on the other.
pub(super) fn query(
    hierarchy: &ContractionHierarchy,
    sources: &[(NodeId, Weight)],
    target: NodeId,
) -> Result<MeetingSearch, SearchError> {
    let num_nodes = hierarchy.num_nodes();
    for &(node, _) in sources {
        if node as usize >= num_nodes {
            return Err(SearchError::SourceOutOfRange { node, num_nodes });
        }
    }
    if target as usize >= num_nodes {
        return Err(SearchError::TargetOutOfRange {
            node: target,
            num_nodes,
        });
    }

    let mut forward = Side::new(hierarchy.upward(), sources);
    let mut backward = Side::new(hierarchy.downward(), &[(target, 0)]);
    let mut best = UNREACHABLE;
    let mut meeting = None;

    loop {
        // Neither half can beat what has been found once its own cheapest
        // remaining node already costs that much.
        if forward.frontier().min(backward.frontier()) >= best {
            break;
        }

        // Advance whichever side is cheaper, so the halves meet near the middle.
        let side = if forward.frontier() <= backward.frontier() {
            &mut forward
        } else {
            &mut backward
        };
        let Some(node) = side.settle_one() else {
            continue;
        };

        if let (Some(up), Some(down)) = (forward.half.cost(node), backward.half.cost(node)) {
            let joined = up.saturating_add(down);
            if joined < best {
                best = joined;
                meeting = Some(node);
            }
        }
    }

    Ok(MeetingSearch {
        forward: forward.half,
        backward: backward.half,
        meeting,
        distance: (best != UNREACHABLE).then_some(best),
        target,
    })
}

/// One direction, mid-search.
struct Side<'a> {
    graph: &'a Graph,
    half: Half,
    queue: BinaryHeap<Reverse<(Weight, NodeId)>>,
}

impl<'a> Side<'a> {
    fn new(graph: &'a Graph, sources: &[(NodeId, Weight)]) -> Self {
        let mut side = Side {
            graph,
            half: Half::default(),
            queue: BinaryHeap::new(),
        };
        for &(node, cost) in sources {
            if side.half.cost(node).is_none_or(|known| cost < known) {
                // A root is its own parent, which is how a path walk knows to
                // stop without needing a sentinel.
                side.half.reached.insert(
                    node,
                    Reached {
                        cost,
                        parent: node,
                        edge: 0,
                    },
                );
                side.queue.push(Reverse((cost, node)));
            }
        }
        side
    }

    fn frontier(&self) -> Weight {
        self.queue
            .peek()
            .map_or(UNREACHABLE, |Reverse((cost, _))| *cost)
    }

    /// Settle the next node and relax its arcs; `None` for a stale entry.
    fn settle_one(&mut self) -> Option<NodeId> {
        let Reverse((cost, node)) = self.queue.pop()?;
        if self.half.cost(node).is_some_and(|known| cost > known) {
            return None;
        }
        self.half.order.push(node);

        for edge in self.graph.out_edges(node) {
            let next = cost.saturating_add(self.graph.weight(edge));
            if next == UNREACHABLE {
                continue;
            }
            let head = self.graph.head(edge);
            if self.half.cost(head).is_none_or(|known| next < known) {
                self.half.reached.insert(
                    head,
                    Reached {
                        cost: next,
                        parent: node,
                        edge,
                    },
                );
                self.queue.push(Reverse((next, head)));
            }
        }
        Some(node)
    }
}

impl ContractionHierarchy {
    /// The original edges of the cheapest path, source first.
    ///
    /// The search ran over the augmented graph, so its answer is in arcs that
    /// may each stand for hundreds of real ones. Unpacking is what turns that
    /// back into something the environment recognises — a technique may search
    /// whatever graph it likes, but it has to answer in the caller's terms.
    pub fn unpack(&self, search: &MeetingSearch) -> Option<Vec<EdgeId>> {
        let meeting = search.meeting?;
        let mut edges = Vec::new();

        // Up from the source: the half's arcs already run in path order.
        for edge in search.forward.edge_path(meeting)? {
            self.expand_into(self.upward_edge(edge), &mut edges);
        }
        // Down to the target: the backward half runs target-ward over reversed
        // arcs, so reading it back to front gives the forward direction.
        for edge in search.backward.edge_path(meeting)?.into_iter().rev() {
            self.expand_into(self.downward_edge(edge), &mut edges);
        }
        Some(edges)
    }

    /// Expand one augmented arc into the original edges beneath it.
    pub fn expand_into(&self, augmented: u32, into: &mut Vec<EdgeId>) {
        // An explicit stack rather than recursion: shortcut chains are shallow
        // in practice but nothing in the structure promises it.
        let mut stack = vec![augmented];
        while let Some(index) = stack.pop() {
            match self.expansion_of(index) {
                super::Expansion::Original(edge) => into.push(edge),
                super::Expansion::Shortcut { first, second } => {
                    stack.push(second);
                    stack.push(first);
                }
            }
        }
    }
}
