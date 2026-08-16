//! Contracting nodes: choosing an order, and inserting the shortcuts.
//!
//! Contraction is the one thing in this crate that needs a graph it can grow, so
//! it works over adjacency lists and builds the CSR graphs once at the end.
//! Contracted nodes are marked rather than removed — their arcs stay in the
//! lists and are skipped, which costs a branch and saves a great deal of
//! bookkeeping.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{Expansion, Ordering, Policy};
use crate::model::graph::{Graph, NodeId, Weight, UNREACHABLE};
use crate::util::progress::Progress;
use crate::util::rng::Rng;

/// One direction of an arc in the working graph. `other` is the far end: the
/// head for an outgoing arc, the tail for an incoming one.
#[derive(Clone, Copy, Debug)]
pub(super) struct Arc {
    pub other: NodeId,
    pub weight: Weight,
    pub augmented: u32,
}

/// An edge of the augmented graph: the originals, plus every shortcut.
#[derive(Clone, Debug)]
pub(super) struct AugmentedEdge {
    pub tail: NodeId,
    pub head: NodeId,
    pub weight: Weight,
    pub expansion: Expansion,
}

/// What contracting a node would cost, and the shortcuts that would cost it.
///
/// The two travel together because they are the same computation. A node's
/// priority is "how many shortcuts would this need", which can only be answered
/// by working out what those shortcuts are — and the witness searches that
/// decide it are the dominant cost of the whole preprocessing step. Asking for
/// the priority and then, unchanged, asking for the shortcuts would run every
/// one of them twice.
struct Cost {
    priority: i64,
    /// `None` when the policy reached its answer without asking — an arbitrary
    /// order never looks at the graph, so it leaves the shortcuts to be worked
    /// out at contraction time rather than for every node up front.
    shortcuts: Option<Vec<AugmentedEdge>>,
}

/// Reusable scratch for the witness searches.
///
/// There is one witness search per pair of neighbours per contraction attempt —
/// millions of them on a city — so they cannot afford to allocate. A stamp per
/// node says which search last wrote a distance, which makes clearing free.
struct Scratch {
    distance: Vec<Weight>,
    stamp: Vec<u32>,
    now: u32,
    queue: BinaryHeap<Reverse<(Weight, u32, NodeId)>>,
}

impl Scratch {
    fn new(num_nodes: usize) -> Self {
        Scratch {
            distance: vec![UNREACHABLE; num_nodes],
            stamp: vec![0; num_nodes],
            now: 0,
            queue: BinaryHeap::new(),
        }
    }

    fn restart(&mut self) {
        self.now += 1;
        self.queue.clear();
    }

    fn distance_of(&self, node: NodeId) -> Weight {
        if self.stamp[node as usize] == self.now {
            self.distance[node as usize]
        } else {
            UNREACHABLE
        }
    }

    fn set(&mut self, node: NodeId, distance: Weight) {
        self.stamp[node as usize] = self.now;
        self.distance[node as usize] = distance;
    }
}

pub(super) struct Builder {
    out: Vec<Vec<Arc>>,
    inc: Vec<Vec<Arc>>,
    contracted: Vec<bool>,
    contracted_neighbours: Vec<u32>,
    scratch: Scratch,
    ordering: Ordering,
    pub edges: Vec<AugmentedEdge>,
}

impl Builder {
    pub(super) fn new(graph: &Graph, ordering: Ordering) -> Self {
        let num_nodes = graph.num_nodes();
        let mut builder = Builder {
            out: vec![Vec::new(); num_nodes],
            inc: vec![Vec::new(); num_nodes],
            contracted: vec![false; num_nodes],
            contracted_neighbours: vec![0; num_nodes],
            scratch: Scratch::new(num_nodes),
            ordering,
            edges: Vec::with_capacity(graph.num_edges()),
        };

        for edge in 0..graph.num_edges() as u32 {
            let (tail, head, weight) = graph.edge(edge);
            if tail == head {
                continue; // a self-loop is never on a shortest path
            }
            builder.push_edge(AugmentedEdge {
                tail,
                head,
                weight,
                expansion: Expansion::Original(edge),
            });
        }
        builder
    }

    /// Add an edge to the augmented list and to both adjacency lists.
    fn push_edge(&mut self, edge: AugmentedEdge) -> u32 {
        let index = self.edges.len() as u32;
        self.out[edge.tail as usize].push(Arc {
            other: edge.head,
            weight: edge.weight,
            augmented: index,
        });
        self.inc[edge.head as usize].push(Arc {
            other: edge.tail,
            weight: edge.weight,
            augmented: index,
        });
        self.edges.push(edge);
        index
    }

    /// Contract every node, cheapest first, and report each one's rank.
    ///
    /// `progress` counts nodes retired. An honest fraction: contraction is
    /// monotone, every node is contracted exactly once, and the total is known
    /// before the first one is.
    pub(super) fn contract_all(&mut self, progress: &Progress) -> Vec<u32> {
        let num_nodes = self.out.len();
        progress.expect("contracting", num_nodes as u64);
        let mut ranks = vec![0u32; num_nodes];
        let mut queue: BinaryHeap<Reverse<(i64, NodeId)>> = BinaryHeap::new();
        for node in 0..num_nodes as NodeId {
            queue.push(Reverse((self.cost(node).priority, node)));
        }

        let mut rank = 0;
        while let Some(Reverse((priority, node))) = queue.pop() {
            if self.contracted[node as usize] {
                continue;
            }
            // Lazy updates: a node's priority goes stale as its neighbours are
            // contracted around it. Rather than updating every neighbour on
            // every contraction, recompute on the way out and put it back if it
            // turns out no longer to be the cheapest.
            let mut shortcuts = None;
            if self.ordering.recomputes() {
                let cost = self.cost(node);
                if cost.priority > priority {
                    if let Some(&Reverse((next, _))) = queue.peek() {
                        if cost.priority > next {
                            queue.push(Reverse((cost.priority, node)));
                            continue;
                        }
                    }
                }
                // Nothing has touched the graph since that recomputation, so
                // the shortcuts it worked out are the ones to insert.
                shortcuts = cost.shortcuts;
            }

            for shortcut in shortcuts.unwrap_or_else(|| self.shortcuts_for(node)) {
                self.add_shortcut(shortcut);
            }
            self.retire(node);
            ranks[node as usize] = rank;
            rank += 1;
            progress.step();
        }
        ranks
    }

    /// What contracting this node would cost, smaller being sooner.
    fn cost(&mut self, node: NodeId) -> Cost {
        match self.ordering.policy {
            Policy::Random { seed } => Cost {
                priority: scramble(seed, node) as i64,
                shortcuts: None,
            },
            Policy::EdgeDifference { deleted_neighbours } => {
                let shortcuts = self.shortcuts_for(node);
                let mut priority = shortcuts.len() as i64 - self.degree(node) as i64;
                if deleted_neighbours {
                    // Contracting next to already-contracted nodes piles the
                    // hierarchy up in one place; this spreads it out.
                    priority += self.contracted_neighbours[node as usize] as i64;
                }
                Cost {
                    priority,
                    shortcuts: Some(shortcuts),
                }
            }
        }
    }

    /// How many live arcs touch this node.
    fn degree(&self, node: NodeId) -> usize {
        let live = |arcs: &Vec<Arc>| {
            arcs.iter()
                .filter(|arc| !self.contracted[arc.other as usize])
                .count()
        };
        live(&self.inc[node as usize]) + live(&self.out[node as usize])
    }

    /// The shortcuts contracting `node` would require: every pair of live
    /// neighbours whose path through it is the only shortest one.
    fn shortcuts_for(&mut self, node: NodeId) -> Vec<AugmentedEdge> {
        let live = |arcs: &[Arc], contracted: &[bool]| -> Vec<Arc> {
            arcs.iter()
                .copied()
                .filter(|arc| !contracted[arc.other as usize])
                .collect()
        };
        let incoming = live(&self.inc[node as usize], &self.contracted);
        let outgoing = live(&self.out[node as usize], &self.contracted);

        let mut shortcuts = Vec::new();
        for entry in &incoming {
            for exit in &outgoing {
                if entry.other == exit.other {
                    continue;
                }
                let weight = entry.weight.saturating_add(exit.weight);
                if weight == UNREACHABLE {
                    continue; // saturated, so no real path is this long
                }
                let witnessed = Self::witness(
                    &self.out,
                    &self.contracted,
                    &mut self.scratch,
                    &self.ordering,
                    entry.other,
                    exit.other,
                    node,
                    weight,
                );
                if !witnessed {
                    shortcuts.push(AugmentedEdge {
                        tail: entry.other,
                        head: exit.other,
                        weight,
                        expansion: Expansion::Shortcut {
                            first: entry.augmented,
                            second: exit.augmented,
                        },
                    });
                }
            }
        }
        shortcuts
    }

    /// Mark a node gone, once its shortcuts are in.
    fn retire(&mut self, node: NodeId) {
        self.contracted[node as usize] = true;
        for direction in [&self.inc[node as usize], &self.out[node as usize]] {
            for arc in direction {
                self.contracted_neighbours[arc.other as usize] += 1;
            }
        }
    }

    /// Add a shortcut, unless something at least as good already runs that way.
    ///
    /// Never rewrites an existing arc, even a longer one. An original edge is
    /// the only record of which edge of the caller's graph it was, and a
    /// hierarchy that overwrites that has quietly lost the ability to answer in
    /// the caller's terms. A superseded arc costs a little space and is simply
    /// never chosen.
    fn add_shortcut(&mut self, shortcut: AugmentedEdge) {
        let best = self.out[shortcut.tail as usize]
            .iter()
            .filter(|arc| arc.other == shortcut.head)
            .map(|arc| arc.weight)
            .min();
        if best.is_some_and(|weight| weight <= shortcut.weight) {
            return;
        }
        self.push_edge(shortcut);
    }

    /// Is there already a path from `from` to `to` no longer than `limit` that
    /// avoids `avoid`?
    ///
    /// A bounded Dijkstra over the uncontracted graph. Bounded three ways: by
    /// the candidate weight, by how many nodes it may settle, and by how many
    /// hops it may take. Giving up early is safe — it only means a shortcut gets
    /// inserted that was not strictly needed, which costs space and query time
    /// and never costs correctness.
    #[allow(clippy::too_many_arguments)]
    fn witness(
        out: &[Vec<Arc>],
        contracted: &[bool],
        scratch: &mut Scratch,
        ordering: &Ordering,
        from: NodeId,
        to: NodeId,
        avoid: NodeId,
        limit: Weight,
    ) -> bool {
        scratch.restart();
        scratch.set(from, 0);
        scratch.queue.push(Reverse((0, 0, from)));

        let mut settled = 0;
        while let Some(Reverse((distance, hops, node))) = scratch.queue.pop() {
            if distance > scratch.distance_of(node) {
                continue;
            }
            if node == to {
                return distance <= limit;
            }
            settled += 1;
            if settled > ordering.max_settled || hops as usize >= ordering.max_hops {
                continue;
            }

            for arc in &out[node as usize] {
                if contracted[arc.other as usize] || arc.other == avoid {
                    continue;
                }
                let next = distance.saturating_add(arc.weight);
                if next > limit {
                    continue; // a witness longer than the shortcut is no witness
                }
                if next < scratch.distance_of(arc.other) {
                    scratch.set(arc.other, next);
                    scratch.queue.push(Reverse((next, hops + 1, arc.other)));
                }
            }
        }
        false
    }
}

/// A node's arbitrary-but-fixed position under the random ordering.
///
/// Seeded per node rather than drawn from one stream, because contraction pops
/// nodes in an order the priorities themselves decide — a shared stream would
/// hand out different numbers depending on when a node was asked about.
fn scramble(seed: u64, node: NodeId) -> u32 {
    (Rng::new(seed ^ u64::from(node)).next() >> 32) as u32
}
