//! Landmarks: bounds from precomputed distances rather than from geometry.
//!
//! A straight-line heuristic knows one thing about a network — that nothing
//! crosses ground faster than the fastest mode — and on a road network that is
//! a weak thing to know, because it must assume every street is a motorway. A
//! landmark bound knows something much stronger: the *actual* distance from a
//! handful of fixed nodes to everywhere.
//!
//! Pick a landmark `L` and precompute `d(L, ·)` and `d(·, L)`. Then for any node
//! `n` and target `t`, the triangle inequality gives two lower bounds on the
//! remaining cost:
//!
//! ```text
//! d(n, t) >= d(L, t) - d(L, n)        // going out through L
//! d(n, t) >= d(n, L) - d(t, L)        // coming back through L
//! ```
//!
//! Both hold for every landmark, so the largest is the sharpest, and taking the
//! maximum keeps it admissible. Landmarks behind the target are the informative
//! ones — which is why they are chosen far apart and near the edges of the map.
//!
//! What this costs: two searches of the whole graph per landmark, and one
//! `Weight` per landmark per node held for the life of the heuristic. What it
//! buys is a bound that knows about one-way streets, dead ends, bridges, and
//! speed limits, because it was measured through them rather than assumed over
//! them. It also needs no coordinates at all, which makes it the heuristic for
//! networks that have no geometry to speak of.

use crate::dijkstra::dijkstra;
use crate::graph::{Graph, NodeId, Weight, UNREACHABLE};
use crate::heuristic::Heuristic;
use crate::search::SearchOptions;

/// How to choose which nodes become landmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Spread them out: each new landmark is the node farthest from every
    /// landmark so far. Costs no more than picking at random — the searches that
    /// measure "farthest" are the same ones that build the tables — and lands
    /// them around the edges of the network, which is where they inform.
    Farthest,
    /// Uniformly at random from the reachable nodes. Cheap, reproducible with a
    /// seed, and a useful control: a landmark set is only worth its memory if it
    /// beats this.
    Random,
}

/// Distances between a set of landmarks and every node, both ways.
#[derive(Debug, Clone)]
pub struct Landmarks {
    nodes: Vec<NodeId>,
    num_nodes: usize,
    /// `from[i * num_nodes + v]` is `d(landmark_i, v)`.
    from: Vec<Weight>,
    /// `to[i * num_nodes + v]` is `d(v, landmark_i)`.
    to: Vec<Weight>,
}

impl Landmarks {
    /// Choose `count` landmarks and measure the network from each of them.
    ///
    /// Runs two full searches per landmark, plus one to find the first one when
    /// spreading them out. On a city street network that is seconds, paid once.
    pub fn build(graph: &Graph, count: usize, selection: Selection, seed: u64) -> Landmarks {
        let num_nodes = graph.num_nodes();
        let count = count.min(num_nodes);
        let reversed = graph.reversed();

        let nodes = match selection {
            Selection::Random => random_nodes(count, num_nodes, seed),
            Selection::Farthest => farthest_nodes(graph, count, seed),
        };

        let mut from = vec![UNREACHABLE; count * num_nodes];
        let mut to = vec![UNREACHABLE; count * num_nodes];
        for (index, &landmark) in nodes.iter().enumerate() {
            let offset = index * num_nodes;
            let outward = dijkstra(graph, &[(landmark, 0)], &SearchOptions::default())
                .expect("a landmark is a node of the graph it was chosen from");
            from[offset..offset + num_nodes].copy_from_slice(&outward.costs);

            let inward = dijkstra(&reversed, &[(landmark, 0)], &SearchOptions::default())
                .expect("the reversed graph has the same nodes");
            to[offset..offset + num_nodes].copy_from_slice(&inward.costs);
        }

        Landmarks {
            nodes,
            num_nodes,
            from,
            to,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The nodes chosen as landmarks.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Bytes of distance table held, for deciding whether a landmark count fits.
    pub fn footprint(&self) -> usize {
        (self.from.len() + self.to.len()) * std::mem::size_of::<Weight>()
    }
}

impl Heuristic for Landmarks {
    fn estimate(&self, node: NodeId, target: NodeId) -> Weight {
        let (node, target) = (node as usize, target as usize);
        let mut best = 0;
        for index in 0..self.nodes.len() {
            let offset = index * self.num_nodes;
            // Each landmark offers two bounds; a landmark the search cannot
            // reach, or that cannot reach the target, offers neither. Skipping
            // it costs sharpness, never correctness — the maximum over fewer
            // valid bounds is still a lower bound.
            let (from_node, from_target) = (self.from[offset + node], self.from[offset + target]);
            if from_node != UNREACHABLE && from_target != UNREACHABLE {
                best = best.max(from_target.saturating_sub(from_node));
            }
            let (to_node, to_target) = (self.to[offset + node], self.to[offset + target]);
            if to_node != UNREACHABLE && to_target != UNREACHABLE {
                best = best.max(to_node.saturating_sub(to_target));
            }
        }
        best
    }

    fn coverage(&self) -> Option<usize> {
        Some(self.num_nodes)
    }
}

/// Landmarks spread as far apart as the network allows.
///
/// Start somewhere, walk to the farthest node from it, and make that the first
/// landmark; from then on each new landmark is whichever node is farthest from
/// all the ones already chosen. The searches doing the measuring are the same
/// ones that would have to run anyway to fill the tables.
fn farthest_nodes(graph: &Graph, count: usize, seed: u64) -> Vec<NodeId> {
    let num_nodes = graph.num_nodes();
    if count == 0 || num_nodes == 0 {
        return Vec::new();
    }

    let mut rng = Rng::new(seed);
    let start = (rng.next() % num_nodes as u64) as NodeId;
    let mut nearest = distances_from(graph, start);
    let mut chosen = Vec::with_capacity(count);

    while chosen.len() < count {
        let next = argmax_reachable(&nearest).unwrap_or(start);
        if chosen.contains(&next) {
            // Everything reachable is already a landmark; the rest of the graph
            // cannot be measured from here, so stop rather than repeat.
            break;
        }
        chosen.push(next);

        let measured = distances_from(graph, next);
        for (node, distance) in measured.iter().enumerate() {
            nearest[node] = nearest[node].min(*distance);
        }
    }
    chosen
}

fn random_nodes(count: usize, num_nodes: usize, seed: u64) -> Vec<NodeId> {
    let mut rng = Rng::new(seed);
    let mut chosen: Vec<NodeId> = Vec::with_capacity(count);
    while chosen.len() < count {
        let candidate = (rng.next() % num_nodes as u64) as NodeId;
        if !chosen.contains(&candidate) {
            chosen.push(candidate);
        }
    }
    chosen
}

fn distances_from(graph: &Graph, source: NodeId) -> Vec<Weight> {
    dijkstra(graph, &[(source, 0)], &SearchOptions::default())
        .expect("a source drawn from the graph's own node range")
        .costs
}

/// The reachable node with the largest distance, or `None` if none are.
fn argmax_reachable(distances: &[Weight]) -> Option<NodeId> {
    distances
        .iter()
        .enumerate()
        .filter(|(_, &distance)| distance != UNREACHABLE)
        .max_by_key(|(_, &distance)| distance)
        .map(|(node, _)| node as NodeId)
}

/// A small deterministic generator, so landmark choice is reproducible without
/// the crate taking on a dependency for it.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Any nonzero state will do; xorshift stays stuck at zero.
        Rng(seed.wrapping_mul(2_685_821_657_736_338_717).max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directed cycle 0->1->2->3->4->0, every hop costing 10. Distances differ
    /// sharply by direction, which is what a landmark bound can see and a
    /// straight-line bound cannot.
    fn ring() -> Graph {
        Graph::from_edges(
            5,
            &[(0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 4, 10), (4, 0, 10)],
        )
        .unwrap()
    }

    #[test]
    fn tables_measure_both_directions() {
        let landmarks = Landmarks::build(&ring(), 1, Selection::Random, 7);
        let landmark = landmarks.nodes()[0] as usize;
        // On a one-way ring, going out and coming back are different distances
        // for every node but the landmark itself.
        let offset = 0;
        for node in 0..5 {
            let out = landmarks.from[offset + node];
            let back = landmarks.to[offset + node];
            assert_eq!(out + back, if node == landmark { 0 } else { 50 });
        }
    }

    #[test]
    fn the_estimate_never_exceeds_the_true_distance() {
        let graph = ring();
        let landmarks = Landmarks::build(&graph, 2, Selection::Farthest, 1);
        for target in 0..5 {
            let truth =
                dijkstra(&graph.reversed(), &[(target, 0)], &SearchOptions::default()).unwrap();
            for node in 0..5 {
                let actual = truth.cost(node).unwrap();
                assert!(
                    landmarks.estimate(node, target) <= actual,
                    "h({node}, {target}) overshot {actual}"
                );
            }
        }
    }

    #[test]
    fn a_landmark_sees_direction_that_geometry_cannot() {
        // Every node sits at distance 10 or 40 from its neighbours depending on
        // which way round you go. Only a measured bound knows the difference.
        let landmarks = Landmarks::build(&ring(), 5, Selection::Farthest, 3);
        assert_eq!(landmarks.estimate(1, 0), 40, "the long way round the ring");
        assert_eq!(landmarks.estimate(0, 1), 10, "and the short way back");
    }

    #[test]
    fn spreading_them_out_picks_distinct_nodes() {
        let landmarks = Landmarks::build(&ring(), 3, Selection::Farthest, 11);
        let mut nodes = landmarks.nodes().to_vec();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn selection_is_reproducible() {
        let first = Landmarks::build(&ring(), 3, Selection::Random, 42);
        let again = Landmarks::build(&ring(), 3, Selection::Random, 42);
        let different = Landmarks::build(&ring(), 3, Selection::Random, 43);
        assert_eq!(first.nodes(), again.nodes());
        assert_ne!(
            first.nodes(),
            different.nodes(),
            "a seed that means something"
        );
    }

    #[test]
    fn asking_for_more_landmarks_than_nodes_is_not_an_error() {
        let landmarks = Landmarks::build(&ring(), 99, Selection::Farthest, 5);
        assert_eq!(landmarks.len(), 5);
    }

    #[test]
    fn unreachable_pairs_fall_back_to_no_estimate() {
        // Two components: nothing measured through a landmark in one of them
        // says anything about the other, so the bound is zero rather than wrong.
        let split = Graph::from_edges(4, &[(0, 1, 10), (2, 3, 10)]).unwrap();
        let landmarks = Landmarks::build(&split, 2, Selection::Random, 2);
        assert_eq!(landmarks.estimate(0, 3), 0);
        assert_eq!(landmarks.estimate(1, 0), 0, "no way back on a one-way edge");
    }

    #[test]
    fn footprint_is_two_weights_per_landmark_per_node() {
        let landmarks = Landmarks::build(&ring(), 2, Selection::Farthest, 1);
        assert_eq!(
            landmarks.footprint(),
            2 * 2 * 5 * std::mem::size_of::<Weight>()
        );
    }
}
