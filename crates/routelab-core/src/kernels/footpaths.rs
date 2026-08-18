//! Closing footpaths under composition.
//!
//! Pyrga et al.'s foot-edges are given as individual links, but a rider who can
//! walk a to b and b to c can walk a to c, and a model that does not know that
//! answers a query dropped on the wrong corner with nothing. Closing them is a
//! shortest-path search over the link graph, which is why it is here with the
//! techniques rather than with the structure it fills in: the container and its
//! lookups are [`crate::model::timetable::Footpaths`], and every schedule-based
//! kernel reads that without needing this.
//!
//! The link graph is a scatter of small components — a street corner, a transit
//! centre — so this is cheap, and is not an all-pairs pass over the network.

use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, NodeId, Weight};
use crate::model::search::SearchOptions;
use crate::model::timetable::{Footpaths, Time};

impl Footpaths {
    /// Build from `(from, to, duration)` triples in any order.
    ///
    /// Links whose stops fall outside `stops` are dropped. Both directions are
    /// the caller's business: a walk that only goes one way is unusual but
    /// not impossible (a one-way turnstile), so nothing is mirrored here.
    pub fn new(stops: usize, links: impl IntoIterator<Item = (NodeId, NodeId, Time)>) -> Self {
        let given: Vec<(NodeId, NodeId, Weight)> = links
            .into_iter()
            .filter(|(from, to, _)| {
                (*from as usize) < stops && (*to as usize) < stops && from != to
            })
            .collect();
        Footpaths::from_closed(stops, Self::closure(stops, &given))
    }

    /// Build from `(from, to, duration)` triples **without** closing them.
    ///
    /// For a set that is already the set a query needs. ULTRA's shortcuts are
    /// that: each one stands for a whole walk between two stops, and the
    /// paper's guarantee is that every intermediate transfer a Pareto-optimal
    /// journey can need is one of them. Chaining them would add every
    /// composition of two — and unlike a radius's foot-edges, which are a
    /// scatter of small components, the shortcut graph spans the network, so
    /// the closure is the quadratic blow-up ULTRA exists to avoid. Twenty-four
    /// thousand shortcuts over six thousand stops close to two hundred and
    /// thirty megabytes of walks no journey takes.
    ///
    /// Every link is direct, so each is its own `via`.
    pub fn from_links(
        stops: usize,
        links: impl IntoIterator<Item = (NodeId, NodeId, Time)>,
    ) -> Self {
        let given: Vec<(NodeId, NodeId, Time, NodeId)> = links
            .into_iter()
            .filter(|(from, to, _)| {
                (*from as usize) < stops && (*to as usize) < stops && from != to
            })
            .map(|(from, to, duration)| (from, to, duration, from))
            .collect();
        Footpaths::from_closed(stops, given)
    }

    /// Every walk reachable by chaining `given`, at its shortest duration, as
    /// `(from, to, duration, via)`.
    ///
    /// The links are a graph, so this is [`crate::kernels::dijkstra::dijkstra`] from each stop
    /// that has one, and `via` is the search's own parent pointer. The
    /// footpath graph is a scatter of small components — a street corner, a
    /// transit centre — so this is cheap; it is not, and need not be, an
    /// all-pairs pass over the network.
    fn closure(
        stops: usize,
        given: &[(NodeId, NodeId, Weight)],
    ) -> Vec<(NodeId, NodeId, Time, NodeId)> {
        let graph = Graph::from_edges(stops, given).expect("links were filtered to the stop set");
        let mut closed = Vec::new();
        for origin in 0..stops as NodeId {
            if graph.out_edges(origin).next().is_none() {
                continue;
            }
            let Ok(reached) = dijkstra(&graph, &[(origin, 0)], &SearchOptions::default()) else {
                continue;
            };
            for &stop in &reached.order {
                if stop == origin {
                    continue;
                }
                let (Some(duration), Some(via)) = (reached.cost(stop), reached.parent(stop)) else {
                    continue;
                };
                closed.push((origin, stop, duration, via));
            }
        }
        closed
    }

    /// Build from **positions in a graph's input edge list**: each named edge
    /// becomes a footpath between its endpoints taking its weight.
    ///
    /// The join between an environment and the models, in one place, and by
    /// input position for the reason [`crate::model::timetable::Timetable::from_input_connections`]
    /// gives — `Graph` permutes edges into CSR order, and whatever produced
    /// the edges holds input positions.
    pub fn from_input_edges(graph: &Graph, positions: impl IntoIterator<Item = u32>) -> Self {
        let to_edge = graph.edges_by_input();
        let links = positions
            .into_iter()
            .filter(|input| (*input as usize) < to_edge.len())
            .map(|input| graph.edge(to_edge[input as usize]));
        Footpaths::new(graph.num_nodes(), links)
    }
}
