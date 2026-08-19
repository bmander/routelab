//! Dijkstra's algorithm on a [`Graph`], with a binary heap and lazy deletion.

use std::convert::Infallible;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::model::graph::{EdgeId, Graph, NodeId, Weight, UNREACHABLE};
use crate::model::search::{
    check_sources, SearchError, SearchOptions, SearchResult, TargetTracker,
};
use crate::model::technique::{Distance, Footprint, Searches, Technique, Unpacks};
use crate::util::progress::Progress;

/// Dijkstra as a configuration: nothing to set, nothing to build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DijkstraTechnique;

impl<'a> Technique<'a> for DijkstraTechnique {
    type Inputs = &'a Graph;
    type Planner = Dijkstra<'a>;
    type Error = Infallible;

    fn bind(&self, graph: &'a Graph, _progress: &Progress) -> Result<Dijkstra<'a>, Infallible> {
        Ok(Dijkstra { graph })
    }
}

/// Dijkstra bound to a graph. Nothing is precomputed, so this borrows: the
/// planner is the graph and the promise to search it.
#[derive(Debug, Clone, Copy)]
pub struct Dijkstra<'a> {
    pub graph: &'a Graph,
}

impl Footprint for Dijkstra<'_> {
    fn footprint(&self) -> usize {
        0
    }

    fn searches(&self) -> (&'static str, usize) {
        ("nodes", self.graph.num_nodes())
    }
}

impl Searches for Dijkstra<'_> {
    type Source = (NodeId, Weight);
    type Query = SearchOptions;
    type Search = SearchResult;
    type Error = SearchError;

    fn search(
        &self,
        sources: &[(NodeId, Weight)],
        query: &SearchOptions,
    ) -> Result<SearchResult, SearchError> {
        dijkstra(self.graph, sources, query)
    }
}

impl Unpacks for Dijkstra<'_> {
    fn edge_path(&self, search: &SearchResult, to: NodeId) -> Option<Vec<EdgeId>> {
        search.edge_path(to)
    }
}

impl Distance for Dijkstra<'_> {
    fn distance(
        &self,
        sources: &[(NodeId, Weight)],
        to: NodeId,
    ) -> Result<Option<Weight>, SearchError> {
        let options = SearchOptions::default().with_any_target([to]);
        Ok(dijkstra(self.graph, sources, &options)?.cost(to))
    }
}

/// Shortest paths from one or more sources, each with an initial cost.
///
/// Multiple sources with initial costs are the shape the transit literature
/// actually needs: "I can reach stop A in 4 minutes and stop B in 7, now route
/// from there." A single source is the special case `&[(node, 0)]`.
///
/// Nodes are settled in order of increasing cost, ties broken by ascending node
/// id, which makes the settle order — and therefore the tree — deterministic.
pub fn dijkstra(
    graph: &Graph,
    sources: &[(NodeId, Weight)],
    options: &SearchOptions,
) -> Result<SearchResult, SearchError> {
    check_sources(graph, sources)?;

    let mut result = SearchResult::new(graph.num_nodes());
    let mut tracker = TargetTracker::new(&options.targets, options.reach, graph.num_nodes());
    if tracker.as_ref().is_some_and(|t| t.done()) {
        return Ok(result);
    }
    let max_cost = options.max_cost.unwrap_or(UNREACHABLE);

    let mut queue: BinaryHeap<Reverse<(Weight, NodeId)>> = BinaryHeap::new();
    for &(node, cost) in sources {
        if cost <= max_cost && cost < result.costs[node as usize] {
            result.costs[node as usize] = cost;
            queue.push(Reverse((cost, node)));
        }
    }

    while let Some(Reverse((cost, node))) = queue.pop() {
        // Lazy deletion: a node can sit in the heap several times, but only the
        // entry matching its current best cost is the real one.
        if cost > result.costs[node as usize] {
            continue;
        }
        if result.settle(node, &mut tracker) {
            break;
        }

        for edge in graph.out_edges(node) {
            let next_cost = cost.saturating_add(graph.weight(edge));
            if next_cost > max_cost || next_cost == UNREACHABLE {
                continue;
            }
            let head = graph.head(edge);
            if next_cost < result.costs[head as usize] {
                result.costs[head as usize] = next_cost;
                result.parent_nodes[head as usize] = node;
                result.parent_edges[head as usize] = edge;
                queue.push(Reverse((next_cost, head)));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::NO_NODE;

    /// 0 -> 1 -> 3 costs 3; 0 -> 2 -> 3 costs 30; 4 is isolated.
    fn diamond() -> Graph {
        Graph::from_edges(5, &[(0, 1, 1), (0, 2, 10), (1, 3, 2), (2, 3, 20)]).unwrap()
    }

    fn from(node: NodeId) -> Vec<(NodeId, Weight)> {
        vec![(node, 0)]
    }

    #[test]
    fn finds_the_cheaper_of_two_routes() {
        let g = diamond();
        let r = dijkstra(&g, &from(0), &SearchOptions::default()).unwrap();
        assert_eq!(r.cost(3), Some(3));
        assert_eq!(r.path(3), Some(vec![0, 1, 3]));
        assert_eq!(g.walk(0, &r.edge_path(3).unwrap()), Some((3, 3)));
        assert_eq!(r.cost(4), None, "isolated node stays unreached");
        assert_eq!(r.parent_nodes[0], NO_NODE);
    }

    #[test]
    fn initial_costs_shift_the_frontier() {
        let g = diamond();
        // Reaching node 2 for free beats reaching node 1 at a 25-unit penalty.
        let r = dijkstra(&g, &[(1, 25), (2, 0)], &SearchOptions::default()).unwrap();
        assert_eq!(r.cost(3), Some(20));
        assert_eq!(r.path(3), Some(vec![2, 3]));
    }

    #[test]
    fn max_cost_bounds_the_search() {
        let g = diamond();
        let r = dijkstra(&g, &from(0), &SearchOptions::default().with_max_cost(2)).unwrap();
        assert_eq!(r.cost(1), Some(1));
        assert_eq!(r.cost(3), None, "3 costs 3, past the bound");
        assert_eq!(r.order, vec![0, 1]);
    }

    #[test]
    fn targets_stop_the_search_early() {
        let g = diamond();
        let r = dijkstra(&g, &from(0), &SearchOptions::default().with_targets([1])).unwrap();
        assert_eq!(r.cost(1), Some(1));
        assert_eq!(r.order, vec![0, 1], "nothing beyond the target is settled");
        // Costs already relaxed when the search stopped are upper bounds, not
        // final distances, except at settled nodes.
        assert_eq!(r.order.last(), Some(&1));
    }

    #[test]
    fn any_target_stops_at_the_nearest_of_them() {
        let g = diamond();
        // Nodes 2 and 3 cost 10 and 3. Asking for all of them settles both;
        // asking for any stops at 3, which is the nearer.
        let all = dijkstra(&g, &from(0), &SearchOptions::default().with_targets([2, 3])).unwrap();
        let any = dijkstra(
            &g,
            &from(0),
            &SearchOptions::default().with_any_target([2, 3]),
        )
        .unwrap();
        assert_eq!(all.order, vec![0, 1, 3, 2]);
        assert_eq!(any.order, vec![0, 1, 3], "stopped once the nearest was in");
        assert_eq!(any.cost(3), Some(3));
    }

    #[test]
    fn unreachable_target_runs_to_exhaustion() {
        let g = diamond();
        let r = dijkstra(&g, &from(0), &SearchOptions::default().with_targets([4])).unwrap();
        assert_eq!(r.cost(4), None);
        // Settle order is by cost: 0 at 0, 1 at 1, 3 at 3, 2 at 10.
        assert_eq!(r.order, vec![0, 1, 3, 2]);
    }

    #[test]
    fn zero_weights_and_self_loops_terminate() {
        let g = Graph::from_edges(3, &[(0, 0, 0), (0, 1, 0), (1, 2, 0), (2, 1, 0)]).unwrap();
        let r = dijkstra(&g, &from(0), &SearchOptions::default()).unwrap();
        assert_eq!(r.costs, vec![0, 0, 0]);
    }

    #[test]
    fn rejects_bad_sources() {
        let g = diamond();
        assert_eq!(
            dijkstra(&g, &[(9, 0)], &SearchOptions::default()).unwrap_err(),
            SearchError::SourceOutOfRange {
                node: 9,
                num_nodes: 5
            }
        );
        assert_eq!(
            dijkstra(&g, &[(0, UNREACHABLE)], &SearchOptions::default()).unwrap_err(),
            SearchError::InfiniteSourceCost { node: 0 }
        );
    }
}
