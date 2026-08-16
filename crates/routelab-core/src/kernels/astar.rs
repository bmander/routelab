//! A*: Dijkstra, ordered by what a search still owes rather than what it has spent.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::model::graph::{Graph, NodeId, Weight, UNREACHABLE};
use crate::model::heuristic::Heuristic;
use crate::model::search::{check_sources, SearchError, SearchOptions, SearchResult};

/// Cheapest path from one or more sources to `target`, guided by `heuristic`.
///
/// The only difference from [`crate::dijkstra`] is the order nodes come out of the
/// queue: by `g + h` rather than `g`, where `g` is the cost of getting to a node
/// and `h` is the estimated cost of getting from it to the target. A heuristic
/// that estimates zero makes this Dijkstra exactly, node for node.
///
/// The result records `g` — the real cost of reaching each node, which is what
/// callers asked for — not the `f` the queue was sorted by.
///
/// `heuristic` must be admissible and consistent; see [`Heuristic`]. An
/// overestimate does not fail loudly, it just returns paths that are not the
/// cheapest, so check a new heuristic against Dijkstra on real instances.
///
/// The goal is the `target` argument. [`SearchOptions::targets`] is not consulted;
/// `max_cost` is, and bounds `g` exactly as it does for Dijkstra.
pub fn astar<H: Heuristic>(
    graph: &Graph,
    sources: &[(NodeId, Weight)],
    target: NodeId,
    heuristic: &H,
    options: &SearchOptions,
) -> Result<SearchResult, SearchError> {
    check_sources(graph, sources)?;
    if target as usize >= graph.num_nodes() {
        return Err(SearchError::TargetOutOfRange {
            node: target,
            num_nodes: graph.num_nodes(),
        });
    }
    if let Some(heuristic_nodes) = heuristic.coverage() {
        if heuristic_nodes != graph.num_nodes() {
            return Err(SearchError::HeuristicCoverage {
                heuristic_nodes,
                num_nodes: graph.num_nodes(),
            });
        }
    }

    let mut result = SearchResult::new(graph.num_nodes());
    let max_cost = options.max_cost.unwrap_or(UNREACHABLE);

    // Entries are (f, node, h), ordered by f = g + h. Ties break on node id, as
    // everywhere else, so the settle order is reproducible — `h` never affects
    // the order, since it is a function of the node the tie already agrees on.
    // Carrying it costs four bytes per queued entry and saves recomputing the
    // estimate on the way out.
    let mut queue: BinaryHeap<Reverse<(Weight, NodeId, Weight)>> = BinaryHeap::new();
    for &(node, cost) in sources {
        if cost <= max_cost && cost < result.costs[node as usize] {
            result.costs[node as usize] = cost;
            let estimate = heuristic.estimate(node, target);
            queue.push(Reverse((cost.saturating_add(estimate), node, estimate)));
        }
    }

    while let Some(Reverse((f, node, estimate))) = queue.pop() {
        let cost = result.costs[node as usize];
        // Lazy deletion, against g rather than f: an entry is stale when the node
        // has since been reached more cheaply than this entry was priced at.
        if f > cost.saturating_add(estimate) {
            continue;
        }
        if result.settle_toward(node, target) {
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
                let estimate = heuristic.estimate(head, target);
                queue.push(Reverse((
                    next_cost.saturating_add(estimate),
                    head,
                    estimate,
                )));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dijkstra::dijkstra;
    use crate::kernels::heuristics::StandardHeuristic;

    /// Four nodes in a line at x = 0, 1, 2, 3, each hop costing 10, plus a
    /// dead-end spur off node 0 that a guided search should not bother with.
    fn corridor() -> (Graph, StandardHeuristic) {
        let graph = Graph::from_edges(
            5,
            &[(0, 1, 10), (1, 2, 10), (2, 3, 10), (0, 4, 10), (4, 0, 10)],
        )
        .unwrap();
        let heuristic =
            StandardHeuristic::euclidean(vec![0.0, 1.0, 2.0, 3.0, -5.0], vec![0.0; 5], 10.0)
                .unwrap();
        (graph, heuristic)
    }

    #[test]
    fn finds_the_cheapest_path_and_reports_real_costs() {
        let (graph, heuristic) = corridor();
        let result = astar(&graph, &[(0, 0)], 3, &heuristic, &SearchOptions::default()).unwrap();
        assert_eq!(result.cost(3), Some(30), "g, not f");
        assert_eq!(result.path(3), Some(vec![0, 1, 2, 3]));
        assert_eq!(graph.walk(0, &result.edge_path(3).unwrap()), Some((3, 30)));
    }

    #[test]
    fn a_zero_heuristic_is_dijkstra_exactly() {
        let (graph, _) = corridor();
        let options = SearchOptions::default();
        let guided = astar(&graph, &[(0, 0)], 3, &StandardHeuristic::Zero, &options).unwrap();
        let plain = dijkstra(&graph, &[(0, 0)], &options.clone().with_targets([3])).unwrap();
        assert_eq!(guided.costs, plain.costs);
        assert_eq!(guided.order, plain.order);
        assert_eq!(guided.parent_nodes, plain.parent_nodes);
    }

    #[test]
    fn guidance_keeps_the_search_off_the_spur() {
        let (graph, heuristic) = corridor();
        let options = SearchOptions::default();
        let guided = astar(&graph, &[(0, 0)], 3, &heuristic, &options).unwrap();
        let plain = astar(&graph, &[(0, 0)], 3, &StandardHeuristic::Zero, &options).unwrap();
        assert!(
            !guided.order.contains(&4),
            "node 4 leads away from the target"
        );
        assert!(plain.order.contains(&4), "unguided, it gets settled anyway");
        assert!(guided.order.len() < plain.order.len());
        assert_eq!(guided.cost(3), plain.cost(3), "same answer, less work");
    }

    #[test]
    fn stops_as_soon_as_the_target_is_settled() {
        let (graph, heuristic) = corridor();
        let result = astar(&graph, &[(0, 0)], 1, &heuristic, &SearchOptions::default()).unwrap();
        assert_eq!(result.order, vec![0, 1]);
        assert_eq!(result.cost(2), None, "never settled, so nothing to report");
    }

    #[test]
    fn max_cost_bounds_the_real_cost_not_the_estimate() {
        let (graph, heuristic) = corridor();
        let options = SearchOptions::default().with_max_cost(20);
        let result = astar(&graph, &[(0, 0)], 3, &heuristic, &options).unwrap();
        assert_eq!(result.cost(2), Some(20), "exactly at the bound");
        assert_eq!(result.cost(3), None, "30 is past it");
    }

    #[test]
    fn initial_costs_shift_the_frontier() {
        let (graph, heuristic) = corridor();
        let result = astar(
            &graph,
            &[(0, 0), (2, 5)],
            3,
            &heuristic,
            &SearchOptions::default(),
        )
        .unwrap();
        assert_eq!(
            result.cost(3),
            Some(15),
            "starting at 2 beats walking the line"
        );
    }

    #[test]
    fn rejects_a_target_outside_the_graph() {
        let (graph, heuristic) = corridor();
        assert_eq!(
            astar(&graph, &[(0, 0)], 9, &heuristic, &SearchOptions::default()).unwrap_err(),
            SearchError::TargetOutOfRange {
                node: 9,
                num_nodes: 5
            }
        );
    }

    #[test]
    fn rejects_a_heuristic_built_for_another_graph() {
        let (graph, _) = corridor();
        let wrong = StandardHeuristic::euclidean(vec![0.0, 1.0], vec![0.0, 0.0], 1.0).unwrap();
        assert_eq!(
            astar(&graph, &[(0, 0)], 3, &wrong, &SearchOptions::default()).unwrap_err(),
            SearchError::HeuristicCoverage {
                heuristic_nodes: 2,
                num_nodes: 5
            }
        );
    }
}
