//! Breadth-first search: the unit-weight special case, kept separate because a
//! FIFO queue beats a heap when every edge costs the same.

use std::collections::VecDeque;

use crate::graph::{Graph, NodeId, UNREACHABLE};
use crate::search::{check_nodes, SearchError, SearchOptions, SearchResult, TargetTracker};

/// Hop counts from one or more sources, all of which start at depth 0.
///
/// Edge weights are ignored — the cost of a node is the number of edges on the
/// path to it. `SearchOptions::max_cost` bounds the depth. Unlike
/// [`crate::dijkstra`], sources take no initial cost: a FIFO queue is only
/// correct when every entry enters at the same depth.
pub fn bfs(
    graph: &Graph,
    sources: &[NodeId],
    options: &SearchOptions,
) -> Result<SearchResult, SearchError> {
    check_nodes(graph, sources.iter().copied())?;

    let mut result = SearchResult::new(graph.num_nodes());
    let mut tracker = TargetTracker::new(&options.targets, options.reach, graph.num_nodes());
    if tracker.as_ref().is_some_and(|t| t.done()) {
        return Ok(result);
    }
    let max_depth = options.max_cost.unwrap_or(UNREACHABLE);

    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for &node in sources {
        if result.costs[node as usize] == UNREACHABLE {
            result.costs[node as usize] = 0;
            queue.push_back(node);
        }
    }

    while let Some(node) = queue.pop_front() {
        let depth = result.costs[node as usize];
        if result.settle(node, &mut tracker) {
            break;
        }
        if depth == max_depth {
            continue;
        }

        for edge in graph.out_edges(node) {
            let head = graph.head(edge);
            if result.costs[head as usize] == UNREACHABLE {
                result.costs[head as usize] = depth + 1;
                result.parent_nodes[head as usize] = node;
                result.parent_edges[head as usize] = edge;
                queue.push_back(head);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dijkstra::dijkstra;

    /// A path 0->1->2->3 with a long shortcut 0->3, plus an isolated node 4.
    fn shortcut() -> Graph {
        Graph::from_edges(5, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 100)]).unwrap()
    }

    #[test]
    fn counts_hops_not_weights() {
        let g = shortcut();
        let r = bfs(&g, &[0], &SearchOptions::default()).unwrap();
        assert_eq!(r.cost(3), Some(1), "the 100-weight shortcut is one hop");
        assert_eq!(r.path(3), Some(vec![0, 3]));
        assert_eq!(r.cost(4), None);
    }

    #[test]
    fn max_depth_bounds_the_frontier() {
        let g = shortcut();
        let r = bfs(&g, &[0], &SearchOptions::default().with_max_cost(1)).unwrap();
        assert_eq!(r.cost(1), Some(1));
        assert_eq!(r.cost(2), None);
    }

    #[test]
    fn multi_source_starts_everyone_at_zero() {
        let g = shortcut();
        let r = bfs(&g, &[0, 2], &SearchOptions::default()).unwrap();
        assert_eq!(r.cost(2), Some(0));
        assert_eq!(r.cost(3), Some(1));
    }

    #[test]
    fn agrees_with_dijkstra_when_all_weights_are_one() {
        let unit = Graph::from_edges(5, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)]).unwrap();
        let by_bfs = bfs(&unit, &[0], &SearchOptions::default()).unwrap();
        let by_dijkstra = dijkstra(&unit, &[(0, 0)], &SearchOptions::default()).unwrap();
        assert_eq!(by_bfs.costs, by_dijkstra.costs);
    }
}
