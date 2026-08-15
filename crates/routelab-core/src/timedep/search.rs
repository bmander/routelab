//! Time-dependent Dijkstra: label-setting with a clock.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{EdgeId, Graph, NodeId, Weight, UNREACHABLE};
use crate::search::{check_sources, SearchError, SearchOptions, SearchResult, TargetTracker};

use super::{Calendar, Departure, WEEK};

/// Follow `edges` from `start`, leaving at `departure`; return where you end up
/// and how long it took, or `None` if that is not a legal timed walk.
///
/// The timed counterpart of [`Graph::walk`], which starts at cost zero with no
/// clock and so cannot check this. A free function taking `(graph, calendar,
/// …)` rather than a method on either, because it belongs to neither: a graph
/// that knows nothing about time should keep knowing nothing, and a schedule is
/// not the thing being walked. What makes a *step* legal is asked of
/// [`Graph::step`], so the two walks cannot drift apart about that.
pub fn walk(
    graph: &Graph,
    calendar: &Calendar,
    start: NodeId,
    edges: &[EdgeId],
    departure: &Departure,
) -> Option<(NodeId, Weight)> {
    if start as usize >= graph.num_nodes() {
        return None;
    }
    let mut node = start;
    let mut elapsed: Weight = 0;
    for &edge in edges {
        let (head, weight) = graph.step(node, edge)?;
        let now = departure.clock.wrapping_add(elapsed) % WEEK;
        let wait = calendar.wait(edge, now, departure.waiting)?;
        elapsed = elapsed.checked_add(wait)?.checked_add(weight)?;
        node = head;
    }
    Some((node, elapsed))
}

/// Earliest arrival from one or more sources, leaving at `departure`.
///
/// Dreyfus (1969): Dijkstra's algorithm generalises to a time-dependent network
/// unchanged, provided arrival is non-decreasing in departure — the FIFO, or
/// non-overtaking, condition. Where it holds, a node settled at the cheapest
/// arrival time is settled for good, exactly as in the static case, and no label
/// ever needs revisiting.
///
/// **This kernel is the FIFO case and only that.** It gets the condition for
/// free rather than checking it: travel times here are constant and waiting is
/// never negative, so leaving later cannot arrive earlier. A network whose
/// travel times genuinely vary — congestion profiles, timetables — may violate
/// FIFO, and then label-*setting* is the wrong family of algorithm entirely.
/// Nothing here would detect that; it would quietly answer wrongly. Hence the
/// narrow contract, stated rather than implied.
///
/// `departure` is on the weekly clock. Costs come back as **elapsed seconds**,
/// including any waiting, which is what makes the result an ordinary
/// [`SearchResult`]: arrival on the clock is `(departure + cost) % WEEK`, and
/// everything downstream that knows how to read a cost keeps working.
pub fn time_dependent_dijkstra(
    graph: &Graph,
    calendar: &Calendar,
    sources: &[(NodeId, Weight)],
    departure: &Departure,
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
        // Lazy deletion, as in `crate::dijkstra`: a node can sit in the heap
        // several times, but only the entry matching its current best cost is
        // the real one.
        if cost > result.costs[node as usize] {
            continue;
        }
        if result.settle(node, &mut tracker) {
            break;
        }

        // The only line that differs from a static search: what time is it here?
        let now = departure.clock.wrapping_add(cost) % WEEK;

        for edge in graph.out_edges(node) {
            // `None` is a shut edge that this policy cannot get through — either
            // waiting is forbidden, or it never opens again.
            let Some(wait) = calendar.wait(edge, now, departure.waiting) else {
                continue;
            };
            let next_cost = cost.saturating_add(wait).saturating_add(graph.weight(edge));
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
