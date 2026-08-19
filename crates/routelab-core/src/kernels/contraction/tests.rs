//! The claim under test is exactness: a hierarchy answers what Dijkstra answers,
//! on every instance, or it is worthless. Everything else — how many shortcuts,
//! how few nodes settled — is only interesting once that holds.

use super::*;
use crate::kernels::dijkstra::dijkstra;
use crate::kernels::oracles::random_graph;
use crate::model::search::SearchOptions;
use crate::util::rng::Rng;

/// A `side` x `side` grid, four-connected both ways, with varied weights — the
/// shape a road network has, and the one hierarchies are good at.
fn grid(side: u32, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let mut edges = Vec::new();
    let at = |row: u32, col: u32| row * side + col;
    for row in 0..side {
        for col in 0..side {
            for (next_row, next_col) in [(row, col + 1), (row + 1, col)] {
                if next_row < side && next_col < side {
                    let weight = 1 + rng.below(20) as u32;
                    edges.push((at(row, col), at(next_row, next_col), weight));
                    edges.push((at(next_row, next_col), at(row, col), weight));
                }
            }
        }
    }
    Graph::from_edges((side * side) as usize, &edges).unwrap()
}

/// Every distance Dijkstra finds, the hierarchy finds too.
fn assert_matches_dijkstra(graph: &Graph, hierarchy: &ContractionHierarchy) {
    for source in 0..graph.num_nodes() as NodeId {
        let truth = dijkstra(graph, &[(source, 0)], &SearchOptions::default()).unwrap();
        for target in 0..graph.num_nodes() as NodeId {
            let found = hierarchy.query(&[(source, 0)], target).unwrap();
            assert_eq!(
                found.distance,
                truth.cost(target),
                "distance {source} -> {target}"
            );

            // And the path it reports is a real one: walking the original graph
            // along those edges must arrive, at exactly that cost.
            if let Some(distance) = found.distance {
                let edges = hierarchy.unpack(&found).expect("a distance implies a path");
                assert_eq!(
                    graph.walk(source, &edges),
                    Some((target, distance)),
                    "unpacked path {source} -> {target}"
                );
            }
        }
    }
}

#[test]
fn a_chain_needs_no_shortcuts_at_all() {
    // Tempting to expect one: remove the middle of a -> b -> c and surely
    // something must stand in for it. But the ends have less to lose and go
    // first, and by the time the middle is contracted nothing routes through it.
    // A hierarchy over a path graph is the path graph.
    let graph = Graph::from_edges(3, &[(0, 1, 5), (1, 2, 7)]).unwrap();
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    assert_eq!(hierarchy.query(&[(0, 0)], 2).unwrap().distance, Some(12));
    assert_eq!(hierarchy.num_shortcuts(), 0);
}

#[test]
fn shortcuts_appear_where_there_is_something_to_route_around() {
    // A grid has alternatives everywhere, so contracting its interior does cost
    // something — that cost is the hierarchy.
    let graph = grid(8, 13);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    assert!(hierarchy.num_shortcuts() > 0);
    assert_eq!(
        hierarchy.num_arcs(),
        graph.num_edges() + hierarchy.num_shortcuts(),
        "the augmented graph is the original plus what contraction added"
    );
}

#[test]
fn a_witness_makes_a_shortcut_unnecessary() {
    // The direct edge is at least as good as the detour, so contracting the
    // middle node costs nothing.
    let graph = Graph::from_edges(3, &[(0, 1, 5), (1, 2, 7), (0, 2, 9)]).unwrap();
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    assert_eq!(hierarchy.num_shortcuts(), 0);
    assert_eq!(hierarchy.query(&[(0, 0)], 2).unwrap().distance, Some(9));
}

#[test]
fn every_node_gets_a_distinct_rank() {
    let graph = grid(6, 3);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let mut ranks: Vec<u32> = (0..graph.num_nodes() as NodeId)
        .map(|node| hierarchy.rank(node))
        .collect();
    ranks.sort_unstable();
    assert_eq!(ranks, (0..graph.num_nodes() as u32).collect::<Vec<_>>());
}

#[test]
fn distances_match_dijkstra_on_a_grid() {
    let graph = grid(7, 11);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    assert_matches_dijkstra(&graph, &hierarchy);
}

#[test]
fn distances_match_dijkstra_on_messy_graphs() {
    // Sparse, dense, disconnected, one-way, self-looping — the instances where a
    // hierarchy is most likely to be quietly wrong.
    for seed in 1..8u64 {
        for (nodes, density) in [(30, 1.2), (40, 3.0), (25, 0.6)] {
            let graph = random_graph(nodes, density, seed);
            let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
            assert_matches_dijkstra(&graph, &hierarchy);
        }
    }
}

#[test]
fn an_arbitrary_order_finds_the_same_distances() {
    // Ordering is a performance choice, never a correctness one.
    let graph = grid(6, 5);
    let careful = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let arbitrary = ContractionHierarchy::build(
        &graph,
        Ordering {
            policy: Policy::Random { seed: 9 },
            ..Ordering::default()
        },
    )
    .unwrap();

    assert_matches_dijkstra(&graph, &arbitrary);
    for source in 0..graph.num_nodes() as NodeId {
        for target in 0..graph.num_nodes() as NodeId {
            assert_eq!(
                careful.query(&[(source, 0)], target).unwrap().distance,
                arbitrary.query(&[(source, 0)], target).unwrap().distance
            );
        }
    }
}

#[test]
fn a_considered_order_builds_a_smaller_hierarchy() {
    // The whole reason ordering is a policy worth configuring.
    let graph = grid(12, 7);
    let careful = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let arbitrary = ContractionHierarchy::build(
        &graph,
        Ordering {
            policy: Policy::Random { seed: 4 },
            ..Ordering::default()
        },
    )
    .unwrap();
    assert!(
        careful.num_shortcuts() < arbitrary.num_shortcuts(),
        "edge difference {} vs random {}",
        careful.num_shortcuts(),
        arbitrary.num_shortcuts()
    );
}

#[test]
fn it_settles_far_less_than_dijkstra() {
    let graph = grid(20, 2);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let corner = (graph.num_nodes() - 1) as NodeId;

    let found = hierarchy.query(&[(0, 0)], corner).unwrap();
    let plain = dijkstra(
        &graph,
        &[(0, 0)],
        &SearchOptions::default().with_targets([corner]),
    )
    .unwrap();
    assert_eq!(found.distance, plain.cost(corner));
    assert!(
        found.settled() < plain.order.len(),
        "hierarchy settled {} against Dijkstra's {}",
        found.settled(),
        plain.order.len()
    );
}

#[test]
fn unreachable_targets_report_nothing() {
    let split = Graph::from_edges(4, &[(0, 1, 3), (2, 3, 3)]).unwrap();
    let hierarchy = ContractionHierarchy::build(&split, Ordering::default()).unwrap();
    let found = hierarchy.query(&[(0, 0)], 3).unwrap();
    assert_eq!(found.distance, None);
    assert_eq!(found.meeting, None);
    assert_eq!(hierarchy.unpack(&found), None);
}

#[test]
fn a_source_that_is_the_target_costs_nothing() {
    let graph = grid(4, 1);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let found = hierarchy.query(&[(5, 0)], 5).unwrap();
    assert_eq!(found.distance, Some(0));
    assert_eq!(hierarchy.unpack(&found), Some(Vec::new()));
}

#[test]
fn several_sources_are_searched_from_at_once() {
    // The shape multimodal access needs: each entry point already costing a walk.
    let graph = grid(5, 8);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    let target = 24;

    let together = hierarchy.query(&[(0, 0), (4, 100)], target).unwrap();
    let separate = [
        dijkstra(&graph, &[(0, 0)], &SearchOptions::default())
            .unwrap()
            .cost(target),
        dijkstra(&graph, &[(4, 100)], &SearchOptions::default())
            .unwrap()
            .cost(target)
            .map(|cost| cost + 100),
    ];
    assert_eq!(
        together.distance,
        separate.into_iter().flatten().min(),
        "the cheaper of the two starts"
    );
}

#[test]
fn rejects_nodes_that_are_not_in_the_graph() {
    let graph = grid(3, 1);
    let hierarchy = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
    assert!(hierarchy.query(&[(99, 0)], 0).is_err());
    assert!(hierarchy.query(&[(0, 0)], 99).is_err());
}
