//! Core-CH's claim, checked: the core is a smaller graph with the same
//! distances between the vertices it kept.
//!
//! The oracle is Dijkstra on the network that went in — not a second
//! contraction, and not the hierarchy beside it. If a walk between two stops
//! is shorter through a street corner than the core says, the core dropped
//! something it should have carried, and that is the only way this fails.

use super::{CoreHierarchy, Ordering, Policy};
use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, NodeId, Weight, UNREACHABLE};
use crate::model::search::SearchOptions;
use crate::util::rng::Rng;

/// A messy road-ish network: mostly local links, a few long ones, both ways.
fn sprawl(seed: u64, nodes: u32, extra: u32) -> Graph {
    let mut rng = Rng::new(seed ^ 0xc0e5);
    let mut edges = Vec::new();
    for node in 0..nodes {
        // A ring, so nothing is stranded and every vertex has somewhere to go.
        let next = (node + 1) % nodes;
        let weight = 10 + rng.below(90) as Weight;
        edges.push((node, next, weight));
        edges.push((next, node, weight));
    }
    for _ in 0..extra {
        let a = rng.below(u64::from(nodes)) as NodeId;
        let b = rng.below(u64::from(nodes)) as NodeId;
        if a == b {
            continue;
        }
        let weight = 5 + rng.below(200) as Weight;
        edges.push((a, b, weight));
        edges.push((b, a, weight));
    }
    Graph::from_edges(nodes as usize, &edges).expect("edges drawn from the node set")
}

/// Distances from `from` over `graph`.
fn reach(graph: &Graph, from: NodeId) -> Vec<Weight> {
    dijkstra(graph, &[(from, 0)], &SearchOptions::default())
        .expect("a source from the graph's own nodes")
        .costs
}

fn every_kth(nodes: u32, k: u32) -> Vec<NodeId> {
    (0..nodes).filter(|node| node % k == 0).collect()
}

#[test]
fn the_core_keeps_every_distance_between_the_vertices_it_kept() {
    // The load-bearing test. Contract most of a network away and the walks
    // between what is left must not have got any longer — nor any shorter,
    // which would mean a shortcut short-circuiting something real.
    for seed in 0..6 {
        let graph = sprawl(seed, 60, 40);
        let keep = every_kth(60, 7);
        let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9)
            .expect("the graph contracts");
        for &from in &keep {
            let truth = reach(&graph, from);
            let core = reach(built.core(), from);
            for &to in &keep {
                assert_eq!(
                    core[to as usize], truth[to as usize],
                    "seed {seed}: {from} -> {to} is {} in the core and {} in the network",
                    core[to as usize], truth[to as usize]
                );
            }
        }
        // And it really did contract: every vertex not kept, since the degree
        // bound here is unreachable.
        assert!(built.is_core(keep[0]));
        assert_eq!(built.num_core(), keep.len());
        assert_eq!(built.num_retired(), 60 - keep.len());
    }
}

#[test]
fn a_kept_vertex_survives_however_ordinary_it_is() {
    // The rule that makes it Core-CH: priority does not get a vote.
    let graph = sprawl(1, 40, 25);
    let keep = vec![3, 17, 29];
    let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).expect("contracts");
    for &node in &keep {
        assert!(
            built.is_core(node),
            "{node} was contracted despite being kept"
        );
    }
    for node in 0..40u32 {
        assert_eq!(built.is_core(node), keep.contains(&node));
    }
}

#[test]
fn stopping_early_leaves_more_standing_and_still_answers() {
    // The second rule. A tight degree bound halts the contraction, so the core
    // holds vertices nobody asked to keep — and the distances between the ones
    // they did ask for are unchanged, which is what makes stopping safe.
    let graph = sprawl(2, 80, 60);
    let keep = every_kth(80, 9);
    let tight = CoreHierarchy::build(&graph, &keep, Ordering::default(), 4.0).expect("contracts");
    let whole = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).expect("contracts");
    assert!(
        tight.num_core() > whole.num_core(),
        "a degree bound of 4 retired as much as no bound at all"
    );
    assert!(
        tight.num_core() > keep.len(),
        "nothing extra was left standing"
    );
    for &from in &keep {
        let truth = reach(&graph, from);
        let core = reach(tight.core(), from);
        for &to in &keep {
            assert_eq!(core[to as usize], truth[to as usize], "{from} -> {to}");
        }
    }
}

#[test]
fn the_core_is_smaller_than_the_network_it_stands_for() {
    // The whole point, and worth a number: a graph of stops where there was a
    // graph of streets.
    let graph = sprawl(3, 200, 150);
    let keep = every_kth(200, 20);
    let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).expect("contracts");
    assert_eq!(built.num_core(), 10);
    assert!(
        built.num_arcs() < graph.num_edges(),
        "core has {} arcs where the network had {}",
        built.num_arcs(),
        graph.num_edges()
    );
}

#[test]
fn keeping_everything_contracts_nothing() {
    let graph = sprawl(4, 30, 20);
    let keep: Vec<NodeId> = (0..30).collect();
    let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).expect("contracts");
    assert_eq!(built.num_retired(), 0);
    assert_eq!(built.num_core(), 30);
    // The core is the network again, so every distance is trivially the same.
    for from in [0u32, 11, 29] {
        assert_eq!(reach(built.core(), from), reach(&graph, from));
    }
}

#[test]
fn an_unreachable_pair_stays_unreachable() {
    // Two rings that never meet: contraction must not invent a way across.
    let mut edges = Vec::new();
    for (first, last) in [(0u32, 5u32), (6, 11)] {
        for node in first..last {
            edges.push((node, node + 1, 10));
            edges.push((node + 1, node, 10));
        }
    }
    let graph = Graph::from_edges(12, &edges).unwrap();
    let keep = vec![0u32, 5, 6, 11];
    let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).expect("contracts");
    let core = reach(built.core(), 0);
    assert_eq!(core[5], 50);
    assert_eq!(core[6], UNREACHABLE);
    assert_eq!(core[11], UNREACHABLE);
}

#[test]
fn the_order_is_a_policy_and_never_an_answer() {
    // The same rule the hierarchy beside this one is held to: a different
    // contraction order builds a different core and answers identically.
    let graph = sprawl(5, 50, 35);
    let keep = every_kth(50, 6);
    let sensible = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).unwrap();
    let arbitrary = CoreHierarchy::build(
        &graph,
        &keep,
        Ordering {
            policy: Policy::Random { seed: 9 },
            ..Ordering::default()
        },
        1e9,
    )
    .unwrap();
    for &from in &keep {
        let a = reach(sensible.core(), from);
        let b = reach(arbitrary.core(), from);
        for &to in &keep {
            assert_eq!(a[to as usize], b[to as usize], "{from} -> {to}");
        }
    }
}

/// A three-part search over a partial hierarchy: climb out of the source, cross
/// the core, climb back down into the target.
///
/// This is the skeleton UCCH's query is built on — Dibbelt, Pajor & Wagner
/// §3.2 — written here rather than in the kernel because what it holds is the
/// *preprocessing*: that the ranks, the two component graphs and the core
/// between them still describe the graph that went in. If this is wrong, every
/// query over it is wrong for the same reason.
fn through_the_core(built: &CoreHierarchy, from: NodeId, to: NodeId) -> Weight {
    let up = reach(built.upward(), from);
    // The downward graph is stored reversed, so climbing it from the target is
    // the same walk read backwards.
    let down = reach(built.downward(), to);
    let nodes = up.len();

    // Meeting in the component: a journey that never needed the core at all.
    let mut best = UNREACHABLE;
    for node in 0..nodes {
        if up[node] != UNREACHABLE && down[node] != UNREACHABLE {
            best = best.min(up[node].saturating_add(down[node]));
        }
    }

    // Or through it: climb in at every core vertex the source reached, cross,
    // and climb out at every one the target did.
    for entry in 0..nodes as NodeId {
        if !built.is_core(entry) || up[entry as usize] == UNREACHABLE {
            continue;
        }
        let across = reach(built.core(), entry);
        for exit in 0..nodes {
            if !built.is_core(exit as NodeId) || down[exit] == UNREACHABLE {
                continue;
            }
            if across[exit] == UNREACHABLE {
                continue;
            }
            best = best.min(
                up[entry as usize]
                    .saturating_add(across[exit])
                    .saturating_add(down[exit]),
            );
        }
    }
    best
}

#[test]
fn a_partial_hierarchy_still_describes_the_graph_it_contracted() {
    // The load-bearing test for the preprocessing UCCH needs. Contract most of
    // a network around a handful of kept vertices, then ask every pair through
    // the hierarchy and compare with plain Dijkstra on what went in.
    for seed in 0..5 {
        let graph = sprawl(seed, 60, 40);
        let keep = every_kth(60, 7);
        let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 4.0)
            .expect("the graph contracts");
        for from in 0..60u32 {
            let truth = reach(&graph, from);
            for to in 0..60u32 {
                assert_eq!(
                    through_the_core(&built, from, to),
                    truth[to as usize],
                    "seed {seed}: {from} -> {to}"
                );
            }
        }
    }
}

#[test]
fn a_kept_vertex_is_unranked_and_a_contracted_one_is_not() {
    // What the two halves of the hierarchy are told apart by: the core is
    // exactly the vertices the contraction never gave a rank to.
    let graph = sprawl(7, 50, 35);
    let keep = every_kth(50, 6);
    let built = CoreHierarchy::build(&graph, &keep, Ordering::default(), 1e9).unwrap();
    let mut ranks: Vec<u32> = Vec::new();
    for node in 0..50u32 {
        assert_eq!(
            built.is_core(node),
            built.rank(node) == crate::kernels::contraction::UNRANKED,
            "{node} disagrees about whether it is in the core"
        );
        if !built.is_core(node) {
            ranks.push(built.rank(node));
        }
    }
    // Ranks are the order it contracted in: dense, from zero, no repeats.
    ranks.sort_unstable();
    assert_eq!(ranks, (0..ranks.len() as u32).collect::<Vec<_>>());
    for &node in &keep {
        assert_eq!(built.rank(node), crate::kernels::contraction::UNRANKED);
    }
}
