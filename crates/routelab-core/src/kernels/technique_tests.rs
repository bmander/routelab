//! The traits in [`crate::model::technique`], held to by every kernel that
//! implements them.
//!
//! The transit kernels are held to each other in
//! [`crate::kernels::timetable::tests`]; this file is the graph side, and the
//! contracts that cut across both families: a one-shot answer agrees with the
//! kept search it is a shortcut for, and an unpacked path costs what the
//! search said it would.

use crate::kernels::astar::{AStar, AStarQuery};
use crate::kernels::bfs::{Bfs, BfsTechnique};
use crate::kernels::contraction::{ContractionHierarchy, ContractionTechnique, Ordering};
use crate::kernels::csa::{ConnectionScanTechnique, ScanQuery};
use crate::kernels::dijkstra::{dijkstra, Dijkstra, DijkstraTechnique};
use crate::kernels::heuristics::StandardHeuristic;
use crate::kernels::oracles::{random_footpaths, random_graph, random_timetable};
use crate::kernels::raptor::{RaptorQuery, RaptorTechnique};
use crate::kernels::timedep::{
    Calendar, Departure, TimeDependentDijkstra, TimeDependentDijkstraTechnique,
    TimeDependentInputs, TimeDependentQuery,
};
use crate::kernels::tripbased::{TripBasedQuery, TripBasedTechnique};
use crate::model::graph::{Graph, NodeId, Weight};
use crate::model::search::SearchOptions;
use crate::model::technique::{
    Distance, Distances, EarliestArrival, Explored, Footprint, Reads, Searches, Technique,
    TransitNetwork, Unpacks,
};
use crate::model::timetable::{Time, Transfer};
use crate::util::progress::Progress;

/// Every distance Dijkstra finds from every source, `planner` finds too.
fn distances_agree<P: Distance>(name: &str, planner: &P, graph: &Graph) {
    for source in 0..graph.num_nodes() as NodeId {
        let truth = dijkstra(graph, &[(source, 0)], &SearchOptions::default()).unwrap();
        for target in 0..graph.num_nodes() as NodeId {
            assert_eq!(
                planner.distance(&[(source, 0)], target).unwrap(),
                truth.cost(target),
                "{name}: {source} -> {target}"
            );
        }
    }
}

/// An unpacked path is a real walk from the source that ends at the target
/// and measures what the search said it cost — for every target the search
/// claims to know, and nothing at all for the ones it does not.
///
/// Two closures for the two things the graph searches do not agree on:
/// `from` writes a source as this planner takes it (a hop count starts every
/// source at depth zero, so BFS takes bare nodes), and `measure` turns the
/// walk into the planner's own units, given its weight and its length in
/// edges.
fn paths_cost_what_they_say<P>(
    name: &str,
    planner: &P,
    graph: &Graph,
    from: impl Fn(NodeId) -> P::Source,
    to_query: impl Fn(NodeId) -> P::Query,
    measure: impl Fn(Weight, usize) -> u32,
) where
    P: Searches + Unpacks,
    P::Error: std::fmt::Debug,
{
    for source in 0..graph.num_nodes() as NodeId {
        for target in 0..graph.num_nodes() as NodeId {
            let search = planner.search(&[from(source)], &to_query(target)).unwrap();
            let Some(cost) = search.cost(target) else {
                assert!(planner.edge_path(&search, target).is_none());
                continue;
            };
            let edges = planner.edge_path(&search, target).unwrap();
            let walked = graph.walk(source, &edges);
            let (ended, weight) =
                walked.unwrap_or_else(|| panic!("{name}: {source} -> {target} is not a walk"));
            assert_eq!(ended, target, "{name}: {source} -> {target}");
            assert_eq!(
                measure(weight, edges.len()),
                cost,
                "{name}: {source} -> {target}"
            );
        }
    }
}

#[test]
fn every_distance_technique_agrees_with_dijkstra() {
    for seed in 0..4 {
        let graph = random_graph(30, 3.0, seed);
        let progress = Progress::new();
        let plain = DijkstraTechnique.bind(&graph, &progress).unwrap();
        distances_agree("Dijkstra", &plain, &graph);
        let zero = StandardHeuristic::Zero;
        let guided = AStar {
            graph: &graph,
            heuristic: &zero,
        };
        distances_agree("A*", &guided, &graph);
        let hierarchy = ContractionTechnique {
            ordering: Ordering::default(),
        }
        .bind(&graph, &progress)
        .unwrap();
        distances_agree("contraction hierarchy", &hierarchy, &graph);
    }
}

#[test]
fn every_unpacked_path_costs_what_the_search_said() {
    let weighed = |weight: Weight, _edges: usize| weight;
    let counted = |_weight: Weight, edges: usize| edges as u32;
    for seed in 0..4 {
        let graph = random_graph(30, 3.0, seed);
        let progress = Progress::new();
        let plain: Dijkstra<'_> = DijkstraTechnique.bind(&graph, &progress).unwrap();
        paths_cost_what_they_say(
            "Dijkstra",
            &plain,
            &graph,
            |node| (node, 0),
            |_| SearchOptions::default(),
            weighed,
        );
        // The hop-count search: bare sources, and a cost measured in edges.
        let hops: Bfs<'_> = BfsTechnique.bind(&graph, &progress).unwrap();
        paths_cost_what_they_say(
            "BFS",
            &hops,
            &graph,
            |node| node,
            |_| SearchOptions::default(),
            counted,
        );
        let zero = StandardHeuristic::Zero;
        let guided = AStar {
            graph: &graph,
            heuristic: &zero,
        };
        paths_cost_what_they_say(
            "A*",
            &guided,
            &graph,
            |node| (node, 0),
            AStarQuery::to,
            weighed,
        );
        let hierarchy: ContractionHierarchy = ContractionTechnique::default()
            .bind(&graph, &progress)
            .unwrap();
        paths_cost_what_they_say(
            "contraction hierarchy",
            &hierarchy,
            &graph,
            |node| (node, 0),
            |target| target,
            weighed,
        );
        // Every edge open at every hour is Dijkstra with a clock, so the same
        // walk must come back — the one binding of the time-dependent planner.
        let calendar = Calendar::unrestricted();
        let timed: TimeDependentDijkstra<'_> = TimeDependentDijkstraTechnique
            .bind(
                TimeDependentInputs {
                    graph: &graph,
                    calendar: &calendar,
                },
                &progress,
            )
            .unwrap();
        paths_cost_what_they_say(
            "time-dependent Dijkstra",
            &timed,
            &graph,
            |node| (node, 0),
            |_| TimeDependentQuery::at(Departure::at(8 * 3_600)),
            weighed,
        );
    }
}

/// The one-shot answer is the kept search's answer for the same target, and
/// a search that rode something has a search space to show for it.
fn one_shot_is_the_kept_search<P>(
    name: &str,
    planner: &P,
    stops: u32,
    to_query: impl Fn(NodeId) -> P::Query,
) where
    P: EarliestArrival + Reads + Explored + Searches<Source = (NodeId, Time)>,
    P::Error: std::fmt::Debug,
{
    for from in 0..stops {
        for to in 0..stops {
            let sources = [(from, 0), ((from + 2) % stops, 300)];
            let search = planner.search(&sources, &to_query(to)).unwrap();
            let kept = planner.itinerary(&search, to);
            assert_eq!(
                kept.as_ref().map(|i| i.arrives),
                planner.earliest_arrival(&sources, to).map(|i| i.arrives),
                "{name}: {sources:?} -> {to}"
            );
            // A search that rode something has something to draw.
            if kept.is_some_and(|i| i.rides().next().is_some()) {
                assert!(
                    !planner.reached(&search).is_empty(),
                    "{name}: {sources:?} -> {to}"
                );
                assert!(search.settled() > 0, "{name}: {sources:?} -> {to}");
            }
        }
    }
}

#[test]
fn a_one_shot_answer_is_what_the_kept_search_holds() {
    for seed in 0..4 {
        let stops = 8;
        let table = random_timetable(seed, stops, 16);
        let paths = random_footpaths(seed, stops, 3);
        let net = TransitNetwork {
            timetable: &table,
            transfer: Transfer::instant(),
            footpaths: &paths,
        };
        let progress = Progress::new();
        let rounds = RaptorTechnique.bind(net, &progress).unwrap();
        one_shot_is_the_kept_search("RAPTOR", &rounds, stops, RaptorQuery::to);
        let scan = ConnectionScanTechnique.bind(net, &progress).unwrap();
        one_shot_is_the_kept_search("CSA", &scan, stops, ScanQuery::to);
        let trips = TripBasedTechnique::default().bind(net, &progress).unwrap();
        one_shot_is_the_kept_search("trip-based", &trips, stops, TripBasedQuery::to);
    }
}

#[test]
fn every_planner_says_what_it_searches_over() {
    let graph = random_graph(12, 2.0, 1);
    let progress = Progress::new();
    let plain = DijkstraTechnique.bind(&graph, &progress).unwrap();
    assert_eq!(plain.searches(), ("nodes", 12));
    assert_eq!(plain.footprint(), 0);
    let hierarchy = ContractionTechnique::default()
        .bind(&graph, &progress)
        .unwrap();
    assert_eq!(hierarchy.searches(), ("nodes", 12));
    assert!(hierarchy.footprint() > 0);

    let table = random_timetable(2, 6, 10);
    let paths = random_footpaths(2, 6, 2);
    let net = TransitNetwork {
        timetable: &table,
        transfer: Transfer::instant(),
        footpaths: &paths,
    };
    assert_eq!(
        RaptorTechnique.bind(net, &progress).unwrap().searches().0,
        "stops"
    );
    assert_eq!(
        TripBasedTechnique::default()
            .bind(net, &progress)
            .unwrap()
            .searches()
            .0,
        "trips"
    );
}
