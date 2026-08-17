//! ULTRA's claim, checked: a kernel that walks only the shortcuts answers
//! exactly as one that walks the whole transfer graph.
//!
//! The oracle is not a second copy of ULTRA. It is the thing ULTRA replaces —
//! every stop-to-stop shortest walk, transitively closed, which is what a
//! timetable kernel needs today to walk without limit and what does not fit
//! on a real network. On instances small enough to close it, the two must
//! agree on every query, and where they differ ULTRA has dropped a transfer
//! some journey needed.

use super::Ultra;
use crate::kernels::csa::ConnectionScan;
use crate::kernels::dijkstra::dijkstra;
use crate::kernels::oracles::random_timetable;
use crate::kernels::raptor::Raptor;
use crate::model::graph::{Graph, NodeId, UNREACHABLE};
use crate::model::search::SearchOptions;
use crate::model::timetable::{Footpaths, Time, Timetable, Transfer};
use crate::util::rng::Rng;

/// A transfer graph that is neither closed nor bounded: short hops between
/// random vertices, both ways, which is what a street network is to a kernel
/// that can only take one-hop transfers.
fn random_transfers(seed: u64, vertices: u32, edges: u32) -> Graph {
    let mut rng = Rng::new(seed ^ 0xa17a);
    let mut links = Vec::new();
    for _ in 0..edges {
        let a = rng.below(u64::from(vertices)) as NodeId;
        let b = rng.below(u64::from(vertices)) as NodeId;
        if a == b {
            continue;
        }
        let walk = 30 + rng.below(240) as Time;
        links.push((a, b, walk));
        links.push((b, a, walk));
    }
    Graph::from_edges(vertices as usize, &links).expect("links were drawn from the vertex set")
}

/// A corridor of neighbourhoods: every trip stays inside one, and the only
/// way across is a short walk between two stops no vehicle joins.
///
/// Random timetables will not do for this. Their trips share stops freely, so
/// a rider changes vehicles standing still and no walk is ever *between* two
/// of them — which leaves nothing for a shortcut to represent, and a
/// comparison that cannot tell a sufficient set from an empty one. Here a
/// journey across the corridor must ride, walk, ride, and the walk is exactly
/// the intermediate transfer ULTRA exists to find.
fn corridor(seed: u64, neighbourhoods: u32, per: u32) -> (Timetable, Graph) {
    let mut rng = Rng::new(seed ^ 0xc0881d0a);
    let stops = neighbourhoods * per;
    let mut connections = Vec::new();
    let mut trip = 0u32;
    for hood in 0..neighbourhoods {
        let first = hood * per;
        // Several vehicles a neighbourhood, each running its stops in order
        // and leaving later the further along the corridor it is, so that a
        // rider crossing it is always catching the next one.
        for _ in 0..3 {
            let mut now = hood * 900 + rng.below(600) as Time;
            for i in 0..per - 1 {
                let departs = now + rng.below(120) as Time;
                let arrives = departs + 60 + rng.below(120) as Time;
                connections.push(crate::kernels::oracles::c(
                    trip,
                    first + i,
                    first + i + 1,
                    departs,
                    arrives,
                ));
                now = arrives;
            }
            trip += 1;
        }
    }
    // The bridges, and nothing else: no walking within a neighbourhood, so a
    // rider cannot simply walk the corridor end to end.
    let mut links = Vec::new();
    for hood in 0..neighbourhoods - 1 {
        let (a, b) = (hood * per + per - 1, (hood + 1) * per);
        let walk = 60 + rng.below(120) as Time;
        links.push((a, b, walk));
        links.push((b, a, walk));
    }
    (
        Timetable::new(stops as usize, connections),
        Graph::from_edges(stops as usize, &links).expect("bridges join real stops"),
    )
}

/// A corridor whose neighbourhoods are joined by *two* walks rather than one,
/// so that a stop has more than one shortcut leaving it.
///
/// [`corridor`] gives every bridging stop exactly one destination, which makes
/// it blind to a merge that collapses by origin alone — the stop pair and the
/// origin are the same thing there. Here the last stop of a neighbourhood can
/// walk to either of the next one's first two stops, and which is better
/// depends on when you arrive.
fn braided(seed: u64, neighbourhoods: u32, per: u32) -> (Timetable, Graph) {
    let (table, transfers) = corridor(seed, neighbourhoods, per);
    let mut links = Vec::new();
    for tail in 0..transfers.num_nodes() as NodeId {
        for edge in transfers.out_edges(tail) {
            links.push((tail, transfers.head(edge), transfers.weight(edge)));
        }
    }
    let mut rng = Rng::new(seed ^ 0xb2a1d);
    for hood in 0..neighbourhoods - 1 {
        let (a, b) = (hood * per + per - 1, (hood + 1) * per + 1);
        let walk = 90 + rng.below(150) as Time;
        links.push((a, b, walk));
        links.push((b, a, walk));
    }
    (
        table,
        Graph::from_edges(transfers.num_nodes(), &links).expect("bridges join real stops"),
    )
}

/// How long it takes to walk from `from` to everywhere.
fn walks_from(transfers: &Graph, from: NodeId) -> Vec<Time> {
    dijkstra(transfers, &[(from, 0)], &SearchOptions::default())
        .expect("a source drawn from the graph's own vertices")
        .costs
}

/// Every stop-to-stop shortest walk, as the closed footpath set a kernel
/// would need to walk this transfer graph without ULTRA.
fn closed(timetable: &Timetable, transfers: &Graph) -> Footpaths {
    let stops = stops_of(timetable, transfers);
    let mut links = Vec::new();
    for &from in &stops {
        let reach = walks_from(transfers, from);
        for &to in &stops {
            if to != from && reach[to as usize] != UNREACHABLE {
                links.push((from, to, reach[to as usize]));
            }
        }
    }
    Footpaths::new(transfers.num_nodes(), links)
}

/// The vertices a vehicle calls at.
fn stops_of(timetable: &Timetable, transfers: &Graph) -> Vec<NodeId> {
    let mut serves = vec![false; transfers.num_nodes()];
    for c in timetable.connections() {
        serves[c.from as usize] = true;
        serves[c.to as usize] = true;
    }
    (0..transfers.num_nodes() as NodeId)
        .filter(|&v| serves[v as usize])
        .collect()
}

/// Earliest arrival walking the closed set: one query, the way a kernel is
/// asked today.
fn by_closure(
    timetable: &Timetable,
    paths: &Footpaths,
    from: NodeId,
    at: Time,
    to: NodeId,
) -> Option<Time> {
    let rounds = Raptor::build(timetable, Transfer::instant(), paths);
    rounds.earliest_arrival(from, at, to).map(|i| i.arrives)
}

/// Earliest arrival walking only ULTRA's shortcuts, with the initial and
/// final transfers searched at query time — which is the query scheme the
/// paper prescribes and the reason the shortcuts need only cover the
/// intermediate ones.
fn by_ultra(
    timetable: &Timetable,
    transfers: &Graph,
    shortcuts: &Footpaths,
    stops: &[NodeId],
    from: NodeId,
    at: Time,
    to: NodeId,
) -> Option<Time> {
    let out = walks_from(transfers, from);
    let back = reversed(transfers);
    let home = walks_from(&back, to);

    // The initial transfer: every stop the source can walk to, each already
    // that far along when the query departs.
    let sources: Vec<(NodeId, Time)> = stops
        .iter()
        .filter(|&&v| out[v as usize] != UNREACHABLE)
        .map(|&v| (v, at.saturating_add(out[v as usize])))
        .collect();

    // A journey that boards nothing: walk the whole way.
    let mut best = if out[to as usize] == UNREACHABLE {
        UNREACHABLE
    } else {
        at.saturating_add(out[to as usize])
    };
    if !sources.is_empty() {
        let rounds = Raptor::build(timetable, Transfer::instant(), shortcuts);
        let search = rounds.search(&sources, None, None, Some(at));
        // The final transfer: get off anywhere and walk the rest.
        for &v in stops {
            let (Some(arrived), reach) = (search.cost(v), home[v as usize]) else {
                continue;
            };
            if reach != UNREACHABLE {
                best = best.min(arrived.saturating_add(reach));
            }
        }
    }
    (best != UNREACHABLE).then_some(best)
}

/// The same walks the other way round, so "who can walk to the target" is a
/// search from it.
fn reversed(transfers: &Graph) -> Graph {
    let mut links = Vec::new();
    for tail in 0..transfers.num_nodes() as NodeId {
        for edge in transfers.out_edges(tail) {
            links.push((transfers.head(edge), tail, transfers.weight(edge)));
        }
    }
    Graph::from_edges(transfers.num_nodes(), &links).expect("the same vertices, reversed")
}

#[test]
fn shortcuts_answer_as_the_whole_transfer_graph_does() {
    // The load-bearing test. Unrestricted walking, worked out two ways: the
    // closure a kernel would need, and the handful of shortcuts ULTRA says
    // are enough.
    let mut instances: Vec<(Timetable, Graph)> = Vec::new();
    for seed in 0..8 {
        instances.push((random_timetable(seed, 8, 16), random_transfers(seed, 8, 10)));
    }
    for seed in 0..6 {
        instances.push(corridor(seed, 3, 3));
        instances.push(corridor(seed, 4, 2));
    }
    for (seed, (table, transfers)) in instances.into_iter().enumerate() {
        let stops = stops_of(&table, &transfers);
        let ultra = Ultra::compute(&table, &transfers);
        let shortcuts = Footpaths::new(transfers.num_nodes(), ultra.shortcuts().to_vec());
        let whole = closed(&table, &transfers);
        for &from in &stops {
            for &to in &stops {
                for at in [0, 900, 2400] {
                    let truth = by_closure(&table, &whole, from, at, to);
                    let got = by_ultra(&table, &transfers, &shortcuts, &stops, from, at, to);
                    assert_eq!(
                        got,
                        truth,
                        "seed {seed}: {from} -> {to} at {at} with {} shortcuts",
                        ultra.len()
                    );
                }
            }
        }
    }
}

#[test]
fn a_connection_scan_reads_the_same_shortcuts() {
    // The paper's claim is that the shortcuts go into any kernel that takes
    // one-hop transfers, not into one of them. CSA is the other one it names.
    for seed in 0..6 {
        let table = random_timetable(seed, 7, 14);
        let transfers = random_transfers(seed, 7, 9);
        let stops = stops_of(&table, &transfers);
        let ultra = Ultra::compute(&table, &transfers);
        let shortcuts = Footpaths::new(transfers.num_nodes(), ultra.shortcuts().to_vec());
        let scan = ConnectionScan::build(&table, Transfer::instant(), &shortcuts);
        let rounds = Raptor::build(&table, Transfer::instant(), &shortcuts);
        for &from in &stops {
            for &to in &stops {
                for at in [0, 1500] {
                    let by_scan = scan.earliest_arrival(from, at, to).map(|i| i.arrives);
                    let by_rounds = rounds.earliest_arrival(from, at, to).map(|i| i.arrives);
                    assert_eq!(by_scan, by_rounds, "seed {seed}: {from} -> {to} at {at}");
                }
            }
        }
    }
}

#[test]
fn a_transfer_graph_of_islands_needs_no_shortcuts() {
    // Nothing walks anywhere, so no intermediate transfer is ever on foot and
    // the shortcut set is empty — the plain model, arrived at rather than
    // asked for.
    let table = random_timetable(3, 6, 12);
    let nothing = Graph::from_edges(6, &[]).unwrap();
    let ultra = Ultra::compute(&table, &nothing);
    assert!(ultra.is_empty());
    assert_eq!(ultra.candidates(), 0);
}

#[test]
fn a_walk_between_two_vehicles_is_kept() {
    // Trip 1 lands at stop 1 at 100; trip 2 leaves stop 2 at 400; stop 1
    // walks to stop 2 in 60. That walk is the only way to reach stop 3, so
    // the shortcut has to be there.
    let table = Timetable::new(
        4,
        [
            crate::kernels::oracles::c(1, 0, 1, 0, 100),
            crate::kernels::oracles::c(2, 2, 3, 400, 500),
        ],
    );
    let transfers = Graph::from_edges(4, &[(1, 2, 60), (2, 1, 60)]).unwrap();
    let ultra = Ultra::compute(&table, &transfers);
    assert_eq!(ultra.shortcuts(), &[(1, 2, 60)]);
    // And it is enough: boarding at 0 reaches 3 at 500.
    let shortcuts = Footpaths::new(4, ultra.shortcuts().to_vec());
    let rounds = Raptor::build(&table, Transfer::instant(), &shortcuts);
    assert_eq!(
        rounds.earliest_arrival(0, 0, 3).map(|i| i.arrives),
        Some(500)
    );
}

#[test]
fn a_walk_no_journey_needs_is_dropped() {
    // The same two trips, but now trip 2 also leaves the stop trip 1 lands
    // at, earlier than the walk could deliver anyone. Walking is pointless,
    // so no shortcut is kept — which is the pruning doing its work.
    let table = Timetable::new(
        4,
        [
            crate::kernels::oracles::c(1, 0, 1, 0, 100),
            crate::kernels::oracles::c(2, 1, 3, 110, 200),
            crate::kernels::oracles::c(3, 2, 3, 400, 500),
        ],
    );
    let transfers = Graph::from_edges(4, &[(1, 2, 60), (2, 1, 60)]).unwrap();
    let ultra = Ultra::compute(&table, &transfers);
    assert!(ultra.shortcuts().is_empty(), "kept {:?}", ultra.shortcuts());
}

#[test]
fn the_agreement_test_has_teeth() {
    // A guard on the guard: if the shortcuts were dropped, the comparison
    // above would have to notice. It does, on the very first seed — so the
    // test is measuring the shortcuts and not the query scheme around them.
    let (table, transfers) = corridor(0, 3, 3);
    let stops = stops_of(&table, &transfers);
    let whole = closed(&table, &transfers);
    let none = Footpaths::none();
    let mut disagreed = 0;
    let mut kept = 0;
    for &from in &stops {
        for &to in &stops {
            for at in [0, 900, 2400] {
                let truth = by_closure(&table, &whole, from, at, to);
                if by_ultra(&table, &transfers, &none, &stops, from, at, to) != truth {
                    disagreed += 1;
                }
                kept += usize::from(truth.is_some());
            }
        }
    }
    assert!(
        kept > 0,
        "the instance answers nothing, so it proves nothing"
    );
    assert!(
        disagreed > 0,
        "walking no shortcut at all answered every query, so the comparison \
         cannot tell a sufficient set from an empty one"
    );
    assert!(!Ultra::compute(&table, &transfers).is_empty());
}

#[test]
fn blocks_of_sources_merge_to_the_shortest_walk() {
    // The sweep hands blocks of source stops to threads and reduces each
    // block to one entry per stop pair as it finishes, so only a corridor
    // long enough to span several blocks exercises the merge at all. Every
    // other instance here is nine stops and fits in one.
    let (table, transfers) = braided(3, 12, 3);
    let stops = stops_of(&table, &transfers);
    assert!(
        stops.len() > super::BLOCK,
        "{} sources is one block, which merges nothing",
        stops.len()
    );

    let ultra = Ultra::compute(&table, &transfers);
    assert!(!ultra.is_empty(), "a corridor is crossed by walking");

    // What reducing per block could get wrong: keeping whichever duration the
    // first block to finish happened to find, rather than the shortest. Every
    // kept shortcut must be the shortest walk between its two stops.
    for &(from, to, duration) in ultra.shortcuts() {
        assert_eq!(
            duration,
            walks_from(&transfers, from)[to as usize],
            "{from} -> {to} kept a walk that is not the shortest"
        );
    }

    // And a pair kept once, not once per block that found it.
    let mut pairs: Vec<(NodeId, NodeId)> =
        ultra.shortcuts().iter().map(|&(a, b, _)| (a, b)).collect();
    let found = pairs.len();
    pairs.sort_unstable();
    pairs.dedup();
    assert_eq!(pairs.len(), found, "a stop pair survived the merge twice");

    // The instance has to be able to tell a merge by pair from a merge by
    // origin, or the assertions above hold for a fixture's reasons rather
    // than the code's. A plain corridor cannot: each of its bridging stops
    // walks to exactly one other, so the two rules agree there.
    let mut origins: Vec<NodeId> = pairs.iter().map(|&(a, _)| a).collect();
    origins.dedup();
    assert!(
        origins.len() < pairs.len(),
        "no stop has two shortcuts leaving it, so this cannot see the difference"
    );

    // And the whole point still holds across the blocks: same answers as the
    // closure, on an instance big enough to have been split.
    let shortcuts = Footpaths::new(transfers.num_nodes(), ultra.shortcuts().to_vec());
    let whole = closed(&table, &transfers);
    for &from in &stops {
        for &to in &stops {
            for at in [0, 900, 2400] {
                assert_eq!(
                    by_ultra(&table, &transfers, &shortcuts, &stops, from, at, to),
                    by_closure(&table, &whole, from, at, to),
                    "{from} -> {to} at {at}"
                );
            }
        }
    }
}
