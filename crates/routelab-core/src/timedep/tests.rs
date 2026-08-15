//! The claim is that this is Dijkstra with a clock — so the load-bearing test is
//! that with nothing scheduled it *is* Dijkstra, node for node, not merely to
//! the same total. Everything after that is about what the clock adds.

use super::*;
use crate::dijkstra::dijkstra;
use crate::graph::{EdgeId, Graph, NodeId, Weight};
use crate::rng::Rng;
use crate::search::{SearchOptions, SearchResult};

const HOUR: Clock = 3_600;
const DAY: Clock = 24 * HOUR;

/// A moment, as a weekday (0 is Monday) and an hour.
fn at(day: Clock, hour: Clock) -> Clock {
    day * DAY + hour * HOUR
}

fn source() -> Vec<(NodeId, Weight)> {
    vec![(0, 0)]
}

/// Node 0 to node 2 two ways: straight through 1 (fast, gated) or round via 3
/// (slow, always open). The shape of the Ballard Locks, in four nodes.
fn gated() -> Graph {
    Graph::from_edges(
        4,
        &[
            (0, 1, 300),  // 5 min up to the gate
            (1, 2, 300),  // 5 min across  <- the gate
            (0, 3, 1800), // 30 min round the long way
            (3, 2, 1800),
        ],
    )
    .unwrap()
}

/// Find the edge from `tail` to `head`. Edge ids are CSR positions, not input
/// positions — `Graph::from_edges` permutes — so a test that hardcodes one is
/// testing whatever happened to land there.
fn edge_between(graph: &Graph, tail: NodeId, head: NodeId) -> EdgeId {
    graph
        .out_edges(tail)
        .find(|&edge| graph.head(edge) == head)
        .unwrap_or_else(|| panic!("no edge {tail} -> {head}"))
}

/// The gate across 1 -> 2, open 07:00-21:00 every day.
fn locks(graph: &Graph) -> Calendar {
    Calendar::from_windows([(
        edge_between(graph, 1, 2),
        Window::daily(7 * HOUR, 21 * HOUR),
    )])
}

/// The whole locks fixture, run for one departure — the only thing the cases
/// below actually vary.
fn through_the_locks(departure: Departure) -> SearchResult {
    let graph = gated();
    time_dependent_dijkstra(
        &graph,
        &locks(&graph),
        &source(),
        &departure,
        &SearchOptions::default(),
    )
    .unwrap()
}

fn messy_graph(seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let edges: Vec<_> = (0..150)
        .map(|_| {
            (
                rng.below(50) as u32,
                rng.below(50) as u32,
                1 + rng.below(600) as u32,
            )
        })
        .collect();
    Graph::from_edges(50, &edges).unwrap()
}

/// A calendar restricting a scattering of edges to arbitrary daily windows,
/// reproducibly. Messy on purpose: the point is that schedules nobody designed
/// cannot break the invariants, not that a tidy one does not.
fn scattered(graph: &Graph, seed: u64) -> Calendar {
    let mut rng = Rng::new(seed);
    let mut entries = Vec::new();
    for edge in 0..graph.num_edges() as EdgeId {
        if rng.below(3) == 0 {
            let start = rng.below(u64::from(DAY)) as Clock;
            entries.push((edge, Window::daily(start, start + 6 * HOUR)));
        }
    }
    Calendar::from_windows(entries)
}

// --- Window and Calendar ----------------------------------------------------

#[test]
fn one_window_is_one_interval_in_the_week_not_a_daily_rule() {
    // Worth pinning, because it is the mistake waiting to be made: a bare
    // `Window` is Monday's 07:00-21:00 and nothing else. Daily schedules go
    // through `daily`, which is why that exists.
    let monday_only = Window::new(7 * HOUR, 21 * HOUR);
    assert!(monday_only.contains(at(0, 12)));
    assert!(
        !monday_only.contains(at(1, 12)),
        "Tuesday is a different day"
    );
}

#[test]
fn a_daily_window_contains_its_hours_on_every_day() {
    let locks = Window::daily(7 * HOUR, 21 * HOUR);
    assert_eq!(locks.len(), 7);
    let open = |t| locks.iter().any(|w| w.contains(t));
    for day in 0..7 {
        assert!(open(at(day, 12)), "day {day} at noon");
        assert!(open(at(day, 7)), "open at the opening instant");
        assert!(!open(at(day, 21)), "shut at the closing instant");
        assert!(!open(at(day, 3)), "day {day} at 3am");
    }
}

#[test]
fn a_daily_window_may_wrap_midnight_and_the_week() {
    let overnight = Window::daily(23 * HOUR, 5 * HOUR);
    let open = |t| overnight.iter().any(|w| w.contains(t));
    assert!(open(at(0, 23)));
    assert!(open(at(1, 2)), "still inside, small hours of the next day");
    assert!(!open(at(1, 12)));
    assert!(open(at(6, 23)));
    assert!(open(at(0, 1)), "Monday 01:00 is inside Sunday's window");
}

#[test]
fn weekday_windows_skip_the_days_they_do_not_name() {
    // `Mo-Fr 05:00-11:00`, the express-lane shape.
    let weekdays = Window::on_days(0..5, 5 * HOUR, 11 * HOUR);
    assert_eq!(weekdays.len(), 5);
    let open = |t| weekdays.iter().any(|w| w.contains(t));
    assert!(open(at(0, 8)), "Monday");
    assert!(open(at(4, 8)), "Friday");
    assert!(!open(at(5, 8)), "Saturday");
    assert!(!open(at(6, 8)), "Sunday");
}

#[test]
fn all_day_is_spelled_as_an_end_of_24_00() {
    let all_day = Window::daily(0, 24 * HOUR);
    let open = |t| all_day.iter().any(|w| w.contains(t));
    for day in 0..7 {
        assert!(open(at(day, 0)) && open(at(day, 13)) && open(at(day, 23)));
    }
}

#[test]
fn waiting_reports_the_gap_to_the_next_opening() {
    let calendar = Calendar::from_windows([(0, Window::daily(7 * HOUR, 21 * HOUR))]);
    assert_eq!(calendar.wait(0, at(0, 12), Waiting::Unrestricted), Some(0));
    // 06:55 is five minutes early — the case the whole policy exists for.
    assert_eq!(
        calendar.wait(0, at(0, 7) - 300, Waiting::Unrestricted),
        Some(300)
    );
    // 21:05 waits until tomorrow morning: nearly ten hours.
    assert_eq!(
        calendar.wait(0, at(0, 21) + 300, Waiting::Unrestricted),
        Some(10 * HOUR - 300)
    );
    // And on Sunday night the next opening is Monday, across the week wrap.
    assert_eq!(
        calendar.wait(0, at(6, 22), Waiting::Unrestricted),
        Some(9 * HOUR)
    );
}

#[test]
fn forbidding_waiting_refuses_a_shut_edge() {
    let calendar = Calendar::from_windows([(0, Window::daily(7 * HOUR, 21 * HOUR))]);
    assert!(calendar.is_open(0, at(0, 12)));
    assert_eq!(calendar.wait(0, at(0, 12), Waiting::Forbidden), Some(0));
    assert!(!calendar.is_open(0, at(0, 3)));
    assert_eq!(calendar.wait(0, at(0, 3), Waiting::Forbidden), None);
}

#[test]
fn an_edge_with_no_windows_is_never_open() {
    let calendar = Calendar::from_windows([(0, vec![])]);
    assert!(!calendar.is_open(0, at(0, 12)));
    assert_eq!(calendar.wait(0, at(0, 12), Waiting::Unrestricted), None);
}

#[test]
fn an_edge_nobody_mentioned_is_always_open() {
    let calendar = Calendar::from_windows([(0, Window::daily(0, HOUR))]);
    assert!(calendar.is_open(7, at(3, 4)), "edge 7 has no entry");
    assert_eq!(calendar.wait(7, at(3, 4), Waiting::Forbidden), Some(0));
    // Including edges past the end of the bitset entirely.
    assert!(calendar.is_open(100_000, at(3, 4)));
}

#[test]
fn windows_named_twice_for_one_edge_are_pooled_not_replaced() {
    // One way splits into many edges and may carry more than one restriction,
    // so a caller accumulating them should not have to group first.
    let calendar = Calendar::from_windows([
        (0, Window::daily(6 * HOUR, 9 * HOUR)),
        (0, Window::daily(15 * HOUR, 18 * HOUR)),
    ]);
    assert!(
        calendar.is_open(0, at(0, 7)),
        "the first restriction survived"
    );
    assert!(calendar.is_open(0, at(0, 16)), "and so did the second");
    assert!(!calendar.is_open(0, at(0, 12)));
    assert_eq!(
        calendar.wait(0, at(0, 12), Waiting::Unrestricted),
        Some(3 * HOUR),
        "at noon the nearer opening is the afternoon one"
    );
}

#[test]
fn input_keyed_windows_are_translated_into_edge_ids() {
    // The hazard this constructor exists for: `from_edges` permutes into CSR
    // order, so windows keyed by input position and read as edge ids would shut
    // the wrong street silently. Input 1 here is 1 -> 2; CSR edge 1 is not.
    let graph = gated();
    let gate = edge_between(&graph, 1, 2);
    assert_ne!(
        gate, 1,
        "the fixture would not test anything if these matched"
    );

    let calendar = Calendar::from_input_windows(&graph, [(1, Window::daily(7 * HOUR, 21 * HOUR))]);
    assert!(
        !calendar.is_open(gate, at(0, 3)),
        "the gate is what got shut"
    );
    assert!(
        calendar.is_open(1, at(0, 3)),
        "and CSR edge 1 was left alone"
    );
}

#[test]
fn the_table_reports_its_own_size() {
    let empty = Calendar::unrestricted();
    assert!(empty.is_empty() && empty.footprint() == 0);
    assert_eq!(empty.len(), 0);

    let calendar = Calendar::from_windows([
        (0, Window::daily(7 * HOUR, 21 * HOUR)),
        (9, Window::daily(0, HOUR)),
    ]);
    assert_eq!(calendar.len(), 2, "two edges restricted, fourteen windows");
    assert!(!calendar.is_empty());
    assert!(calendar.footprint() >= 14 * std::mem::size_of::<Window>());
}

// --- The search -------------------------------------------------------------

#[test]
fn with_nothing_scheduled_it_is_plain_dijkstra() {
    // Not "the same costs" — the same search. Same settle order, same parents,
    // same edges. If this drifts, the kernel is not Dijkstra with a clock.
    for seed in 0..12 {
        let graph = messy_graph(seed);
        let options = SearchOptions::default();
        let timed = time_dependent_dijkstra(
            &graph,
            &Calendar::unrestricted(),
            &source(),
            &Departure::at(at(2, 9)),
            &options,
        )
        .unwrap();
        let plain = dijkstra(&graph, &source(), &options).unwrap();

        assert_eq!(timed.costs, plain.costs, "seed {seed}");
        assert_eq!(timed.order, plain.order, "seed {seed}");
        assert_eq!(timed.parent_nodes, plain.parent_nodes, "seed {seed}");
        assert_eq!(timed.parent_edges, plain.parent_edges, "seed {seed}");
    }
}

#[test]
fn the_departure_time_does_not_matter_when_nothing_is_scheduled() {
    let graph = messy_graph(3);
    let calendar = Calendar::unrestricted();
    let run = |clock| {
        time_dependent_dijkstra(
            &graph,
            &calendar,
            &source(),
            &Departure::at(clock),
            &SearchOptions::default(),
        )
        .unwrap()
        .costs
    };
    let noon = run(at(0, 12));
    for departure in [at(0, 0), at(3, 4), at(6, 23), WEEK - 1] {
        assert_eq!(run(departure), noon, "departing {departure}");
    }
}

#[test]
fn an_open_gate_is_simply_the_short_way() {
    let result = through_the_locks(Departure::at(at(0, 12)));
    assert_eq!(result.cost(2), Some(600));
    assert_eq!(result.path(2), Some(vec![0, 1, 2]));
}

#[test]
fn arriving_just_before_opening_waits_rather_than_detouring() {
    // Leave at 06:50, reach the gate at 06:55, wait five minutes, cross.
    // 300 + 300 (wait) + 300 = 900, against 3600 the long way.
    let result = through_the_locks(Departure::at(at(0, 7) - 600));
    assert_eq!(result.cost(2), Some(900));
    assert_eq!(
        result.path(2),
        Some(vec![0, 1, 2]),
        "waited, did not detour"
    );
}

#[test]
fn arriving_after_closing_takes_the_long_way_round() {
    // Leave at 21:00. The gate shuts as we arrive, and waiting until 07:00 is
    // ten hours — so the thirty-minute detour wins, without being told to.
    let result = through_the_locks(Departure::at(at(0, 21)));
    assert_eq!(result.cost(2), Some(3600));
    assert_eq!(result.path(2), Some(vec![0, 3, 2]));
}

#[test]
fn refusing_to_wait_detours_where_waiting_would_not() {
    // The same 06:50 departure that waits five minutes above. Told it cannot
    // stop, it pays an hour instead — which is what shows the policy is real.
    let result = through_the_locks(Departure::at(at(0, 7) - 600).waiting(Waiting::Forbidden));
    assert_eq!(result.cost(2), Some(3600));
    assert_eq!(result.path(2), Some(vec![0, 3, 2]));
}

#[test]
fn a_search_may_cross_midnight_and_the_week_boundary() {
    // Sunday 20:50: reach the gate at 20:55 and cross with five minutes to
    // spare, while the clock wraps into Monday during the walk.
    let spare = through_the_locks(Departure::at(at(6, 21) - 600));
    assert_eq!(spare.cost(2), Some(600));
    // Ten minutes later it has shut, so it is the long way.
    let missed = through_the_locks(Departure::at(at(6, 21)));
    assert_eq!(missed.cost(2), Some(3600));
}

#[test]
fn a_permanently_shut_edge_is_unreachable_however_long_you_wait() {
    let graph = gated();
    let calendar = Calendar::from_windows([(edge_between(&graph, 1, 2), vec![])]);
    let result = time_dependent_dijkstra(
        &graph,
        &calendar,
        &source(),
        &Departure::at(at(0, 12)),
        &SearchOptions::default(),
    )
    .unwrap();
    assert_eq!(result.cost(2), Some(3600), "forced the long way");
}

#[test]
fn forbidding_waiting_never_beats_allowing_it() {
    // The control, over schedules nobody designed. Refusing to wait can only
    // cost you, never save you.
    for seed in 0..8 {
        let graph = messy_graph(seed);
        let calendar = scattered(&graph, seed);

        for clock in [at(0, 2), at(3, 14)] {
            let run = |departure: Departure| {
                time_dependent_dijkstra(
                    &graph,
                    &calendar,
                    &source(),
                    &departure,
                    &SearchOptions::default(),
                )
                .unwrap()
            };
            let patient = run(Departure::at(clock));
            let hurried = run(Departure::at(clock).waiting(Waiting::Forbidden));

            for node in 0..graph.num_nodes() as NodeId {
                if let Some(hurried_cost) = hurried.cost(node) {
                    let patient_cost = patient.cost(node).expect("waiting cannot lose an answer");
                    assert!(
                        patient_cost <= hurried_cost,
                        "seed {seed} node {node}: waiting {patient_cost} > not waiting {hurried_cost}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_path_it_returns_is_a_legal_timed_walk() {
    // The falsifiability check: walk the edges back with the same clock and the
    // same policy, and land where it claimed at the cost it claimed.
    for seed in 0..12 {
        let graph = messy_graph(seed);
        let calendar = scattered(&graph, seed ^ 0xc10c);

        for waiting in [Waiting::Unrestricted, Waiting::Forbidden] {
            let departure = Departure::at(at(1, 9)).waiting(waiting);
            let result = time_dependent_dijkstra(
                &graph,
                &calendar,
                &source(),
                &departure,
                &SearchOptions::default(),
            )
            .unwrap();
            for node in 0..graph.num_nodes() as NodeId {
                let Some(cost) = result.cost(node) else {
                    continue;
                };
                let edges = result.edge_path(node).unwrap();
                assert_eq!(
                    walk(&graph, &calendar, 0, &edges, &departure),
                    Some((node, cost)),
                    "seed {seed} node {node} waiting {waiting:?}"
                );
            }
        }
    }
}

#[test]
fn an_unscheduled_timed_walk_agrees_with_the_plain_one() {
    // The two walk checkers must not drift: with nothing shut they are the same
    // question, and `Graph::step` is the rule they share.
    let graph = gated();
    let open = Calendar::unrestricted();
    let departure = Departure::at(at(0, 12));
    let edges = vec![edge_between(&graph, 0, 1), edge_between(&graph, 1, 2)];
    assert_eq!(
        walk(&graph, &open, 0, &edges, &departure),
        graph.walk(0, &edges)
    );
    // And both refuse the same nonsense.
    let backwards = vec![edges[1], edges[0]];
    assert!(walk(&graph, &open, 0, &backwards, &departure).is_none());
    assert!(graph.walk(0, &backwards).is_none());
    assert!(walk(&graph, &open, 99, &edges, &departure).is_none());
    assert!(graph.walk(99, &edges).is_none());
}

#[test]
fn bounds_and_targets_work_as_they_do_for_any_search() {
    let graph = gated();
    let run = |options| {
        time_dependent_dijkstra(
            &graph,
            &locks(&graph),
            &source(),
            &Departure::at(at(0, 12)),
            &options,
        )
        .unwrap()
    };
    let bounded = run(SearchOptions::default().with_max_cost(400));
    assert_eq!(bounded.cost(1), Some(300));
    assert_eq!(bounded.cost(2), None, "600 is past the bound");

    let stopped = run(SearchOptions::default().with_targets([1]));
    assert_eq!(stopped.order, vec![0, 1]);
}
