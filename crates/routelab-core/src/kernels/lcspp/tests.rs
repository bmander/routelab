//! LCSPP's claim, checked: the answer is the best journey the language admits,
//! and no better.
//!
//! The oracles are the searches this one generalises. Allow every mode and it
//! must be the time-dependent search over the merged network; allow no riding
//! and it must be plain Dijkstra over the arcs that are left. Between those two
//! sits the case the kernel exists for — walk, ride, walk — and the fixture is
//! built so that the constrained answer and the unconstrained one differ, since
//! a test where the constraint changes nothing cannot tell whether it was
//! applied.

use super::{label_constrained, Modes, Multimodal};
use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, NodeId, UNREACHABLE};
use crate::model::search::SearchOptions;
use crate::model::timetable::{Leg, Time, Timetable};

/// The modes, as symbols on the arcs.
const FOOT: usize = 0;
const LINK: usize = 1;
const RAIL: usize = 2;

/// Streets `0-1-2-3` end to end, two stops `4` and `5` beside `1` and `2`, and
/// one train between them.
///
/// Walking the whole way is slow and always possible; riding is quicker but
/// only if you walk to a stop, wait, ride, and walk off — which is the journey
/// shape the language has to admit for the train to be usable at all.
fn corridor() -> (Graph, Vec<u8>, Timetable) {
    let mut edges = Vec::new();
    let mut labels = Vec::new();
    let mut arc = |from: NodeId, to: NodeId, weight: Time, mode: usize| {
        edges.push((from, to, weight));
        labels.push(mode as u8);
    };
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        arc(a, b, 1000, FOOT);
        arc(b, a, 1000, FOOT);
    }
    for (street, stop) in [(1, 4), (2, 5)] {
        arc(street, stop, 10, LINK);
        arc(stop, street, 10, LINK);
    }
    let graph = Graph::from_edges(6, &edges).expect("real vertices");
    let timetable = Timetable::new(6, [crate::kernels::oracles::c(0, 4, 5, 1100, 1150)]);
    (graph, labels, timetable)
}

/// The paper's Figure 1(a): a foot state and a rail state, joined by link.
fn foot_and_rail() -> Modes {
    Modes::new(2, 3)
        .within(0, FOOT)
        .within(1, RAIL)
        .on(0, LINK, 1)
        .on(1, LINK, 0)
        .starting(0)
        .accepting(0)
}

/// Everything the pavement offers and nothing else.
fn on_foot() -> Modes {
    Modes::new(1, 3)
        .within(0, FOOT)
        .within(0, LINK)
        .starting(0)
        .accepting(0)
}

fn ask(modes: &Modes, from: NodeId, at: Time, to: NodeId) -> Option<Time> {
    let (graph, labels, timetable) = corridor();
    let network = Multimodal {
        scalar: &graph,
        labels: &labels,
        timetable: &timetable,
        riding: RAIL as u8,
    };
    label_constrained(&network, modes, &[(from, at)], to).map(|journey| journey.arrives)
}

#[test]
fn hello_world() {
    // Walk to the stop, wait for the train, ride, walk off: 1000 to the stop's
    // street, 10 onto the platform, the 11:00 train arriving 1150, 10 back onto
    // the street, 1000 to the end.
    assert_eq!(ask(&foot_and_rail(), 0, 0, 3), Some(2160));
}

#[test]
fn a_language_that_forbids_riding_is_dijkstra_on_what_is_left() {
    // The load-bearing oracle in one direction. With no rail state the train is
    // unreachable however convenient it is, and what is left is an ordinary
    // static graph — so the answer must be the one `dijkstra` gives on it.
    let (graph, _, _) = corridor();
    let truth = dijkstra(&graph, &[(0, 0)], &SearchOptions::default())
        .expect("a source from the graph's own nodes")
        .costs;
    for to in 0..6u32 {
        let got = ask(&on_foot(), 0, 0, to);
        let expected = match truth[to as usize] {
            UNREACHABLE => None,
            cost => Some(cost),
        };
        assert_eq!(got, expected, "on foot to {to}");
    }
}

#[test]
fn the_constraint_is_what_decides() {
    // And the other direction: the same query, the same network, two languages,
    // two answers. Walking the corridor is 3000 and riding it is 2160, so a
    // language that admits the train is worth 840 seconds — which is what makes
    // the test above a claim about the language rather than about the fixture.
    assert_eq!(ask(&on_foot(), 0, 0, 3), Some(3000));
    assert_eq!(ask(&foot_and_rail(), 0, 0, 3), Some(2160));
}

#[test]
fn a_journey_must_end_in_a_state_that_accepts() {
    // Riding is allowed but stepping back onto the pavement is not, so the
    // journey cannot end where it wants to. The train is still reachable — the
    // stop it serves is — which is the difference between forbidding a mode and
    // forbidding an ending.
    let stuck = Modes::new(2, 3)
        .within(0, FOOT)
        .within(1, RAIL)
        .on(0, LINK, 1)
        .starting(0)
        .accepting(1);
    assert_eq!(
        ask(&stuck, 0, 0, 5),
        Some(1150),
        "the platform is reachable"
    );
    assert_eq!(ask(&stuck, 0, 0, 3), None, "but the far pavement is not");
}

#[test]
fn the_legs_are_the_arcs_that_were_taken() {
    // A journey is told as what it did, in order, so a caller can draw it: four
    // walks either side of one ride, and the ride is the connection that was
    // boarded rather than a duration someone worked out afterwards.
    let (graph, labels, timetable) = corridor();
    let network = Multimodal {
        scalar: &graph,
        labels: &labels,
        timetable: &timetable,
        riding: RAIL as u8,
    };
    let journey = label_constrained(&network, &foot_and_rail(), &[(0, 0)], 3)
        .expect("the corridor is walked");
    let rides: Vec<_> = journey
        .legs
        .iter()
        .filter_map(|leg| match leg {
            Leg::Ride(ride) => Some((ride.from, ride.to, ride.departs, ride.arrives)),
            Leg::Walk(_) => None,
        })
        .collect();
    assert_eq!(rides, vec![(4, 5, 1100, 1150)]);
    assert_eq!(journey.legs.len(), 5, "walk, link, ride, link, walk");
    // And they join up, from the origin to the target.
    let mut at = 0;
    for leg in &journey.legs {
        assert_eq!(leg.from(), at, "a leg starts where the last one ended");
        at = leg.to();
    }
    assert_eq!(at, 3);
    assert_eq!(journey.arrives, 2160);
}

#[test]
fn an_automaton_that_accepts_nothing_answers_nothing() {
    let nowhere = Modes::new(1, 3).within(0, FOOT);
    assert!(nowhere.is_empty());
    assert_eq!(ask(&nowhere, 0, 0, 3), None);
}

#[test]
fn waiting_for_a_train_that_has_gone_is_not_riding_it() {
    // Leave late enough and the only train has left, so the language admits the
    // rail state and it buys nothing: the answer falls back to walking. This is
    // the time-dependent relaxation being time-dependent.
    assert_eq!(ask(&foot_and_rail(), 0, 1000, 3), Some(4000));
    assert_eq!(ask(&on_foot(), 0, 1000, 3), Some(4000));
}
