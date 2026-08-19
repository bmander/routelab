//! UCCH's claim, checked: the same answers as the search it accelerates.
//!
//! It is a speedup for [`super::super::label_constrained`] and nothing else, so
//! the oracle is that search on the network that went in, uncontracted. Where
//! they differ the hierarchy has lost something — a shortcut that should have
//! stood in for a path, or a modal transfer the core forgot it could make.

use super::super::{label_constrained, LabelConstrainedTechnique, Modes, Multimodal};
use super::{Ucch, UcchInputs, UcchTechnique};
use crate::kernels::contraction::Ordering;
use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, NodeId, UNREACHABLE};
use crate::model::search::SearchOptions;
use crate::model::technique::{EarliestArrival, Footprint, Technique};
use crate::model::timetable::{Time, Timetable};
use crate::util::progress::Progress;

/// The pavements alone, the whole scalar network, its labels, the link arcs, and
/// the schedule.
type Corridor = (
    Graph,
    Graph,
    Vec<u8>,
    Vec<(NodeId, NodeId, Time)>,
    Timetable,
);

const FOOT: usize = 0;
const LINK: usize = 1;
const RAIL: usize = 2;

/// Ten street corners in a line, two stops beside the third and the eighth, and
/// one train between them.
///
/// The corners in between are degree two and belong to no link arc, so they are
/// exactly what the contraction is free to take — which is the point: if nothing
/// contracts, the hierarchy is the network and the tests prove nothing.
fn corridor() -> Corridor {
    let mut pavement = Vec::new();
    for corner in 0..9u32 {
        pavement.push((corner, corner + 1, 500));
        pavement.push((corner + 1, corner, 500));
    }
    let links = vec![(2u32, 10u32, 10u32), (10, 2, 10), (7, 11, 10), (11, 7, 10)];

    // The whole scalar network, foot and link together, which is what the
    // uncontracted search reads and what a walk is told in.
    let mut arcs = pavement.clone();
    let mut labels = vec![FOOT as u8; pavement.len()];
    for &(tail, head, weight) in &links {
        arcs.push((tail, head, weight));
        labels.push(LINK as u8);
    }

    let timetable = Timetable::new(12, [crate::kernels::oracles::c(0, 10, 11, 1200, 1300)]);
    (
        Graph::from_edges(12, &pavement).expect("real corners"),
        Graph::from_edges(12, &arcs).expect("real corners"),
        labels,
        links,
        timetable,
    )
}

fn foot_and_rail() -> Modes {
    Modes::new(2, 3)
        .within(0, FOOT)
        .within(1, RAIL)
        .on(0, LINK, 1)
        .on(1, LINK, 0)
        .starting(0)
        .accepting(0)
}

fn on_foot() -> Modes {
    Modes::new(1, 3)
        .within(0, FOOT)
        .within(0, LINK)
        .starting(0)
        .accepting(0)
}

/// Both searches over the same network, for one query.
fn both(
    modes: &Modes,
    from: NodeId,
    at: Time,
    to: NodeId,
    degree: f64,
) -> (Option<Time>, Option<Time>) {
    let (pavement, scalar, labels, links, timetable) = corridor();
    let network = Multimodal {
        scalar: &scalar,
        labels: &labels,
        timetable: &timetable,
        riding: RAIL as u8,
    };
    let plain = label_constrained(&network, modes, &[(from, at)], to).map(|j| j.arrives);
    let ucch = Ucch::build(
        &pavement,
        FOOT as u8,
        &links,
        LINK as u8,
        &[10, 11],
        Ordering::default(),
        degree,
    )
    .expect("the pavements contract");
    let fast = ucch
        .earliest_arrival(&network, modes, &[(from, at)], to)
        .map(|j| j.arrives);
    (plain, fast)
}

#[test]
fn it_answers_as_the_search_it_accelerates() {
    // The load-bearing test. Every pair, every hour that matters, both
    // languages, and two degree bounds so the core is a different size each
    // time.
    for degree in [2.0, 1e9] {
        for modes in [on_foot(), foot_and_rail()] {
            for from in 0..12u32 {
                for to in 0..12u32 {
                    for at in [0, 900, 1250, 2000] {
                        let (plain, fast) = both(&modes, from, at, to, degree);
                        assert_eq!(fast, plain, "degree {degree}: {from} -> {to} at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn bound_through_the_trait_it_still_answers_as_the_search_it_accelerates() {
    // The same claim, asked the way a caller generic over techniques asks it:
    // configure, bind, and hold both planners to one trait.
    let (pavement, scalar, labels, links, timetable) = corridor();
    let network = Multimodal {
        scalar: &scalar,
        labels: &labels,
        timetable: &timetable,
        riding: RAIL as u8,
    };
    let progress = Progress::new();
    for modes in [on_foot(), foot_and_rail()] {
        let plain = LabelConstrainedTechnique {
            modes: modes.clone(),
        }
        .bind(network, &progress)
        .unwrap();
        let fast = UcchTechnique {
            modes,
            walking: FOOT as u8,
            link_label: LINK as u8,
            ordering: Ordering::default(),
            max_degree: 2.0,
        }
        .bind(
            UcchInputs {
                network,
                walkable: &pavement,
                links: &links,
                served: &[10, 11],
            },
            &progress,
        )
        .unwrap();
        assert_eq!(plain.searches().0, "states");
        assert!(
            fast.searches().1 < plain.searches().1,
            "the core is smaller"
        );
        for from in 0..12u32 {
            for to in 0..12u32 {
                for at in [0, 900, 1250, 2000] {
                    assert_eq!(
                        fast.earliest_arrival(&[(from, at)], to).map(|j| j.arrives),
                        plain.earliest_arrival(&[(from, at)], to).map(|j| j.arrives),
                        "{from} -> {to} at {at}"
                    );
                }
            }
        }
    }
}

#[test]
fn hello_world() {
    // Walk to the third corner, onto the platform, ride, and walk off: riding
    // beats the pavement, which is what makes the corridor worth crossing.
    let (plain, fast) = both(&foot_and_rail(), 0, 0, 9, 2.0);
    assert_eq!(fast, Some(2310));
    assert_eq!(plain, Some(2310));
    // And on foot it is ten corners at five hundred seconds each, less one.
    assert_eq!(both(&on_foot(), 0, 0, 9, 2.0).1, Some(4500));
}

#[test]
fn the_core_is_smaller_than_the_network() {
    // The whole point of the preprocessing, and worth a number: what a query
    // searches instead of the network. The transfer nodes survive whatever
    // their importance, which is the rule that makes it UCCH.
    let (pavement, _, _, links, _) = corridor();
    let built = Ucch::build(
        &pavement,
        FOOT as u8,
        &links,
        LINK as u8,
        &[10, 11],
        Ordering::default(),
        1e9,
    )
    .expect("contracts");
    assert!(built.num_core() < 12, "nothing was contracted");
    for &(tail, head, _) in &links {
        assert!(
            built.is_core(tail),
            "{tail} is a transfer node and was contracted"
        );
        assert!(
            built.is_core(head),
            "{head} is a transfer node and was contracted"
        );
    }
}

#[test]
fn a_walk_is_told_as_the_arcs_it_was_made_of() {
    // A shortcut stands for a path, so a journey that walked one has to be told
    // hop by hop — otherwise a caller cannot draw it or say which layer it came
    // from. Every leg here is one arc of the uncontracted network.
    let (pavement, scalar, labels, links, timetable) = corridor();
    let network = Multimodal {
        scalar: &scalar,
        labels: &labels,
        timetable: &timetable,
        riding: RAIL as u8,
    };
    let built = Ucch::build(
        &pavement,
        FOOT as u8,
        &links,
        LINK as u8,
        &[10, 11],
        Ordering::default(),
        2.0,
    )
    .expect("contracts");
    let journey = built
        .earliest_arrival(&network, &foot_and_rail(), &[(0, 0)], 9)
        .expect("the corridor is crossed");
    let mut at = 0;
    for leg in &journey.legs {
        assert_eq!(leg.from(), at, "a leg starts where the last one ended");
        at = leg.to();
    }
    assert_eq!(at, 9);
    assert_eq!(journey.legs.last().map(|leg| leg.to()), Some(9));
    assert_eq!(journey.arrives, 2310);
    // Two corners walked, a link, the ride, a link, then two corners: no leg
    // spans more than one arc of the input.
    for leg in &journey.legs {
        let (tail, head) = (leg.from(), leg.to());
        let direct = (0..scalar.num_edges() as u32).any(|edge| {
            scalar.head(edge) == head
                && (0..12u32).any(|node| node == tail && scalar.out_edges(node).contains(&edge))
        });
        assert!(
            direct || matches!(leg, crate::model::timetable::Leg::Ride(_)),
            "{tail} -> {head} is not an arc of the network"
        );
    }
}

#[test]
fn a_language_that_forbids_riding_is_dijkstra_on_the_pavements() {
    let (_, scalar, _, _, _) = corridor();
    let truth = dijkstra(&scalar, &[(0, 0)], &SearchOptions::default())
        .expect("a real source")
        .costs;
    for to in 0..12u32 {
        let (_, fast) = both(&on_foot(), 0, 0, to, 2.0);
        let expected = match truth[to as usize] {
            UNREACHABLE => None,
            cost => Some(cost),
        };
        assert_eq!(fast, expected, "on foot to {to}");
    }
}
