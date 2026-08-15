//! Estimates of the cost remaining to a target, for goal-directed search.
//!
//! A heuristic is the one thing A* needs that the graph cannot supply. The graph
//! knows edges and weights; it does not know that two nodes are 400 metres apart,
//! or that nothing in this network moves faster than 25 m/s. That knowledge comes
//! from outside — in routelab, from the layers of an
//! [`Environment`](../../routelab/environment.py) — and arrives here as a compact
//! object the search can query in constant time.

use std::fmt;

use crate::graph::{NodeId, Weight, UNREACHABLE};

/// A lower bound on the cost of getting from `node` to `target`.
///
/// The bound must be **admissible**: never greater than the true remaining cost.
/// An overestimate makes A* faster and wrong — it will happily return a path that
/// is not the cheapest, with nothing in the result to say so.
///
/// It should also be **consistent** (`h(a) <= w(a, b) + h(b)` for every edge),
/// which is what lets the search settle each node once and never revisit it. Every
/// heuristic here is consistent; a custom one that is merely admissible needs
/// re-expansion that [`crate::astar`] does not do.
pub trait Heuristic {
    /// The estimated cost from `node` to `target`.
    fn estimate(&self, node: NodeId, target: NodeId) -> Weight;

    /// How many nodes this heuristic holds data for, if it is node-indexed.
    ///
    /// Lets a search reject a heuristic built for a different graph up front,
    /// rather than reading past the end of its tables mid-query.
    fn coverage(&self) -> Option<usize> {
        None
    }

    /// Bytes of precomputed table this heuristic holds.
    ///
    /// Zero for one that computes rather than remembers. What a caller weighs
    /// against how much the bound sharpens.
    fn footprint(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HeuristicError {
    /// Coordinate arrays of differing lengths.
    MismatchedCoordinates { xs: usize, ys: usize },
    /// A scale factor that is negative, infinite, or NaN.
    InvalidCostPerDistance(f64),
}

impl fmt::Display for HeuristicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicError::MismatchedCoordinates { xs, ys } => write!(
                f,
                "coordinate arrays must be the same length, got {xs} x and {ys} y"
            ),
            HeuristicError::InvalidCostPerDistance(value) => write!(
                f,
                "cost_per_distance must be finite and non-negative, got {value}"
            ),
        }
    }
}

impl std::error::Error for HeuristicError {}

/// The heuristics routelab ships, in one type so the Python bindings can hand a
/// search whichever the caller asked for without a trait object.
#[derive(Debug, Clone)]
pub enum StandardHeuristic {
    /// Estimates nothing. A* with this is Dijkstra — the control case a
    /// benchmark needs, and the thing every other heuristic must beat.
    Zero,
    /// Straight-line distance priced at the fastest rate anything in the network
    /// travels: `distance * cost_per_distance`.
    ///
    /// Admissible exactly because of that rate. `cost_per_distance` is the *least*
    /// cost any layer charges to cover one unit of distance, so no real path can
    /// undercut it. Price it at the walking rate in a network that also has
    /// trains, and the estimate becomes an overestimate for every trip that takes
    /// the train.
    Euclidean {
        xs: Vec<f64>,
        ys: Vec<f64>,
        cost_per_distance: f64,
    },
    /// Distances measured from a handful of fixed nodes, combined by the
    /// triangle inequality. Needs no geometry — only the graph and the time to
    /// walk it. See [`crate::landmark`].
    Landmarks(crate::landmark::Landmarks),
}

impl StandardHeuristic {
    /// Build a Euclidean heuristic from per-node coordinates.
    pub fn euclidean(
        xs: Vec<f64>,
        ys: Vec<f64>,
        cost_per_distance: f64,
    ) -> Result<Self, HeuristicError> {
        if xs.len() != ys.len() {
            return Err(HeuristicError::MismatchedCoordinates {
                xs: xs.len(),
                ys: ys.len(),
            });
        }
        if !cost_per_distance.is_finite() || cost_per_distance < 0.0 {
            return Err(HeuristicError::InvalidCostPerDistance(cost_per_distance));
        }
        Ok(StandardHeuristic::Euclidean {
            xs,
            ys,
            cost_per_distance,
        })
    }
}

impl Heuristic for StandardHeuristic {
    #[inline]
    fn estimate(&self, node: NodeId, target: NodeId) -> Weight {
        match self {
            StandardHeuristic::Zero => 0,
            StandardHeuristic::Euclidean {
                xs,
                ys,
                cost_per_distance,
            } => {
                let dx = xs[node as usize] - xs[target as usize];
                let dy = ys[node as usize] - ys[target as usize];
                // Not `hypot`: its overflow-safe scaling costs branches and a
                // division on the hottest arithmetic in the search, and guards a
                // range no coordinate system reaches — `dx * dx` would need dx
                // above 1e154 to overflow.
                let estimate = (dx * dx + dy * dy).sqrt() * cost_per_distance;
                // Floor, never round: half a second of rounding up is enough to
                // make the bound inadmissible and the answer wrong. The cast
                // saturates, and UNREACHABLE is reserved, so clamp below it.
                (estimate.floor() as Weight).min(UNREACHABLE - 1)
            }
            StandardHeuristic::Landmarks(landmarks) => landmarks.estimate(node, target),
        }
    }

    fn coverage(&self) -> Option<usize> {
        match self {
            StandardHeuristic::Zero => None,
            StandardHeuristic::Euclidean { xs, .. } => Some(xs.len()),
            StandardHeuristic::Landmarks(landmarks) => landmarks.coverage(),
        }
    }

    fn footprint(&self) -> usize {
        match self {
            StandardHeuristic::Zero => 0,
            StandardHeuristic::Euclidean { xs, ys, .. } => {
                (xs.len() + ys.len()) * std::mem::size_of::<f64>()
            }
            StandardHeuristic::Landmarks(landmarks) => landmarks.footprint(),
        }
    }
}

impl fmt::Display for StandardHeuristic {
    /// How a heuristic describes itself, so that callers — including the Python
    /// bindings — do not each grow a match arm per variant.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StandardHeuristic::Zero => write!(f, "zero"),
            StandardHeuristic::Euclidean {
                xs,
                cost_per_distance,
                ..
            } => write!(
                f,
                "euclidean({} nodes, cost_per_distance={cost_per_distance})",
                xs.len()
            ),
            StandardHeuristic::Landmarks(landmarks) => write!(
                f,
                "landmarks({} over {} nodes)",
                landmarks.len(),
                landmarks.num_nodes()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclidean() -> StandardHeuristic {
        // Three nodes on a line at x = 0, 300, 1000, priced at 0.5 cost per unit.
        StandardHeuristic::euclidean(vec![0.0, 300.0, 1000.0], vec![0.0, 0.0, 0.0], 0.5).unwrap()
    }

    #[test]
    fn zero_estimates_nothing() {
        assert_eq!(StandardHeuristic::Zero.estimate(0, 7), 0);
        assert_eq!(StandardHeuristic::Zero.coverage(), None);
    }

    #[test]
    fn euclidean_prices_the_straight_line() {
        let h = euclidean();
        assert_eq!(h.estimate(0, 2), 500);
        assert_eq!(h.estimate(1, 2), 350);
        assert_eq!(h.estimate(2, 2), 0, "no distance left at the target");
        assert_eq!(h.coverage(), Some(3));
    }

    #[test]
    fn euclidean_rounds_down_to_stay_admissible() {
        // 1 unit apart at 0.75 per unit is 0.75, which must not become 1.
        let h = StandardHeuristic::euclidean(vec![0.0, 1.0], vec![0.0, 0.0], 0.75).unwrap();
        assert_eq!(h.estimate(0, 1), 0);
    }

    #[test]
    fn huge_distances_stay_below_the_unreachable_sentinel() {
        let h = StandardHeuristic::euclidean(vec![0.0, 1e30], vec![0.0, 0.0], 1e30).unwrap();
        assert_eq!(h.estimate(0, 1), UNREACHABLE - 1);
    }

    #[test]
    fn rejects_unusable_inputs() {
        assert_eq!(
            StandardHeuristic::euclidean(vec![0.0], vec![], 1.0).unwrap_err(),
            HeuristicError::MismatchedCoordinates { xs: 1, ys: 0 }
        );
        assert_eq!(
            StandardHeuristic::euclidean(vec![0.0], vec![0.0], -1.0).unwrap_err(),
            HeuristicError::InvalidCostPerDistance(-1.0)
        );
        assert!(StandardHeuristic::euclidean(vec![0.0], vec![0.0], f64::NAN).is_err());
    }
}
