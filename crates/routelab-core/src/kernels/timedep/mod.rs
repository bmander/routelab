//! Routing when an edge is not always there.
//!
//! Every other kernel in this crate answers the same question: cheapest path
//! over fixed weights. This one answers a different one — *leaving at this time,
//! when do I arrive?* — because real networks are not always open. A drawbridge
//! lifts, a park path is locked overnight, a reversible lane runs the other way
//! before eleven.
//!
//! The scope is deliberately narrow and worth stating up front: **travel times
//! are constant; only availability varies.** That is the subclass OpenStreetMap
//! actually describes with its `:conditional` tags, and it is the subclass in
//! which Dijkstra still works unmodified (see [`time_dependent_dijkstra`]).
//! Networks whose travel times themselves vary with the clock — congestion
//! profiles, timetables — are a harder problem and a different algorithm.
//!
//! Nothing here changes [`crate::model::graph::Graph`]. The graph holds the same
//! weights it always did; the schedule lives beside it in a [`Calendar`] passed
//! per query. A technique may need extra data, but it should not make every
//! other technique carry it.

mod calendar;
mod search;

#[cfg(test)]
mod tests;

pub use calendar::{Calendar, Window};
pub use search::{time_dependent_dijkstra, walk};

/// A moment, in seconds since Monday 00:00.
pub type Clock = u32;

/// The length of the clock's cycle. See [`calendar`] for why a week.
pub const WEEK: Clock = 7 * 24 * 60 * 60;

/// When a journey starts, and what it may do when it meets a shut edge.
///
/// The two travel together because neither means anything without the other —
/// the same shape as [`crate::kernels::contraction::Ordering`], which carries a
/// contraction policy alongside the limits that policy is applied under. It also
/// keeps them from being two bare `u32`-ish positionals next to the sources.
///
/// ```
/// # use routelab_core::kernels::timedep::{Departure, Waiting};
/// let friday_rush = Departure::at(4 * 86_400 + 17 * 3_600);
/// assert_eq!(friday_rush.waiting(Waiting::Forbidden).waiting, Waiting::Forbidden);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Departure {
    pub clock: Clock,
    pub waiting: Waiting,
}

impl Departure {
    /// Leaving at `clock`, willing to wait.
    pub fn at(clock: Clock) -> Self {
        Departure {
            clock: clock % WEEK,
            waiting: Waiting::Unrestricted,
        }
    }

    /// The same departure under a different waiting policy.
    pub fn waiting(self, waiting: Waiting) -> Self {
        Departure { waiting, ..self }
    }
}

/// What you may do when you reach an edge that is shut.
///
/// Both are in the literature (Orda & Rom, *JACM* 37(3), 1990, treat waiting
/// policies explicitly), and they are a policy rather than an implementation
/// detail, so — like landmark selection or contraction order — you choose one
/// and the choice is comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waiting {
    /// Wait for it to open, and pay the wait as travel time.
    ///
    /// What [`Departure::at`] assumes, and self-correcting: waiting is costed,
    /// so a five-minute wait beats a long detour and a ten-hour one loses to it,
    /// without anyone having to say which. No policy switch decides that; the
    /// clock does.
    Unrestricted,
    /// Treat a shut edge as absent.
    ///
    /// The control. It answers a genuinely different question — "what if I
    /// cannot stop?" — and it is what shows that waiting is doing real work
    /// rather than decorating a result that would have happened anyway.
    Forbidden,
}
