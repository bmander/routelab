//! When each edge may be travelled, and how long you would have to wait.
//!
//! The clock is **weekly**: seconds since Monday 00:00, wrapping at
//! [`WEEK`]. That is not a shortcut, it is the range of the data. Every
//! restriction this library reads from OpenStreetMap is either a daily window
//! (`07:00-21:00`) or a weekday-qualified one (`Mo-Fr 05:00-11:00`); nothing
//! expresses a date, a month or a public holiday, because schedules that do are
//! refused at parse time rather than approximated. A weekly clock represents
//! exactly what can be represented and nothing it cannot.
//!
//! The table is **sparse**, and the cost that matters is not what a restricted
//! edge pays but what every other edge pays to ask. On a city extract a couple
//! of hundred edges out of six hundred thousand carry a restriction, so a query
//! is a third of a million lookups that almost all miss — and a hash lookup to
//! prove a negative, measured, costs about a fifth of the whole search. Hence
//! the bitset in front of the map: one shift and a mask answers the common case,
//! and the map is only consulted once the bit says there is something to find.

use std::collections::HashMap;

use crate::model::graph::{EdgeId, Graph, Weight};

use super::{Clock, Waiting, WEEK};

/// A half-open interval `[start, end)` on the weekly clock.
///
/// `end <= start` means the window wraps midnight — or the week — which is how
/// `23:00-05:00` is stored. Wrapping is handled here, once, so that nothing
/// downstream has to think about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: Clock,
    pub end: Clock,
}

impl Window {
    /// A window from `start` to `end`, both in seconds on the weekly clock.
    pub fn new(start: Clock, end: Clock) -> Self {
        Window {
            start: start % WEEK,
            end: end % WEEK,
        }
    }

    /// The seven weekly windows a *daily* time range expands to.
    ///
    /// A window is one interval on the weekly clock, so "every day 07:00-21:00"
    /// is seven of them rather than a repeating rule. Expanding here keeps
    /// [`Window`] a single dumb concept — an interval — instead of growing a
    /// recurrence grammar that the search would then have to interpret on every
    /// relaxation. Seven intervals per edge is nothing; a rule engine in the
    /// hot loop is not.
    ///
    /// `from` and `to` are seconds within a day. `to <= from` wraps past
    /// midnight into the following day, which is how `23:00-05:00` is written,
    /// and `to == 24:00` is the spelling for all day.
    pub fn daily(from: Clock, to: Clock) -> Vec<Window> {
        Window::on_days(0..7, from, to)
    }

    /// The same, restricted to certain weekdays — 0 is Monday, 6 is Sunday.
    pub fn on_days(days: impl IntoIterator<Item = u32>, from: Clock, to: Clock) -> Vec<Window> {
        const DAY: Clock = 24 * 60 * 60;
        let length = if to > from {
            to - from
        } else {
            to + DAY - from
        };
        days.into_iter()
            .map(|day| {
                let start = day * DAY + from;
                Window::new(start, start + length)
            })
            .collect()
    }

    /// Is `at` inside this window? `at` must already be reduced modulo [`WEEK`].
    pub fn contains(&self, at: Clock) -> bool {
        if self.end <= self.start {
            at >= self.start || at < self.end // wraps the end of the week
        } else {
            at >= self.start && at < self.end
        }
    }

    /// Seconds from `at` until this window next opens.
    ///
    /// Only meaningful when it is not open now, which every caller has already
    /// established — so there is no "already open" case to handle here.
    fn opens_in(&self, at: Clock) -> Weight {
        (self.start + WEEK - at) % WEEK
    }
}

/// When each edge may be travelled. Edges with no entry are always open.
#[derive(Debug, Clone, Default)]
pub struct Calendar {
    entries: HashMap<EdgeId, Vec<Window>>,
    /// One bit per edge id: does `entries` hold anything for it? Consulted
    /// before the map on every relaxation, which is the only reason the map's
    /// cost stays off the hot path. See the module docstring.
    restricted: Vec<u64>,
}

impl Calendar {
    /// A calendar in which nothing is ever shut.
    ///
    /// Not a special case in the search — it is the ordinary empty table, which
    /// is what makes "time-dependent Dijkstra with an empty calendar equals
    /// plain Dijkstra" a real test of the kernel rather than of a bypass.
    pub fn unrestricted() -> Self {
        Calendar::default()
    }

    /// Build from `(edge, windows)` pairs, **keyed by graph edge id**.
    ///
    /// Windows for an edge named more than once are pooled, not replaced: one
    /// OpenStreetMap way splits into many edges and may carry more than one
    /// restriction, so a caller accumulating them should not have to group
    /// first. An edge named with no windows at all is **never** open, which is
    /// how a permanently shut way is expressed.
    ///
    /// Edge ids are CSR positions, which are not the order the edges were
    /// handed to [`Graph::from_edges`]. A loader holding its own edge order
    /// wants [`Calendar::from_input_windows`] instead.
    pub fn from_windows(entries: impl IntoIterator<Item = (EdgeId, Vec<Window>)>) -> Self {
        let mut calendar = Calendar::default();
        for (edge, windows) in entries {
            calendar.entries.entry(edge).or_default().extend(windows);
            let word = edge as usize / 64;
            if word >= calendar.restricted.len() {
                calendar.restricted.resize(word + 1, 0);
            }
            calendar.restricted[word] |= 1 << (edge % 64);
        }
        calendar
    }

    /// Build from windows keyed by **position in the edge list `graph` was built
    /// from**, rather than by graph edge id.
    ///
    /// The distinction is worth a constructor of its own because getting it
    /// wrong is silent: [`Graph::from_edges`] permutes edges into CSR order, so
    /// a calendar keyed by input position and read as if it were keyed by edge
    /// id shuts the wrong streets and every test still passes. Anything that
    /// produced the edge list — an OSM reader, a layer — is holding input
    /// positions, so this is the constructor it wants.
    pub fn from_input_windows(
        graph: &Graph,
        entries: impl IntoIterator<Item = (u32, Vec<Window>)>,
    ) -> Self {
        let to_edge = graph.edges_by_input();
        Calendar::from_windows(
            entries
                .into_iter()
                .filter(|(input, _)| (*input as usize) < to_edge.len())
                .map(|(input, windows)| (to_edge[input as usize], windows)),
        )
    }

    /// How many edges carry a restriction — the honest denominator when
    /// reporting what a schedule actually changed.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        let windows: usize = self.entries.values().map(Vec::len).sum();
        self.entries.len() * (std::mem::size_of::<EdgeId>() + std::mem::size_of::<Vec<Window>>())
            + windows * std::mem::size_of::<Window>()
            + self.restricted.len() * std::mem::size_of::<u64>()
    }

    /// The windows during which `edge` is open, empty if it never is.
    #[inline]
    fn windows_for(&self, edge: EdgeId) -> Option<&[Window]> {
        let word = edge as usize / 64;
        if word >= self.restricted.len() || self.restricted[word] & (1 << (edge % 64)) == 0 {
            return None; // the overwhelmingly common case, and the cheap one
        }
        self.entries.get(&edge).map(Vec::as_slice)
    }

    /// Does `edge` carry a schedule at all?
    ///
    /// Distinct from [`Calendar::is_open`], and the distinction is the one that
    /// answers "why did the hour make no difference": an edge nobody scheduled
    /// is open at every hour, which looks exactly like an edge that happens to
    /// be open now.
    pub fn is_restricted(&self, edge: EdgeId) -> bool {
        self.windows_for(edge).is_some()
    }

    /// Is `edge` open at weekly time `at`?
    pub fn is_open(&self, edge: EdgeId, at: Clock) -> bool {
        self.wait(edge, at, Waiting::Forbidden) == Some(0)
    }

    /// How long you must wait at `at` before `edge` can be entered.
    ///
    /// `Some(0)` when it is open already. `None` when it cannot be entered at
    /// all — either it is shut and waiting is forbidden, or it is shut and never
    /// opens again within the week, which on a weekly clock means never.
    pub fn wait(&self, edge: EdgeId, at: Clock, policy: Waiting) -> Option<Weight> {
        let Some(windows) = self.windows_for(edge) else {
            return Some(0); // unrestricted
        };
        let at = at % WEEK;
        if windows.iter().any(|window| window.contains(at)) {
            return Some(0);
        }
        match policy {
            Waiting::Forbidden => None,
            // The soonest any of its windows opens. A week is the whole domain,
            // so a window that never opens within one never opens — and an edge
            // with no windows at all has no opening, which `min` reports as
            // `None` without needing a special case.
            Waiting::Unrestricted => windows.iter().map(|window| window.opens_in(at)).min(),
        }
    }
}
