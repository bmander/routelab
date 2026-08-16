//! How far along a long piece of preprocessing is.
//!
//! Preprocessing here is measured in seconds, not milliseconds — six for a
//! city's contraction hierarchy — and anything watching it wants to tell "still
//! working" from "hung". That is the whole ambition: a counter the work bumps
//! and anyone can read.
//!
//! **A counter, not a callback.** A callback would have to be handed in, kept
//! somewhere, and called under whatever locks the work happens to hold — and
//! across the Python boundary it would mean taking the GIL back mid-search,
//! which is exactly what preprocessing releases it to avoid. A [`Progress`] is
//! read from another thread entirely while the work runs on holding nothing:
//! two relaxed loads and, for the phase name, a mutex over a `&'static str`
//! that is written once per phase and never contended for long.
//!
//! ## Phases, because one number was a lie
//!
//! Building a contraction hierarchy is two pieces of work: contracting the
//! nodes, and then assembling the result into CSR graphs. Counting only the
//! first is what a first attempt does, and on a walking network it reported
//! 100% with thirty-four seconds still to run — a bar that sits at full for a
//! quarter of the job reads as *hung*, which is the exact thing it was added to
//! rule out. So a fraction is always **within a named phase**, and the name is
//! part of the report. "assembling 20%" is honest where "100%" was not.
//!
//! Reporting is opt-in and additive. Every preprocessing entry point that can
//! report has a `_reporting` twin; the plain one stays plain, and calls the twin
//! with a [`Progress`] nobody reads. Nothing that does not care about progress
//! grew a parameter.
//!
//! ## What can honestly report, and what cannot
//!
//! A fraction is only worth showing when the work has a monotone count of steps
//! against a total it knows in advance. Contraction does — a node is contracted
//! and never uncontracted — and so does building a landmark table, which is a
//! fixed number of searches. Reading a file through a parser that yields no
//! counts does not, and inventing a number for it would be worse than showing
//! none: a bar that lies about being nearly finished is a bar nobody believes
//! again. Work like that is simply never handed a counter, and a counter nobody
//! wrote to reports [`Progress::fraction`] as `None`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A step counter for work that takes long enough to watch.
///
/// Cloning shares the counter, which is the point: the work keeps one and
/// whoever is watching keeps another.
#[derive(Debug, Clone, Default)]
pub struct Progress {
    done: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    /// What is happening, as a literal. Behind a mutex rather than an atomic
    /// because it is written once per phase — three times in the longest build
    /// here — and a lock nobody contends for costs nothing to be correct.
    phase: Arc<Mutex<&'static str>>,
}

impl Progress {
    pub fn new() -> Self {
        Progress::default()
    }

    /// Begin a named phase of `total` steps, from zero.
    ///
    /// Each phase counts on its own. Work with two unequal halves reports the
    /// second as a fresh count under a new name rather than letting the first
    /// finish at 100% and then carry on — see this module's docstring for the
    /// bar that made the case.
    pub fn expect(&self, phase: &'static str, total: u64) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        if let Ok(mut held) = self.phase.lock() {
            *held = phase;
        }
    }

    /// One step done. A relaxed increment — the reader wants to know roughly
    /// where this has got to, and paying for ordering it does not need would
    /// show up in a loop that runs once per node.
    pub fn step(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// Steps finished, and how many there are. `total` is zero when the work
    /// never said, which is a counter nothing wrote to.
    pub fn read(&self) -> (u64, u64) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }

    /// What is happening, or `""` before anything says.
    pub fn phase(&self) -> &'static str {
        self.phase.lock().map(|held| *held).unwrap_or("")
    }

    /// How far along, from 0 to 1, or `None` when the work cannot say.
    pub fn fraction(&self) -> Option<f64> {
        let (done, total) = self.read();
        (total > 0).then(|| (done as f64 / total as f64).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contraction::{ContractionHierarchy, Ordering};
    use crate::graph::Graph;
    use crate::landmark::{Landmarks, Selection};

    #[test]
    fn a_counter_nobody_wrote_to_cannot_say() {
        // Which is what work with no honest measure of its own reports: the
        // plain entry points pass one of these and never touch it.
        assert_eq!(Progress::new().fraction(), None);
        assert_eq!(Progress::new().read(), (0, 0));
    }

    #[test]
    fn clones_share_one_counter() {
        // The whole point: the work holds one and the watcher holds another.
        let work = Progress::new();
        let watcher = work.clone();
        work.expect("counting", 4);
        work.step();
        work.step();
        assert_eq!(watcher.read(), (2, 4));
        assert_eq!(watcher.fraction(), Some(0.5));
        assert_eq!(watcher.phase(), "counting");
    }

    #[test]
    fn a_new_phase_starts_from_zero() {
        let progress = Progress::new();
        progress.expect("first", 2);
        progress.step();
        progress.step();
        assert_eq!(progress.fraction(), Some(1.0));
        progress.expect("second", 10);
        // Not 100% with work still to do — the whole reason phases exist.
        assert_eq!(progress.read(), (0, 10));
        assert_eq!(progress.phase(), "second");
    }

    #[test]
    fn contraction_finishes_at_one() {
        let graph =
            Graph::from_edges(5, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (0, 4, 9)]).unwrap();
        let progress = Progress::new();
        ContractionHierarchy::build_reporting(&graph, Ordering::default(), &progress).unwrap();
        // The last phase is the one still showing, and it is finished. A bar
        // that stops short of full is its own bug report; so is one that
        // reaches full while the work goes on.
        assert_eq!(progress.phase(), "assembling");
        assert_eq!(progress.fraction(), Some(1.0));
    }

    #[test]
    fn landmarks_finish_at_one() {
        let graph =
            Graph::from_edges(6, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]).unwrap();
        for selection in [Selection::Farthest, Selection::Random] {
            let progress = Progress::new();
            Landmarks::build_reporting(&graph, 3, selection, 7, &progress);
            assert_eq!(progress.fraction(), Some(1.0), "{selection:?}");
            assert_eq!(progress.phase(), "measuring");
        }
    }

    #[test]
    fn reporting_changes_nothing_about_the_answer() {
        // The plain entry point calls the reporting one with a counter nobody
        // reads, so this is really a check that it stayed that way.
        let graph =
            Graph::from_edges(6, &[(0, 1, 4), (1, 2, 3), (0, 2, 9), (2, 3, 2), (3, 4, 5)]).unwrap();
        let quiet = ContractionHierarchy::build(&graph, Ordering::default()).unwrap();
        let watched =
            ContractionHierarchy::build_reporting(&graph, Ordering::default(), &Progress::new())
                .unwrap();
        assert_eq!(quiet.num_shortcuts(), watched.num_shortcuts());
        assert_eq!(quiet.num_arcs(), watched.num_arcs());
        for node in 0..graph.num_nodes() as u32 {
            assert_eq!(quiet.rank(node), watched.rank(node));
        }
    }
}
