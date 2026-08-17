//! The labels: two hub sets per event, and the same folded per stop.
//!
//! A *reachability labeling* gives every vertex a forward label `L_f(v)` —
//! hubs reachable from `v` — and a backward label `L_b(v)` — hubs that reach
//! `v` — such that whenever `u` reaches `w`, some hub lies in both `L_f(u)`
//! and `L_b(w)` (the *cover property*). Then "is there a journey" is an
//! intersection of two sorted lists, and no search at all.
//!
//! ## Which labeling algorithm
//!
//! The paper computes its labels with RXL (Delling, Goldberg, Pajor & Werneck,
//! ESA 2014), a sampling-based greedy scheme it uses as a black box — "any
//! label generation scheme … creates two event labels for every vertex". This
//! module uses **pruned labeling** instead (Akiba, Iwata & Yoshida, SIGMOD 2013,
//! in the reachability form of Yano, Akiba, Iwata & Yoshida, CIKM 2013 — the
//! paper's own references [3] and [21]): take the vertices in some order,
//! most important first, and from each run a forward and a backward search
//! that adds the vertex as a hub to everything it reaches — but *prunes* any
//! vertex the labels built so far already cover. It is a page of code, it is
//! provably complete, and its labels are canonical for the order. What is
//! given up is RXL's smaller labels; what is kept is the paper's every query.
//!
//! Correctness, in one paragraph. For any `u` that reaches `w`, let `m` be the
//! earliest vertex in the order among all vertices on all `u`–`w` paths. When
//! `m` is processed, its forward search reaches `w` along some path, and no
//! vertex `x` on that path is pruned: pruning `x` needs an earlier hub `h`
//! with `m → h → x`, and then `h` would lie on a `u`–`w` path and precede
//! `m`. So `m` lands in `L_b(w)`, and symmetrically in `L_f(u)`.
//!
//! The order is **descending degree, ties broken at random** — the
//! pruned-labeling papers' own recipe, with a tie-break that turns out to
//! matter a great deal on this graph; see [`hub_order`]. It is a policy and
//! never a correctness choice, the way a contraction order is.
//!
//! ## Stop labels (§4)
//!
//! Event labels cover *all* pairs of events, most of them dominated: a hub
//! reachable from the 08:12 departure is reachable from the 08:00 one too, by
//! waiting. The paper folds each stop's event labels into one **stop label**
//! per direction — for each hub, only the *latest* departure at the stop that
//! reaches it, and only the *earliest* arrival at the stop it reaches — which
//! obeys a *tight journey cover property*: every journey that cannot be left
//! later or arrived earlier has a hub in `SL_f(s) ∩ SL_b(t)`. Those are what
//! a profile query sweeps.
//!
//! Beside every event-label entry sits the next vertex toward the hub, which
//! is how a hub match unpacks to a path and a path to a journey — the "known
//! path unpacking techniques" the paper's §5 points at.

use crate::model::graph::NodeId;
use crate::model::timetable::Time;
use crate::util::progress::Progress;
use crate::util::rng::Rng;

use super::events::EventGraph;

/// Event labels, CSR per event, each sorted by hub id — which is time order.
#[derive(Debug)]
pub(super) struct Labels {
    fwd_first: Vec<u32>,
    fwd_hub: Vec<u32>,
    /// The vertex after this one on a path to the hub (the hub itself for
    /// the entry a hub carries for itself).
    fwd_via: Vec<u32>,
    bwd_first: Vec<u32>,
    bwd_hub: Vec<u32>,
    /// The vertex before this one on a path from the hub.
    bwd_via: Vec<u32>,
}

impl Labels {
    /// Pruned labeling over `graph`, hubs in [`hub_order`].
    ///
    /// `progress` counts hubs processed, one phase named `"labeling"`.
    pub(super) fn build(graph: &EventGraph, progress: &Progress) -> Self {
        let events = graph.num_events();
        let order = hub_order(graph);

        let mut fwd: Vec<Vec<(u32, u32)>> = vec![Vec::new(); events];
        let mut bwd: Vec<Vec<(u32, u32)>> = vec![Vec::new(); events];
        // Two stamp arrays: which hubs the current hub's own label holds,
        // and which vertices this search has already seen. Stamps rather
        // than clearing, so a search costs what it visits and no more.
        // (Bitsets, cleared by walking what set them, were measured here:
        // 54 KB apiece against 1.7 MB, and worth 2% of the build on a city
        // — not worth four clear-loops a reader has to check mirror the
        // four that set them.)
        let mut marked = vec![0u32; events];
        let mut seen = vec![0u32; events];
        let mut stamp = 0u32;
        let mut queue: Vec<(u32, u32)> = Vec::new();

        progress.expect("labeling", events as u64);
        for &hub in &order {
            // Forward: everything `hub` reaches gains it in its backward label.
            stamp += 1;
            for &(h, _) in &fwd[hub as usize] {
                marked[h as usize] = stamp;
            }
            queue.clear();
            queue.push((hub, hub));
            seen[hub as usize] = stamp;
            let mut head = 0;
            while head < queue.len() {
                let (u, prev) = queue[head];
                head += 1;
                if u != hub
                    && bwd[u as usize]
                        .iter()
                        .any(|&(h, _)| marked[h as usize] == stamp)
                {
                    continue;
                }
                bwd[u as usize].push((hub, prev));
                for (w, _) in graph.out(u) {
                    if seen[w as usize] != stamp {
                        seen[w as usize] = stamp;
                        queue.push((w, u));
                    }
                }
            }

            // Backward: everything that reaches `hub` gains it in its forward label.
            stamp += 1;
            for &(h, _) in &bwd[hub as usize] {
                marked[h as usize] = stamp;
            }
            queue.clear();
            queue.push((hub, hub));
            seen[hub as usize] = stamp;
            let mut head = 0;
            while head < queue.len() {
                let (u, next) = queue[head];
                head += 1;
                if u != hub
                    && fwd[u as usize]
                        .iter()
                        .any(|&(h, _)| marked[h as usize] == stamp)
                {
                    continue;
                }
                fwd[u as usize].push((hub, next));
                for w in graph.into(u) {
                    if seen[w as usize] != stamp {
                        seen[w as usize] = stamp;
                        queue.push((w, u));
                    }
                }
            }
            progress.step();
        }

        let (fwd_first, fwd_hub, fwd_via) = pack(fwd);
        let (bwd_first, bwd_hub, bwd_via) = pack(bwd);
        Labels {
            fwd_first,
            fwd_hub,
            fwd_via,
            bwd_first,
            bwd_hub,
            bwd_via,
        }
    }

    /// `L_f(event)`: hubs reachable from it, sorted, with the next vertex
    /// toward each.
    pub(super) fn forward(&self, event: u32) -> (&[u32], &[u32]) {
        let (a, b) = (
            self.fwd_first[event as usize] as usize,
            self.fwd_first[event as usize + 1] as usize,
        );
        (&self.fwd_hub[a..b], &self.fwd_via[a..b])
    }

    /// `L_b(event)`: hubs that reach it, sorted, with the vertex before it
    /// on the way from each.
    pub(super) fn backward(&self, event: u32) -> (&[u32], &[u32]) {
        let (a, b) = (
            self.bwd_first[event as usize] as usize,
            self.bwd_first[event as usize + 1] as usize,
        );
        (&self.bwd_hub[a..b], &self.bwd_via[a..b])
    }

    /// The next vertex from `event` toward `hub`, if `hub` is in its forward
    /// label.
    pub(super) fn next_toward(&self, event: u32, hub: u32) -> Option<u32> {
        let (hubs, via) = self.forward(event);
        hubs.binary_search(&hub).ok().map(|i| via[i])
    }

    /// The vertex before `event` on the way from `hub`, if `hub` is in its
    /// backward label.
    pub(super) fn previous_from(&self, event: u32, hub: u32) -> Option<u32> {
        let (hubs, via) = self.backward(event);
        hubs.binary_search(&hub).ok().map(|i| via[i])
    }

    /// Entries held, both directions: the size of the labeling.
    pub(super) fn total_hubs(&self) -> usize {
        self.fwd_hub.len() + self.bwd_hub.len()
    }

    pub(super) fn footprint(&self) -> usize {
        std::mem::size_of::<u32>()
            * (self.fwd_first.len()
                + self.fwd_hub.len()
                + self.fwd_via.len()
                + self.bwd_first.len()
                + self.bwd_hub.len()
                + self.bwd_via.len())
    }
}

/// A hub in both of two sorted labels, and how many entries were read
/// finding it — the paper's "hubs" measure of a query's work.
pub(super) fn meet(forward: &[u32], backward: &[u32]) -> (Option<u32>, usize) {
    let (mut i, mut j) = (0, 0);
    while i < forward.len() && j < backward.len() {
        match forward[i].cmp(&backward[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return (Some(forward[i]), i + j + 2),
        }
    }
    (None, i + j)
}

/// Sort each vertex's entries by hub and lay them out CSR-style.
fn pack(labels: Vec<Vec<(u32, u32)>>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let entries: usize = labels.iter().map(Vec::len).sum();
    let mut first = Vec::with_capacity(labels.len() + 1);
    let mut hubs = Vec::with_capacity(entries);
    let mut via = Vec::with_capacity(entries);
    first.push(0u32);
    for mut label in labels {
        label.sort_unstable();
        for (h, v) in label {
            hubs.push(h);
            via.push(v);
        }
        first.push(hubs.len() as u32);
    }
    (first, hubs, via)
}

/// Stop labels (§4): per stop and per hub, the latest departure at the stop
/// that reaches the hub, and the earliest arrival at the stop from it.
/// Hubs and times in separate arrays, as the paper stores them, so a sweep
/// touches times only for hubs that match.
#[derive(Debug)]
pub(super) struct StopLabels {
    fwd_first: Vec<u32>,
    fwd_hub: Vec<u32>,
    fwd_time: Vec<Time>,
    bwd_first: Vec<u32>,
    bwd_hub: Vec<u32>,
    bwd_time: Vec<Time>,
}

impl StopLabels {
    /// Fold `labels` per stop. `progress` counts stops, one phase named
    /// `"merging"`.
    pub(super) fn build(graph: &EventGraph, labels: &Labels, progress: &Progress) -> Self {
        let stops = graph.num_stops();
        let events = graph.num_events();
        let mut seen = vec![0u32; events];
        let mut stamp = 0u32;
        let (mut fwd_first, mut fwd_hub, mut fwd_time) = (vec![0u32], Vec::new(), Vec::new());
        let (mut bwd_first, mut bwd_hub, mut bwd_time) = (vec![0u32], Vec::new(), Vec::new());
        let mut entries: Vec<(u32, Time)> = Vec::new();

        progress.expect("merging", stops as u64);
        for stop in 0..stops as NodeId {
            let list = graph.events_at(stop);

            // Forward: latest event first, so the first sighting of a hub is
            // the latest departure that reaches it.
            stamp += 1;
            entries.clear();
            for &event in list.iter().rev() {
                let time = graph.time(event);
                for &h in labels.forward(event).0 {
                    if seen[h as usize] != stamp {
                        seen[h as usize] = stamp;
                        entries.push((h, time));
                    }
                }
            }
            entries.sort_unstable();
            fwd_hub.extend(entries.iter().map(|&(h, _)| h));
            fwd_time.extend(entries.iter().map(|&(_, t)| t));
            fwd_first.push(fwd_hub.len() as u32);

            // Backward: earliest event first.
            stamp += 1;
            entries.clear();
            for &event in list {
                let time = graph.time(event);
                for &h in labels.backward(event).0 {
                    if seen[h as usize] != stamp {
                        seen[h as usize] = stamp;
                        entries.push((h, time));
                    }
                }
            }
            entries.sort_unstable();
            bwd_hub.extend(entries.iter().map(|&(h, _)| h));
            bwd_time.extend(entries.iter().map(|&(_, t)| t));
            bwd_first.push(bwd_hub.len() as u32);
            progress.step();
        }

        StopLabels {
            fwd_first,
            fwd_hub,
            fwd_time,
            bwd_first,
            bwd_hub,
            bwd_time,
        }
    }

    /// `SL_f(stop)`: hubs, and the latest departure at `stop` reaching each.
    pub(super) fn forward(&self, stop: NodeId) -> (&[u32], &[Time]) {
        let stop = stop as usize;
        if stop + 1 >= self.fwd_first.len() {
            return (&[], &[]);
        }
        let (a, b) = (
            self.fwd_first[stop] as usize,
            self.fwd_first[stop + 1] as usize,
        );
        (&self.fwd_hub[a..b], &self.fwd_time[a..b])
    }

    /// `SL_b(stop)`: hubs, and the earliest arrival at `stop` from each.
    pub(super) fn backward(&self, stop: NodeId) -> (&[u32], &[Time]) {
        let stop = stop as usize;
        if stop + 1 >= self.bwd_first.len() {
            return (&[], &[]);
        }
        let (a, b) = (
            self.bwd_first[stop] as usize,
            self.bwd_first[stop + 1] as usize,
        );
        (&self.bwd_hub[a..b], &self.bwd_time[a..b])
    }

    pub(super) fn total_hubs(&self) -> usize {
        self.fwd_hub.len() + self.bwd_hub.len()
    }

    pub(super) fn footprint(&self) -> usize {
        std::mem::size_of::<u32>()
            * (self.fwd_first.len()
                + self.fwd_hub.len()
                + self.fwd_time.len()
                + self.bwd_first.len()
                + self.bwd_hub.len()
                + self.bwd_time.len())
    }
}

/// The order hubs are taken in: descending degree, ties broken at random.
///
/// Degree is the pruned-labeling papers' recipe — an event with many arcs is
/// one at a busy stop, where journeys cross. The tie-break is the part that
/// matters here, and it is not cosmetic. Almost every event has the same
/// small degree, and breaking ties by id would take a stop's events in a
/// block, in time order; each of those, when it is labeled, finds nothing
/// yet processed between it and its whole future, and labels all of it. On a
/// two-hour window of a city that is 200 hubs per label; taking the ties at
/// random — so the early hubs are spread across the day and the network, and
/// something already stands between most pairs — is 20. Random tie-break is
/// a seeded shuffle, so a build is reproducible.
fn hub_order(graph: &EventGraph) -> Vec<u32> {
    let mut rng = Rng::new(0x9e37_79b9);
    let mut keyed: Vec<(std::cmp::Reverse<usize>, u64, u32)> = (0..graph.num_events() as u32)
        .map(|e| (std::cmp::Reverse(graph.degree(e)), rng.next(), e))
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, _, e)| e).collect()
}
