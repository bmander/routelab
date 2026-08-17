//! ULTRA: unlimited transfers, worked out once.
//!
//! Baum, Buchhold, Sauer, Wagner & Zündorf, *UnLimited TRAnsfers for
//! Multi-Modal Route Planning: An Efficient Solution* (ESA 2019). Every
//! timetable kernel here takes its walks as **one-hop transfers between
//! stops**, closed under composition — which is why they must be restricted,
//! by a radius or by a change-time bound, before the closure explodes. The
//! paper's own words for the state of the art this fixes: existing approaches
//! "only support limited walking between stops, either by imposing a maximum
//! transfer distance or by requiring the transfer graph to be transitively
//! closed". This library does both, and `Footpaths(feed, within=200)` is
//! exactly the first.
//!
//! ULTRA removes the restriction without changing the kernels. Its
//! observation is that where a walk sits in a journey decides how much it
//! matters: *initial* transfers (source to the first stop) and *final* ones
//! (last stop to the target) are common and can be searched at query time,
//! while *intermediate* transfers — between two vehicles — are rare enough
//! that the paths worth walking can be enumerated ahead of time. What comes
//! out is a small set of **shortcuts**, one per intermediate transfer that
//! some Pareto-optimal journey needs, and those are one-hop stop-to-stop
//! transfers like any other. A kernel reading them is unchanged and now walks
//! without limit.
//!
//! ## What it computes
//!
//! A **candidate** is a journey of exactly two trips whose initial and final
//! transfers are both empty: board at the source stop, ride, walk, ride, and
//! stop where the second vehicle does. Every shortcut a query could need is
//! the intermediate transfer of some candidate, because a longer journey
//! decomposes into two-trip subjourneys (the paper's Lemma 1). So the
//! preprocessing enumerates candidates and keeps the ones no **witness** —
//! any journey of at most two trips that is as good — beats.
//!
//! That enumeration is a two-round RAPTOR from each stop in turn, with the
//! transfer phase replaced by a Dijkstra search over the transfer graph — the
//! paper's *MR*, restricted to two rounds. Round 0 is the initial transfer, a
//! Dijkstra from the source; round 1 boards and rides and walks; round 2
//! boards and rides again. A stop still holding a ride label after round 2's
//! walk is a candidate, and its shortcut runs from where the first vehicle
//! was left to where the second was boarded. A witness that arrives no later
//! takes the label instead, and no shortcut is written — which is the whole
//! of the pruning.
//!
//! ## What is faithful, and what is not here
//!
//! The candidate and witness definitions, the two-round MR that enumerates
//! them, the restriction to departure times occurring **at** the source stop
//! (the paper's `DT`, which is what makes candidates the only journeys
//! explored with an empty initial transfer), one Dijkstra per source reused
//! across its runs, and the parent pointers that read an intermediate
//! transfer back out — that is Algorithm 1 as written, in its *stop-to-stop*
//! variant, the one RAPTOR and CSA take unchanged.
//!
//! The **canonical MR** of §3.1 is here, and is far less code than it is
//! paper. §3.1 spends its length defining a *tiebreaking sequence* — a total
//! order on equivalent journeys, read off the last leg backwards — but Lemma 2
//! then says two mechanical changes to MR suffice to find journeys in
//! increasing order of it, after which ordinary weak domination discards the
//! rest. Both are here: routes are scanned in route-index order (`ride` walks
//! `0..num_lines()`), and the transfer phase's priority queue is keyed on
//! `⟨arrival, vertex index⟩` rather than on arrival alone. That a journey
//! ending in a trip beats an equal one ending in a walk falls out of running
//! `ride` before the transfer phase within a round. No sequence is ever built.
//!
//! Both orderings are load-bearing and trivial to break by accident — a
//! `.rev()`, a plain-integer queue key — so
//! `canonical_mr_breaks_a_tie_by_route_index` and `..._by_vertex_index` pin
//! each on an instance whose shortcut changes when it is reversed.
//!
//! The **self-pruning** of §3.2 is here too: runs from one source go in
//! descending departure order and keep each other's labels, so a run that
//! leaves earlier and arrives no sooner stops instead of propagating. Worth
//! 2.5x to 2.9x, with the shortcut set unchanged. Its **repair** — the
//! three-part dominance rule of [`accepts`] that stops self-pruning from
//! discarding a canonical journey — is implemented as stated, and is live:
//! stub it out and a fixture's shortcuts drop from seven to four.
//!
//! What is *not* established is that the repair is **necessary**. Across
//! roughly 10,400 generated instances — 4,400 random, 6,000 corridors, the
//! shape built precisely so that a walk between vehicles is load-bearing —
//! 756 had a different shortcut set with it than without, and not one of those
//! answered any query wrongly without it. A guess at why, which is a guess and
//! not the paper's: Theorem 3 promises that every intermediate transfer of
//! every *canonical* journey is represented, and that is strictly stronger
//! than what a correct query needs, which is that *some* optimal journey is.
//! Dropping the repair may lose a canonical journey's shortcut while leaving
//! an equally good journey answerable by shortcuts that were kept. The repair
//! stays regardless: it is what the paper says, and a superset is the safe
//! direction.
//!
//! ## §3.3, optimization by optimization
//!
//! **Route collection.** Algorithm 1's line 9 is here: round 2 scans only the
//! routes serving what round 1 marked, in route-index order. Line 6 — round 1's
//! set, the routes boardable in this run's departure window, precomputed with
//! `DT` — is not. Round 1 still scans every route, which a profile puts at 12%
//! of the time, so that is the size of what is left on the table.
//!
//! **Limited Dijkstra searches.** Both stopping criteria are here, each
//! counting what its round can: see [`Until`]. The witness limit is here as
//! `WITNESS_LIMIT`, a choice rather than a value from the paper. What is *not*
//! here is the part that makes stopping free rather than merely safe — keeping
//! leftover labels queued across runs, in two separate queues, removing
//! dominated labels from them explicitly, and inheriting a new label's run from
//! its parent instead of the current run. Without it, stopping early loses
//! witnesses that would have pruned non-canonical candidates, so the cost is
//! superfluous shortcuts. Measured on the harness the criteria are worth about
//! 10%, which on a contended machine is inside the noise; the harness averages
//! under one candidate a run, so it likely understates them.
//!
//! **Pruning with found shortcuts.** Here, in [`Final`]: a candidate whose
//! intermediate transfer is already a shortcut this thread found is demoted to
//! a witness. Cuts candidates six-fold on the harness — 11,205 to 1,840 — for
//! the same 175 shortcuts, and costs nothing measurable either way. The paper's
//! own note that per-thread shortcuts are enough is what makes it safe beside
//! the parallel sweep. Not here: turning *queued* candidates into witnesses
//! when a shortcut is inserted, which needs the per-stop candidate lists.
//!
//! **Transfer graph contraction.** Core-CH is here, though a rung up — the
//! caller passes the core, see [`Ultra::compute`]. Contracting cliques of stops
//! a zero-length walk apart, so they make one run instead of several, is *not
//! applicable*: every walk this library can build is `max(1, ceil(...))`
//! seconds, so no two stops are ever zero apart and the clique is always a
//! single stop.
//!
//! **Parallelization.** Here, over blocks of source stops.
//!
//! Still out, and each its own increment: the **event-to-event** variant, which
//! computes shortcuts between stop events rather than stops and is what a
//! trip-based query wants, and the two pieces named above.

#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::model::graph::{Graph, NodeId, NO_NODE, UNREACHABLE};
use crate::model::lines::Lines;
use crate::model::timetable::{Footpaths, Time, Timetable};
use crate::util::progress::Progress;

/// What one round knows about reaching a vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Label {
    arrival: Time,
    /// Where the last leg began: the stop a vehicle was boarded at when
    /// [`Label::rode`], and the stop a walk started from when it did not.
    /// The paper's `p_k[v]`.
    parent: NodeId,
    /// Does the journey end aboard a vehicle? A walk away from the stop it
    /// alighted at makes a final transfer non-empty, and a candidate has none.
    rode: bool,
    /// Is the journey's *initial* transfer empty — did it board its first
    /// vehicle at the source stop rather than walking somewhere first? The
    /// paper's `⊥`, carried as a flag rather than a sentinel parent because
    /// the parent is wanted either way. A journey with this set is a *prefix
    /// of a candidate*, which is what the repaired dominance rule protects.
    direct: bool,
    /// Which run wrote this, so that a label carried over from a
    /// later-departing run can be told from one this run made. The paper's
    /// `run(v, i)`, and the third of the three conditions in §3.2.
    run: u32,
}

/// No run has written this label yet.
const NO_RUN: u32 = u32::MAX;

impl Label {
    const UNREACHED: Label = Label {
        arrival: UNREACHABLE,
        parent: NO_NODE,
        rode: false,
        direct: false,
        run: NO_RUN,
    };
}

/// Does `fresh` replace the label already at this vertex? §3.2's dominance
/// rule, with the repair.
///
/// Plain rRAPTOR discards a journey weakly dominated by what is already there,
/// and that is exactly what makes self-pruning work: a run at an earlier
/// departure that arrives no sooner than a later-departing run stops dead
/// instead of propagating. It is also what can throw a *canonical* journey
/// away, because keeping labels across runs implicitly maximises departure
/// time as a third criterion nobody asked for — Figure 2 of the paper is a
/// network where every Pareto-optimal journey contains a suboptimal
/// subjourney, so the problem cannot be defined away.
///
/// The repair: a weakly dominated journey still survives when all three of
/// the paper's discard conditions fail — it is a candidate prefix, it is not
/// *strongly* dominated, and the label holding it back was written by some
/// earlier run rather than this one.
fn accepts(now: Label, before: Label, fresh: Label, run: u32) -> bool {
    if fresh.arrival < now.arrival {
        // Not dominated at all: an outright improvement.
        return true;
    }
    // Weakly dominated, so rRAPTOR would drop it. Strong domination is
    // domination in both criteria — sooner in this round, or no later with one
    // trip fewer.
    let strongly = now.arrival < fresh.arrival || before.arrival <= fresh.arrival;
    fresh.direct && !strongly && now.run != run
}

/// A set of transfer shortcuts: the intermediate transfers a query can need,
/// and no others worth keeping.
///
/// Built once from a [`Timetable`] and an unrestricted transfer graph, and
/// read as one-hop transfers between stops — which is what every timetable
/// kernel here already takes.
#[derive(Debug, Clone, Default)]
pub struct Ultra {
    /// `(from, to, duration)`, deduplicated, each the shortest path between
    /// its two stops in the transfer graph.
    shortcuts: Vec<(NodeId, NodeId, Time)>,
    /// Candidates found before deduplication — the work the pruning did.
    candidates: usize,
}

impl Ultra {
    /// Work out the shortcuts of `timetable` over `transfers`.
    ///
    /// `transfers` is the transfer graph: any non-schedule-based mode, its
    /// weights durations rather than moments, its vertices including every
    /// stop the timetable serves. It is **not** required to be transitively
    /// closed or bounded in any way, which is the point.
    ///
    /// Algorithm 1 takes a *core* graph, and on a street network that is what
    /// to pass: [`CoreHierarchy`](crate::CoreHierarchy) contracts away every
    /// vertex no vehicle calls at and leaves the same distances between the
    /// ones it keeps, which is all an intermediate transfer runs between.
    pub fn compute(timetable: &Timetable, transfers: &Graph) -> Self {
        Self::compute_reporting(timetable, transfers, &Progress::new())
    }

    /// [`Ultra::compute`], counting source stops into `progress`.
    pub fn compute_reporting(
        timetable: &Timetable,
        transfers: &Graph,
        progress: &Progress,
    ) -> Self {
        let lines = Lines::from_timetable(timetable);
        let vertices = transfers.num_nodes().max(timetable.num_stops());

        // The stops worth starting from: those a vehicle actually calls at.
        // A vertex of the transfer graph that no trip serves can begin no
        // journey with an empty initial transfer, so it begins no candidate.
        let mut serves: Vec<bool> = vec![false; vertices];
        for c in timetable.connections() {
            serves[c.from as usize] = true;
            serves[c.to as usize] = true;
        }
        let stops: Vec<NodeId> = (0..vertices as NodeId)
            .filter(|&v| serves[v as usize])
            .collect();

        // Every source stop is judged on its own — its runs read the
        // timetable and the transfer graph and write nothing either of them
        // can see — so a thread per core takes blocks of stops until there
        // are none, the way the trip-based preprocessing does.
        progress.expect("finding shortcuts", stops.len() as u64);
        let sweep = Sweep {
            lines: &lines,
            transfers,
            serves: &serves,
            vertices,
        };
        let blocks = stops.len().div_ceil(BLOCK);
        let next = AtomicUsize::new(0);
        let done: Mutex<Vec<Option<Block>>> = Mutex::new((0..blocks).map(|_| None).collect());
        let threads = std::thread::available_parallelism()
            .map_or(1, |cores| cores.get())
            .min(blocks.max(1));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (next, done, sweep, stops) = (&next, &done, &sweep, &stops);
                scope.spawn(move || {
                    let mut worker = Worker::new(sweep);
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= blocks {
                            break;
                        }
                        let block = &stops[index * BLOCK..((index + 1) * BLOCK).min(stops.len())];
                        let made = worker.shortcuts_from(block, progress);
                        done.lock().expect("preprocessing lock")[index] = Some(made);
                    }
                });
            }
        });

        let mut found: Vec<(NodeId, NodeId, Time)> = Vec::new();
        let mut candidates = 0usize;
        for block in done.into_inner().expect("preprocessing lock") {
            let (made, counted) = block.expect("every block was claimed");
            found.extend(made);
            candidates += counted;
        }

        keep_shortest(&mut found);
        Ultra {
            shortcuts: found,
            candidates,
        }
    }

    /// The shortcuts, as `(from, to, duration)` — one-hop transfers between
    /// stops, which is what a timetable kernel reads.
    pub fn shortcuts(&self) -> &[(NodeId, NodeId, Time)] {
        &self.shortcuts
    }

    /// How many shortcuts were kept.
    pub fn len(&self) -> usize {
        self.shortcuts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shortcuts.is_empty()
    }

    /// Candidates found before duplicates were dropped — what the enumeration
    /// produced, against what it kept.
    pub fn candidates(&self) -> usize {
        self.candidates
    }

    /// The shortcuts as the one-hop transfer set a timetable kernel takes.
    ///
    /// `stops` is how many vertices the kernel will number, which is the
    /// transfer graph's count rather than the timetable's: a shortcut runs
    /// between stops, but they are numbered among the graph's vertices.
    pub fn footpaths(&self, stops: usize) -> Footpaths {
        Footpaths::new(stops, self.shortcuts.iter().copied())
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.shortcuts.len() * std::mem::size_of::<(NodeId, NodeId, Time)>()
    }
}

/// What one block of source stops yields: the shortcuts their candidates
/// produced, already reduced to one per stop pair, and how many candidates
/// that was before the reduction.
type Block = (Vec<(NodeId, NodeId, Time)>, usize);

/// One entry per stop pair, the shortest walk between them.
///
/// Applied to each block as it finishes and once more to the blocks merged,
/// which it may be because it is idempotent: the shortest of the per-block
/// shortest is the shortest. Doing it per block is not a tidiness — it is what
/// keeps the memory this phase holds proportional to the *shortcuts* rather
/// than to the *candidates*, and candidates outnumber shortcuts by a wide
/// margin. Every block's findings stay live until the sweep ends, so raw
/// candidates accumulating there is gigabytes on a city where the answer is
/// megabytes.
fn keep_shortest(found: &mut Vec<(NodeId, NodeId, Time)>) {
    // Lexicographic, so for a given pair the shortest duration sorts first and
    // is the one `dedup_by` keeps.
    found.sort_unstable();
    found.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
}

/// Source stops per block of preprocessing work. Small enough that a core
/// which draws a run of busy stops does not gate the phase — a stop on twenty
/// lines does far more work than one on a single loop — and large enough that
/// claiming a block costs nothing beside doing it.
const BLOCK: usize = 16;

/// What every source stop's runs read, shared across the threads.
struct Sweep<'a> {
    lines: &'a Lines,
    transfers: &'a Graph,
    serves: &'a [bool],
    vertices: usize,
}

/// One thread's scratch space: the three rounds of labels and a relaxation,
/// kept between blocks rather than reallocated per source.
struct Worker<'a> {
    sweep: &'a Sweep<'a>,
    walk: Relaxation,
    rounds: [Vec<Label>; 3],
    /// Every vertex written since the source changed, so the labels can be
    /// cleared without walking the numbering.
    touched: Dirty,
    /// What the current round improved: the transfer phase's frontier, and
    /// what the next round collects its routes from.
    marked: Vec<NodeId>,
    /// The routes this round scans, in route-index order — Algorithm 1's
    /// lines 6 and 9. Round 1 takes every route, since round 0 reached
    /// everything the source can walk to; round 2 takes only those serving a
    /// stop round 1 improved, which is most of what this saves.
    routes: Vec<u32>,
    /// Which routes are already in `routes`, so collecting them is linear.
    listed: Vec<bool>,
    /// The candidates the last round's riding found, for its transfer search
    /// to count down — §3.3's stopping criterion.
    awaited: Awaited,
}

/// How long the final transfer relaxation keeps going after the last candidate
/// has been settled — §3.3's witness limit, τ_wit.
///
/// Stopping the moment the candidates are all settled is correct but loses
/// witnesses that would have dominated non-canonical candidates in *later*
/// runs, and each one lost is a superfluous shortcut. The paper leaves the
/// limit as a tunable and does not fix a value; this one is a choice, measured
/// on the harness rather than taken from the paper.
const WITNESS_LIMIT: Time = 300;

/// The stops a round's riding improved with a candidate label — what §3.3's
/// stopping criterion counts down as the transfer search settles them.
///
/// Only the final round's search can use this so simply. There, a candidate is
/// written by the route scan and never by a walk, so the set is fixed before
/// the search starts and only shrinks. In round 1 a walk out of a candidate
/// prefix is itself a candidate prefix, so the set grows as the search runs and
/// the criterion needs the queue counted rather than a fixed list — which is
/// why only this half is here.
struct Awaited {
    flag: Vec<bool>,
    list: Vec<NodeId>,
    left: usize,
}

impl Awaited {
    fn new(vertices: usize) -> Self {
        Awaited {
            flag: vec![false; vertices],
            list: Vec::new(),
            left: 0,
        }
    }

    fn mark(&mut self, v: NodeId) {
        if !self.flag[v as usize] {
            self.flag[v as usize] = true;
            self.list.push(v);
            self.left += 1;
        }
    }

    fn settle(&mut self, v: NodeId) {
        if self.flag[v as usize] {
            self.flag[v as usize] = false;
            self.left -= 1;
        }
    }

    fn done(&self) -> bool {
        self.left == 0
    }

    fn reset(&mut self) {
        for &v in &self.list {
            self.flag[v as usize] = false;
        }
        self.list.clear();
        self.left = 0;
    }
}

/// What a round writes down as it goes: the vertices it touched, the ones it
/// marked for the phase after it, and which run is doing the writing.
struct Writing<'a> {
    touched: &'a mut Dirty,
    marked: &'a mut Vec<NodeId>,
    run: u32,
}

/// A shortcut set keyed on both its stops at once.
type Known = HashSet<u64, BuildHasherDefault<PairHash>>;

/// `(from, to)` as one key.
fn pair(from: NodeId, to: NodeId) -> u64 {
    (u64::from(from) << 32) | u64::from(to)
}

/// A multiplicative hash for those keys.
///
/// The standard hasher is SipHash, which is the right default for keys a
/// stranger chose and the wrong one for a lookup on this inner loop: measured,
/// it cost more than the pruning it enables saved, turning a 6x cut in
/// candidates into a net loss. These keys are two of our own node ids.
#[derive(Default)]
struct PairHash(u64);

impl Hasher for PairHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 ^= self.0 >> 29;
    }
}

/// What only the final round has business with: the candidates its transfer
/// search must settle before it may stop, and the shortcuts already found.
///
/// The second is §3.3's last pruning. A candidate whose intermediate transfer
/// is already a shortcut has nothing left to contribute, so it is demoted to a
/// witness — where it still prunes others, and no longer has to be waited for.
/// The set is this thread's own, which the paper says is enough: missing a
/// shortcut another thread found costs a duplicate, not an answer.
struct Final<'a> {
    awaited: &'a mut Awaited,
    known: &'a Known,
}

/// What a round's transfer search waits for before it is allowed to stop.
///
/// Two criteria because the two rounds count different things, which is how
/// §3.3 states it. In the last round a candidate is written by the route scan
/// and never by a walk, so the set is fixed before the search and only shrinks
/// — a flag per stop. In round 1 a walk out of a candidate prefix is itself a
/// candidate prefix, so the set grows as the search runs and what has to be
/// counted is how many prefix labels are still in the queue.
enum Until<'a> {
    /// Every candidate the route scan found has been settled.
    Candidates(&'a mut Awaited),
    /// No candidate prefix is left in the queue.
    Prefixes,
}

/// The vertices written since the source changed, each listed once.
///
/// The clear, the carry-forward and the extraction all walk this list, and all
/// three are idempotent — so pushing a vertex again on every improvement, which
/// under self-pruning means many times over a source's runs, is repetition
/// three loops then pay for. A seen flag keeps it to one entry a vertex, which
/// also bounds the list by the core rather than by the work done in it.
struct Dirty {
    seen: Vec<bool>,
    list: Vec<NodeId>,
}

impl Dirty {
    fn new(vertices: usize) -> Self {
        Dirty {
            seen: vec![false; vertices],
            list: Vec::new(),
        }
    }

    fn mark(&mut self, v: NodeId) {
        if !self.seen[v as usize] {
            self.seen[v as usize] = true;
            self.list.push(v);
        }
    }

    fn clear(&mut self) {
        for &v in &self.list {
            self.seen[v as usize] = false;
        }
        self.list.clear();
    }
}

/// The routes serving anything in `marked`, in route-index order.
///
/// Algorithm 1 line 9. The sort is not tidiness: canonical MR needs routes
/// scanned by index, and [`accepts`] resolves an equal arrival in favour of
/// whichever route reached it first.
fn routes_serving(lines: &Lines, marked: &[NodeId], listed: &mut [bool], out: &mut Vec<u32>) {
    out.clear();
    for &v in marked {
        for &(line, _) in lines.lines_at(v) {
            if !listed[line as usize] {
                listed[line as usize] = true;
                out.push(line);
            }
        }
    }
    for &line in out.iter() {
        listed[line as usize] = false;
    }
    out.sort_unstable();
}

impl<'a> Worker<'a> {
    fn new(sweep: &'a Sweep<'a>) -> Self {
        let vertices = sweep.vertices;
        Worker {
            sweep,
            walk: Relaxation::new(vertices),
            rounds: [
                vec![Label::UNREACHED; vertices],
                vec![Label::UNREACHED; vertices],
                vec![Label::UNREACHED; vertices],
            ],
            touched: Dirty::new(vertices),
            marked: Vec::new(),
            routes: Vec::new(),
            listed: vec![false; sweep.lines.num_lines()],
            awaited: Awaited::new(vertices),
        }
    }

    /// Every shortcut the candidates from `sources` produce — one per stop
    /// pair, since this is held until the whole sweep ends — and how many
    /// candidates that was.
    fn shortcuts_from(&mut self, sources: &[NodeId], progress: &Progress) -> Block {
        let Sweep {
            lines,
            transfers,
            serves,
            ..
        } = *self.sweep;
        let mut found = Vec::new();
        let mut candidates = 0usize;
        // Shortcuts this thread has already produced, so a candidate that
        // would only repeat one is demoted to a witness before it is waited on.
        let mut known: Known = Known::default();
        for &source in sources {
            // One Dijkstra per source, reused by every run from it: the
            // initial transfer does not depend on when you leave.
            self.walk.reach_of(transfers, source);
            let reach = std::mem::take(&mut self.walk.reached);

            // Self-pruning (§3.2): the labels are cleared once per source, not
            // once per run. Runs go in descending departure order, so a run
            // that leaves earlier and arrives no sooner than one that left
            // later is dominated by labels still standing from that later run,
            // and stops instead of propagating. Clearing here would be what
            // makes each run pay for the whole timetable again.
            for round in &mut self.rounds {
                for &v in &self.touched.list {
                    round[v as usize] = Label::UNREACHED;
                }
            }
            self.touched.clear();

            for (run, departure) in departures_at(lines, source).into_iter().enumerate() {
                let run = run as u32;

                // Round 0: the initial transfer. Only the source itself is
                // reached without walking, so only it can begin a candidate.
                // Departures descend, so every one of these improves on the
                // run before — it is the rounds above that self-prune.
                for &(v, duration) in &reach {
                    let fresh = Label {
                        arrival: departure.saturating_add(duration),
                        parent: NO_NODE,
                        rode: false,
                        direct: v == source,
                        run,
                    };
                    if accepts(self.rounds[0][v as usize], Label::UNREACHED, fresh, run) {
                        self.rounds[0][v as usize] = fresh;
                        self.touched.mark(v);
                    }
                }

                // Round 1 scans everything, because round 0 reached every stop
                // the source can walk to; round 2 scans only what round 1
                // touched.
                self.routes.clear();
                self.routes.extend(0..lines.num_lines() as u32);
                for round in 1..=2 {
                    // A round begins knowing whatever one trip fewer knew:
                    // "at most `k` trips" is what makes a witness with fewer
                    // trips prune a candidate with more. Improve rather than
                    // overwrite, because what stands here may be a better
                    // label from a later-departing run.
                    let (earlier, later) = self.rounds.split_at_mut(round);
                    for &v in &self.touched.list {
                        let carried = earlier[round - 1][v as usize];
                        if carried.arrival < later[0][v as usize].arrival {
                            later[0][v as usize] = carried;
                        }
                    }
                    self.marked.clear();
                    let last = round == 2;
                    let mut writing = Writing {
                        touched: &mut self.touched,
                        marked: &mut self.marked,
                        run,
                    };
                    let mut closing = last.then_some(Final {
                        awaited: &mut self.awaited,
                        known: &known,
                    });
                    ride(
                        lines,
                        &self.routes,
                        &earlier[round - 1],
                        &mut later[0],
                        &mut writing,
                        closing.as_mut(),
                    );
                    self.walk.relax(
                        transfers,
                        &earlier[round - 1],
                        &mut later[0],
                        &mut writing,
                        if last {
                            Until::Candidates(&mut self.awaited)
                        } else {
                            Until::Prefixes
                        },
                    );
                    if last {
                        self.awaited.reset();
                    }
                    if round == 1 {
                        routes_serving(lines, &self.marked, &mut self.listed, &mut self.routes);
                    }
                }

                // A stop still aboard after the last walk is a candidate: its
                // final transfer is empty, and nothing at most as long got
                // there sooner. Only what this run found — the labels outlive
                // the run now, and an older run's candidate was already taken.
                for &v in &self.touched.list {
                    let arrived = self.rounds[2][v as usize];
                    if arrived.run != run || !arrived.rode || !arrived.direct || !serves[v as usize]
                    {
                        continue;
                    }
                    let boarded = arrived.parent;
                    let before = self.rounds[1][boarded as usize];
                    // Boarding where the first vehicle was left is no
                    // transfer at all, and needs no shortcut.
                    if before.rode || before.parent == NO_NODE {
                        continue;
                    }
                    let left = before.parent;
                    candidates += 1;
                    let duration = before.arrival - self.rounds[1][left as usize].arrival;
                    found.push((left, boarded, duration));
                    known.insert(pair(left, boarded));
                }
            }
            self.walk.reached = reach;
            progress.step();
        }
        // Reduced here rather than only at the end: this vector outlives the
        // block, waiting on every other block to finish, and what it holds
        // raw is candidates.
        keep_shortest(&mut found);
        (found, candidates)
    }
}

/// Every distinct moment a vehicle leaves `stop`, latest first.
///
/// The paper's `DT`: a candidate boards its first vehicle at the source
/// without walking, so these are the only departures that can begin one, and
/// running from any other moment would find witnesses and no candidates.
fn departures_at(lines: &Lines, stop: NodeId) -> Vec<Time> {
    let mut times: Vec<Time> = Vec::new();
    for &(line, position) in lines.lines_at(stop) {
        if !lines.can_board(line, position) {
            continue;
        }
        for trip in lines.trips_of(line) {
            times.push(lines.time(trip, position).departure);
        }
    }
    times.sort_unstable_by(|a, b| b.cmp(a));
    times.dedup();
    times
}

/// One round of boarding and riding: RAPTOR's route scan, reading `before`
/// and improving `now`. Whatever it improves is `marked`, which is where the
/// transfer phase after it starts from.
fn ride(
    lines: &Lines,
    routes: &[u32],
    before: &[Label],
    now: &mut [Label],
    writing: &mut Writing<'_>,
    last: Option<&mut Final<'_>>,
) {
    let run = writing.run;
    let mut last = last;
    for &line in routes {
        let stops = lines.stops_of(line);
        let mut aboard: Option<u32> = None;
        let mut boarded_at = 0u32;
        for i in 0..stops.len() as u32 {
            let stop = stops[i as usize];
            if let Some(trip) = aboard {
                let arrival = lines.time(trip, i).arrival;
                let boarded = stops[boarded_at as usize];
                let mut fresh = Label {
                    arrival,
                    parent: boarded,
                    rode: true,
                    direct: before[boarded as usize].direct,
                    run,
                };
                // The shortcut this would produce, if it is a candidate: the
                // walk it took to reach where it boarded. Already found makes
                // it a witness instead — §3.3's last pruning.
                if fresh.direct {
                    if let Some(last) = last.as_deref() {
                        let prior = before[boarded as usize];
                        if !prior.rode
                            && prior.parent != NO_NODE
                            && last.known.contains(&pair(prior.parent, boarded))
                        {
                            fresh.direct = false;
                        }
                    }
                }
                if accepts(now[stop as usize], before[stop as usize], fresh, run) {
                    now[stop as usize] = fresh;
                    writing.touched.mark(stop);
                    writing.marked.push(stop);
                    // A candidate: still aboard, and boarded its first vehicle
                    // at the source. The transfer search after this has to
                    // settle it before it can stop.
                    if fresh.direct {
                        if let Some(last) = last.as_deref_mut() {
                            last.awaited.mark(stop);
                        }
                    }
                }
            }
            // Board, or step onto an earlier trip, on the strength of what
            // one trip fewer reached this stop with.
            let ready = before[stop as usize].arrival;
            if ready == UNREACHABLE || !lines.can_board(line, i) {
                continue;
            }
            let earlier = match aboard {
                None => true,
                Some(trip) => ready <= lines.time(trip, i).departure,
            };
            if earlier {
                if let Some(trip) = lines.earliest_trip(line, i, ready, aboard) {
                    if aboard != Some(trip) {
                        aboard = Some(trip);
                        boarded_at = i;
                    }
                }
            }
        }
    }
}

/// The transfer phase, as a Dijkstra over the transfer graph rather than a
/// hop along a closed set — which is the whole difference between MR and
/// RAPTOR, and the reason the walking need not be restricted.
struct Relaxation {
    queue: BinaryHeap<Reverse<(Time, NodeId, bool)>>,
    /// Reused between sources: `(vertex, duration)` for everything one source
    /// can walk to.
    reached: Vec<(NodeId, Time)>,
    distance: Vec<Time>,
}

impl Relaxation {
    fn new(vertices: usize) -> Self {
        Relaxation {
            queue: BinaryHeap::new(),
            reached: Vec::new(),
            distance: vec![UNREACHABLE; vertices],
        }
    }

    /// Everything `source` can walk to, and how long it takes, left in
    /// [`Relaxation::reached`] for the runs from that source to read.
    fn reach_of(&mut self, transfers: &Graph, source: NodeId) {
        for &(v, _) in &self.reached {
            self.distance[v as usize] = UNREACHABLE;
        }
        self.reached.clear();
        self.queue.clear();
        if (source as usize) < self.distance.len() {
            self.distance[source as usize] = 0;
            self.queue.push(Reverse((0, source, false)));
        }
        while let Some(Reverse((duration, v, _))) = self.queue.pop() {
            if duration > self.distance[v as usize] {
                continue;
            }
            self.reached.push((v, duration));
            for edge in transfers.out_edges(v) {
                let next = duration.saturating_add(transfers.weight(edge));
                let head = transfers.head(edge);
                if next < self.distance[head as usize] {
                    self.distance[head as usize] = next;
                    self.queue.push(Reverse((next, head, false)));
                }
            }
        }
    }

    /// Walk on from every vertex this round's riding improved, carrying the
    /// stop each walk began at so an intermediate transfer can be read back
    /// whole.
    ///
    /// Seeded from `marked` rather than from everything the source ever
    /// touched: under self-pruning the labels outlive the run, and a vertex
    /// nothing improved this round was already walked on from when it was.
    fn relax(
        &mut self,
        transfers: &Graph,
        before: &[Label],
        round: &mut [Label],
        writing: &mut Writing<'_>,
        until: Until<'_>,
    ) {
        let run = writing.run;
        // §3.3's stopping criterion. Once nothing left in the queue can prune a
        // candidate, the rest of the walk only builds witnesses for later runs
        // — worth `WITNESS_LIMIT` more and no further. Stopping cannot lose a
        // candidate, only a witness that would have pruned a non-canonical
        // one, so what it costs is superfluous shortcuts rather than answers.
        let mut deadline: Option<Time> = None;
        let mut until = until;
        // Queue entries carry whether their label was a candidate prefix when
        // pushed, which is what round 1's criterion counts. Third in the tuple
        // so the ordering stays ⟨arrival, vertex⟩ as canonical MR needs.
        let mut prefixes = 0usize;
        self.queue.clear();
        for &v in writing.marked.iter() {
            let label = round[v as usize];
            if label.arrival != UNREACHABLE {
                if label.direct {
                    prefixes += 1;
                }
                self.queue.push(Reverse((label.arrival, v, label.direct)));
            }
        }
        while let Some(Reverse((arrival, v, was_prefix))) = self.queue.pop() {
            if was_prefix {
                prefixes -= 1;
            }
            if arrival > round[v as usize].arrival {
                continue;
            }
            if let Some(limit) = deadline {
                if arrival > limit {
                    break;
                }
            }
            let spent = match &mut until {
                Until::Candidates(awaited) => {
                    awaited.settle(v);
                    awaited.done()
                }
                Until::Prefixes => prefixes == 0,
            };
            // Not latched. Round 1's criterion is not monotone — a walk out of
            // a candidate prefix is another candidate prefix, so the count can
            // fall to zero and rise again as the search runs. Committing to a
            // deadline the first time it empties stops the search before
            // prefixes it had not yet found, and that loses candidates rather
            // than witnesses.
            if spent {
                if deadline.is_none() {
                    deadline = Some(arrival.saturating_add(WITNESS_LIMIT));
                }
            } else {
                deadline = None;
            }
            let here = round[v as usize];
            // The walk's origin, not the step before it: a transfer is the
            // whole path from the stop a vehicle was left at.
            let began = if here.rode { v } else { here.parent };
            for edge in transfers.out_edges(v) {
                let next = arrival.saturating_add(transfers.weight(edge));
                let head = transfers.head(edge);
                let fresh = Label {
                    arrival: next,
                    parent: began,
                    rode: false,
                    direct: here.direct,
                    run,
                };
                if accepts(round[head as usize], before[head as usize], fresh, run) {
                    round[head as usize] = fresh;
                    writing.touched.mark(head);
                    writing.marked.push(head);
                    if fresh.direct {
                        prefixes += 1;
                    }
                    self.queue.push(Reverse((next, head, fresh.direct)));
                }
            }
        }
    }
}
