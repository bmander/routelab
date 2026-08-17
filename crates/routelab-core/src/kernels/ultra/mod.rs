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
//! Not here, and each its own increment: the **canonical** tiebreaking of
//! §3.1, which shrinks the set by choosing one journey among equals — without
//! it this keeps every intermediate transfer of every Pareto-optimal two-trip
//! journey, a superset of the paper's, and so sufficient but larger than it
//! needs to be. The **self-pruning** of §3.2, where runs at descending
//! departure times reuse each other's labels, and the repair that self-pruning
//! then needs. **Core-CH**, which contracts away every vertex that is not a
//! stop so the transfer relaxation runs on a graph of stops rather than of
//! streets — this runs plain Dijkstra instead, which is why a city-sized
//! street graph is out of reach here and a stop-to-stop transfer graph is not.
//! And the **event-to-event** variant, which computes shortcuts between stop
//! events rather than stops and is what a trip-based query wants.

#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
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
    /// the parent is wanted either way.
    direct: bool,
}

impl Label {
    const UNREACHED: Label = Label {
        arrival: UNREACHABLE,
        parent: NO_NODE,
        rode: false,
        direct: false,
    };
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
    touched: Vec<NodeId>,
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
            touched: Vec::new(),
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
        for &source in sources {
            // One Dijkstra per source, reused by every run from it: the
            // initial transfer does not depend on when you leave.
            self.walk.reach_of(transfers, source);
            let reach = std::mem::take(&mut self.walk.reached);

            for departure in departures_at(lines, source) {
                for round in &mut self.rounds {
                    for &v in &self.touched {
                        round[v as usize] = Label::UNREACHED;
                    }
                }
                self.touched.clear();

                // Round 0: the initial transfer. Only the source itself is
                // reached without walking, so only it can begin a candidate.
                for &(v, duration) in &reach {
                    self.rounds[0][v as usize] = Label {
                        arrival: departure.saturating_add(duration),
                        parent: NO_NODE,
                        rode: false,
                        direct: v == source,
                    };
                    self.touched.push(v);
                }

                for round in 1..=2 {
                    // A round begins knowing whatever one trip fewer knew:
                    // "at most `k` trips" is what makes a witness with fewer
                    // trips prune a candidate with more.
                    let (earlier, later) = self.rounds.split_at_mut(round);
                    // Only what this run has written. Everything else is
                    // UNREACHED in both, since the rounds were cleared over
                    // `touched` when the run began — and that clear is what
                    // keeps a run proportional to what it reached rather than
                    // to the numbering, which a whole-array copy here would
                    // undo. On a multimodal environment the numbering is every
                    // street corner and the run touches the core.
                    for &v in &self.touched {
                        later[0][v as usize] = earlier[round - 1][v as usize];
                    }
                    ride(lines, &earlier[round - 1], &mut later[0], &mut self.touched);
                    self.walk.relax(transfers, &mut later[0], &mut self.touched);
                }

                // A stop still aboard after the last walk is a candidate: its
                // final transfer is empty, and nothing at most as long got
                // there sooner.
                for &v in &self.touched {
                    let arrived = self.rounds[2][v as usize];
                    if !arrived.rode || !arrived.direct || !serves[v as usize] {
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
/// and improving `now`.
fn ride(lines: &Lines, before: &[Label], now: &mut [Label], touched: &mut Vec<NodeId>) {
    for line in 0..lines.num_lines() as u32 {
        let stops = lines.stops_of(line);
        let mut aboard: Option<u32> = None;
        let mut boarded_at = 0u32;
        for i in 0..stops.len() as u32 {
            let stop = stops[i as usize];
            if let Some(trip) = aboard {
                let arrival = lines.time(trip, i).arrival;
                if arrival < now[stop as usize].arrival {
                    now[stop as usize] = Label {
                        arrival,
                        parent: stops[boarded_at as usize],
                        rode: true,
                        direct: before[stops[boarded_at as usize] as usize].direct,
                    };
                    touched.push(stop);
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
    queue: BinaryHeap<Reverse<(Time, NodeId)>>,
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
            self.queue.push(Reverse((0, source)));
        }
        while let Some(Reverse((duration, v))) = self.queue.pop() {
            if duration > self.distance[v as usize] {
                continue;
            }
            self.reached.push((v, duration));
            for edge in transfers.out_edges(v) {
                let next = duration.saturating_add(transfers.weight(edge));
                let head = transfers.head(edge);
                if next < self.distance[head as usize] {
                    self.distance[head as usize] = next;
                    self.queue.push(Reverse((next, head)));
                }
            }
        }
    }

    /// Walk on from every vertex a round has reached, carrying the stop each
    /// walk began at so an intermediate transfer can be read back whole.
    fn relax(&mut self, transfers: &Graph, round: &mut [Label], touched: &mut Vec<NodeId>) {
        self.queue.clear();
        for &v in touched.iter() {
            let label = round[v as usize];
            if label.arrival != UNREACHABLE {
                self.queue.push(Reverse((label.arrival, v)));
            }
        }
        while let Some(Reverse((arrival, v))) = self.queue.pop() {
            if arrival > round[v as usize].arrival {
                continue;
            }
            let here = round[v as usize];
            // The walk's origin, not the step before it: a transfer is the
            // whole path from the stop a vehicle was left at.
            let began = if here.rode { v } else { here.parent };
            for edge in transfers.out_edges(v) {
                let next = arrival.saturating_add(transfers.weight(edge));
                let head = transfers.head(edge);
                if next < round[head as usize].arrival {
                    round[head as usize] = Label {
                        arrival: next,
                        parent: began,
                        rode: false,
                        direct: here.direct,
                    };
                    touched.push(head);
                    self.queue.push(Reverse((next, head)));
                }
            }
        }
    }
}
