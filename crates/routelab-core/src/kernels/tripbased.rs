//! Trip-based routing: trips, and the transfers between them.
//!
//! Witt, *Trip-Based Public Transit Routing* (ESA 2015). RAPTOR labels stops
//! and scans routes; CSA labels stops and scans connections. Witt's
//! observation is that once a rider is aboard a trip at a known stop, their
//! future is fully determined by the timetable — every stop the trip reaches,
//! and every other trip they could change onto — so it is **trips** that
//! should carry labels, and the changes between them can be worked out once,
//! ahead of any query, and stored as a **transfer set**.
//!
//! **Preprocessing** (§3.1). For every trip `t` and every stop `p_t^i` it
//! reaches, every line at that stop or a footpath away has one earliest trip
//! `u` a rider alighting there could catch; each is a *transfer*
//! `t@i → u@j` (Algorithm 1). Transfers within a line to a later trip are not
//! kept — staying in your seat is never worse — and neither are *U-turns*
//! (Algorithm 2), where `u`'s next stop is `t`'s previous one and the change
//! could have been made a stop earlier. Then a **reduction** (Algorithm 3)
//! walks each trip backwards keeping the earliest arrival it can achieve at
//! every stop, with and without each transfer, and drops the transfers that
//! improve nothing: reduction is exact, and it removes most of the set —
//! 84% of London's, and about the same share here.
//!
//! **Query** (§3.2). Given a source, target and departure, the trips reachable
//! from the source become *trip segments* — a trip and the stop it was boarded
//! at — in a queue for round 0. Round `n` scans each segment once: if the
//! trip reaches the target (or a stop that walks to it) it offers a journey
//! with `n` transfers; if the trip cannot beat the best arrival so far it is
//! pruned; otherwise every transfer out of every stop the segment covers
//! enqueues a new segment for round `n+1`. A per-trip label `R(t)` — the
//! first stop already reached on `t`, lowered for `t` and every later trip of
//! its line — is what keeps a trip from being scanned twice and what makes
//! the search a breadth-first sweep with no priority queue. The result is
//! the Pareto set over arrival time and number of transfers, fewest
//! transfers first, exactly as RAPTOR's rounds produce it.
//!
//! **Profile** (§3.3). The same loop, once per departure the source offers in
//! a window, latest first, keeping the labels: a trip reached at a stop after
//! `n` transfers by a later departure need not be scanned again by an earlier
//! one, since whatever it would find is dominated. Labels become `R_n(t)`, one
//! per transfer count, and pruning compares against the best arrival with at
//! most `n+1` transfers found so far. What comes out is the Pareto set over
//! (departure, arrival, transfers).
//!
//! ## What is faithful, and what is not here
//!
//! Lines as the paper defines them (a stop sequence whose trips never
//! overtake — [`Lines`]), Algorithms 1–4 as written, journeys read back off
//! a pointer per queue entry (§3.5), the profile of §3.3, and the query's
//! three prunings: same-line dominance in `R`, the target check before the
//! transfer scan, and the arrival bound. Changing vehicles is instantaneous:
//! the paper's `Δτ_ch(p)` is [`Transfer::minimum`], zero, so its two
//! reduction labels `τ_A`/`τ_C` coincide (its footnote 3) and only `τ_A` is
//! kept; a minimum change time is a new `Transfer` constructor, and this is
//! the second kernel here — after RAPTOR — that could honour one, since a
//! transfer knows both trips. Preprocessing is parallel over trips, as the
//! paper says it trivially is — a thread per core drawing blocks of trips
//! until there are none. Not here: the paper's SIMD and three-loop query
//! layout (§3.4) and its transfer preferences, each its own increment.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::model::graph::{NodeId, UNREACHABLE};
use crate::model::lines::Lines;
use crate::model::technique::{
    Distances, EarliestArrival, Explored, Footprint, Front, Profiled, Reads, Searches, Technique,
    TransitNetwork,
};
use crate::model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Time, Timetable, Transfer, TripId, Walk,
};
use crate::util::progress::Progress;

/// What a trip-based query takes, beyond where you start.
///
/// The target is not optional: the paper's query is point-to-point, and the
/// lines that reach the target are what every segment is checked against.
/// There is no one-to-all form, so there is no `Default` here — a query has
/// to be aimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TripBasedQuery {
    /// Where the query is aimed.
    pub target: NodeId,
    /// Stop after this many changes.
    pub max_transfers: Option<usize>,
    /// What an elapsed cost is measured from; `None` for the earliest source.
    pub departing: Option<Time>,
}

impl TripBasedQuery {
    /// A query aimed at `target`, with no cap and no fixed departure.
    pub fn to(target: NodeId) -> Self {
        TripBasedQuery {
            target,
            max_transfers: None,
            departing: None,
        }
    }
}

/// Trip-based routing as a configuration: whether to reduce transfers
/// (Algorithm 3), which is the paper's own control and on by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TripBasedTechnique {
    pub reduce: bool,
}

impl Default for TripBasedTechnique {
    fn default() -> Self {
        TripBasedTechnique { reduce: true }
    }
}

impl<'a> Technique<'a> for TripBasedTechnique {
    type Inputs = TransitNetwork<'a>;
    type Planner = TripBased;
    type Error = Infallible;

    fn bind(&self, net: TransitNetwork<'a>, progress: &Progress) -> Result<TripBased, Infallible> {
        Ok(TripBased::build_reporting(
            net.timetable,
            net.transfer,
            net.footpaths,
            self.reduce,
            progress,
        ))
    }
}

/// "None" in the arrays that index trips, positions and segments.
const NONE: u32 = u32::MAX;

/// A timetable as trips and the transfers between them.
///
/// Built once from a [`Timetable`] and a [`Footpaths`], the way
/// [`crate::kernels::raptor::Raptor`] is, and reused for every query. Building
/// is the expensive half here — seconds on a city — which is why it reports
/// progress.
#[derive(Debug)]
pub struct TripBased {
    lines: Lines,
    footpaths: Footpaths,
    /// The same footpaths reversed: a query asks which stops walk *to* the
    /// target, which is [`Footpaths::from`] on this.
    incoming: Footpaths,
    /// The last position of each trip — where boarding is useless and `R(t)`
    /// starts.
    last: Vec<u32>,
    /// Transfers out of each `(trip, position)`, CSR by [`Lines::slot`].
    transfers_start: Vec<u32>,
    /// `(trip, position)` to board.
    transfers: Vec<(u32, u32)>,
    /// How many transfers Algorithms 1 and 2 produced before reduction.
    initial_transfers: usize,
}

impl TripBased {
    /// Compute and reduce the transfer set of `timetable`, with `footpaths`
    /// between stops.
    ///
    /// The `transfer` is the paper's change time at every stop; only
    /// [`Transfer::instant`] exists today.
    pub fn build(timetable: &Timetable, transfer: Transfer, footpaths: &Footpaths) -> Self {
        Self::build_reporting(timetable, transfer, footpaths, true, &Progress::new())
    }

    /// [`TripBased::build`], counting trips into `progress` through both
    /// phases — and, with `reduce` false, keeping every transfer Algorithms 1
    /// and 2 produce. That is the paper's own control (its Table 3): the
    /// answers are identical, the queries slower.
    pub fn build_reporting(
        timetable: &Timetable,
        transfer: Transfer,
        footpaths: &Footpaths,
        reduce: bool,
        progress: &Progress,
    ) -> Self {
        let lines = Lines::from_timetable(timetable);
        let change = transfer.minimum();
        let trips = lines.num_trips();
        let last: Vec<u32> = (0..trips as u32).map(|t| lines.len(t) - 1).collect();

        let table = Precompute {
            lines: &lines,
            footpaths,
            change,
        };
        // Algorithms 1 and 2. Both phases answer in the same shape — the
        // transfers a block of trips makes, and where each of its slots ends
        // within them — so one assembly puts either back together.
        progress.expect("computing transfers", trips as u64);
        let (mut transfers, mut transfers_start) = stitch(
            lines.num_slots(),
            table.in_parallel(trips, |range| table.transfers(range, progress)),
        );
        let initial_transfers = transfers.len();

        // Algorithm 3: keep a transfer only if some stop is reached sooner
        // through it than the trip — or a transfer taken later along it —
        // already reaches it. Backwards along the trip, so "later" is known.
        if reduce {
            progress.expect("reducing transfers", trips as u64);
            let (kept, kept_start) = stitch(
                lines.num_slots(),
                table.in_parallel(trips, |range| {
                    table.reduce(range, &transfers, &transfers_start, progress)
                }),
            );
            transfers = kept;
            transfers_start = kept_start;
        }

        TripBased {
            lines,
            footpaths: footpaths.clone(),
            incoming: footpaths.reversed(),
            last,
            transfers_start,
            transfers,
            initial_transfers,
        }
    }

    pub fn num_stops(&self) -> usize {
        self.lines.num_stops()
    }

    /// Lines in the paper's sense — distinct stop sequences whose trips never
    /// overtake — which is more than a feed's own count of routes.
    pub fn num_lines(&self) -> usize {
        self.lines.num_lines()
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    pub fn num_trips(&self) -> usize {
        self.lines.num_trips()
    }

    /// Transfers kept: the set a query scans.
    pub fn num_transfers(&self) -> usize {
        self.transfers.len()
    }

    /// Transfers Algorithms 1 and 2 produced, before reduction — what
    /// [`TripBased::num_transfers`] would be with `reduce` false.
    pub fn num_initial_transfers(&self) -> usize {
        self.initial_transfers
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.lines.footprint()
            + self.footpaths.footprint()
            + self.incoming.footprint()
            + (self.last.len() + self.transfers_start.len()) * std::mem::size_of::<u32>()
            + self.transfers.len() * std::mem::size_of::<(u32, u32)>()
    }

    /// The transfers out of `trip` at `position`.
    fn transfers_from(&self, trip: u32, position: u32) -> &[(u32, u32)] {
        let slot = self.lines.slot(trip, position);
        &self.transfers
            [self.transfers_start[slot] as usize..self.transfers_start[slot + 1] as usize]
    }

    /// The paper's `ℒ`: every `(line, position, walk)` at the target or a
    /// footpath away from it, sorted by line so a segment can find its own.
    fn target_lines(&self, target: NodeId) -> Vec<(u32, u32, Time)> {
        let mut found: Vec<(u32, u32, Time)> = Vec::new();
        if (target as usize) >= self.num_stops() {
            return found;
        }
        found.extend(self.lines.lines_at(target).iter().map(|&(l, i)| (l, i, 0)));
        for (q, walk) in self.incoming.from(target) {
            found.extend(self.lines.lines_at(q).iter().map(|&(l, i)| (l, i, walk)));
        }
        found.sort_unstable();
        found
    }

    /// The entries of `ℒ` on `line`.
    fn on_line(target_lines: &[(u32, u32, Time)], line: u32) -> &[(u32, u32, Time)] {
        let start = target_lines.partition_point(|&(l, _, _)| l < line);
        let end = start + target_lines[start..].partition_point(|&(l, _, _)| l == line);
        &target_lines[start..end]
    }

    /// The trips a rider standing at `stop` at `at` can board first, each with
    /// where — one per `(line, position)` at `stop` or a footpath away — and,
    /// for a profile, only those the rider could still leave for by `until`:
    /// a journey is dated by the moment it leaves the stop, walk included,
    /// and one leaving after the window closes is not the window's.
    fn boardable(&self, stop: NodeId, at: Time, until: Time) -> Vec<(u32, u32)> {
        let mut found = Vec::new();
        if (stop as usize) >= self.num_stops() {
            return found;
        }
        let reachable = std::iter::once((stop, 0)).chain(self.footpaths.from(stop));
        for (q, walk) in reachable {
            let ready = at.saturating_add(walk);
            for &(l, i) in self.lines.lines_at(q) {
                if !self.lines.can_board(l, i) {
                    continue;
                }
                if let Some(t) = self.lines.earliest_trip(l, i, ready, None) {
                    if self.lines.time(t, i).departure.saturating_sub(walk) <= until {
                        found.push((t, i));
                    }
                }
            }
        }
        found
    }

    // --- Earliest arrival (§3.2) --------------------------------------------

    /// The Pareto front for `to`: one itinerary per number of changes that
    /// arrives strictly earlier than any journey with fewer, fewest first.
    pub fn pareto(&self, from: NodeId, at: Time, to: NodeId) -> Vec<Itinerary> {
        let search = self.search(&[(from, at)], &TripBasedQuery::to(to));
        self.itineraries(&search, to)
    }

    /// Run the query from `sources` — each a stop and the time you are
    /// standing there — toward the query's target, stopping after its
    /// `max_transfers` changes if given.
    ///
    /// The query is the paper's and needs its target: the lines that reach it
    /// are what a segment is checked against, and its best arrival is what
    /// prunes the rest. There is no one-to-all form.
    ///
    /// The query's `departing` is what an elapsed cost is measured from — the
    /// moment the question was asked, which is not the same as the earliest
    /// source when every source carries a head start. `None` means the
    /// earliest source.
    pub fn search(&self, sources: &[(NodeId, Time)], query: &TripBasedQuery) -> TripBasedSearch {
        let TripBasedQuery {
            target,
            max_transfers,
            departing,
        } = *query;
        let mut sweep = Sweep::new(self, false);
        let target_lines = self.target_lines(target);
        let mut earliest_source = UNREACHABLE;
        let mut found: Vec<Found> = Vec::new();

        // A journey that boards nothing: standing at the target, or walking
        // to it. What every arrival must then beat.
        let mut best = UNREACHABLE;
        for &(stop, at) in sources {
            if (stop as usize) >= self.num_stops() {
                continue;
            }
            earliest_source = earliest_source.min(at);
            let walked = if stop == target {
                Some(at)
            } else {
                self.footpaths
                    .duration(stop, target)
                    .map(|w| at.saturating_add(w))
            };
            if let Some(arrives) = walked {
                if arrives < best {
                    best = arrives;
                    found.clear();
                    found.push(Found {
                        arrives,
                        transfers: NONE,
                        segment: NONE,
                        alight: stop,
                    });
                }
            }
        }
        // Round 0: the trips a rider can board first.
        for &(stop, at) in sources {
            for (t, i) in self.boardable(stop, at, UNREACHABLE) {
                sweep.board(t, i, stop, at);
            }
        }

        // The rounds. Round `n` holds every segment reached with `n`
        // transfers; a cap ends the sweep after round `cap`.
        let mut begin = 0;
        let mut round = 0u32;
        while begin < sweep.segments.len() && !max_transfers.is_some_and(|cap| round as usize > cap)
        {
            let end = sweep.segments.len();
            let more = !max_transfers.is_some_and(|cap| round as usize >= cap);
            for index in begin..end {
                let seg = sweep.segments[index];
                sweep.scanned += 1;
                let line = self.lines.line_of(seg.trip);
                // Does this trip reach the target beyond where it was boarded?
                for &(_, i, walk) in Self::on_line(&target_lines, line) {
                    if i > seg.from {
                        let arrives = self.lines.time(seg.trip, i).arrival.saturating_add(walk);
                        if arrives < best {
                            best = arrives;
                            let hit = Found {
                                arrives,
                                transfers: round,
                                segment: index as u32,
                                alight: i,
                            };
                            match found.last_mut() {
                                Some(last) if last.transfers == round => *last = hit,
                                _ => found.push(hit),
                            }
                        }
                    }
                }
                // Prune: a trip that cannot beat the best arrival at its very
                // next stop leads nowhere worth going.
                if !more || self.lines.time(seg.trip, seg.from + 1).arrival >= best {
                    continue;
                }
                for i in seg.from + 1..=seg.to {
                    for &(u, j) in self.transfers_from(seg.trip, i) {
                        sweep.transfer(u, j, round, index as u32, i);
                    }
                }
            }
            begin = end;
            round += 1;
        }

        let settled = sweep.settled();
        TripBasedSearch {
            target,
            segments: sweep.segments,
            found,
            settled,
            scanned: sweep.scanned,
            departing: departing.unwrap_or(earliest_source),
        }
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    pub fn path(&self, search: &TripBasedSearch, stop: NodeId) -> Option<Vec<NodeId>> {
        Some(self.itinerary(search, stop)?.stops(stop))
    }

    /// The earliest arrival at `to` in `search`, however many changes it
    /// takes. Only the target the search was run toward has one.
    pub fn itinerary(&self, search: &TripBasedSearch, to: NodeId) -> Option<Itinerary> {
        if to != search.target {
            return None;
        }
        let found = *search.found.last()?;
        Some(self.rebuild(&search.segments, search.target, found, search.settled))
    }

    /// One itinerary per number of changes that arrives strictly earlier than
    /// any journey with fewer — the Pareto front, fewest changes first. Empty
    /// if `to` is not the target or was never reached.
    pub fn itineraries(&self, search: &TripBasedSearch, to: NodeId) -> Vec<Itinerary> {
        if to != search.target {
            return Vec::new();
        }
        search
            .found
            .iter()
            .map(|&found| self.rebuild(&search.segments, search.target, found, search.settled))
            .collect()
    }

    /// Follow the pointers behind `found` back to its source (§3.5): the
    /// segment that reached the target, the segment it was transferred from,
    /// and so on, one ride leg per hop and one walk per given link.
    fn rebuild(
        &self,
        segments: &[Segment],
        target: NodeId,
        found: Found,
        settled: usize,
    ) -> Itinerary {
        let mut legs: Vec<Leg> = Vec::new();
        if found.segment == NONE {
            // Walked, or already there.
            let source = found.alight;
            if source != target {
                let duration = self.footpaths.duration(source, target).unwrap_or(0);
                let walk = Walk {
                    from: source,
                    to: target,
                    departs: found.arrives - duration,
                    arrives: found.arrives,
                };
                legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
            }
            return Itinerary {
                arrives: found.arrives,
                legs,
                settled,
            };
        }
        // Built last leg first, then reversed.
        let mut index = found.segment;
        let mut alight = found.alight;
        let seg = segments[index as usize];
        let stops = self.lines.stops_of(self.lines.line_of(seg.trip));
        let off = stops[alight as usize];
        if off != target {
            let walk = Walk {
                from: off,
                to: target,
                departs: self.lines.time(seg.trip, alight).arrival,
                arrives: found.arrives,
            };
            legs.extend(self.footpaths.expand(walk).into_iter().rev().map(Leg::Walk));
        }
        loop {
            let seg = segments[index as usize];
            let stops = self.lines.stops_of(self.lines.line_of(seg.trip));
            for k in (seg.from..alight).rev() {
                legs.push(Leg::Ride(Connection {
                    trip: self.lines.trip_id(seg.trip),
                    from: stops[k as usize],
                    to: stops[k as usize + 1],
                    departs: self.lines.time(seg.trip, k).departure,
                    arrives: self.lines.time(seg.trip, k + 1).arrival,
                }));
            }
            // The walk onto this trip, from wherever the rider came: the stop
            // they alighted at, or the source they started from. Either way
            // it is a leg only if that stop is not this one.
            let boarding = stops[seg.from as usize];
            let (from, at) = match seg.boarded {
                Boarded::Source { origin, at } => (origin, at),
                Boarded::Transfer { segment, alighted } => {
                    let parent = segments[segment as usize];
                    let line = self.lines.line_of(parent.trip);
                    (
                        self.lines.stops_of(line)[alighted as usize],
                        self.lines.time(parent.trip, alighted).arrival,
                    )
                }
            };
            if from != boarding {
                let duration = self.footpaths.duration(from, boarding).unwrap_or(0);
                let walk = Walk {
                    from,
                    to: boarding,
                    departs: at,
                    arrives: at.saturating_add(duration),
                };
                legs.extend(self.footpaths.expand(walk).into_iter().rev().map(Leg::Walk));
            }
            match seg.boarded {
                Boarded::Source { .. } => break,
                Boarded::Transfer { segment, alighted } => {
                    index = segment;
                    alight = alighted;
                }
            }
        }
        legs.reverse();
        Itinerary {
            arrives: found.arrives,
            legs,
            settled,
        }
    }

    // --- Profile (§3.3) --------------------------------------------------------

    /// Every journey worth leaving `source` on for `target` between `from` and
    /// `until`: the Pareto set over departure, arrival and changes.
    ///
    /// The earliest-arrival loop, run once per moment a trip leaves the source
    /// (or a stop it walks to, less the walk), latest first, keeping one label
    /// per trip and transfer count across the runs. A departure is the latest
    /// moment you can leave and still make that journey. Journeys the direct
    /// walk from `source` to `target` would beat are left out, as a profile of
    /// departures cannot hold a walk that leaves whenever you do.
    pub fn profile(
        &self,
        source: NodeId,
        target: NodeId,
        from: Time,
        until: Time,
    ) -> TripBasedProfile {
        let mut profile = TripBasedProfile {
            source,
            target,
            segments: Vec::new(),
            found: Vec::new(),
            settled: 0,
            scanned: 0,
            runs: 0,
        };
        if (source as usize) >= self.num_stops()
            || (target as usize) >= self.num_stops()
            || until < from
        {
            return profile;
        }
        let target_lines = self.target_lines(target);
        let walk = if source == target {
            Some(0)
        } else {
            self.footpaths.duration(source, target)
        };

        // The moments a rider standing at the source can leave: a trip's
        // departure from the source, or from a stop it walks to, less the
        // walk. Latest first, each once.
        let mut departures: Vec<Time> = Vec::new();
        let reachable = std::iter::once((source, 0)).chain(self.footpaths.from(source));
        for (q, walked) in reachable {
            for &(l, i) in self.lines.lines_at(q) {
                if !self.lines.can_board(l, i) {
                    continue;
                }
                for t in self.lines.trips_of(l) {
                    let leaves = self.lines.time(t, i).departure;
                    if leaves >= walked && leaves - walked >= from && leaves - walked <= until {
                        departures.push(leaves - walked);
                    }
                }
            }
        }
        departures.sort_unstable_by(|a, b| b.cmp(a));
        departures.dedup();

        let mut sweep = Sweep::new(self, true);
        // `best[n]`: the earliest arrival with at most `n` changes found so
        // far — by this run or a later-leaving one, either of which
        // dominates whatever this run could add beyond it.
        let mut best: Vec<Time> = Vec::new();
        for &at in &departures {
            profile.runs += 1;
            let begun = sweep.segments.len();
            for (t, i) in self.boardable(source, at, until) {
                sweep.board(t, i, source, at);
            }
            let mut begin = begun;
            let mut round = 0u32;
            while begin < sweep.segments.len() {
                let end = sweep.segments.len();
                for index in begin..end {
                    let seg = sweep.segments[index];
                    sweep.scanned += 1;
                    let line = self.lines.line_of(seg.trip);
                    for &(_, i, walked) in Self::on_line(&target_lines, line) {
                        if i > seg.from {
                            let arrives =
                                self.lines.time(seg.trip, i).arrival.saturating_add(walked);
                            if arrives < best_within(&best, round) {
                                lower(&mut best, round, arrives);
                                let hit = ProfileFound {
                                    departs: at,
                                    found: Found {
                                        arrives,
                                        transfers: round,
                                        segment: index as u32,
                                        alight: i,
                                    },
                                };
                                match profile.found.last_mut() {
                                    Some(last)
                                        if last.departs == at && last.found.transfers == round =>
                                    {
                                        *last = hit
                                    }
                                    _ => profile.found.push(hit),
                                }
                            }
                        }
                    }
                    if self.lines.time(seg.trip, seg.from + 1).arrival
                        >= best_within(&best, round + 1)
                    {
                        continue;
                    }
                    for i in seg.from + 1..=seg.to {
                        for &(u, j) in self.transfers_from(seg.trip, i) {
                            sweep.transfer(u, j, round, index as u32, i);
                        }
                    }
                }
                begin = end;
                round += 1;
            }
        }
        // Earliest departure first, then fewest changes; and nothing the
        // walk beats.
        profile
            .found
            .retain(|f| walk.is_none_or(|w| f.found.arrives < f.departs.saturating_add(w)));
        profile
            .found
            .sort_by_key(|f| (f.departs, f.found.transfers));
        profile.settled = sweep.settled();
        profile.scanned = sweep.scanned;
        profile.segments = sweep.segments;
        profile
    }

    /// The direct walk from the profile's source to its target, if there is
    /// one: a journey with no departure time, which a profile cannot hold and
    /// its entries were measured against.
    pub fn walk(&self, profile: &TripBasedProfile) -> Option<Time> {
        if profile.source == profile.target {
            return None;
        }
        self.footpaths.duration(profile.source, profile.target)
    }

    /// The profile's Pareto set as `(departs, arrives, transfers)`, earliest
    /// departure first.
    pub fn triples(&self, profile: &TripBasedProfile) -> Vec<(Time, Time, usize)> {
        profile
            .found
            .iter()
            .map(|f| (f.departs, f.found.arrives, f.found.transfers as usize))
            .collect()
    }

    /// The journey behind every triple, with its departure, earliest first.
    pub fn journeys(&self, profile: &TripBasedProfile) -> Vec<(Time, Itinerary)> {
        profile
            .found
            .iter()
            .map(|f| {
                (
                    f.departs,
                    self.rebuild(&profile.segments, profile.target, f.found, profile.settled),
                )
            })
            .collect()
    }
}

/// Trips per block of preprocessing work. Small enough that a core which
/// draws a run of long, busy lines does not gate the phase — the work per
/// trip varies by orders of magnitude — and large enough that claiming one
/// costs nothing beside doing it.
const BLOCK: usize = 32;

/// What one block of preprocessing answers with: the transfers its trips make
/// or keep, and where each of their `(trip, position)` slots ends within them.
/// Both algorithms speak it, which is what lets one assembly serve both.
type Block = (Vec<(u32, u32)>, Vec<u32>);

/// Put the blocks back together into one CSR: their transfers appended in
/// trip order, and each slot's end offset by everything before it.
///
/// Both algorithms answer in this shape, so this is written once. Slots run
/// trip by trip and position by position, so appending in block order is
/// already the order [`Lines::slot`] numbers them in.
fn stitch(slots: usize, blocks: Vec<Block>) -> Block {
    let mut all: Vec<(u32, u32)> =
        Vec::with_capacity(blocks.iter().map(|(made, _)| made.len()).sum());
    let mut start = vec![0u32; slots + 1];
    let mut slot = 0;
    for (made, ends) in blocks {
        let base = all.len() as u32;
        all.extend_from_slice(&made);
        for end in ends {
            slot += 1;
            start[slot] = base + end;
        }
    }
    (all, start)
}

/// What Algorithms 1–3 read, shared across the threads that run them.
struct Precompute<'a> {
    lines: &'a Lines,
    footpaths: &'a Footpaths,
    /// The paper's `Δτ_ch`, one number for every stop.
    change: Time,
}

impl Precompute<'_> {
    /// Run `block` over every [`BLOCK`]-sized run of `trips`, a thread per
    /// core taking the next unclaimed block until there are none, and collect
    /// what they return in trip order.
    ///
    /// Blocks rather than one run per core because the work per trip is wildly
    /// uneven — a long line's trip makes thousands of transfers where a short
    /// one makes a handful — and a static split leaves every core waiting on
    /// whichever drew the busiest run. Claiming a block is one atomic add.
    fn in_parallel<T: Send>(
        &self,
        trips: usize,
        block: impl Fn(std::ops::Range<u32>) -> T + Sync,
    ) -> Vec<T> {
        let blocks = trips.div_ceil(BLOCK);
        let next = AtomicUsize::new(0);
        let done: Mutex<Vec<Option<T>>> = Mutex::new((0..blocks).map(|_| None).collect());
        let threads = std::thread::available_parallelism()
            .map_or(1, |cores| cores.get())
            .min(blocks.max(1));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (next, done, block) = (&next, &done, &block);
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= blocks {
                        break;
                    }
                    let range = (index * BLOCK) as u32..((index + 1) * BLOCK).min(trips) as u32;
                    let made = block(range);
                    done.lock().expect("preprocessing lock")[index] = Some(made);
                });
            }
        });
        done.into_inner()
            .expect("preprocessing lock")
            .into_iter()
            .map(|made| made.expect("every block was claimed"))
            .collect()
    }

    /// Algorithms 1 and 2 for the trips in `range`: every transfer they make,
    /// and where each `(trip, position)` slot ends within them.
    fn transfers(&self, range: std::ops::Range<u32>, progress: &Progress) -> Block {
        let lines = self.lines;
        let mut made: Vec<(u32, u32)> = Vec::new();
        let mut ends: Vec<u32> = Vec::new();
        for t in range {
            let line = lines.line_of(t);
            let stops = lines.stops_of(line);
            for i in 0..stops.len() as u32 {
                // Nobody alights where they boarded: position 0 has no
                // transfers out of it.
                if i > 0 {
                    let arrival = lines.time(t, i).arrival;
                    let here = stops[i as usize];
                    let reachable =
                        std::iter::once((here, self.change)).chain(self.footpaths.from(here));
                    for (q, walk) in reachable {
                        let ready = arrival.saturating_add(walk);
                        for &(l, j) in lines.lines_at(q) {
                            // ... and nobody boards a trip at its last stop.
                            if !lines.can_board(l, j) {
                                continue;
                            }
                            let onto = lines.stops_of(l);
                            let Some(u) = lines.earliest_trip(l, j, ready, None) else {
                                continue;
                            };
                            // A later trip of your own line is never worth
                            // changing onto: stay in your seat. Trips are
                            // numbered in line order, so "earlier" is `<`.
                            if l == line && u >= t && j >= i {
                                continue;
                            }
                            // Algorithm 2: a U-turn — `u` goes straight back
                            // to the stop `t` just left, and the change could
                            // have been made there — adds nothing.
                            if stops[i as usize - 1] == onto[j as usize + 1]
                                && lines.time(t, i - 1).arrival.saturating_add(self.change)
                                    <= lines.time(u, j + 1).departure
                            {
                                continue;
                            }
                            made.push((u, j));
                        }
                    }
                }
                ends.push(made.len() as u32);
            }
            progress.step();
        }
        (made, ends)
    }

    /// Algorithm 3 for the trips in `range`: the transfers worth keeping, in
    /// the same shape [`Precompute::transfers`] answers in.
    ///
    /// A trip is judged backwards, since what a transfer must improve on is
    /// what the stops after it already reach, and emitted forwards, since that
    /// is the order the slots are numbered in — so the verdicts are held per
    /// trip and read back out in order.
    fn reduce(
        &self,
        range: std::ops::Range<u32>,
        transfers: &[(u32, u32)],
        transfers_start: &[u32],
        progress: &Progress,
    ) -> Block {
        let lines = self.lines;
        let mut arrival_at = vec![UNREACHABLE; lines.num_stops()];
        let mut touched: Vec<NodeId> = Vec::new();
        let mut kept: Vec<(u32, u32)> = Vec::new();
        let mut ends: Vec<u32> = Vec::new();
        let mut keep: Vec<bool> = Vec::new();
        for t in range {
            let stops = lines.stops_of(lines.line_of(t));
            let first = transfers_start[lines.slot(t, 0)] as usize;
            let past = transfers_start[lines.slot(t, lines.len(t) - 1) + 1] as usize;
            keep.clear();
            keep.resize(past - first, false);
            {
                let mut relax = |stop: NodeId, at: Time| -> bool {
                    if at < arrival_at[stop as usize] {
                        if arrival_at[stop as usize] == UNREACHABLE {
                            touched.push(stop);
                        }
                        arrival_at[stop as usize] = at;
                        true
                    } else {
                        false
                    }
                };
                for i in (1..lines.len(t)).rev() {
                    let arrival = lines.time(t, i).arrival;
                    let here = stops[i as usize];
                    relax(here, arrival);
                    for (q, walk) in self.footpaths.from(here) {
                        relax(q, arrival.saturating_add(walk));
                    }
                    let slot = lines.slot(t, i);
                    for index in transfers_start[slot] as usize..transfers_start[slot + 1] as usize
                    {
                        let (u, j) = transfers[index];
                        let onto = lines.stops_of(lines.line_of(u));
                        let mut improves = false;
                        for k in j + 1..onto.len() as u32 {
                            let there = lines.time(u, k).arrival;
                            improves |= relax(onto[k as usize], there);
                            for (q, walk) in self.footpaths.from(onto[k as usize]) {
                                improves |= relax(q, there.saturating_add(walk));
                            }
                        }
                        keep[index - first] = improves;
                    }
                }
            }
            for stop in touched.drain(..) {
                arrival_at[stop as usize] = UNREACHABLE;
            }
            for i in 0..lines.len(t) {
                let slot = lines.slot(t, i);
                for index in transfers_start[slot] as usize..transfers_start[slot + 1] as usize {
                    if keep[index - first] {
                        kept.push(transfers[index]);
                    }
                }
                ends.push(kept.len() as u32);
            }
            progress.step();
        }
        (kept, ends)
    }
}

/// The earliest arrival with at most `transfers` changes found so far. Past
/// the end of the table it is the last entry: more changes never have to
/// arrive later, so the bound carries.
fn best_within(best: &[Time], transfers: u32) -> Time {
    best.get(transfers as usize)
        .or(best.last())
        .copied()
        .unwrap_or(UNREACHABLE)
}

/// Record an arrival with `transfers` changes: a bound on every count from
/// there up, since more changes never have to arrive later.
fn lower(best: &mut Vec<Time>, transfers: u32, arrives: Time) {
    let n = transfers as usize;
    if best.len() <= n {
        let carry = best.last().copied().unwrap_or(UNREACHABLE);
        best.resize(n + 1, carry);
    }
    for entry in &mut best[n..] {
        *entry = (*entry).min(arrives);
    }
}

/// One entry of the queue: a trip boarded at `from`, whose transfers are
/// scanned up to `to` — the stop an earlier entry already reached it at —
/// and how it was reached, for reading the journey back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
    trip: u32,
    from: u32,
    to: u32,
    round: u32,
    /// How the rider got aboard: off an earlier segment, or from a source.
    boarded: Boarded,
}

/// How a trip segment was reached — the pointer journey extraction follows
/// back, one leg at a time, to whichever source the journey left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boarded {
    /// From a source stop, standing there at that time; the walk to the
    /// boarding stop, if the two differ, is the journey's first leg.
    Source { origin: NodeId, at: Time },
    /// Off the queue entry at `segment`, alighting at `alighted` on its trip.
    Transfer { segment: u32, alighted: u32 },
}

/// A journey that reached the target: the round, the segment and the
/// position it alighted at — or, with no segment, walked from `alight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Found {
    arrives: Time,
    transfers: u32,
    segment: u32,
    alight: u32,
}

/// [`Found`], with the departure it was found for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProfileFound {
    departs: Time,
    found: Found,
}

/// One search's working state: the queue of segments and the labels
/// `R_n(t)` — for an earliest-arrival query one row, for a profile one per
/// transfer count.
struct Sweep<'a> {
    kernel: &'a TripBased,
    segments: Vec<Segment>,
    /// `reached[n][t]`: the first position already reached on `t` with at
    /// most `n` changes. Rows are added as rounds need them, each starting
    /// as the row before.
    reached: Vec<Vec<u32>>,
    scanned: usize,
    /// One row for a query, as many as there are rounds for a profile.
    per_round: bool,
}

impl<'a> Sweep<'a> {
    fn new(kernel: &'a TripBased, per_round: bool) -> Self {
        Sweep {
            kernel,
            segments: Vec::new(),
            reached: vec![kernel.last.clone()],
            scanned: 0,
            per_round,
        }
    }

    /// Board `trip` at `position` from a source: round 0, and the only kind of
    /// segment a journey ends its walk back at.
    fn board(&mut self, trip: u32, position: u32, origin: NodeId, at: Time) {
        self.enqueue(trip, position, 0, Boarded::Source { origin, at });
    }

    /// Change onto `trip` at `position` off the segment at `segment`, having
    /// alighted at `alighted` on its trip: one more change than it took.
    fn transfer(&mut self, trip: u32, position: u32, round: u32, segment: u32, alighted: u32) {
        self.enqueue(
            trip,
            position,
            round + 1,
            Boarded::Transfer { segment, alighted },
        );
    }

    /// The paper's `enqueue`: if `trip` has not been reached at or before
    /// `position` with `round` changes, queue the segment from there to where
    /// it had been, and mark that position reached on this trip and every
    /// later trip of its line — none of which can improve on it.
    fn enqueue(&mut self, trip: u32, position: u32, round: u32, boarded: Boarded) {
        let row = if self.per_round { round as usize } else { 0 };
        while self.reached.len() <= row {
            let previous = self.reached.last().cloned().unwrap();
            self.reached.push(previous);
        }
        if position >= self.reached[row][trip as usize] {
            return;
        }
        self.segments.push(Segment {
            trip,
            from: position,
            to: self.reached[row][trip as usize],
            round,
            boarded,
        });
        // `R(u) ← min(R(u), i)` for every later `u` of the line — and, for a
        // profile, every later row, since a stop reached with `n` changes is
        // reached with more. A later trip already lowered this far needs no
        // visit: every lowering of `t` was a lowering of it too, so nothing
        // past it is higher either.
        let lines = &self.kernel.lines;
        let line_end = lines.trips_of(lines.line_of(trip)).end;
        for r in row..self.reached.len() {
            for u in trip..line_end {
                if self.reached[r][u as usize] <= position {
                    break;
                }
                self.reached[r][u as usize] = position;
            }
        }
    }

    /// Distinct trips reached: the paper labels trips, so this is what a
    /// query settles.
    fn settled(&self) -> usize {
        let last_row = self.reached.last().unwrap();
        last_row
            .iter()
            .zip(&self.kernel.last)
            .filter(|(&reached, &last)| reached < last)
            .count()
    }
}

impl Footprint for TripBased {
    fn footprint(&self) -> usize {
        TripBased::footprint(self)
    }

    fn searches(&self) -> (&'static str, usize) {
        ("trips", self.num_trips())
    }
}

impl Searches for TripBased {
    type Source = (NodeId, Time);
    type Query = TripBasedQuery;
    type Search = TripBasedSearch;
    type Error = Infallible;

    fn search(
        &self,
        sources: &[(NodeId, Time)],
        query: &TripBasedQuery,
    ) -> Result<TripBasedSearch, Infallible> {
        Ok(TripBased::search(self, sources, query))
    }
}

impl Reads for TripBased {
    fn itinerary(&self, search: &TripBasedSearch, to: NodeId) -> Option<Itinerary> {
        TripBased::itinerary(self, search, to)
    }
}

impl Front for TripBased {
    fn itineraries(&self, search: &TripBasedSearch, to: NodeId) -> Vec<Itinerary> {
        TripBased::itineraries(self, search, to)
    }
}

impl Explored for TripBased {
    type Step = (usize, TripId, Vec<NodeId>);

    fn reached(&self, search: &TripBasedSearch) -> Vec<Self::Step> {
        search.reached(self)
    }
}

impl EarliestArrival for TripBased {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        let search = TripBased::search(self, sources, &TripBasedQuery::to(to));
        TripBased::itinerary(self, &search, to)
    }
}

impl Profiled for TripBased {
    fn departures(
        &self,
        from: NodeId,
        to: NodeId,
        opens: Time,
        closes: Time,
    ) -> Vec<(Time, Itinerary)> {
        self.journeys(&self.profile(from, to, opens, closes))
    }
}

/// What a query found: the segments it scanned and the journeys that reached
/// the target, one per number of changes that improved on fewer. Plain data;
/// the [`TripBased`] it came from reads the itineraries out of it.
#[derive(Debug, Clone)]
pub struct TripBasedSearch {
    target: NodeId,
    segments: Vec<Segment>,
    found: Vec<Found>,
    /// Distinct trips reached — a share of the network's trips, which is
    /// what this kernel labels.
    pub settled: usize,
    /// Trip segments scanned — the paper's own measure of work.
    pub scanned: usize,
    /// What an elapsed cost is measured from: the moment the question was
    /// asked, or the earliest source if the caller did not say.
    pub departing: Time,
}

impl TripBasedSearch {
    /// The stop the query ran toward.
    pub fn target(&self) -> NodeId {
        self.target
    }

    /// Earliest arrival at `stop`, if it is the target and was reached. No
    /// other stop holds a label: the query is point-to-point.
    pub fn cost(&self, stop: NodeId) -> Option<Time> {
        (stop == self.target)
            .then(|| self.found.last().map(|f| f.arrives))
            .flatten()
    }

    /// Rounds run: one more than the most changes any segment was reached
    /// with.
    pub fn rounds(&self) -> usize {
        self.segments.last().map_or(0, |s| s.round as usize + 1)
    }

    /// Every segment scanned, as the number of changes it was reached with,
    /// the vehicle it rode and the stops it covers, boarded first — the search
    /// space, for drawing.
    pub fn reached(&self, kernel: &TripBased) -> Vec<(usize, TripId, Vec<NodeId>)> {
        self.segments
            .iter()
            .map(|seg| {
                let stops = kernel.lines.stops_of(kernel.lines.line_of(seg.trip));
                (
                    seg.round as usize,
                    kernel.lines.trip_id(seg.trip),
                    stops[seg.from as usize..=seg.to as usize].to_vec(),
                )
            })
            .collect()
    }
}

impl Distances for TripBasedSearch {
    fn cost(&self, stop: NodeId) -> Option<u32> {
        TripBasedSearch::cost(self, stop)
    }

    fn settled(&self) -> usize {
        self.settled
    }
}

/// What a profile query found: the Pareto set of departures for one source
/// and target, and the segments behind them.
#[derive(Debug, Clone)]
pub struct TripBasedProfile {
    source: NodeId,
    target: NodeId,
    segments: Vec<Segment>,
    found: Vec<ProfileFound>,
    /// Distinct trips reached over every run.
    pub settled: usize,
    /// Trip segments scanned over every run.
    pub scanned: usize,
    /// Departures the source offered in the window: how many times the
    /// query loop ran.
    pub runs: usize,
}

impl TripBasedProfile {
    pub fn source(&self) -> NodeId {
        self.source
    }

    pub fn target(&self) -> NodeId {
        self.target
    }

    /// Journeys in the set.
    pub fn len(&self) -> usize {
        self.found.len()
    }

    pub fn is_empty(&self) -> bool {
        self.found.is_empty()
    }
}
