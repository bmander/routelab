//! RAPTOR: rounds over routes, and no graph at all.
//!
//! Delling, Pajor & Werneck, *Round-Based Public Transit Routing* (ALENEX
//! 2012; Transportation Science 2015). Both of Pyrga et al.'s models make a
//! timetable into a graph and hand it to a shortest-path search; the priority
//! queue and the graph are the cost. RAPTOR's observation is that a timetable
//! has structure a graph throws away — **routes** (ordered stop sequences) and
//! the **trips** along them — and that with that structure the search is a
//! few array scans and no heap.
//!
//! Round `k` finds, for every stop, the earliest arrival using at most `k`
//! trips. It scans each route touched by a stop improved in round `k-1` once,
//! from the earliest such stop onward, riding the earliest trip that can be
//! caught and stepping onto an earlier one whenever a stop's round-`k-1` label
//! allows it. Then footpaths are relaxed one hop from every stop the round
//! improved. The search stops when a round improves nothing, or at a cap.
//! What comes out is not one answer but a set: one journey per round that
//! arrived earlier than the round before — the Pareto front over arrival time
//! and number of changes, which is what a rider actually chooses among.
//!
//! ## What is faithful, and what is not here
//!
//! Routes with non-overtaking trips (a route whose trips overtake is split, as
//! the paper prescribes), rounds, marked stops, the queue of (route, earliest
//! marked stop), the re-boarding check at every stop, local pruning against
//! the best label so far and target pruning against the target's, and one-hop
//! footpath relaxation over a set closed under composition — that is the
//! paper's basic algorithm with its pruning. Not here: McRAPTOR (more criteria
//! than changes), rRAPTOR (a range of departure times), and the parallel
//! variant. Changing vehicles is instantaneous, so the five timetable kernels
//! answer the same question and can be checked against each other; a minimum
//! change time is a new [`Transfer`] constructor, and this is one of the two
//! kernels — with [`crate::kernels::tripbased`] — that could honour it: a
//! round-`k` label knows the boarding was a change.
//!
//! ## What a search hands back
//!
//! RAPTOR is one-to-all by nature: after the rounds every stop holds its
//! earliest arrival by round, so a [`RaptorSearch`] keeps the labels and reads
//! any target's itinerary — or its whole Pareto set — back out of them. That
//! is also why it has a search space worth drawing: which round first reached
//! each stop is the picture in the paper.

use std::convert::Infallible;

use crate::model::graph::{NodeId, UNREACHABLE};
use crate::model::lines::Lines;
use crate::model::technique::{
    BindError, Distances, EarliestArrival, Explored, Footprint, Front, Reads, Searches, Technique,
    TransitNetwork,
};
use crate::model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Time, Timetable, Transfer, Walk,
};
use crate::util::progress::Progress;

/// What a RAPTOR query takes, beyond where you start.
///
/// A knob RAPTOR does not have is a field this struct does not declare — there
/// is no `max_cost` here, and no way to pass one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaptorQuery {
    /// Prune toward this stop; `None` for the one-to-all search.
    pub target: Option<NodeId>,
    /// Stop after this many rounds — round `k` allows `k` trips.
    pub max_rounds: Option<usize>,
    /// What an elapsed cost is measured from; `None` for the earliest source.
    pub departing: Option<Time>,
}

impl RaptorQuery {
    /// A query aimed at `target`, with everything else at its default.
    pub fn to(target: NodeId) -> Self {
        RaptorQuery {
            target: Some(target),
            ..RaptorQuery::default()
        }
    }
}

/// RAPTOR as a configuration: nothing to set, so nothing to hold. Binding it
/// to a [`TransitNetwork`] lays the timetable out as routes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaptorTechnique;

impl<'a> Technique<'a> for RaptorTechnique {
    type Inputs = TransitNetwork<'a>;
    type Planner = Raptor;

    fn bind(&self, net: TransitNetwork<'a>, _progress: &Progress) -> Result<Raptor, BindError> {
        Ok(Raptor::build(net.timetable, net.transfer, net.footpaths))
    }
}

/// A timetable in the paper's own layout: routes, the trips along each in
/// departure order, and a time per (trip, position) — the [`Lines`] every
/// route-reading kernel shares — with the footpaths between stops.
///
/// Built once from a [`Timetable`] and a [`Footpaths`], the way
/// [`crate::kernels::timetable::TimeExpanded`] is, and reused for every query.
#[derive(Debug)]
pub struct Raptor {
    lines: Lines,
    footpaths: Footpaths,
}

impl Raptor {
    /// Lay `timetable` out as routes and trips, with `footpaths` between stops.
    ///
    /// The `transfer` is accepted for parity with the other timetable models
    /// and is only ever [`Transfer::instant`] today.
    pub fn build(timetable: &Timetable, _transfer: Transfer, footpaths: &Footpaths) -> Self {
        Raptor {
            lines: Lines::from_timetable(timetable),
            footpaths: footpaths.clone(),
        }
    }

    pub fn num_stops(&self) -> usize {
        self.lines.num_stops()
    }

    /// Routes in the paper's sense — distinct stop sequences, split so that no
    /// trip overtakes another. More than a feed's own count of routes.
    pub fn num_routes(&self) -> usize {
        self.lines.num_lines()
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    pub fn num_trips(&self) -> usize {
        self.lines.num_trips()
    }

    /// Hops between consecutive stops, summed over trips — the connection count.
    pub fn num_connections(&self) -> usize {
        self.lines.num_connections()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.lines.footprint() + self.footpaths.footprint()
    }

    /// The Pareto front for `to`: one itinerary per number of changes that
    /// arrives strictly earlier than any journey with fewer, fewest first.
    pub fn pareto(&self, from: NodeId, at: Time, to: NodeId) -> Vec<Itinerary> {
        let search = self.search(&[(from, at)], &RaptorQuery::to(to));
        self.itineraries(&search, to)
    }

    /// Run the rounds from `sources` — each a stop and the time you are
    /// standing there — pruning toward the query's target if it has one, and
    /// stopping after its `max_rounds` (round `k` allows `k` trips) or when a
    /// round improves nothing.
    ///
    /// Without a target this is the one-to-all search: every stop's earliest
    /// arrival by round. With one, stops that cannot beat the target's best
    /// arrival are not labelled, which is what keeps a query cheap.
    ///
    /// The query's `departing` is what an elapsed cost is measured from — the
    /// moment the question was asked, which is not the same as the earliest
    /// source when every source carries a head start. `None` means the
    /// earliest source.
    pub fn search(&self, sources: &[(NodeId, Time)], query: &RaptorQuery) -> RaptorSearch {
        let RaptorQuery {
            target,
            max_rounds,
            departing,
        } = *query;
        let target = target.filter(|&t| (t as usize) < self.num_stops());
        let mut rounds = Rounds::new(self.num_stops());
        let mut earliest_source = UNREACHABLE;

        // Round 0: the sources, and what they can walk to.
        for &(stop, at) in sources {
            if (stop as usize) < self.num_stops() {
                earliest_source = earliest_source.min(at);
                if at < rounds.labels[0][stop as usize] {
                    rounds.labels[0][stop as usize] = at;
                    rounds.parents[0][stop as usize] = Parent::Origin;
                    rounds.mark(stop);
                }
            }
        }
        rounds.relax(self, target);

        let mut first_position: Vec<u32> = vec![u32::MAX; self.num_routes()];
        let mut touched: Vec<u32> = Vec::new();
        while !rounds.marked.is_empty() && !max_rounds.is_some_and(|cap| rounds.round() >= cap) {
            let round = rounds.open();

            // The routes to scan, each from its earliest marked stop.
            for &stop in &rounds.marked {
                for &(route, position) in self.lines.lines_at(stop) {
                    if position < first_position[route as usize] {
                        if first_position[route as usize] == u32::MAX {
                            touched.push(route);
                        }
                        first_position[route as usize] = position;
                    }
                }
            }
            rounds.clear_marks();

            for route in touched.drain(..) {
                let start = first_position[route as usize];
                first_position[route as usize] = u32::MAX;
                rounds.scanned += 1;
                let stops = self.lines.stops_of(route);
                let mut aboard: Option<u32> = None;
                let mut boarded_at: u32 = 0;
                for i in start..stops.len() as u32 {
                    let stop = stops[i as usize];
                    if let Some(trip) = aboard {
                        // Alight if that improves this stop — and, with a
                        // target, only if it could still improve the target.
                        let arrival = self.lines.time(trip, i).arrival;
                        if rounds.improves(round, stop, arrival, target) {
                            rounds.improve(
                                round,
                                stop,
                                arrival,
                                Parent::Ride {
                                    route,
                                    trip,
                                    board: boarded_at,
                                    alight: i,
                                },
                            );
                        }
                    }
                    // Board, or step onto an earlier trip: on the strength of
                    // the *previous* round's label, which is what makes one
                    // round one trip.
                    let ready = rounds.labels[round - 1][stop as usize];
                    if ready == UNREACHABLE {
                        continue;
                    }
                    let can_catch_earlier = match aboard {
                        None => true,
                        Some(trip) => ready <= self.lines.time(trip, i).departure,
                    };
                    if can_catch_earlier {
                        if let Some(trip) = self.lines.earliest_trip(route, i, ready, aboard) {
                            if aboard != Some(trip) {
                                aboard = Some(trip);
                                boarded_at = i;
                            }
                        }
                    }
                }
            }

            rounds.relax(self, target);
            if rounds.marked.is_empty() {
                // A round that improved nothing is not a round anyone rode:
                // its row is the one before, so it is not kept.
                rounds.close();
            }
        }

        let settled = rounds.labels[rounds.round()]
            .iter()
            .filter(|&&arrival| arrival != UNREACHABLE)
            .count();
        RaptorSearch {
            labels: rounds.labels,
            parents: rounds.parents,
            settled,
            scanned: rounds.scanned,
            departing: departing.unwrap_or(earliest_source),
        }
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    pub fn path(&self, search: &RaptorSearch, stop: NodeId) -> Option<Vec<NodeId>> {
        Some(self.itinerary(search, stop)?.stops(stop))
    }

    /// The earliest arrival at `to` in `search`, however many changes it takes.
    pub fn itinerary(&self, search: &RaptorSearch, to: NodeId) -> Option<Itinerary> {
        let round = search.best_round(to)?;
        Some(self.rebuild(search, round, to))
    }

    /// One itinerary per round that arrived at `to` strictly earlier than the
    /// round before — the Pareto front over arrival and changes, fewest
    /// changes first. Empty if `to` was never reached.
    pub fn itineraries(&self, search: &RaptorSearch, to: NodeId) -> Vec<Itinerary> {
        search
            .improving_rounds(to)
            .map(|round| self.rebuild(search, round, to))
            .collect()
    }

    /// Read the journey behind round `round`'s label at `to` back off the
    /// parent pointers, one leg per connection, one walk per given link.
    fn rebuild(&self, search: &RaptorSearch, round: usize, to: NodeId) -> Itinerary {
        let arrives = search.labels[round][to as usize];
        let mut legs: Vec<Leg> = Vec::new();
        let mut here = to;
        let mut round = round;
        loop {
            match search.parents[round][here as usize] {
                Parent::Origin => break,
                Parent::Inherited => {
                    if round == 0 {
                        break;
                    }
                    round -= 1;
                }
                Parent::Ride {
                    route,
                    trip,
                    board,
                    alight,
                } => {
                    let stops = self.lines.stops_of(route);
                    for i in (board..alight).rev() {
                        legs.push(Leg::Ride(Connection {
                            trip: self.lines.trip_id(trip),
                            from: stops[i as usize],
                            to: stops[i as usize + 1],
                            departs: self.lines.time(trip, i).departure,
                            arrives: self.lines.time(trip, i + 1).arrival,
                        }));
                    }
                    here = stops[board as usize];
                    round -= 1;
                }
                Parent::Walk { from } => {
                    let landed = search.labels[round][here as usize];
                    let duration = self.footpaths.duration(from, here).unwrap_or(0);
                    let walk = Walk {
                        from,
                        to: here,
                        departs: landed - duration,
                        arrives: landed,
                    };
                    legs.extend(self.footpaths.expand(walk).into_iter().rev().map(Leg::Walk));
                    here = from;
                }
            }
        }
        legs.reverse();
        Itinerary {
            arrives,
            legs,
            settled: search.settled,
        }
    }
}

/// One search's working state: a label and a parent per stop per round, and
/// the stops the current round improved.
///
/// Together rather than as seven locals, because every one of them is written
/// at the same three moments — a stop is improved, a round opens, a round
/// closes — and a footpath relaxation needs all of them at once.
struct Rounds {
    labels: Vec<Vec<Time>>,
    parents: Vec<Vec<Parent>>,
    marked: Vec<NodeId>,
    is_marked: Vec<bool>,
    scanned: usize,
}

impl Rounds {
    fn new(stops: usize) -> Self {
        Rounds {
            labels: vec![vec![UNREACHABLE; stops]],
            parents: vec![vec![Parent::Inherited; stops]],
            marked: Vec::new(),
            is_marked: vec![false; stops],
            scanned: 0,
        }
    }

    /// The round now being filled.
    fn round(&self) -> usize {
        self.labels.len() - 1
    }

    /// Start the next round. Its labels begin as the last round's, so the row
    /// is both "with at most `k` trips" and "the best known so far", which is
    /// what the paper prunes against.
    fn open(&mut self) -> usize {
        self.labels.push(self.labels[self.round()].clone());
        self.parents
            .push(vec![Parent::Inherited; self.is_marked.len()]);
        self.round()
    }

    /// Drop a round that improved nothing: its row is the one before it.
    fn close(&mut self) {
        self.labels.pop();
        self.parents.pop();
    }

    fn mark(&mut self, stop: NodeId) {
        if !self.is_marked[stop as usize] {
            self.is_marked[stop as usize] = true;
            self.marked.push(stop);
        }
    }

    fn clear_marks(&mut self) {
        for &stop in &self.marked {
            self.is_marked[stop as usize] = false;
        }
        self.marked.clear();
    }

    /// Would arriving at `stop` at `arrival` improve on what this round holds
    /// — and, with a target, on what the target holds? The second is the
    /// paper's target pruning: a stop no earlier than the target's own best
    /// arrival cannot lead anywhere that beats it.
    fn improves(&self, round: usize, stop: NodeId, arrival: Time, target: Option<NodeId>) -> bool {
        let bound = match target {
            Some(t) => self.labels[round][t as usize].min(self.labels[round][stop as usize]),
            None => self.labels[round][stop as usize],
        };
        arrival < bound
    }

    fn improve(&mut self, round: usize, stop: NodeId, arrival: Time, parent: Parent) {
        self.labels[round][stop as usize] = arrival;
        self.parents[round][stop as usize] = parent;
        self.mark(stop);
    }

    /// One hop of walking from every stop marked this round. One hop is enough
    /// because the footpaths are closed under composition; walking on from a
    /// stop reached on foot could only tie the direct link the origin already
    /// used.
    fn relax(&mut self, raptor: &Raptor, target: Option<NodeId>) {
        let round = self.round();
        // The stops marked on entry — walking adds more as it goes, and those
        // are the ones a second hop would be.
        let walking_from = self.marked.len();
        for i in 0..walking_from {
            let from = self.marked[i];
            let at = self.labels[round][from as usize];
            for (to, duration) in raptor.footpaths.from(from) {
                if to as usize >= raptor.num_stops() {
                    continue;
                }
                let arrival = at.saturating_add(duration);
                if self.improves(round, to, arrival, target) {
                    self.improve(round, to, arrival, Parent::Walk { from });
                }
            }
        }
    }
}

/// How a round's label at a stop was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    /// Not improved this round: the label is the previous round's.
    Inherited,
    /// A source, standing there at the given time.
    Origin,
    /// Rode `trip` of `route` from position `board` to position `alight`.
    Ride {
        route: u32,
        trip: u32,
        board: u32,
        alight: u32,
    },
    /// Walked here from `from`, in the same round.
    Walk { from: NodeId },
}

impl Footprint for Raptor {
    fn footprint(&self) -> usize {
        Raptor::footprint(self)
    }

    fn searches(&self) -> (&'static str, usize) {
        ("stops", self.num_stops())
    }
}

impl Searches for Raptor {
    type Source = (NodeId, Time);
    type Query = RaptorQuery;
    type Search = RaptorSearch;
    type Error = Infallible;

    fn search(
        &self,
        sources: &[(NodeId, Time)],
        query: &RaptorQuery,
    ) -> Result<RaptorSearch, Infallible> {
        Ok(Raptor::search(self, sources, query))
    }
}

impl Reads for Raptor {
    fn itinerary(&self, search: &RaptorSearch, to: NodeId) -> Option<Itinerary> {
        Raptor::itinerary(self, search, to)
    }
}

impl Front for Raptor {
    fn itineraries(&self, search: &RaptorSearch, to: NodeId) -> Vec<Itinerary> {
        Raptor::itineraries(self, search, to)
    }
}

impl Explored for Raptor {
    type Step = (NodeId, usize, Time);

    fn reached(&self, search: &RaptorSearch) -> Vec<Self::Step> {
        search.reached()
    }
}

impl EarliestArrival for Raptor {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        let search = Raptor::search(self, sources, &RaptorQuery::to(to));
        Raptor::itinerary(self, &search, to)
    }
}

/// What a round-based search found: every stop's earliest arrival by round,
/// and how each was reached. Plain data; the [`Raptor`] it came from reads a
/// target's itinerary — or its whole Pareto front — back out of it.
#[derive(Debug, Clone)]
pub struct RaptorSearch {
    labels: Vec<Vec<Time>>,
    parents: Vec<Vec<Parent>>,
    /// Distinct stops that received a label — a share of the network's stops,
    /// the way the time-dependent model's settled count is.
    pub settled: usize,
    /// Route scans, summed over rounds — the paper's own measure of work.
    pub scanned: usize,
    /// What an elapsed cost is measured from: the moment the question was
    /// asked, or the earliest source if the caller did not say.
    pub departing: Time,
}

impl RaptorSearch {
    /// Rounds that improved something: `k` means some stop was first reached,
    /// or reached sooner, with `k` trips.
    pub fn rounds(&self) -> usize {
        self.labels.len() - 1
    }

    /// Earliest arrival at `stop` over every round, if it was reached.
    pub fn cost(&self, stop: NodeId) -> Option<Time> {
        let best = *self.labels.last()?.get(stop as usize)?;
        (best != UNREACHABLE).then_some(best)
    }

    /// The round that first reached `stop`: 0 for a source and what it can
    /// walk to, `k` for a stop first reached with `k` trips.
    ///
    /// Read off the labels rather than recorded: a round's row starts as the
    /// one before it, so the first row where a stop is reachable is the round
    /// that reached it.
    pub fn round_reached(&self, stop: NodeId) -> Option<usize> {
        let s = stop as usize;
        if s >= self.labels[0].len() {
            return None;
        }
        self.labels.iter().position(|row| row[s] != UNREACHABLE)
    }

    /// The round whose label at `to` is the best — the fewest trips that
    /// achieve the earliest arrival.
    fn best_round(&self, to: NodeId) -> Option<usize> {
        self.improving_rounds(to).last()
    }

    /// Every round that arrived at `to` strictly earlier than the round before
    /// it: the Pareto front's rounds, fewest trips first.
    fn improving_rounds(&self, to: NodeId) -> impl Iterator<Item = usize> + '_ {
        let s = to as usize;
        let mut best = UNREACHABLE;
        (0..self.labels.len()).filter(move |&round| match self.labels[round].get(s) {
            Some(&arrival) if arrival < best => {
                best = arrival;
                true
            }
            _ => false,
        })
    }

    /// Every stop reached, with the round that first reached it and the
    /// arrival *that round* achieved — the search space, for drawing.
    ///
    /// Deliberately not [`Self::cost`], which is the best arrival over every
    /// round: a later round often improves a stop it did not discover, and
    /// pairing that arrival with this round would describe a journey no round
    /// ever made. A frontier is a picture of where each round got to, so both
    /// halves of the pair have to come from the same round. Ask `cost` for the
    /// answer and this for the search.
    pub fn reached(&self) -> Vec<(NodeId, usize, Time)> {
        (0..self.labels[0].len())
            .filter_map(|s| {
                let stop = s as NodeId;
                let round = self.round_reached(stop)?;
                // Reachable in that row by construction — it is the first row
                // where the stop is not UNREACHABLE.
                Some((stop, round, self.labels[round][s]))
            })
            .collect()
    }
}

impl Distances for RaptorSearch {
    fn cost(&self, stop: NodeId) -> Option<u32> {
        RaptorSearch::cost(self, stop)
    }

    fn settled(&self) -> usize {
        self.settled
    }
}
