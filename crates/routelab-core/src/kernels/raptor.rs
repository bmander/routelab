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
//! variant. Changing vehicles is instantaneous, so the three timetable kernels
//! answer the same question and can be checked against each other; a minimum
//! change time is a new [`Transfer`] constructor, and this is the one kernel
//! that could honour it — a round-`k` label knows the boarding was a change.
//!
//! ## What a search hands back
//!
//! RAPTOR is one-to-all by nature: after the rounds every stop holds its
//! earliest arrival by round, so a [`RaptorSearch`] keeps the labels and reads
//! any target's itinerary — or its whole Pareto set — back out of them. That
//! is also why it has a search space worth drawing: which round first reached
//! each stop is the picture in the paper.

use crate::model::graph::{NodeId, UNREACHABLE};

use crate::model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Time, Timetable, Transfer, Walk,
};

/// When a trip reaches a stop and when it leaves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StopTime {
    arrival: Time,
    departure: Time,
}

/// A timetable in the paper's own layout: routes, the trips along each in
/// departure order, and a time per (trip, position).
///
/// Built once from a [`Timetable`] and a [`Footpaths`], the way
/// [`crate::kernels::timetable::TimeExpanded`] is, and reused for every query.
#[derive(Debug)]
pub struct Raptor {
    stops: usize,
    /// Route `r`'s stop sequence is `route_stops[route_stops_start[r]..route_stops_start[r+1]]`.
    /// A stop a route revisits appears once per visit.
    route_stops_start: Vec<u32>,
    route_stops: Vec<NodeId>,
    /// Route `r`'s trips are the internal trips `route_trips_start[r]..route_trips_start[r+1]`,
    /// in departure order and never overtaking one another.
    route_trips_start: Vec<u32>,
    /// Trip `t` at position `i` is `stop_times[trip_times_start[t] + i]`.
    trip_times_start: Vec<u32>,
    stop_times: Vec<StopTime>,
    /// The `Connection::trip` each internal trip came from — a feed's trip
    /// whose chain of connections was broken becomes several internal trips.
    trip_ids: Vec<u32>,
    /// Every `(route, position)` a stop occupies, CSR by stop.
    stop_routes_start: Vec<u32>,
    stop_routes: Vec<(u32, u32)>,
    footpaths: Footpaths,
}

/// One trip as the builder assembles it, before routes are known.
struct Chain {
    stops: Vec<NodeId>,
    times: Vec<StopTime>,
    trip: u32,
}

impl Chain {
    /// Turn a run of connections into the trip a rider would ride: its stops
    /// in order, and when it reaches and leaves each. Position 0 has no
    /// arrival and the last no departure — you never alight where you got on,
    /// or board where the trip ends — so both borrow the time beside them.
    fn from_run(run: &mut Vec<Connection>, chains: &mut Vec<Chain>) {
        let Some(&first) = run.first() else { return };
        let mut stops = Vec::with_capacity(run.len() + 1);
        let mut times = Vec::with_capacity(run.len() + 1);
        stops.push(first.from);
        times.push(StopTime {
            arrival: first.departs,
            departure: first.departs,
        });
        for (i, c) in run.iter().enumerate() {
            stops.push(c.to);
            times.push(StopTime {
                arrival: c.arrives,
                departure: run.get(i + 1).map_or(c.arrives, |next| next.departs),
            });
        }
        chains.push(Chain {
            stops,
            times,
            trip: first.trip,
        });
        run.clear();
    }
}

impl Raptor {
    /// Lay `timetable` out as routes and trips, with `footpaths` between stops.
    ///
    /// The `transfer` is accepted for parity with the other two models and is
    /// only ever [`Transfer::instant`] today.
    pub fn build(timetable: &Timetable, _transfer: Transfer, footpaths: &Footpaths) -> Self {
        let stops = timetable.num_stops();

        // Connections back into trips: sorted by trip and then by time, so a
        // trip's hops are contiguous and in riding order — robust to a trip
        // that revisits a stop, which chaining `from` to `to` would not be.
        let mut connections: Vec<Connection> = timetable.connections().to_vec();
        connections.sort_by_key(|c| (c.trip, c.departs, c.arrives));

        // Cut each trip's run into chains. A feed's trip that lost a hop (not
        // boardable, no times) or a hand-made timetable that never joined up
        // is several trips as far as a rider is concerned.
        let mut chains: Vec<Chain> = Vec::new();
        let mut run: Vec<Connection> = Vec::new();
        for c in connections {
            let breaks = match run.last() {
                Some(prev) => prev.trip != c.trip || prev.to != c.from || c.departs < prev.arrives,
                None => false,
            };
            if breaks {
                Chain::from_run(&mut run, &mut chains);
            }
            run.push(c);
        }
        Chain::from_run(&mut run, &mut chains);

        // Group by stop sequence, deterministically: identical sequences are
        // contiguous, and within one they come in departure order.
        chains.sort_by(|a, b| {
            a.stops
                .cmp(&b.stops)
                .then(a.times[0].departure.cmp(&b.times[0].departure))
                .then(
                    a.times
                        .last()
                        .unwrap()
                        .arrival
                        .cmp(&b.times.last().unwrap().arrival),
                )
                .then(a.trip.cmp(&b.trip))
        });

        // Then split each group so that no trip overtakes another: the paper's
        // routes are FIFO, which is what lets "the earliest trip you can catch
        // at this stop" be a binary search over departures and lets an earlier
        // trip never arrive later downstream. Greedy: each trip joins the first
        // sub-route whose last trip it does not overtake at any position.
        let mut route_stops_start = vec![0u32];
        let mut route_stops: Vec<NodeId> = Vec::new();
        let mut route_trips_start = vec![0u32];
        let mut trip_times_start = vec![0u32];
        let mut stop_times: Vec<StopTime> = Vec::new();
        let mut trip_ids: Vec<u32> = Vec::new();

        let mut group_start = 0;
        while group_start < chains.len() {
            let mut group_end = group_start + 1;
            while group_end < chains.len() && chains[group_end].stops == chains[group_start].stops {
                group_end += 1;
            }
            let group = &chains[group_start..group_end];
            // Sub-routes as lists of chain indices into `group`.
            let mut sub_routes: Vec<Vec<usize>> = Vec::new();
            for (i, chain) in group.iter().enumerate() {
                let fits = sub_routes.iter().position(|members| {
                    let last = &group[*members.last().unwrap()];
                    last.times
                        .iter()
                        .zip(&chain.times)
                        .all(|(a, b)| a.arrival <= b.arrival && a.departure <= b.departure)
                });
                match fits {
                    Some(r) => sub_routes[r].push(i),
                    None => sub_routes.push(vec![i]),
                }
            }
            for members in sub_routes {
                route_stops.extend_from_slice(&group[0].stops);
                route_stops_start.push(route_stops.len() as u32);
                for i in members {
                    let chain = &group[i];
                    stop_times.extend_from_slice(&chain.times);
                    trip_times_start.push(stop_times.len() as u32);
                    trip_ids.push(chain.trip);
                }
                route_trips_start.push(trip_ids.len() as u32);
            }
            group_start = group_end;
        }

        // Every (route, position) a stop occupies, CSR by stop.
        let num_routes = route_stops_start.len() - 1;
        let mut stop_routes_start = vec![0u32; stops + 1];
        for r in 0..num_routes {
            for i in route_stops_start[r]..route_stops_start[r + 1] {
                stop_routes_start[route_stops[i as usize] as usize + 1] += 1;
            }
        }
        for s in 0..stops {
            stop_routes_start[s + 1] += stop_routes_start[s];
        }
        let mut fill = stop_routes_start.clone();
        let mut stop_routes = vec![(0u32, 0u32); route_stops.len()];
        for r in 0..num_routes {
            let start = route_stops_start[r];
            for i in start..route_stops_start[r + 1] {
                let stop = route_stops[i as usize] as usize;
                stop_routes[fill[stop] as usize] = (r as u32, i - start);
                fill[stop] += 1;
            }
        }

        Raptor {
            stops,
            route_stops_start,
            route_stops,
            route_trips_start,
            trip_times_start,
            stop_times,
            trip_ids,
            stop_routes_start,
            stop_routes,
            footpaths: footpaths.clone(),
        }
    }

    pub fn num_stops(&self) -> usize {
        self.stops
    }

    /// Routes in the paper's sense — distinct stop sequences, split so that no
    /// trip overtakes another. More than a feed's own count of routes.
    pub fn num_routes(&self) -> usize {
        self.route_stops_start.len() - 1
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    pub fn num_trips(&self) -> usize {
        self.trip_ids.len()
    }

    /// Hops between consecutive stops, summed over trips — the connection count.
    pub fn num_connections(&self) -> usize {
        self.stop_times.len() - self.trip_ids.len()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        std::mem::size_of::<u32>()
            * (self.route_stops_start.len()
                + self.route_stops.len()
                + self.route_trips_start.len()
                + self.trip_times_start.len()
                + self.trip_ids.len()
                + self.stop_routes_start.len())
            + self.stop_times.len() * std::mem::size_of::<StopTime>()
            + self.stop_routes.len() * std::mem::size_of::<(u32, u32)>()
            + self.footpaths.footprint()
    }

    fn stops_of(&self, route: u32) -> &[NodeId] {
        let r = route as usize;
        &self.route_stops
            [self.route_stops_start[r] as usize..self.route_stops_start[r + 1] as usize]
    }

    fn routes_at(&self, stop: NodeId) -> &[(u32, u32)] {
        let s = stop as usize;
        &self.stop_routes
            [self.stop_routes_start[s] as usize..self.stop_routes_start[s + 1] as usize]
    }

    #[inline]
    fn time(&self, trip: u32, position: u32) -> StopTime {
        self.stop_times[(self.trip_times_start[trip as usize] + position) as usize]
    }

    /// The earliest trip of `route` leaving `position` at or after `at`,
    /// looking no later than `before` (exclusive) — the paper's observation
    /// that once aboard trip `t`, only trips before `t` need looking at.
    ///
    /// A binary search, which is what the non-overtaking split buys: a route's
    /// trips leave every one of its stops in the same order.
    fn earliest_trip(
        &self,
        route: u32,
        position: u32,
        at: Time,
        before: Option<u32>,
    ) -> Option<u32> {
        let first = self.route_trips_start[route as usize];
        let end = before.unwrap_or(self.route_trips_start[route as usize + 1]);
        let trips = &self.trip_times_start[first as usize..end as usize];
        let found = trips
            .partition_point(|&start| self.stop_times[(start + position) as usize].departure < at)
            as u32;
        (first + found < end).then_some(first + found)
    }

    /// Earliest arrival at `to`, leaving `from` no earlier than `at`.
    pub fn earliest_arrival(&self, from: NodeId, at: Time, to: NodeId) -> Option<Itinerary> {
        let search = self.search(&[(from, at)], Some(to), None, None);
        self.itinerary(&search, to)
    }

    /// The Pareto front for `to`: one itinerary per number of changes that
    /// arrives strictly earlier than any journey with fewer, fewest first.
    pub fn pareto(&self, from: NodeId, at: Time, to: NodeId) -> Vec<Itinerary> {
        let search = self.search(&[(from, at)], Some(to), None, None);
        self.itineraries(&search, to)
    }

    /// Run the rounds from `sources` — each a stop and the time you are
    /// standing there — pruning toward `target` if one is given, and stopping
    /// after `max_rounds` rounds (round `k` allows `k` trips) or when a round
    /// improves nothing.
    ///
    /// Without a target this is the one-to-all search: every stop's earliest
    /// arrival by round. With one, stops that cannot beat the target's best
    /// arrival are not labelled, which is what keeps a query cheap.
    ///
    /// `departing` is what an elapsed cost is measured from — the moment the
    /// question was asked, which is not the same as the earliest source when
    /// every source carries a head start. `None` means the earliest source.
    pub fn search(
        &self,
        sources: &[(NodeId, Time)],
        target: Option<NodeId>,
        max_rounds: Option<usize>,
        departing: Option<Time>,
    ) -> RaptorSearch {
        let target = target.filter(|&t| (t as usize) < self.stops);
        let mut rounds = Rounds::new(self.stops);
        let mut earliest_source = UNREACHABLE;

        // Round 0: the sources, and what they can walk to.
        for &(stop, at) in sources {
            if (stop as usize) < self.stops {
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
                for &(route, position) in self.routes_at(stop) {
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
                let stops = self.stops_of(route);
                let mut aboard: Option<u32> = None;
                let mut boarded_at: u32 = 0;
                for i in start..stops.len() as u32 {
                    let stop = stops[i as usize];
                    if let Some(trip) = aboard {
                        // Alight if that improves this stop — and, with a
                        // target, only if it could still improve the target.
                        let arrival = self.time(trip, i).arrival;
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
                        Some(trip) => ready <= self.time(trip, i).departure,
                    };
                    if can_catch_earlier {
                        if let Some(trip) = self.earliest_trip(route, i, ready, aboard) {
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
        let itinerary = self.itinerary(search, stop)?;
        let mut path: Vec<NodeId> = itinerary
            .legs
            .first()
            .map(|leg| vec![leg.from()])
            .unwrap_or_default();
        path.extend(itinerary.legs.iter().map(|leg| leg.to()));
        if path.is_empty() {
            path.push(stop);
        }
        Some(path)
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
                    let stops = self.stops_of(route);
                    for i in (board..alight).rev() {
                        legs.push(Leg::Ride(Connection {
                            trip: self.trip_ids[trip as usize],
                            from: stops[i as usize],
                            to: stops[i as usize + 1],
                            departs: self.time(trip, i).departure,
                            arrives: self.time(trip, i + 1).arrival,
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
                if to as usize >= raptor.stops {
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

    /// Every stop reached, with the round that first reached it and its
    /// earliest arrival — the search space, for drawing.
    pub fn reached(&self) -> Vec<(NodeId, usize, Time)> {
        (0..self.labels[0].len())
            .filter_map(|s| {
                let stop = s as NodeId;
                Some((stop, self.round_reached(stop)?, self.cost(stop)?))
            })
            .collect()
    }
}
