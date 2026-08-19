//! CSA: one array of connections, scanned once.
//!
//! Dibbelt, Pajor, Strasser & Wagner, *Intriguingly Simple and Fast Transit
//! Routing* (SEA 2013). RAPTOR stopped building a graph and read the
//! timetable's routes and trips instead; CSA's observation is that even routes
//! are more structure than the question needs. A connection is *reachable* if
//! you are already aboard its trip or standing at its departure stop in time —
//! and because a timetable is aperiodic, the connections can be sorted by
//! departure once and that property checked in one linear pass. So: sort every
//! connection into a single array, keep a label per stop and a flag per trip,
//! and scan.
//!
//! **Earliest arrival** (§3): labels start at the sources; the scan begins at
//! the first connection departing at or after the earliest source (the
//! paper's *start criterion*, a binary search), and for each connection in
//! turn asks whether it is reachable — its trip already flagged, or its
//! departure stop labelled no later than it leaves. A reachable connection
//! flags its trip and, if it improves the label at its arrival stop, relaxes
//! that stop's footpaths one hop. With a target, the scan stops at the first
//! connection departing no earlier than the target's label (the *stop
//! criterion*); without one it runs to the end of the day and every stop holds
//! its earliest arrival — a one-to-all search, which is why this kernel has a
//! [`ScanSearch`] to keep and something to draw.
//!
//! **Profile** (§4, pCSA-P): the same array scanned the other way. For a
//! target, every stop keeps a partial *profile* — Pareto-optimal (departure,
//! arrival) pairs in decreasing order of departure — and each connection,
//! visited latest first, learns the earliest it can deliver a rider to the
//! target: by getting off at its arrival stop (and walking, if that stop walks
//! to the target), by staying aboard (the trip's own arrival time, which stands
//! in for the trip flag), or by transferring (its arrival stop's profile,
//! evaluated at its arrival). That value becomes a candidate pair at its
//! departure stop — and, walked back along every incoming footpath, at each
//! stop that can walk to it — inserted unless a pair already there dominates
//! it. When the scan reaches the start of the window, the source's profile is
//! the answer: one journey per moment worth leaving.
//!
//! ## What is faithful, and what is not here
//!
//! The paper's basic algorithm with its improvements — start criterion, stop
//! criterion, trip flags — and its footpath rule (relax on improvement, over
//! footpaths that are transitively closed, which ours are), plus its profile
//! variant with Pareto arrays evaluated by a scan from the back, footpaths at
//! the departure stop inserted along incoming links, footpaths to the target
//! folded into evaluation, and its one-to-one pruning against the source's
//! profile. Changing vehicles is instantaneous, so the five timetable kernels
//! answer the same question and check each other; a minimum change time is a
//! new [`Transfer`] constructor, which this kernel would honour by subtracting
//! it from a departure as the paper does. Not here: pseudoconnections and time
//! indexing (pCSA-C, pCSA-CT — the paper's cache-friendlier layouts for the
//! same profile), the multi-criteria profile (mcpCSA), and the minimum
//! expected arrival time problem (§5), each its own increment. Journey
//! extraction is not spelled out in the paper; the pointers here are the
//! obvious ones — the connection that reached each stop, the connection each
//! trip was entered with — and every itinerary is checked against
//! [`Itinerary::is_valid`].
//!
//! Connections that arrive when they depart are kept, as the [`Timetable`]
//! keeps them; the paper filters them out, and a chain of them at one instant
//! across several trips is the one shape a departure-ordered scan cannot
//! promise to see in riding order.

use std::convert::Infallible;

use crate::model::graph::{NodeId, UNREACHABLE};
use crate::model::technique::{
    BindError, Distances, EarliestArrival, Explored, Footprint, Profiled, Reads, Searches,
    Technique, TransitNetwork,
};
use crate::model::timetable::{
    Connection, Footpaths, Itinerary, Leg, Time, Timetable, Transfer, TripId, Walk,
};
use crate::util::progress::Progress;

/// What a connection scan takes, beyond where you start. No cap on changes:
/// a scan does not count them, so there is no field to count them with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanQuery {
    /// Stop early once this stop's arrival is settled; `None` labels every stop.
    pub target: Option<NodeId>,
    /// What an elapsed cost is measured from; `None` for the earliest source.
    pub departing: Option<Time>,
}

impl ScanQuery {
    /// A query aimed at `target`, with everything else at its default.
    pub fn to(target: NodeId) -> Self {
        ScanQuery {
            target: Some(target),
            ..ScanQuery::default()
        }
    }
}

/// Connection scan as a configuration: nothing to set. Binding it sorts the
/// timetable's connections by departure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConnectionScanTechnique;

impl<'a> Technique<'a> for ConnectionScanTechnique {
    type Inputs = TransitNetwork<'a>;
    type Planner = ConnectionScan;

    fn bind(
        &self,
        net: TransitNetwork<'a>,
        _progress: &Progress,
    ) -> Result<ConnectionScan, BindError> {
        Ok(ConnectionScan::build(
            net.timetable,
            net.transfer,
            net.footpaths,
        ))
    }
}

/// "No connection", in the arrays that index them.
const NONE: u32 = u32::MAX;

/// A timetable in the paper's own layout: every connection, in departure
/// order, with the trip it belongs to and the one that follows it aboard.
///
/// Built once from a [`Timetable`] and a [`Footpaths`], the way
/// [`crate::kernels::raptor::Raptor`] is, and reused for every query.
#[derive(Debug)]
pub struct ConnectionScan {
    stops: usize,
    /// The array, sorted by departure. Each connection keeps the feed's own
    /// [`TripId`], which is what an answer names.
    connections: Vec<Connection>,
    /// The internal trip each connection belongs to, dense and parallel to
    /// `connections`: what the trip flags are indexed by. Kept beside the
    /// array rather than written over `Connection::trip`, so that the id a
    /// rider is told and the index a scan flags cannot be confused for one
    /// another — they are not even the same type.
    aboard: Vec<u32>,
    /// The next connection of the same internal trip, or `NONE` at its end —
    /// the paper's `c_next`.
    next_in_trip: Vec<u32>,
    /// The feed trip each internal trip came from — a feed's trip whose chain
    /// of connections was broken becomes several.
    trip_ids: Vec<TripId>,
    footpaths: Footpaths,
    /// The same footpaths reversed: the profile scan walks a candidate back to
    /// every stop that can walk to it, which is [`Footpaths::from`] on this.
    incoming: Footpaths,
}

impl ConnectionScan {
    /// Sort `timetable`'s connections into the array, with `footpaths`
    /// between stops.
    ///
    /// The `transfer` is accepted for parity with the other three models and
    /// is only ever [`Transfer::instant`] today.
    pub fn build(timetable: &Timetable, _transfer: Transfer, footpaths: &Footpaths) -> Self {
        let stops = timetable.num_stops();

        // Connections back into trips: sorted by trip and then by time, so a
        // trip's hops are contiguous and in riding order. A feed's trip that
        // lost a hop, or a hand-made timetable that never joined up, is
        // several trips as far as a rider is concerned — the same cut RAPTOR
        // makes, because "still aboard" must mean the vehicle really carried
        // you from one hop to the next.
        let mut by_trip: Vec<Connection> = timetable.connections().to_vec();
        by_trip.sort_unstable_by_key(|c| (c.trip, c.departs, c.arrives, c.from, c.to));
        let mut trip_ids: Vec<TripId> = Vec::new();
        // A connection and the internal trip it belongs to, travelling
        // together: the sort below has to keep them paired, which is what
        // storing the index in the connection used to buy and cost.
        let mut riding: Vec<(Connection, u32)> = Vec::with_capacity(by_trip.len());
        let mut previous: Option<Connection> = None;
        for c in by_trip {
            let breaks = match previous {
                Some(prev) => prev.trip != c.trip || prev.to != c.from || c.departs < prev.arrives,
                None => true,
            };
            if breaks {
                trip_ids.push(c.trip);
            }
            previous = Some(c);
            riding.push((c, (trip_ids.len() - 1) as u32));
        }

        // Then into departure order — a stable sort, which is what keeps a
        // trip's own hops in riding order under the key: the only way two of
        // them can tie on it is a chain of hops that arrive when they depart,
        // and those are already laid out in riding order above.
        riding.sort_by_key(|(c, trip)| (c.departs, c.arrives, *trip));
        let (connections, aboard): (Vec<Connection>, Vec<u32>) = riding.into_iter().unzip();

        let mut next_in_trip = vec![NONE; connections.len()];
        let mut last_of_trip = vec![NONE; trip_ids.len()];
        for (i, &trip) in aboard.iter().enumerate() {
            let last = &mut last_of_trip[trip as usize];
            if *last != NONE {
                next_in_trip[*last as usize] = i as u32;
            }
            *last = i as u32;
        }

        ConnectionScan {
            stops,
            connections,
            aboard,
            next_in_trip,
            trip_ids,
            incoming: footpaths.reversed(),
            footpaths: footpaths.clone(),
        }
    }

    pub fn num_stops(&self) -> usize {
        self.stops
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    pub fn num_trips(&self) -> usize {
        self.trip_ids.len()
    }

    /// The length of the array.
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// Bytes held, as every other preprocessed structure here reports it.
    pub fn footprint(&self) -> usize {
        self.connections.len() * std::mem::size_of::<Connection>()
            + (self.next_in_trip.len() + self.aboard.len()) * std::mem::size_of::<u32>()
            + self.trip_ids.len() * std::mem::size_of::<TripId>()
            + self.footpaths.footprint()
            + self.incoming.footprint()
    }

    /// Who can walk to `stop`, as `(from, duration)`.
    fn walks_to(&self, stop: NodeId) -> impl Iterator<Item = (NodeId, Time)> + '_ {
        self.incoming.from(stop)
    }

    /// The connection at `index`. It already wears the feed's trip id, which
    /// is what an answer names.
    fn ride(&self, index: u32) -> Connection {
        self.connections[index as usize]
    }

    // --- Earliest arrival (§3) ---------------------------------------------

    /// Scan from `sources` — each a stop and the time you are standing there
    /// — stopping early if the query names a target, else labelling every stop.
    ///
    /// The query's `departing` is what an elapsed cost is measured from — the
    /// moment the question was asked, which is not the same as the earliest
    /// source when every source carries a head start. `None` means the
    /// earliest source.
    pub fn search(&self, sources: &[(NodeId, Time)], query: &ScanQuery) -> ScanSearch {
        let ScanQuery { target, departing } = *query;
        let target = target.filter(|&t| (t as usize) < self.stops);
        let mut earliest = vec![UNREACHABLE; self.stops];
        let mut arrived_by = vec![Parent::Origin; self.stops];
        let mut entered = vec![NONE; self.trip_ids.len()];
        let mut earliest_source = UNREACHABLE;

        for &(stop, at) in sources {
            if (stop as usize) < self.stops && at < earliest[stop as usize] {
                earliest[stop as usize] = at;
                arrived_by[stop as usize] = Parent::Origin;
                earliest_source = earliest_source.min(at);
            }
        }
        // A walk from where you are standing needs no connection to set it
        // off, so the sources' footpaths are relaxed before the scan; one hop,
        // because the set is closed under composition.
        for &(stop, _) in sources {
            if (stop as usize) < self.stops {
                let at = earliest[stop as usize];
                for (to, duration) in self.footpaths.from(stop) {
                    let arrival = at.saturating_add(duration);
                    if (to as usize) < self.stops && arrival < earliest[to as usize] {
                        earliest[to as usize] = arrival;
                        arrived_by[to as usize] = Parent::Walk { from: stop };
                    }
                }
            }
        }

        // The start criterion: nothing departing before the earliest source
        // can be reached, so a binary search skips it.
        let start = self
            .connections
            .partition_point(|c| c.departs < earliest_source);
        let mut scanned = 0;
        for i in start..self.connections.len() {
            let c = self.connections[i];
            // The stop criterion: once the array has reached the target's
            // label, nothing left in it can arrive sooner.
            if target.is_some_and(|t| c.departs >= earliest[t as usize]) {
                break;
            }
            scanned += 1;
            let trip = self.aboard[i] as usize;
            if entered[trip] == NONE {
                if earliest[c.from as usize] > c.departs {
                    continue;
                }
                entered[trip] = i as u32;
            }
            if c.arrives < earliest[c.to as usize] {
                earliest[c.to as usize] = c.arrives;
                arrived_by[c.to as usize] = Parent::Ride(i as u32);
                for (to, duration) in self.footpaths.from(c.to) {
                    let arrival = c.arrives.saturating_add(duration);
                    if (to as usize) < self.stops && arrival < earliest[to as usize] {
                        earliest[to as usize] = arrival;
                        arrived_by[to as usize] = Parent::Walk { from: c.to };
                    }
                }
            }
        }

        let settled = earliest.iter().filter(|&&t| t != UNREACHABLE).count();
        ScanSearch {
            earliest,
            arrived_by,
            entered,
            settled,
            scanned,
            departing: departing.unwrap_or(earliest_source),
        }
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    pub fn path(&self, search: &ScanSearch, stop: NodeId) -> Option<Vec<NodeId>> {
        Some(self.itinerary(search, stop)?.stops(stop))
    }

    /// The earliest arrival at `to` in `search`, read back off the pointers:
    /// one leg per connection, one walk per given link.
    pub fn itinerary(&self, search: &ScanSearch, to: NodeId) -> Option<Itinerary> {
        let arrives = search.cost(to)?;
        let mut legs: Vec<Leg> = Vec::new();
        let mut here = to;
        loop {
            match search.arrived_by[here as usize] {
                Parent::Origin => break,
                Parent::Walk { from } => {
                    let landed = search.earliest[here as usize];
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
                Parent::Ride(alighted) => {
                    // Aboard from the connection the trip was entered with to
                    // the one that landed here, in riding order.
                    let boarded = search.entered[self.aboard[alighted as usize] as usize];
                    let mut ride = Vec::new();
                    let mut i = boarded;
                    loop {
                        ride.push(Leg::Ride(self.ride(i)));
                        if i == alighted {
                            break;
                        }
                        i = self.next_in_trip[i as usize];
                    }
                    legs.extend(ride.into_iter().rev());
                    here = self.connections[boarded as usize].from;
                }
            }
        }
        legs.reverse();
        Some(Itinerary {
            arrives,
            legs,
            settled: search.settled,
        })
    }

    // --- Profile (§4) --------------------------------------------------------

    /// The profile toward `target` for every stop, over journeys leaving no
    /// earlier than `from` — the paper's all-to-one query, which is what one
    /// backward scan computes.
    ///
    /// `prune` is the paper's one-to-one rule: given the stop the question is
    /// really about, a candidate pair anywhere that the source's own profile
    /// already dominates is not inserted, since nothing through it can reach
    /// the source undominated. The other stops' profiles are then partial;
    /// the source's is exact.
    pub fn profile(&self, target: NodeId, from: Time, prune: Option<NodeId>) -> ScanProfile {
        let stops = self.stops;
        let mut profile = ScanProfile {
            target,
            entries: Vec::new(),
            profiles: vec![Vec::new(); stops],
            decisions: vec![Decision::Unscanned; self.connections.len()],
            settled: 0,
            scanned: 0,
        };
        if (target as usize) >= stops {
            return profile;
        }
        let prune = prune.filter(|&p| (p as usize) < stops);

        // Getting off and walking the rest: the footpaths into the target,
        // folded into evaluation as the paper says, so the identity profile
        // at the target is a case and not a table.
        let mut to_target = vec![UNREACHABLE; stops];
        to_target[target as usize] = 0;
        for (stop, duration) in self.walks_to(target) {
            to_target[stop as usize] = duration;
        }

        // The trip's arrival time at the target if you stay aboard from here
        // — what the trip flag becomes when the answer is a time.
        let mut trip_arrival = vec![UNREACHABLE; self.trip_ids.len()];

        // Latest departure first, down to the first connection departing
        // inside the window; connections leaving after it are still scanned,
        // because a journey that leaves in time may ride them later.
        let end = self.connections.partition_point(|c| c.departs < from);
        profile.scanned = self.connections.len() - end;
        for i in (end..self.connections.len()).rev() {
            let c = self.connections[i];
            let walk = to_target[c.to as usize];
            let exit = if walk == UNREACHABLE {
                UNREACHABLE
            } else {
                c.arrives.saturating_add(walk)
            };
            let aboard = trip_arrival[self.aboard[i] as usize];
            let (transfer, via) = profile.evaluate(c.to, c.arrives);
            let best = exit.min(aboard).min(transfer);
            // Nothing this connection can do reaches the target: it keeps the
            // `Unscanned` decision it was born with, and the trip's own
            // arrival is already `UNREACHABLE`, since `best` is at most that.
            if best == UNREACHABLE {
                continue;
            }
            profile.decisions[i] = if best == exit {
                Decision::Exit
            } else if best == aboard {
                Decision::Continue
            } else {
                Decision::Transfer(via)
            };
            trip_arrival[self.aboard[i] as usize] = best;
            // A candidate at the departure stop, and — walked back along each
            // incoming footpath — at every stop that can walk to it.
            profile.offer(prune, c.from, c.departs, best, i as u32, false);
            for (stop, duration) in self.walks_to(c.from) {
                if stop != target && c.departs >= duration {
                    profile.offer(prune, stop, c.departs - duration, best, i as u32, true);
                }
            }
        }
        profile.settled = profile.profiles.iter().filter(|p| !p.is_empty()).count();
        profile
    }

    /// The direct walk from `stop` to the profile's target, if there is one:
    /// a journey with no departure time, which a profile of pairs cannot
    /// hold and a caller should measure the pairs against.
    pub fn walk(&self, profile: &ScanProfile, stop: NodeId) -> Option<Time> {
        if stop == profile.target {
            return None;
        }
        self.footpaths.duration(stop, profile.target)
    }

    /// The Pareto pairs at `stop` leaving within `[from, until]`, earliest
    /// first, as `(departs, arrives)` — a departure being the latest moment
    /// you can leave and still make that arrival. Pairs the direct walk
    /// dominates are left out.
    pub fn pairs(
        &self,
        profile: &ScanProfile,
        stop: NodeId,
        from: Time,
        until: Time,
    ) -> Vec<(Time, Time)> {
        self.entries_within(profile, stop, from, until)
            .map(|entry| (entry.dep, entry.arr))
            .collect()
    }

    /// The journey behind every pair [`ConnectionScan::pairs`] would list,
    /// with its departure, earliest first.
    pub fn journeys(
        &self,
        profile: &ScanProfile,
        stop: NodeId,
        from: Time,
        until: Time,
    ) -> Vec<(Time, Itinerary)> {
        self.entries_within(profile, stop, from, until)
            .map(|entry| (entry.dep, self.rebuild(profile, stop, entry)))
            .collect()
    }

    fn entries_within<'a>(
        &'a self,
        profile: &'a ScanProfile,
        stop: NodeId,
        from: Time,
        until: Time,
    ) -> impl Iterator<Item = Entry> + 'a {
        let walk = self.walk(profile, stop).unwrap_or(UNREACHABLE);
        let listed: &[u32] = if (stop as usize) < self.stops {
            &profile.profiles[stop as usize]
        } else {
            &[]
        };
        listed
            .iter()
            .rev()
            .map(move |&e| profile.entries[e as usize])
            .filter(move |entry| {
                entry.dep >= from
                    && entry.dep <= until
                    && (walk == UNREACHABLE || entry.arr < entry.dep.saturating_add(walk))
            })
    }

    /// Follow one profile entry's decisions to the target: walk to the first
    /// connection if the entry was reached on foot, ride, and at each stop
    /// do what the scan decided there — get off, stay on, or change.
    fn rebuild(&self, profile: &ScanProfile, stop: NodeId, entry: Entry) -> Itinerary {
        let mut legs: Vec<Leg> = Vec::new();
        let mut i = entry.conn;
        if entry.walked {
            let first = self.connections[i as usize];
            let walk = Walk {
                from: stop,
                to: first.from,
                departs: entry.dep,
                arrives: first.departs,
            };
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        loop {
            let c = self.connections[i as usize];
            legs.push(Leg::Ride(self.ride(i)));
            match profile.decisions[i as usize] {
                Decision::Exit => {
                    if c.to != profile.target {
                        let walk = Walk {
                            from: c.to,
                            to: profile.target,
                            departs: c.arrives,
                            arrives: entry.arr,
                        };
                        legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
                    }
                    break;
                }
                Decision::Continue => i = self.next_in_trip[i as usize],
                Decision::Transfer(next) => {
                    let next = profile.entries[next as usize];
                    if next.walked {
                        let boarding = self.connections[next.conn as usize];
                        let walk = Walk {
                            from: c.to,
                            to: boarding.from,
                            departs: next.dep,
                            arrives: boarding.departs,
                        };
                        legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
                    }
                    i = next.conn;
                }
                Decision::Unscanned => break,
            }
        }
        Itinerary {
            arrives: entry.arr,
            legs,
            settled: profile.settled,
        }
    }
}

/// How a stop's label was reached in an earliest-arrival scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    /// A source, standing there at the given time — or never reached, which
    /// a label of `UNREACHABLE` says first.
    Origin,
    /// Landed by the connection at this index of the array.
    Ride(u32),
    /// Walked here from `from`.
    Walk { from: NodeId },
}

impl Footprint for ConnectionScan {
    fn footprint(&self) -> usize {
        ConnectionScan::footprint(self)
    }

    fn searches(&self) -> (&'static str, usize) {
        ("stops", self.num_stops())
    }
}

impl Searches for ConnectionScan {
    type Source = (NodeId, Time);
    type Query = ScanQuery;
    type Search = ScanSearch;
    type Error = Infallible;

    fn search(
        &self,
        sources: &[(NodeId, Time)],
        query: &ScanQuery,
    ) -> Result<ScanSearch, Infallible> {
        Ok(ConnectionScan::search(self, sources, query))
    }
}

impl Reads for ConnectionScan {
    fn itinerary(&self, search: &ScanSearch, to: NodeId) -> Option<Itinerary> {
        ConnectionScan::itinerary(self, search, to)
    }
}

impl Explored for ConnectionScan {
    type Step = (NodeId, Time);

    fn reached(&self, search: &ScanSearch) -> Vec<Self::Step> {
        search.reached()
    }
}

impl EarliestArrival for ConnectionScan {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        let search = ConnectionScan::search(self, sources, &ScanQuery::to(to));
        ConnectionScan::itinerary(self, &search, to)
    }
}

impl Profiled for ConnectionScan {
    /// The one-to-one profile: scanned toward `to`, pruned at `from`, read at
    /// `from` for the window.
    fn departures(
        &self,
        from: NodeId,
        to: NodeId,
        opens: Time,
        closes: Time,
    ) -> Vec<(Time, Itinerary)> {
        let profile = self.profile(to, opens, Some(from));
        self.journeys(&profile, from, opens, closes)
    }
}

/// What an earliest-arrival scan found: every stop's label and how it was
/// reached, and which connection each trip was entered with. Plain data; the
/// [`ConnectionScan`] it came from reads any target's itinerary out of it.
#[derive(Debug, Clone)]
pub struct ScanSearch {
    earliest: Vec<Time>,
    arrived_by: Vec<Parent>,
    entered: Vec<u32>,
    /// Distinct stops that received a label — a share of the network's stops.
    pub settled: usize,
    /// Connections scanned — the paper's own measure of work.
    pub scanned: usize,
    /// What an elapsed cost is measured from: the moment the question was
    /// asked, or the earliest source if the caller did not say.
    pub departing: Time,
}

impl ScanSearch {
    /// Earliest arrival at `stop`, if it was reached.
    pub fn cost(&self, stop: NodeId) -> Option<Time> {
        let best = *self.earliest.get(stop as usize)?;
        (best != UNREACHABLE).then_some(best)
    }

    /// Every stop reached, with its earliest arrival — the search space, for
    /// drawing: the sweep, as far as it got.
    pub fn reached(&self) -> Vec<(NodeId, Time)> {
        self.earliest
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t != UNREACHABLE)
            .map(|(stop, &t)| (stop as NodeId, t))
            .collect()
    }
}

impl Distances for ScanSearch {
    fn cost(&self, stop: NodeId) -> Option<u32> {
        ScanSearch::cost(self, stop)
    }

    fn settled(&self) -> usize {
        self.settled
    }
}

/// One Pareto pair in a stop's profile: leave at `dep`, arrive at `arr`, by
/// boarding the connection at `conn` — walking to it first if `walked`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    dep: Time,
    arr: Time,
    conn: u32,
    walked: bool,
}

/// What a connection's rider does at its arrival stop, decided when it was
/// scanned: the pointer journey extraction follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    Unscanned,
    /// Get off — you are at the target, or a walk away from it.
    Exit,
    /// Stay aboard for the next connection of the trip.
    Continue,
    /// Change, into the journey this entry of the arrival stop's profile holds.
    Transfer(u32),
}

/// What a profile scan found: a Pareto profile per stop toward one target,
/// and the decisions that let any pair be read back as a journey.
#[derive(Debug, Clone)]
pub struct ScanProfile {
    target: NodeId,
    /// Every entry ever inserted, append-only, so a decision can point at one
    /// that a later, better pair has since pushed out of its profile.
    entries: Vec<Entry>,
    /// Per stop, entry indices in decreasing order of departure.
    profiles: Vec<Vec<u32>>,
    decisions: Vec<Decision>,
    /// Stops whose profile holds at least one pair.
    pub settled: usize,
    /// Connections scanned.
    pub scanned: usize,
}

impl ScanProfile {
    /// The stop the profiles run toward.
    pub fn target(&self) -> NodeId {
        self.target
    }

    /// The earliest arrival at the target leaving `stop` at or after `at`,
    /// with the entry that achieves it: a scan from the back of the profile,
    /// where the latest-inserted, earliest-departing pairs are.
    fn evaluate(&self, stop: NodeId, at: Time) -> (Time, u32) {
        for &e in self.profiles[stop as usize].iter().rev() {
            let entry = self.entries[e as usize];
            if entry.dep >= at {
                return (entry.arr, e);
            }
        }
        (UNREACHABLE, NONE)
    }

    /// Insert `(dep, arr)` at `stop` unless a pair there dominates it,
    /// dropping the pairs it dominates — and, with `prune`, unless the
    /// source's own profile already dominates it.
    fn offer(
        &mut self,
        prune: Option<NodeId>,
        stop: NodeId,
        dep: Time,
        arr: Time,
        conn: u32,
        walked: bool,
    ) {
        if let Some(source) = prune {
            if source != stop && self.evaluate(source, dep).0 <= arr {
                return;
            }
        }
        let list = &mut self.profiles[stop as usize];
        // Pairs are in decreasing order of departure; a footpath-shifted pair
        // may land short of the back, so find its place from there.
        let mut at = list.len();
        while at > 0 && self.entries[list[at - 1] as usize].dep < dep {
            at -= 1;
        }
        // The pair just before departs no earlier: if it also arrives no
        // later, it dominates. If it departs at the same moment and arrives
        // later, the new pair dominates it instead — so step back onto it and
        // let the sweep below carry it off with the rest.
        if at > 0 {
            let before = self.entries[list[at - 1] as usize];
            if before.arr <= arr {
                return;
            }
            if before.dep == dep {
                at -= 1;
            }
        }
        // Everything from here on departs no later and, where it also arrives
        // no earlier, is dominated — and those are contiguous.
        let mut dropped = at;
        while dropped < list.len() && self.entries[list[dropped] as usize].arr >= arr {
            dropped += 1;
        }
        list.drain(at..dropped);
        self.entries.push(Entry {
            dep,
            arr,
            conn,
            walked,
        });
        list.insert(at, (self.entries.len() - 1) as u32);
    }
}
