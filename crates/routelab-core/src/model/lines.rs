//! A timetable as lines and the trips along them.
//!
//! The structure a timetable has that a graph throws away: **lines** — ordered
//! stop sequences — and the **trips** that run along each, in departure order
//! and never overtaking one another. Delling, Pajor & Werneck call them
//! *routes* and RAPTOR scans them round by round; Witt calls them *lines* and
//! the trip-based search transfers between their trips. Two kernels read the
//! same layout, which is the test for living here rather than beside one of
//! them; neither decides anything about how it is built.
//!
//! Built once from a [`Timetable`]. A feed's trip whose chain of connections
//! was broken — a hop that was not boardable, a hand-made timetable that never
//! joined up — is several trips as far as a rider is concerned, and a stop
//! sequence whose trips overtake one another is split into as many lines as
//! it takes for none to, greedily. That split is what lets "the earliest trip
//! you can catch here" be a binary search, and lets an earlier trip never
//! arrive later downstream — both papers rest on it.

use crate::model::graph::NodeId;
use crate::model::timetable::{Connection, Time, Timetable};

/// When a trip reaches a stop and when it leaves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopTime {
    pub arrival: Time,
    pub departure: Time,
}

/// Lines, the trips along each in departure order, and a time per
/// `(trip, position)`.
///
/// Trips are numbered densely across lines, so line `r`'s trips are a
/// contiguous range and a trip's line is a lookup; positions along a line
/// count from zero, and a stop a line revisits occupies one position per visit.
#[derive(Debug, Clone)]
pub struct Lines {
    stops: usize,
    /// Line `r`'s stop sequence is `line_stops[line_stops_start[r]..line_stops_start[r+1]]`.
    line_stops_start: Vec<u32>,
    line_stops: Vec<NodeId>,
    /// Line `r`'s trips are `line_trips_start[r]..line_trips_start[r+1]`, in
    /// departure order and never overtaking one another.
    line_trips_start: Vec<u32>,
    /// The line each trip runs along.
    trip_line: Vec<u32>,
    /// Trip `t` at position `i` is `stop_times[trip_times_start[t] + i]`.
    trip_times_start: Vec<u32>,
    stop_times: Vec<StopTime>,
    /// The `Connection::trip` each trip came from — a feed's trip whose chain
    /// of connections was broken becomes several.
    trip_ids: Vec<u32>,
    /// Every `(line, position)` a stop occupies, CSR by stop.
    stop_lines_start: Vec<u32>,
    stop_lines: Vec<(u32, u32)>,
}

/// One trip as the builder assembles it, before lines are known.
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

impl Lines {
    /// Lay `timetable` out as lines and trips.
    pub fn from_timetable(timetable: &Timetable) -> Self {
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

        // Then split each group so that no trip overtakes another: lines are
        // FIFO, which is what lets "the earliest trip you can catch at this
        // stop" be a binary search over departures and lets an earlier trip
        // never arrive later downstream. Greedy: each trip joins the first
        // sub-line whose last trip it does not overtake at any position.
        let mut line_stops_start = vec![0u32];
        let mut line_stops: Vec<NodeId> = Vec::new();
        let mut line_trips_start = vec![0u32];
        let mut trip_line: Vec<u32> = Vec::new();
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
            // Sub-lines as lists of chain indices into `group`.
            let mut sub_lines: Vec<Vec<usize>> = Vec::new();
            for (i, chain) in group.iter().enumerate() {
                let fits = sub_lines.iter().position(|members| {
                    let last = &group[*members.last().unwrap()];
                    last.times
                        .iter()
                        .zip(&chain.times)
                        .all(|(a, b)| a.arrival <= b.arrival && a.departure <= b.departure)
                });
                match fits {
                    Some(r) => sub_lines[r].push(i),
                    None => sub_lines.push(vec![i]),
                }
            }
            for members in sub_lines {
                let line = line_stops_start.len() as u32 - 1;
                line_stops.extend_from_slice(&group[0].stops);
                line_stops_start.push(line_stops.len() as u32);
                for i in members {
                    let chain = &group[i];
                    stop_times.extend_from_slice(&chain.times);
                    trip_times_start.push(stop_times.len() as u32);
                    trip_ids.push(chain.trip);
                    trip_line.push(line);
                }
                line_trips_start.push(trip_ids.len() as u32);
            }
            group_start = group_end;
        }

        // Every (line, position) a stop occupies, CSR by stop.
        let num_lines = line_stops_start.len() - 1;
        let mut stop_lines_start = vec![0u32; stops + 1];
        for r in 0..num_lines {
            for i in line_stops_start[r]..line_stops_start[r + 1] {
                stop_lines_start[line_stops[i as usize] as usize + 1] += 1;
            }
        }
        for s in 0..stops {
            stop_lines_start[s + 1] += stop_lines_start[s];
        }
        let mut fill = stop_lines_start.clone();
        let mut stop_lines = vec![(0u32, 0u32); line_stops.len()];
        for r in 0..num_lines {
            let start = line_stops_start[r];
            for i in start..line_stops_start[r + 1] {
                let stop = line_stops[i as usize] as usize;
                stop_lines[fill[stop] as usize] = (r as u32, i - start);
                fill[stop] += 1;
            }
        }

        Lines {
            stops,
            line_stops_start,
            line_stops,
            line_trips_start,
            trip_line,
            trip_times_start,
            stop_times,
            trip_ids,
            stop_lines_start,
            stop_lines,
        }
    }

    pub fn num_stops(&self) -> usize {
        self.stops
    }

    /// Lines in the papers' sense — distinct stop sequences, split so that no
    /// trip overtakes another. More than a feed's own count of routes.
    pub fn num_lines(&self) -> usize {
        self.line_stops_start.len() - 1
    }

    /// Trips in the papers' sense: one per unbroken chain of connections.
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
            * (self.line_stops_start.len()
                + self.line_stops.len()
                + self.line_trips_start.len()
                + self.trip_line.len()
                + self.trip_times_start.len()
                + self.trip_ids.len()
                + self.stop_lines_start.len())
            + self.stop_times.len() * std::mem::size_of::<StopTime>()
            + self.stop_lines.len() * std::mem::size_of::<(u32, u32)>()
    }

    /// The stops of `line`, in order.
    pub fn stops_of(&self, line: u32) -> &[NodeId] {
        let r = line as usize;
        &self.line_stops[self.line_stops_start[r] as usize..self.line_stops_start[r + 1] as usize]
    }

    /// The trips of `line`, earliest first, as a range of trip numbers.
    pub fn trips_of(&self, line: u32) -> std::ops::Range<u32> {
        let r = line as usize;
        self.line_trips_start[r]..self.line_trips_start[r + 1]
    }

    /// The line `trip` runs along.
    pub fn line_of(&self, trip: u32) -> u32 {
        self.trip_line[trip as usize]
    }

    /// Every `(line, position)` at `stop`.
    pub fn lines_at(&self, stop: NodeId) -> &[(u32, u32)] {
        let s = stop as usize;
        if s + 1 < self.stop_lines_start.len() {
            &self.stop_lines
                [self.stop_lines_start[s] as usize..self.stop_lines_start[s + 1] as usize]
        } else {
            &[]
        }
    }

    /// How many stops `trip` serves.
    pub fn len(&self, trip: u32) -> u32 {
        let t = trip as usize;
        self.trip_times_start[t + 1] - self.trip_times_start[t]
    }

    /// The `Connection::trip` `trip` came from.
    pub fn trip_id(&self, trip: u32) -> u32 {
        self.trip_ids[trip as usize]
    }

    /// Where `trip` at `position` sits in the flat table of stop times — a
    /// dense id for one (trip, position), which is what a per-stop-of-trip
    /// side table indexes by.
    #[inline]
    pub fn slot(&self, trip: u32, position: u32) -> usize {
        (self.trip_times_start[trip as usize] + position) as usize
    }

    /// One more than the largest [`Lines::slot`].
    pub fn num_slots(&self) -> usize {
        self.stop_times.len()
    }

    #[inline]
    pub fn time(&self, trip: u32, position: u32) -> StopTime {
        self.stop_times[self.slot(trip, position)]
    }

    /// Can a rider board `line` at `position`?
    ///
    /// Not at its last stop, where the line ends and nothing leaves. The other
    /// half of the boarding rule to [`Lines::earliest_trip`], and here rather
    /// than in the kernels because it is a fact about the layout: every kernel
    /// reading these lines asks it, and none of them decides it.
    pub fn can_board(&self, line: u32, position: u32) -> bool {
        position as usize + 1 < self.stops_of(line).len()
    }

    /// The earliest trip of `line` leaving `position` at or after `at`,
    /// looking no later than `before` (exclusive) — a bound worth having
    /// because a search already aboard one of a line's trips need only look
    /// at the ones before it.
    ///
    /// A binary search, which is what the non-overtaking split buys: a line's
    /// trips leave every one of its stops in the same order.
    pub fn earliest_trip(
        &self,
        line: u32,
        position: u32,
        at: Time,
        before: Option<u32>,
    ) -> Option<u32> {
        let first = self.line_trips_start[line as usize];
        let end = before.unwrap_or(self.line_trips_start[line as usize + 1]);
        let trips = &self.trip_times_start[first as usize..end as usize];
        let found = trips
            .partition_point(|&start| self.stop_times[(start + position) as usize].departure < at)
            as u32;
        (first + found < end).then_some(first + found)
    }
}
