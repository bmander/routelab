//! The two questions, answered from the labels: earliest arrival from event
//! labels (§3), a profile from stop labels (§4), walks at either end folded
//! in as superlabels (§5), and every answer unpacked to a journey.

use crate::model::graph::NodeId;
use crate::model::timetable::{Itinerary, Leg, Time, Walk};

use super::events::{WAIT, WALK};
use super::labels::meet;
use super::PublicTransitLabeling;

/// Where a query enters the event graph: an event, and the walk that got
/// there from a source, if one did.
struct Seed {
    event: u32,
    walk: Option<Walk>,
}

/// The best finish found so far: which seed reached which event, the walk
/// from there to the target if it ends on foot, and when.
struct Found {
    seed: usize,
    event: u32,
    walk_in: Option<Walk>,
    arrives: Time,
}

impl PublicTransitLabeling {
    /// Earliest arrival at `to` from `sources` — each a stop and the time
    /// you are standing there.
    ///
    /// The paper's earliest-arrival query with event labels, pruning and
    /// binary search, run from every seed the sources give — the first event
    /// at each source at or after its time, and at each stop a source can
    /// walk to — and toward every stop the target can be walked to from,
    /// since a journey that ends on foot arrives at a stop plus a walk, not
    /// at an event. `settled` on the answer is label entries read: the
    /// paper's own measure of a query's work.
    pub fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        let stops = self.num_stops();
        if to as usize >= stops {
            return None;
        }

        // Standing at the target already, or walking straight to it: an
        // answer with no event to find, compared at the end.
        let mut standing: Option<(Time, Option<Walk>)> = None;
        let mut consider_standing = |arrives: Time, walk: Option<Walk>| {
            if standing.is_none_or(|(known, _)| arrives < known) {
                standing = Some((arrives, walk));
            }
        };
        let mut seeds: Vec<Seed> = Vec::new();
        let seed = |stop: NodeId, ready: Time, walk: Option<Walk>, seeds: &mut Vec<Seed>| {
            if let Some(event) = self.events.first_event_at(stop, ready) {
                // One seed per event: two sources reaching the same event
                // enter identically, and the first to claim it keeps it.
                if !seeds.iter().any(|s| s.event == event) {
                    seeds.push(Seed { event, walk });
                }
            }
        };
        for &(from, at) in sources {
            if from as usize >= stops {
                continue;
            }
            if from == to {
                consider_standing(at, None);
            }
            seed(from, at, None, &mut seeds);
            for (next, duration) in self.footpaths.from(from) {
                let walk = Walk {
                    from,
                    to: next,
                    departs: at,
                    arrives: at.saturating_add(duration),
                };
                if next == to {
                    consider_standing(walk.arrives, Some(walk));
                } else {
                    seed(next, walk.arrives, Some(walk), &mut seeds);
                }
            }
        }

        // Where a journey can finish: at the target, or at any stop that
        // walks to it, plus the walk.
        // Shortest walk first, so the best so far tightens early and the
        // rest are pruned harder.
        let mut finishes: Vec<(NodeId, Time)> = std::iter::once((to, 0))
            .chain(self.incoming.from(to))
            .collect();
        finishes.sort_unstable_by_key(|&(_, walk)| walk);

        let mut best: Option<Found> = None;
        let mut read = 0usize;
        for (index, seed) in seeds.iter().enumerate() {
            let (forward, _) = self.labels.forward(seed.event);
            let since = self.events.time(seed.event);
            for &(stop, walk) in &finishes {
                let list = self.events.events_at(stop);
                // Prune: nothing before the seed is reachable, and nothing
                // that would not beat the best so far is worth probing.
                let lo = list.partition_point(|&e| self.events.time(e) < since);
                let hi = match &best {
                    Some(found) => list.partition_point(|&e| {
                        self.events.time(e).saturating_add(walk) < found.arrives
                    }),
                    None => list.len(),
                };
                if lo >= hi {
                    continue;
                }
                let mut reachable = |event: u32| {
                    let (hub, touched) = meet(forward, self.labels.backward(event).0);
                    read += touched;
                    hub.is_some()
                };
                // Reachability along a stop's events is monotone — a later
                // one is reached from an earlier one by waiting — so the
                // first reachable event is a binary search.
                if !reachable(list[hi - 1]) {
                    continue;
                }
                let (mut low, mut high) = (lo, hi - 1);
                while low < high {
                    let mid = (low + high) / 2;
                    if reachable(list[mid]) {
                        high = mid;
                    } else {
                        low = mid + 1;
                    }
                }
                let event = list[low];
                let arrives = self.events.time(event).saturating_add(walk);
                let walk_in = (stop != to).then(|| Walk {
                    from: stop,
                    to,
                    departs: self.events.time(event),
                    arrives,
                });
                best = Some(Found {
                    seed: index,
                    event,
                    walk_in,
                    arrives,
                });
            }
        }

        // Standing still wins ties: nothing ridden can arrive sooner than
        // not having to.
        if let Some((arrives, walk)) = standing {
            if best.as_ref().is_none_or(|found| arrives <= found.arrives) {
                return Some(self.standing_still(arrives, walk, read));
            }
        }
        let found = best?;
        let seed = &seeds[found.seed];
        let mut legs = Vec::new();
        if let Some(walk) = seed.walk {
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        let (hub, touched) = meet(
            self.labels.forward(seed.event).0,
            self.labels.backward(found.event).0,
        );
        read += touched;
        let hub = hub.expect("a reachable event shares a hub with the seed");
        self.unpack(seed.event, hub, found.event, &mut legs);
        if let Some(walk) = found.walk_in {
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        Some(Itinerary {
            arrives: found.arrives,
            legs,
            settled: read,
        })
    }

    /// Already there, or a walk away: an answer with no event to find.
    fn standing_still(&self, arrives: Time, walk: Option<Walk>, read: usize) -> Itinerary {
        Itinerary {
            arrives,
            legs: walk
                .map(|walk| {
                    self.footpaths
                        .expand(walk)
                        .into_iter()
                        .map(Leg::Walk)
                        .collect()
                })
                .unwrap_or_default(),
            settled: read,
        }
    }

    /// The path from `from` up to `hub` and down to `to`, read off the label
    /// pointers, as legs: a connection arc is a ride, a foot arc a walk
    /// leaving when the rider stood at its tail, a waiting arc nothing.
    fn unpack(&self, from: u32, hub: u32, to: u32, legs: &mut Vec<Leg>) {
        let mut arcs: Vec<(u32, u32)> = Vec::new();
        let mut here = from;
        while here != hub {
            let next = self
                .labels
                .next_toward(here, hub)
                .expect("a forward label points the way to its hub");
            arcs.push((here, next));
            here = next;
        }
        let mut descent: Vec<(u32, u32)> = Vec::new();
        let mut here = to;
        while here != hub {
            let previous = self
                .labels
                .previous_from(here, hub)
                .expect("a backward label points the way from its hub");
            descent.push((previous, here));
            here = previous;
        }
        arcs.extend(descent.into_iter().rev());

        for (tail, head) in arcs {
            let kind = self
                .events
                .arc_kind(tail, head)
                .expect("a label pointer follows an arc");
            match kind {
                WAIT => {}
                WALK => {
                    let (from, to) = (self.events.stop(tail), self.events.stop(head));
                    let departs = self.events.time(tail);
                    let duration = self.footpaths.duration(from, to).unwrap_or(0);
                    let walk = Walk {
                        from,
                        to,
                        departs,
                        arrives: departs.saturating_add(duration),
                    };
                    legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
                }
                index => legs.push(Leg::Ride(self.connections[index as usize])),
            }
        }
    }

    // --- Profile (§4) --------------------------------------------------------

    /// Every journey worth leaving `from` on for `to` within `[opens,
    /// closes]`, as `(departure, itinerary)`, earliest first: one per
    /// Pareto-optimal (leave at, arrive by) pair, a departure being the
    /// latest moment that still makes its arrival. Pairs the direct walk
    /// from `from` to `to` would beat are left out, as they are for CSA.
    ///
    /// The paper's stop-label profile query, over the superlabels of `from`
    /// with the stops it walks to and of `to` with the stops that walk to
    /// it.
    pub fn profile(
        &self,
        from: NodeId,
        to: NodeId,
        opens: Time,
        closes: Time,
    ) -> Vec<(Time, Itinerary)> {
        let stops = self.num_stops();
        if from as usize >= stops || to as usize >= stops || from == to || closes < opens {
            return Vec::new();
        }

        // The superlabels (§5), assembled during the query as the paper
        // suggests: the origin's forward stop label merged with those of the
        // stops it walks to, each departure shifted earlier by the walk, and
        // for each hub the latest; the target's backward label merged with
        // those of the stops that walk to it, arrivals shifted later, and for
        // each hub the earliest. Only hubs from `opens` on: hubs are numbered
        // by time, and a journey leaving after `opens` passes only through
        // events after it. Departures after `closes` are kept, since they may
        // dominate ones inside the window.
        let ahead: Vec<(NodeId, Time)> = std::iter::once((from, 0))
            .chain(self.footpaths.from(from))
            .collect();
        let forward = merge_labels(
            ahead.iter().map(|&(stop, walk)| {
                let (hubs, times) = self.stop_labels.forward(stop);
                (hubs, times, stop, walk)
            }),
            |hub| self.events.time(hub) < opens,
            |time, walk| (time >= walk).then(|| time - walk),
            |a, b| a > b,
        );
        let behind: Vec<(NodeId, Time)> = std::iter::once((to, 0))
            .chain(self.incoming.from(to))
            .collect();
        let backward = merge_labels(
            behind.iter().map(|&(stop, walk)| {
                let (hubs, times) = self.stop_labels.backward(stop);
                (hubs, times, stop, walk)
            }),
            |hub| self.events.time(hub) < opens,
            |time, walk| Some(time.saturating_add(walk)),
            |a, b| a < b,
        );

        // The coordinated sweep over the two superlabels.
        let (mut i, mut j) = (0, 0);
        let mut read = 0usize;
        let mut candidates: Vec<(Time, Time, u32, NodeId, Time, NodeId, Time)> = Vec::new();
        while i < forward.len() && j < backward.len() {
            read += 1;
            let (hf, dep, p, ptime) = forward[i];
            let (hb, arr, q, qtime) = backward[j];
            match hf.cmp(&hb) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if dep >= opens && arr >= dep {
                        candidates.push((dep, arr, hf, p, ptime, q, qtime));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        // Keep the tight pairs: latest departure first, and only what
        // arrives strictly earlier than everything leaving later. What the
        // direct walk beats is not worth listing.
        let walk = self.footpaths.duration(from, to);
        candidates.sort_unstable_by_key(|&(dep, arr, hub, ..)| (std::cmp::Reverse(dep), arr, hub));
        let mut kept: Vec<(Time, Itinerary)> = Vec::new();
        let mut best: Option<Time> = None;
        for (dep, arr, hub, p, ptime, q, qtime) in candidates {
            if best.is_some_and(|b| arr >= b) {
                continue;
            }
            if walk.is_some_and(|w| arr >= dep.saturating_add(w)) {
                continue;
            }
            best = Some(arr);
            // The window trims after dominance, not before: a pair leaving
            // inside it that a later departure past its close would beat is
            // not a step of the profile, only a moment before one.
            if dep > closes {
                continue;
            }
            let mut legs = Vec::new();
            if p != from {
                let out = Walk {
                    from,
                    to: p,
                    departs: dep,
                    arrives: ptime,
                };
                legs.extend(self.footpaths.expand(out).into_iter().map(Leg::Walk));
            }
            let start = self
                .events
                .event_at(p, ptime)
                .expect("a stop label names one of the stop's events");
            let finish = self
                .events
                .event_at(q, qtime)
                .expect("a stop label names one of the stop's events");
            self.unpack(start, hub, finish, &mut legs);
            if q != to {
                let inward = Walk {
                    from: q,
                    to,
                    departs: qtime,
                    arrives: arr,
                };
                legs.extend(self.footpaths.expand(inward).into_iter().map(Leg::Walk));
            }
            kept.push((
                dep,
                Itinerary {
                    arrives: arr,
                    legs,
                    settled: read,
                },
            ));
        }
        kept.reverse();
        kept
    }
}

/// A position in one stop label: its hubs and times, the stop, the walk to
/// or from it, and how far along it the merge has read.
type Cursor<'a> = (&'a [u32], &'a [Time], NodeId, Time, usize);

/// Merge several stop labels, each sorted by hub, into one sorted by hub with
/// one entry per hub: `(hub, shifted time, stop, time at that stop)`, the
/// entry kept being the one `better` prefers. `skip` drops leading hubs a
/// query cannot use; `shift` moves a time by the walk to or from the stop,
/// or drops the entry.
fn merge_labels<'a>(
    labels: impl Iterator<Item = (&'a [u32], &'a [Time], NodeId, Time)>,
    skip: impl Fn(u32) -> bool,
    shift: impl Fn(Time, Time) -> Option<Time>,
    better: impl Fn(Time, Time) -> bool,
) -> Vec<(u32, Time, NodeId, Time)> {
    // Cursors into each label, and a heap on their current hub.
    let mut cursors: Vec<Cursor<'a>> = labels
        .map(|(hubs, times, stop, walk)| {
            let start = hubs.partition_point(|&h| skip(h));
            (hubs, times, stop, walk, start)
        })
        .filter(|(hubs, _, _, _, start)| *start < hubs.len())
        .collect();
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = cursors
        .iter()
        .enumerate()
        .map(|(c, (hubs, _, _, _, start))| std::cmp::Reverse((hubs[*start], c)))
        .collect();
    let mut merged: Vec<(u32, Time, NodeId, Time)> = Vec::new();
    while let Some(std::cmp::Reverse((hub, c))) = heap.pop() {
        let (hubs, times, stop, walk, index) = &mut cursors[c];
        if let Some(shifted) = shift(times[*index], *walk) {
            match merged.last_mut() {
                Some(last) if last.0 == hub => {
                    if better(shifted, last.1) {
                        *last = (hub, shifted, *stop, times[*index]);
                    }
                }
                _ => merged.push((hub, shifted, *stop, times[*index])),
            }
        }
        *index += 1;
        if *index < hubs.len() {
            heap.push(std::cmp::Reverse((hubs[*index], c)));
        }
    }
    merged
}
