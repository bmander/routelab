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

/// The best finish found so far: which seed reached which event, over which
/// hub, the walk from there to the target if it ends on foot, and when.
struct Found {
    seed: usize,
    event: u32,
    /// The hub the seed and the event share — kept from the probe that found
    /// it, since unpacking the journey needs it and the intersection has
    /// already been paid for.
    hub: u32,
    walk_in: Option<Walk>,
    arrives: Time,
}

/// A journey the profile sweep found: the entry on either side that reaches
/// the hub they share.
#[derive(Clone, Copy)]
struct Candidate {
    leaving: Reach,
    arriving: Reach,
}

impl Candidate {
    /// The hub the journey passes through — the same on both sides, which is
    /// what made it a candidate.
    fn hub(&self) -> u32 {
        self.leaving.hub
    }

    /// The latest moment the rider can leave the query's origin.
    fn departs(&self) -> Time {
        self.leaving.shifted
    }

    /// When they are at the query's target.
    fn arrives(&self) -> Time {
        self.arriving.shifted
    }
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

        let (seeds, standing) = self.enter(sources, to);

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
                    hub
                };
                // Reachability along a stop's events is monotone — a later
                // one is reached from an earlier one by waiting — so the
                // first reachable event is a binary search. `high` is always
                // an event known to be reachable, so the hub that answered
                // for it is the one the winner is unpacked over.
                let Some(mut hub) = reachable(list[hi - 1]) else {
                    continue;
                };
                let (mut low, mut high) = (lo, hi - 1);
                while low < high {
                    let mid = (low + high) / 2;
                    match reachable(list[mid]) {
                        Some(over) => {
                            hub = over;
                            high = mid;
                        }
                        None => low = mid + 1,
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
                    hub,
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
        self.unpack(seed.event, found.hub, found.event, &mut legs);
        if let Some(walk) = found.walk_in {
            legs.extend(self.footpaths.expand(walk).into_iter().map(Leg::Walk));
        }
        Some(Itinerary {
            arrives: found.arrives,
            legs,
            settled: read,
        })
    }

    /// Where a query enters the graph, and the answer that needs no event.
    ///
    /// One seed per source — the first event at it at or after its time —
    /// and one per stop a source can walk to, remembering the walk, since a
    /// walk from where the rider is standing is not an arc of the graph.
    /// Standing at the target already, or walking straight to it, is an
    /// answer with no arrival event at all, so it is carried out separately
    /// and compared at the end.
    fn enter(
        &self,
        sources: &[(NodeId, Time)],
        to: NodeId,
    ) -> (Vec<Seed>, Option<(Time, Option<Walk>)>) {
        let stops = self.num_stops();
        let mut seeds: Vec<Seed> = Vec::new();
        let mut standing: Option<(Time, Option<Walk>)> = None;
        let mut consider_standing = |arrives: Time, walk: Option<Walk>| {
            if standing.is_none_or(|(known, _)| arrives < known) {
                standing = Some((arrives, walk));
            }
        };
        for &(from, at) in sources {
            if from as usize >= stops {
                continue;
            }
            if from == to {
                consider_standing(at, None);
            }
            let onward = std::iter::once((from, 0)).chain(self.footpaths.from(from));
            for (stop, duration) in onward {
                let walk = (stop != from).then(|| Walk {
                    from,
                    to: stop,
                    departs: at,
                    arrives: at.saturating_add(duration),
                });
                let ready = at.saturating_add(duration);
                if stop == to {
                    if let Some(walk) = walk {
                        consider_standing(walk.arrives, Some(walk));
                    }
                    continue;
                }
                if let Some(event) = self.events.first_event_at(stop, ready) {
                    // One seed per event: two sources reaching the same event
                    // enter identically, and the first to claim it keeps it.
                    if !seeds.iter().any(|held| held.event == event) {
                        seeds.push(Seed { event, walk });
                    }
                }
            }
        }
        (seeds, standing)
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
        let mut candidates: Vec<Candidate> = Vec::new();
        while i < forward.len() && j < backward.len() {
            read += 1;
            let (leaving, arriving) = (forward[i], backward[j]);
            match leaving.hub.cmp(&arriving.hub) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if leaving.shifted >= opens && arriving.shifted >= leaving.shifted {
                        candidates.push(Candidate { leaving, arriving });
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
        candidates.sort_unstable_by_key(|candidate| {
            (
                std::cmp::Reverse(candidate.departs()),
                candidate.arrives(),
                candidate.hub(),
            )
        });
        let mut kept: Vec<(Time, Itinerary)> = Vec::new();
        let mut best: Option<Time> = None;
        for candidate in candidates {
            let (dep, arr) = (candidate.departs(), candidate.arrives());
            if best.is_some_and(|known| arr >= known) {
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
            let Candidate { leaving, arriving } = candidate;
            let mut legs = Vec::new();
            if leaving.stop != from {
                let out = Walk {
                    from,
                    to: leaving.stop,
                    departs: dep,
                    arrives: leaving.time,
                };
                legs.extend(self.footpaths.expand(out).into_iter().map(Leg::Walk));
            }
            let start = self
                .events
                .event_at(leaving.stop, leaving.time)
                .expect("a stop label names one of the stop's events");
            let finish = self
                .events
                .event_at(arriving.stop, arriving.time)
                .expect("a stop label names one of the stop's events");
            self.unpack(start, candidate.hub(), finish, &mut legs);
            if arriving.stop != to {
                let inward = Walk {
                    from: arriving.stop,
                    to,
                    departs: arriving.time,
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

/// One entry of a superlabel: a hub, when the query can leave or arrive by
/// way of it, and which stop and stop-label time that came from.
///
/// The stop and its own time are provenance rather than arithmetic: the
/// journey is rebuilt by looking up the *event* at `(stop, time)`, which is
/// only honest if the pair came out of a stop label rather than being
/// re-derived from `shifted` and a walk.
#[derive(Clone, Copy)]
struct Reach {
    hub: u32,
    /// The moment at the query's own origin or target, the walk applied.
    shifted: Time,
    stop: NodeId,
    time: Time,
}

/// A position in one stop label: its hubs and times, the stop, the walk to
/// or from it, and how far along it the merge has read.
struct Cursor<'a> {
    hubs: &'a [u32],
    times: &'a [Time],
    stop: NodeId,
    walk: Time,
    at: usize,
}

/// Merge several stop labels, each sorted by hub, into one sorted by hub with
/// one [`Reach`] per hub, the entry kept being the one `better` prefers.
/// `skip` drops leading hubs a query cannot use; `shift` moves a time by the
/// walk to or from the stop, or drops the entry.
fn merge_labels<'a>(
    labels: impl Iterator<Item = (&'a [u32], &'a [Time], NodeId, Time)>,
    skip: impl Fn(u32) -> bool,
    shift: impl Fn(Time, Time) -> Option<Time>,
    better: impl Fn(Time, Time) -> bool,
) -> Vec<Reach> {
    // Cursors into each label, and a heap on their current hub.
    let mut cursors: Vec<Cursor<'a>> = labels
        .map(|(hubs, times, stop, walk)| Cursor {
            hubs,
            times,
            stop,
            walk,
            at: hubs.partition_point(|&h| skip(h)),
        })
        .filter(|cursor| cursor.at < cursor.hubs.len())
        .collect();
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = cursors
        .iter()
        .enumerate()
        .map(|(c, cursor)| std::cmp::Reverse((cursor.hubs[cursor.at], c)))
        .collect();
    let left: usize = cursors
        .iter()
        .map(|cursor| cursor.hubs.len() - cursor.at)
        .sum();
    let mut merged: Vec<Reach> = Vec::with_capacity(left);
    while let Some(std::cmp::Reverse((hub, c))) = heap.pop() {
        let cursor = &mut cursors[c];
        let time = cursor.times[cursor.at];
        if let Some(shifted) = shift(time, cursor.walk) {
            let reach = Reach {
                hub,
                shifted,
                stop: cursor.stop,
                time,
            };
            match merged.last_mut() {
                Some(last) if last.hub == hub => {
                    if better(shifted, last.shifted) {
                        *last = reach;
                    }
                }
                _ => merged.push(reach),
            }
        }
        cursor.at += 1;
        if cursor.at < cursor.hubs.len() {
            heap.push(std::cmp::Reverse((cursor.hubs[cursor.at], c)));
        }
    }
    merged
}
