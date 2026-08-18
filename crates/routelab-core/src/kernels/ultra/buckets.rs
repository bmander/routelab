//! Bucket-CH: the one-to-many searches an initial and a final transfer are.
//!
//! ULTRA's shortcuts cover the *intermediate* transfers — the walk between two
//! vehicles. The walks at either end of a journey are not shortcuts and cannot
//! be: one endpoint of each is the query's own source or target, which
//! preprocessing has never seen. §4.1 of the paper says what to do instead:
//!
//! > Therefore, initial and final transfers can be explored with two additional
//! > one-to-many queries on the original transfer graph: a forward query to
//! > compute distances from `s` to all stops, and a backward query for the
//! > distances from all stops to `t`. ULTRA uses Bucket-CH for this task, as it
//! > is one of the fastest known one-to-many algorithms.
//!
//! A plain Dijkstra answers the same question and is what this kernel used to
//! do. It is also the reason a query over a city's pavement cost most of a
//! second: two searches of a half-million-vertex graph, per query, to find a
//! walk that is usually a few hundred metres.
//!
//! ## How a bucket answers it
//!
//! Contract the transfer graph once. Every shortest path in a contraction
//! hierarchy goes *up* then *down*, so a distance from `s` to a stop `w`
//! splits at the highest vertex `v` on the way: `dist(s,w) = dist↑(s,v) +
//! dist↓(v,w)`. The second half does not depend on the query, so it is
//! precomputed: run an upward search from every stop and file the result at
//! each vertex it settles. The **forward bucket** of `v` holds one entry per
//! stop that can be reached from `v` on the way down.
//!
//! A query then climbs from `s` — a few hundred vertices, however far away the
//! target is, because a hierarchy's search space is a property of the ranks
//! rather than of the distance — and reads the buckets of what it settled.
//! Backward buckets answer the mirror question for the final transfer.
//!
//! ## The pruning is the paper's, and it is what makes local queries quick
//!
//! Algorithm 2 starts with a bidirectional query for the *direct* walk from
//! `s` to `t`, and then keeps only the transfers that beat it:
//!
//! > An initial transfer to a stop `v` cannot be part of an optimal journey if
//! > `τtra(s,v) ≥ τtra(s,t)`, since any journey containing the initial transfer
//! > will be dominated by the direct transfer from `s` to `t`.
//!
//! Entries are sorted by distance when the buckets are built, so a scan stops
//! at the first one that cannot beat the direct walk rather than reading the
//! whole bucket — the paper's own refinement, and the reason a short query
//! touches a handful of stops instead of every stop in the city.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::super::contraction::{ContractionHierarchy, Ordering};
use crate::model::graph::{EdgeId, Graph, GraphError, NodeId, Weight, UNREACHABLE};
use crate::model::search::SearchError;
use crate::util::progress::Progress;

/// One end of a precomputed half-path: a stop, and what it costs between that
/// stop and the vertex whose bucket holds this.
#[derive(Debug, Clone, Copy)]
struct Entry {
    /// Index into [`Buckets::stops`], not a node id — buckets are read far
    /// more often than they are built, and an index addresses the answer array
    /// directly.
    stop: u32,
    distance: Weight,
}

/// Buckets in CSR, the way every other table here is laid out: one flat run of
/// entries and an offset per vertex, rather than a vector of vectors whose
/// headers alone would outweigh the data on a half-million-vertex graph.
#[derive(Debug, Clone)]
struct Filed {
    offsets: Vec<u32>,
    entries: Vec<Entry>,
}

impl Filed {
    fn of(&self, node: NodeId) -> &[Entry] {
        let start = self.offsets[node as usize] as usize;
        let stop = self.offsets[node as usize + 1] as usize;
        &self.entries[start..stop]
    }

    /// File `(vertex, stop, distance)` triples, sorted within each vertex by
    /// distance so a scan can stop early.
    fn build(num_nodes: usize, mut found: Vec<(NodeId, u32, Weight)>) -> Self {
        let mut offsets = vec![0u32; num_nodes + 1];
        for &(node, _, _) in &found {
            offsets[node as usize + 1] += 1;
        }
        for index in 0..num_nodes {
            offsets[index + 1] += offsets[index];
        }
        let mut entries = vec![
            Entry {
                stop: 0,
                distance: 0
            };
            found.len()
        ];
        let mut cursor = offsets.clone();
        for (node, stop, distance) in found.drain(..) {
            let at = &mut cursor[node as usize];
            entries[*at as usize] = Entry { stop, distance };
            *at += 1;
        }
        for index in 0..num_nodes {
            let run = offsets[index] as usize..offsets[index + 1] as usize;
            entries[run].sort_unstable_by_key(|entry| entry.distance);
        }
        Filed { offsets, entries }
    }

    fn footprint(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<u32>()
            + self.entries.len() * std::mem::size_of::<Entry>()
    }
}

/// What a query needs to know about the walks at either end of a journey.
#[derive(Debug, Clone)]
pub struct Endpoints {
    /// The direct walk from source to target, if one exists — the journey
    /// every ride has to beat, and the bound both scans were pruned by.
    pub direct: Option<Weight>,
    /// Reaching each stop from the source, indexed as [`Buckets::stops`] is.
    /// `UNREACHABLE` where no walk beats `direct`.
    pub to_stop: Vec<Weight>,
    /// Reaching the target from each stop, indexed the same way.
    pub from_stop: Vec<Weight>,
}

/// A contracted transfer graph, and one bucket per vertex in each direction.
#[derive(Debug, Clone)]
pub struct Buckets {
    hierarchy: ContractionHierarchy,
    /// The stops, ascending. A bucket entry names one by index into this.
    stops: Vec<NodeId>,
    forward: Filed,
    backward: Filed,
}

impl Buckets {
    /// Contract `graph` and file a bucket entry for every vertex each stop's
    /// upward search settles.
    ///
    /// Two searches per stop, which on a city's pavement is the smaller half
    /// of what this costs; the contraction itself is the rest.
    pub fn build(graph: &Graph, stops: &[NodeId], progress: &Progress) -> Result<Self, GraphError> {
        let hierarchy =
            ContractionHierarchy::build_reporting(graph, Ordering::default(), progress)?;
        let mut stops = stops.to_vec();
        stops.sort_unstable();
        stops.dedup();

        progress.expect("filing buckets", stops.len() as u64);
        let mut forward = Vec::new();
        let mut backward = Vec::new();
        // One scratch pair reused across every search: a stop's search space is
        // a few hundred vertices and the graph is half a million, so a result
        // sized to the graph would spend all its time being cleared.
        let mut scratch = Scratch::new(graph.num_nodes());
        for (index, &stop) in stops.iter().enumerate() {
            // Down-halves: the search over `downward` from a stop climbs, and
            // its cost at `v` is what the descent from `v` to this stop costs.
            scratch.search(hierarchy.downward(), stop, &mut |node, distance| {
                forward.push((node, index as u32, distance))
            });
            // Up-halves, for the mirror question a final transfer asks.
            scratch.search(hierarchy.upward(), stop, &mut |node, distance| {
                backward.push((node, index as u32, distance))
            });
            progress.step();
        }

        let num_nodes = graph.num_nodes();
        Ok(Buckets {
            forward: Filed::build(num_nodes, forward),
            backward: Filed::build(num_nodes, backward),
            hierarchy,
            stops,
        })
    }

    /// The stops these buckets were built for, ascending.
    pub fn stops(&self) -> &[NodeId] {
        &self.stops
    }

    /// Algorithm 2, lines 1–3: the direct walk, and the initial and final
    /// transfers worth having.
    ///
    /// `sources` are `(vertex, head start)` pairs, so a query standing at
    /// several places each already some seconds along is one search rather
    /// than several — the same shape every other technique here takes.
    pub fn endpoints(
        &self,
        sources: &[(NodeId, Weight)],
        target: NodeId,
    ) -> Result<Endpoints, SearchError> {
        let search = self.hierarchy.query(sources, target)?;
        // Everything is measured against the direct walk. Without one there is
        // no bound and every reachable stop is a candidate, which is the same
        // answer the unbounded search gave and the case the paper's pruning
        // has nothing to say about.
        let limit = search.distance.unwrap_or(UNREACHABLE);

        let mut to_stop = vec![UNREACHABLE; self.stops.len()];
        for &node in &search.forward.order {
            let Some(climbed) = search.forward.cost(node) else {
                continue;
            };
            if climbed >= limit {
                continue;
            }
            for entry in self.forward.of(node) {
                let total = climbed.saturating_add(entry.distance);
                // Sorted by distance, so the first entry that cannot beat the
                // direct walk means none of the rest can either.
                if total >= limit {
                    break;
                }
                let best = &mut to_stop[entry.stop as usize];
                if total < *best {
                    *best = total;
                }
            }
        }

        let mut from_stop = vec![UNREACHABLE; self.stops.len()];
        for &node in &search.backward.order {
            let Some(descended) = search.backward.cost(node) else {
                continue;
            };
            if descended >= limit {
                continue;
            }
            for entry in self.backward.of(node) {
                let total = descended.saturating_add(entry.distance);
                if total >= limit {
                    break;
                }
                let best = &mut from_stop[entry.stop as usize];
                if total < *best {
                    *best = total;
                }
            }
        }

        Ok(Endpoints {
            direct: search.distance,
            to_stop,
            from_stop,
        })
    }

    /// Where the cheapest walk to `to` started, and the edges it crossed, in
    /// the transfer graph's own numbering.
    ///
    /// A distance is all a query needs to route; a *journey* needs the path,
    /// and this is where it comes from — one hierarchy query, unpacked back
    /// into original edges, asked once per walk that survived into the answer
    /// rather than once per stop that might have.
    ///
    /// Takes sources rather than a source because the walk out of a query's
    /// origin may have several to choose between, each already some seconds
    /// along, and which one it left from is part of the answer.
    pub fn path(
        &self,
        sources: &[(NodeId, Weight)],
        to: NodeId,
    ) -> Result<Option<(NodeId, Vec<EdgeId>)>, SearchError> {
        let search = self.hierarchy.query(sources, to)?;
        let Some(edges) = self.hierarchy.unpack(&search) else {
            return Ok(None);
        };
        // `origin` reads back down the forward half to whichever source the
        // meeting node was reached from; with one source it is that source.
        let from = search.origin().unwrap_or(to);
        Ok(Some((from, edges)))
    }

    pub fn footprint(&self) -> usize {
        self.hierarchy.footprint()
            + self.forward.footprint()
            + self.backward.footprint()
            + self.stops.len() * std::mem::size_of::<NodeId>()
    }

    /// Bucket entries filed, both directions — the number that says what the
    /// buckets cost against what a query saves.
    pub fn num_entries(&self) -> usize {
        self.forward.entries.len() + self.backward.entries.len()
    }
}

/// A Dijkstra over one climbing graph, reusing its arrays between searches.
///
/// Dense costs with a dirty list rather than a hash map: a stop's search space
/// is small, but there are thousands of stops, and clearing only what was
/// touched is what keeps the build linear in the work actually done.
struct Scratch {
    cost: Vec<Weight>,
    touched: Vec<NodeId>,
    queue: BinaryHeap<Reverse<(Weight, NodeId)>>,
}

impl Scratch {
    fn new(num_nodes: usize) -> Self {
        Scratch {
            cost: vec![UNREACHABLE; num_nodes],
            touched: Vec::new(),
            queue: BinaryHeap::new(),
        }
    }

    /// Settle everything reachable from `source`, reporting each vertex once
    /// with its final distance.
    fn search(&mut self, graph: &Graph, source: NodeId, found: &mut dyn FnMut(NodeId, Weight)) {
        for &node in &self.touched {
            self.cost[node as usize] = UNREACHABLE;
        }
        self.touched.clear();
        self.queue.clear();

        self.cost[source as usize] = 0;
        self.touched.push(source);
        self.queue.push(Reverse((0, source)));

        while let Some(Reverse((cost, node))) = self.queue.pop() {
            if cost > self.cost[node as usize] {
                continue;
            }
            found(node, cost);
            for edge in graph.out_edges(node) {
                let head = graph.head(edge);
                let next = cost.saturating_add(graph.weight(edge));
                if next < self.cost[head as usize] {
                    if self.cost[head as usize] == UNREACHABLE {
                        self.touched.push(head);
                    }
                    self.cost[head as usize] = next;
                    self.queue.push(Reverse((next, head)));
                }
            }
        }
    }
}
