//! UCCH: contraction for label-constrained routing, with the language still a
//! query input.
//!
//! Dibbelt, Pajor & Wagner, *User-Constrained Multi-Modal Route Planning*
//! (ALENEX 2012) §3. The speedup for [`super::label_constrained`], and the
//! middle of three corners: [`crate::Ultra`] precomputes for minutes and answers
//! in milliseconds, plain LCSPP precomputes nothing and searches the whole
//! graph, and this contracts the streets in seconds.
//!
//! ## Why an ordinary hierarchy will not do
//!
//! Contract the merged graph and a shortcut can span two modes. Its label
//! becomes the concatenation `lbl(e₁)···lbl(e_k)`, and where consecutive labels
//! differ the shortcut has a **modal transfer baked into it**. A query that
//! forbids that transfer cannot use the shortcut — but the path avoiding it may
//! already have been discarded by the witness search, so the answer is wrong.
//! Repairing that with a witness search per automaton state works, and the paper
//! calls it SDCH, but it is slower, adds far more shortcuts, and fixes the
//! automaton at preprocessing time. Which is the one thing LCSPP exists not to
//! do.
//!
//! ## What UCCH does
//!
//! Two rules. **Contract each mode's subnetwork on its own**, so a shortcut
//! never spans a modal boundary. And **never contract a transfer node** — any
//! vertex incident to a link arc — so those stay at the top of the hierarchy and
//! are what is left standing: the core. Witness searches are restricted to the
//! mode's own subnetwork, which errs only towards keeping shortcuts nobody
//! needed; widening them would be unsound, since a shorter path through another
//! mode is no witness if the query forbids that mode.
//!
//! Because the shortcuts leave every core-to-core shortest path inside the core,
//! the preprocessing says nothing about the automaton, and the language stays a
//! query input.
//!
//! Here there is exactly one subnetwork to contract, which is the paper's own
//! *practical variant*: link arcs have transfer nodes at both ends so nothing in
//! them can be contracted, and time-dependent networks are left alone because
//! contracting them is much less effective. So every shortcut is a walk, and no
//! label has to survive the contraction.
//!
//! ## The query
//!
//! Three parts, and §3.3's two observations are what make them cheap.
//!
//! *"By definition [an automaton's] state may only change when traversing link
//! edges. In particular, when searching inside the component, there is never a
//! state transition. Thus, we use the automaton only on the core."* So the two
//! component searches are ordinary one-mode climbs — one distance a vertex, no
//! product — and the automaton is paid for on the core alone.
//!
//! *"Another alternative is not applying bidirectional search on the core at
//! all. The forward search continues regularly, while the backward search does
//! not scan edges incident to core nodes. This approach turns out most
//! effective."* Which also settles the awkward question of searching a
//! time-dependent core backwards: it never happens. The climb from the target
//! stays in the component, which is time-independent by construction, so it
//! yields a walking duration from each core vertex to the target and needs no
//! clock.
//!
//! A walk in the answer may be a shortcut, which is a path and not an arc, so it
//! is told hop by hop by searching the uncontracted subnetwork between its ends —
//! the same obligation [`crate::ContractionHierarchy`] meets by unpacking, and
//! the same way [`crate::Ultra`]'s query tells its walks.

#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{Modes, Multimodal};
use crate::kernels::contraction::{CoreHierarchy, Ordering};
use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, GraphError, NodeId, Weight, UNREACHABLE};
use crate::model::search::SearchOptions;
use crate::model::timetable::{Itinerary, Leg, Time, Walk};
use crate::util::progress::Progress;

/// A hierarchy over one mode's subnetwork, contracted around the vertices where
/// the networks join.
pub struct Ucch {
    hierarchy: CoreHierarchy,
    /// The mode of the subnetwork that was contracted, and so of every shortcut.
    walking: u8,
    /// The core's scalar arcs: what the contraction left standing, plus every
    /// link arc, since those are never contracted.
    core: Graph,
    /// The mode of each core arc, by position in the list it was built from.
    core_labels: Vec<u8>,
}

impl Ucch {
    /// Contract `walkable` around the endpoints of `links`.
    ///
    /// `walkable` is one mode's subnetwork — the pavements — over the whole
    /// numbering, so a stop is the vertex it always was even where no pavement
    /// reaches it. `links` are the arcs that join networks; they are never
    /// contracted, and their endpoints are therefore the core.
    pub fn build(
        walkable: &Graph,
        walking: u8,
        links: &[(NodeId, NodeId, Weight)],
        link_label: u8,
        ordering: Ordering,
        max_degree: f64,
    ) -> Result<Self, GraphError> {
        Self::build_reporting(
            walkable,
            walking,
            links,
            link_label,
            ordering,
            max_degree,
            &Progress::new(),
        )
    }

    /// [`Ucch::build`], counting contracted vertices into `progress`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_reporting(
        walkable: &Graph,
        walking: u8,
        links: &[(NodeId, NodeId, Weight)],
        link_label: u8,
        ordering: Ordering,
        max_degree: f64,
        progress: &Progress,
    ) -> Result<Self, GraphError> {
        // The transfer nodes: every end of every link arc. These are what the
        // contraction may not touch, and what is left when it stops.
        let mut transfer: Vec<NodeId> = Vec::with_capacity(links.len() * 2);
        for &(tail, head, _) in links {
            transfer.push(tail);
            transfer.push(head);
        }
        transfer.sort_unstable();
        transfer.dedup();

        let hierarchy =
            CoreHierarchy::build_reporting(walkable, &transfer, ordering, max_degree, progress)?;

        // The core: what the contraction left of the pavements, and every link
        // arc. A shortcut among the former is a walk, because only one
        // subnetwork was contracted — so no label had to survive contraction.
        let mut arcs: Vec<(NodeId, NodeId, Weight)> = Vec::new();
        let mut core_labels: Vec<u8> = Vec::new();
        let standing = hierarchy.core();
        for tail in 0..standing.num_nodes() as NodeId {
            for edge in standing.out_edges(tail) {
                arcs.push((tail, standing.head(edge), standing.weight(edge)));
                core_labels.push(walking);
            }
        }
        for &(tail, head, weight) in links {
            arcs.push((tail, head, weight));
            core_labels.push(link_label);
        }

        Ok(Ucch {
            core: Graph::from_edges(walkable.num_nodes(), &arcs)?,
            core_labels,
            hierarchy,
            walking,
        })
    }

    /// Is this vertex in the core?
    pub fn is_core(&self, node: NodeId) -> bool {
        self.hierarchy.is_core(node)
    }

    /// How many vertices the core holds — what a query searches instead of the
    /// whole network.
    pub fn num_core(&self) -> usize {
        self.hierarchy.num_core()
    }

    /// Arcs in the core, links included.
    pub fn num_arcs(&self) -> usize {
        self.core_labels.len()
    }

    pub fn footprint(&self) -> usize {
        self.hierarchy.footprint() + self.core_labels.len()
    }

    /// The earliest arrival at `to` by a journey `allowed` admits.
    ///
    /// `network` is the *uncontracted* network — the same one
    /// [`super::label_constrained`] takes. Its schedule is what the core rides;
    /// its scalar graph is read only to tell a shortcut hop by hop.
    pub fn earliest_arrival(
        &self,
        network: &Multimodal<'_>,
        allowed: &Modes,
        sources: &[(NodeId, Time)],
        to: NodeId,
    ) -> Option<Itinerary> {
        let vertices = self.core.num_nodes();
        if to as usize >= vertices || allowed.is_empty() || sources.is_empty() {
            return None;
        }
        let departure = sources.iter().map(|&(_, at)| at).min()?;

        // Walking has to be allowed somewhere for a climb to be usable: in an
        // initial state to walk out of the source, in a final one to walk into
        // the target. The component is one mode, so that is the only question
        // the automaton is asked below the core.
        let leaving = self.within(allowed, allowed.initial);
        let arriving = self.within(allowed, allowed.accepting);

        let access = self.climb(self.hierarchy.upward(), sources, departure, leaving);
        let egress = self.climb(
            self.hierarchy.downward(),
            &[(to, departure)],
            departure,
            arriving,
        );

        // A journey that never needed the core: the two climbs meeting below it.
        let mut best = UNREACHABLE;
        let mut answer: Option<Found> = None;
        for node in 0..vertices {
            let (there, back) = (access[node], egress[node]);
            if there == UNREACHABLE || back == UNREACHABLE {
                continue;
            }
            let arrives = departure.saturating_add(there).saturating_add(back);
            if arrives < best {
                best = arrives;
                answer = Some(Found::Walked(node as NodeId));
            }
        }

        // Or through it. This is where the automaton is paid for.
        let crossing = self.cross(network, allowed, &access, departure);
        let states = allowed.num_states();
        for (node, &back) in egress.iter().enumerate().take(vertices) {
            if back == UNREACHABLE || !self.is_core(node as NodeId) {
                continue;
            }
            for state in 0..states {
                if !allowed.accepts(state) {
                    continue;
                }
                let at = crossing.arrivals[node * states + state];
                if at == UNREACHABLE {
                    continue;
                }
                // Only a final state that walks may use a non-empty egress.
                if back > 0 && allowed.step(state, self.walking) & (1 << state) == 0 {
                    continue;
                }
                let arrives = at.saturating_add(back);
                if arrives < best {
                    best = arrives;
                    answer = Some(Found::Rode {
                        exit: node as NodeId,
                        state,
                    });
                }
            }
        }

        let found = answer?;
        let legs = self.tell(network, sources, departure, &crossing, allowed, found, to);
        Some(Itinerary {
            arrives: best,
            legs,
            settled: crossing.settled,
        })
    }

    /// Does any state in `set` travel `self.walking` without leaving itself?
    fn within(&self, allowed: &Modes, set: u32) -> bool {
        (0..allowed.num_states()).any(|state| {
            set & (1 << state) != 0 && allowed.step(state, self.walking) & (1 << state) != 0
        })
    }

    /// A one-mode climb through the component, as a duration from `departure`.
    ///
    /// `allowed` says whether the mode may be walked at all in the states this
    /// climb belongs to; where it may not, only standing still is free.
    fn climb(
        &self,
        graph: &Graph,
        sources: &[(NodeId, Time)],
        departure: Time,
        allowed: bool,
    ) -> Vec<Time> {
        let seeds: Vec<(NodeId, Weight)> = sources
            .iter()
            .filter(|&&(node, _)| (node as usize) < graph.num_nodes())
            .map(|&(node, at)| (node, at.saturating_sub(departure)))
            .collect();
        if !allowed {
            let mut standing = vec![UNREACHABLE; graph.num_nodes()];
            for (node, cost) in seeds {
                standing[node as usize] = standing[node as usize].min(cost);
            }
            return standing;
        }
        dijkstra(graph, &seeds, &SearchOptions::default())
            .map(|found| found.costs)
            .unwrap_or_else(|_| vec![UNREACHABLE; graph.num_nodes()])
    }

    /// The product search over the core: Dijkstra on `(core vertex, state)`.
    fn cross(
        &self,
        network: &Multimodal<'_>,
        allowed: &Modes,
        access: &[Time],
        departure: Time,
    ) -> Crossing {
        let states = allowed.num_states();
        let vertices = self.core.num_nodes();
        let mut crossing = Crossing {
            arrivals: vec![UNREACHABLE; vertices * states],
            parents: vec![None; vertices * states],
            settled: 0,
        };
        let mut queue: BinaryHeap<Reverse<(Time, usize)>> = BinaryHeap::new();

        // Climb in wherever the source's own search reached the core.
        for (node, &there) in access.iter().enumerate().take(vertices) {
            if there == UNREACHABLE || !self.is_core(node as NodeId) {
                continue;
            }
            let at = departure.saturating_add(there);
            for state in 0..states {
                if allowed.initial & (1 << state) == 0 {
                    continue;
                }
                // A walk to get here needs a state that walks; standing on the
                // source needs nothing.
                if there > 0 && allowed.step(state, self.walking) & (1 << state) == 0 {
                    continue;
                }
                let product = node * states + state;
                if at < crossing.arrivals[product] {
                    crossing.arrivals[product] = at;
                    queue.push(Reverse((at, product)));
                }
            }
        }

        while let Some(Reverse((now, product))) = queue.pop() {
            if now > crossing.arrivals[product] {
                continue;
            }
            crossing.settled += 1;
            let vertex = (product / states) as NodeId;
            let state = product % states;

            for edge in self.core.out_edges(vertex) {
                let given = self.core.input_index(edge) as usize;
                let symbol = self.core_labels.get(given).copied().unwrap_or(u8::MAX);
                let mut next = allowed.step(state, symbol);
                if next == 0 {
                    continue;
                }
                let head = self.core.head(edge);
                let arrives = now.saturating_add(self.core.weight(edge));
                let leg = Leg::Walk(Walk {
                    from: vertex,
                    to: head,
                    departs: now,
                    arrives,
                });
                while next != 0 {
                    let onto = next.trailing_zeros() as usize;
                    next &= next - 1;
                    crossing.improve(
                        head as usize * states + onto,
                        arrives,
                        product,
                        leg,
                        symbol,
                        &mut queue,
                    );
                }
            }

            // The one relaxation a static graph does not have.
            let riding = allowed.step(state, network.riding);
            if riding == 0 {
                continue;
            }
            for (head, connections, first) in network.timetable.edges_from(vertex) {
                let boarding = connections.partition_point(|c| c.departs < now);
                if boarding == connections.len() {
                    continue;
                }
                let connection = network.timetable.soonest_from(first + boarding);
                let mut onto_states = riding;
                while onto_states != 0 {
                    let onto = onto_states.trailing_zeros() as usize;
                    onto_states &= onto_states - 1;
                    crossing.improve(
                        head as usize * states + onto,
                        connection.arrives,
                        product,
                        Leg::Ride(connection),
                        u8::MAX,
                        &mut queue,
                    );
                }
            }
        }
        crossing
    }

    /// The journey, as the arcs it actually took.
    #[allow(clippy::too_many_arguments)]
    fn tell(
        &self,
        network: &Multimodal<'_>,
        sources: &[(NodeId, Time)],
        departure: Time,
        crossing: &Crossing,
        allowed: &Modes,
        found: Found,
        to: NodeId,
    ) -> Vec<Leg> {
        let states = allowed.num_states();
        let mut moves: Vec<Move> = Vec::new();
        match found {
            Found::Walked(meeting) => {
                moves.push(Move::Walk {
                    from: self.nearest_source(sources, meeting, network),
                    to: meeting,
                });
                moves.push(Move::Walk { from: meeting, to });
            }
            Found::Rode { exit, state } => {
                // Back through the core, then the climbs either side of it.
                let mut product = exit as usize * states + state;
                let mut crossed: Vec<Move> = Vec::new();
                while let Some((previous, leg, mode)) = crossing.parents[product] {
                    crossed.push(match leg {
                        // Only the contracted mode's arcs can be shortcuts; a
                        // link arc was never contracted, so it is already an arc.
                        Leg::Walk(walk) if mode == self.walking => Move::Walk {
                            from: walk.from,
                            to: walk.to,
                        },
                        Leg::Walk(walk) => Move::Arc {
                            from: walk.from,
                            to: walk.to,
                            weight: walk.arrives.saturating_sub(walk.departs),
                        },
                        Leg::Ride(ride) => Move::Ride(ride),
                    });
                    product = previous;
                }
                crossed.reverse();
                let entry = (product / states) as NodeId;
                moves.push(Move::Walk {
                    from: self.nearest_source(sources, entry, network),
                    to: entry,
                });
                moves.extend(crossed);
                moves.push(Move::Walk { from: exit, to });
            }
        }

        // Now spend the clock on them, expanding every walk into the hops it was
        // made of — a shortcut stands for a path and a caller wants the path.
        let mut legs: Vec<Leg> = Vec::new();
        let mut clock = departure;
        for step in moves {
            match step {
                Move::Ride(ride) => {
                    clock = ride.arrives;
                    legs.push(Leg::Ride(ride));
                }
                Move::Arc { from, to, weight } => {
                    let arrives = clock.saturating_add(weight);
                    legs.push(Leg::Walk(Walk {
                        from,
                        to,
                        departs: clock,
                        arrives,
                    }));
                    clock = arrives;
                }
                Move::Walk { from, to } => {
                    if from == to {
                        continue;
                    }
                    for (tail, head, weight) in self.hops(network, from, to) {
                        let arrives = clock.saturating_add(weight);
                        legs.push(Leg::Walk(Walk {
                            from: tail,
                            to: head,
                            departs: clock,
                            arrives,
                        }));
                        clock = arrives;
                    }
                }
            }
        }
        legs
    }

    /// Which source a journey through `entry` actually left from.
    fn nearest_source(
        &self,
        sources: &[(NodeId, Time)],
        entry: NodeId,
        network: &Multimodal<'_>,
    ) -> NodeId {
        if sources.len() == 1 {
            return sources[0].0;
        }
        sources
            .iter()
            .min_by_key(|&&(node, at)| {
                let hops = self.hops(network, node, entry);
                let walked: u64 = hops.iter().map(|&(_, _, weight)| u64::from(weight)).sum();
                if node != entry && hops.is_empty() {
                    u64::MAX
                } else {
                    u64::from(at) + walked
                }
            })
            .map(|&(node, _)| node)
            .unwrap_or(entry)
    }

    /// One walk, as the arcs of the uncontracted subnetwork it is made of.
    ///
    /// Only arcs of the contracted mode are followed, so this never wanders into
    /// another network to shorten a walk the automaton allowed only as a walk.
    fn hops(
        &self,
        network: &Multimodal<'_>,
        from: NodeId,
        to: NodeId,
    ) -> Vec<(NodeId, NodeId, Weight)> {
        if from == to {
            return Vec::new();
        }
        let graph = network.scalar;
        let mut costs = vec![UNREACHABLE; graph.num_nodes()];
        let mut arrived: Vec<Option<(NodeId, Weight)>> = vec![None; graph.num_nodes()];
        let mut queue: BinaryHeap<Reverse<(Weight, NodeId)>> = BinaryHeap::new();
        costs[from as usize] = 0;
        queue.push(Reverse((0, from)));
        while let Some(Reverse((cost, node))) = queue.pop() {
            if cost > costs[node as usize] {
                continue;
            }
            if node == to {
                break;
            }
            for edge in graph.out_edges(node) {
                let given = graph.input_index(edge) as usize;
                if network.labels.get(given).copied() != Some(self.walking) {
                    continue;
                }
                let head = graph.head(edge);
                let next = cost.saturating_add(graph.weight(edge));
                if next < costs[head as usize] {
                    costs[head as usize] = next;
                    arrived[head as usize] = Some((node, graph.weight(edge)));
                    queue.push(Reverse((next, head)));
                }
            }
        }
        if costs[to as usize] == UNREACHABLE {
            return Vec::new();
        }
        let mut hops = Vec::new();
        let mut at = to;
        while let Some((previous, weight)) = arrived[at as usize] {
            hops.push((previous, at, weight));
            at = previous;
        }
        hops.reverse();
        hops
    }
}

/// One stretch of the answer before the clock has been spent on it.
#[derive(Clone, Copy)]
enum Move {
    /// A walk to be told hop by hop: a shortcut, or a climb through the
    /// component, either of which stands for a path.
    Walk {
        from: NodeId,
        to: NodeId,
    },
    /// An arc of the network already — a link, say — with nothing to unpack.
    Arc {
        from: NodeId,
        to: NodeId,
        weight: Weight,
    },
    Ride(crate::model::timetable::Ride),
}

/// Which of the two shapes the answer took.
#[derive(Clone, Copy)]
enum Found {
    /// The climbs met below the core: a journey entirely on foot.
    Walked(NodeId),
    /// It crossed the core, leaving it at `exit` in `state`.
    Rode { exit: NodeId, state: usize },
}

/// What the product search over the core found.
struct Crossing {
    arrivals: Vec<Time>,
    /// How each product vertex was reached, and by an arc of which mode —
    /// `u8::MAX` for a ride, which is no mode of the scalar network.
    parents: Vec<Option<(usize, Leg, u8)>>,
    settled: usize,
}

impl Crossing {
    #[allow(clippy::too_many_arguments)]
    fn improve(
        &mut self,
        product: usize,
        arrives: Time,
        from: usize,
        leg: Leg,
        mode: u8,
        queue: &mut BinaryHeap<Reverse<(Time, usize)>>,
    ) {
        if arrives < self.arrivals[product] {
            self.arrivals[product] = arrives;
            self.parents[product] = Some((from, leg, mode));
            queue.push(Reverse((arrives, product)));
        }
    }
}
