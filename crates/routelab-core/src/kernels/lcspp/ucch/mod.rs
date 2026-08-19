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

use super::{Modes, Multimodal, StateSet};
use crate::kernels::contraction::{CoreHierarchy, Ordering};
use crate::kernels::dijkstra::dijkstra;
use crate::model::graph::{Graph, GraphError, NodeId, Weight, UNREACHABLE};
use crate::model::search::{SearchOptions, SearchResult};
use crate::model::technique::{BindError, EarliestArrival, Footprint, Technique};
use crate::model::timetable::{Itinerary, Leg, Time, Walk};
use crate::util::progress::Progress;

/// What a UCCH is built from: the uncontracted network every query reads,
/// plus the pieces of it the contraction works on — one mode's subnetwork,
/// the link arcs that are never contracted, and the vertices a schedule
/// serves.
#[derive(Clone, Copy)]
pub struct UcchInputs<'a> {
    pub network: Multimodal<'a>,
    /// One mode's subnetwork over the whole numbering — the pavements.
    pub walkable: &'a Graph,
    /// The arcs that join networks; their endpoints are the core.
    pub links: &'a [(NodeId, NodeId, Weight)],
    /// Vertices a schedule serves, kept in the core.
    pub served: &'a [NodeId],
}

/// UCCH as a configuration: the language, which mode gets contracted and
/// which the links count as, and how the contraction is run.
#[derive(Debug, Clone)]
pub struct UcchTechnique {
    pub modes: Modes,
    pub walking: u8,
    pub link_label: u8,
    pub ordering: Ordering,
    pub max_degree: f64,
}

impl<'a> Technique<'a> for UcchTechnique {
    type Inputs = UcchInputs<'a>;
    type Planner = UcchPlanner<'a>;

    fn bind(
        &self,
        inputs: UcchInputs<'a>,
        progress: &Progress,
    ) -> Result<UcchPlanner<'a>, BindError> {
        let hierarchy = Ucch::build_reporting(
            inputs.walkable,
            self.walking,
            inputs.links,
            self.link_label,
            inputs.served,
            self.ordering,
            self.max_degree,
            progress,
        )?;
        Ok(UcchPlanner {
            hierarchy,
            network: inputs.network,
            modes: self.modes.clone(),
        })
    }
}

/// A UCCH bound to the network it was contracted from and the language it
/// searches under. The [`Ucch`] is the data; this is what answers.
pub struct UcchPlanner<'a> {
    pub hierarchy: Ucch,
    pub network: Multimodal<'a>,
    pub modes: Modes,
}

impl Footprint for UcchPlanner<'_> {
    fn footprint(&self) -> usize {
        self.hierarchy.footprint()
    }

    /// The core's product: every core vertex in every state.
    fn searches(&self) -> (&'static str, usize) {
        (
            "states",
            self.hierarchy.num_core() * self.modes.num_states(),
        )
    }
}

impl EarliestArrival for UcchPlanner<'_> {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        self.hierarchy
            .earliest_arrival(&self.network, &self.modes, sources, to)
    }
}

/// A vertex the contraction retired, so the core has no number for it.
const NOT_CORE: u32 = u32::MAX;

/// A hierarchy over one mode's subnetwork, contracted around the vertices where
/// the networks join.
pub struct Ucch {
    hierarchy: CoreHierarchy,
    /// The mode of the subnetwork that was contracted, and so of every shortcut.
    walking: u8,
    /// The core's scalar arcs: what the contraction left standing, plus every
    /// link arc, since those are never contracted. Numbered `0..num_core`, not
    /// by the network's own ids — §3.3's node reordering, and the reason a
    /// query's tables are the size of the core rather than of the city.
    core: Graph,
    /// The mode of each core arc, by position in the list it was built from.
    core_labels: Vec<u8>,
    /// Which vertex each core number is.
    core_nodes: Vec<NodeId>,
    /// Which core number each vertex has, [`NOT_CORE`] for one it retired.
    core_of: Vec<u32>,
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
        served: &[NodeId],
        ordering: Ordering,
        max_degree: f64,
    ) -> Result<Self, GraphError> {
        Self::build_reporting(
            walkable,
            walking,
            links,
            link_label,
            served,
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
        served: &[NodeId],
        ordering: Ordering,
        max_degree: f64,
        progress: &Progress,
    ) -> Result<Self, GraphError> {
        // The transfer nodes: every end of every link arc. These are what the
        // contraction may not touch, and what is left when it stops.
        //
        // And every vertex a vehicle calls at, which the practical variant does
        // not contract either. Most of them are transfer nodes already, but a
        // stop the pavements never reach is joined to nothing at all — so
        // nothing would protect it, and a trip riding *through* it would find
        // the core had no number for it.
        let mut transfer: Vec<NodeId> = Vec::with_capacity(links.len() * 2 + served.len());
        for &(tail, head, _) in links {
            transfer.push(tail);
            transfer.push(head);
        }
        transfer.extend_from_slice(served);
        transfer.sort_unstable();
        transfer.dedup();

        let hierarchy =
            CoreHierarchy::build_reporting(walkable, &transfer, ordering, max_degree, progress)?;

        // Number the core densely. §3.3 reorders vertices so that the core sits
        // at the front, because "most of the time is spent on the core" — here
        // it is given a numbering of its own instead, which is the same idea
        // taken to its end: a query's tables are then the size of the core, not
        // of the city, and 2% of a city fits in cache where all of it does not.
        let mut core_of = vec![NOT_CORE; walkable.num_nodes()];
        let mut core_nodes: Vec<NodeId> = Vec::new();
        for node in 0..walkable.num_nodes() as NodeId {
            if hierarchy.is_core(node) {
                core_of[node as usize] = core_nodes.len() as u32;
                core_nodes.push(node);
            }
        }

        // The core: what the contraction left of the pavements, and every link
        // arc. A shortcut among the former is a walk, because only one
        // subnetwork was contracted — so no label had to survive contraction.
        let mut arcs: Vec<(NodeId, NodeId, Weight)> = Vec::new();
        let mut core_labels: Vec<u8> = Vec::new();
        let standing = hierarchy.core();
        for tail in 0..standing.num_nodes() as NodeId {
            let from = core_of[tail as usize];
            if from == NOT_CORE {
                continue;
            }
            for edge in standing.out_edges(tail) {
                let to = core_of[standing.head(edge) as usize];
                if to == NOT_CORE {
                    continue;
                }
                arcs.push((from, to, standing.weight(edge)));
                core_labels.push(walking);
            }
        }
        for &(tail, head, weight) in links {
            let (from, to) = (core_of[tail as usize], core_of[head as usize]);
            if from == NOT_CORE || to == NOT_CORE {
                continue;
            }
            arcs.push((from, to, weight));
            core_labels.push(link_label);
        }

        Ok(Ucch {
            core: Graph::from_edges(core_nodes.len(), &arcs)?,
            core_labels,
            core_nodes,
            core_of,
            hierarchy,
            walking,
        })
    }

    /// Is this vertex in the core?
    pub fn is_core(&self, node: NodeId) -> bool {
        self.core_of.get(node as usize).copied().unwrap_or(NOT_CORE) != NOT_CORE
    }

    /// How many vertices the core holds — what a query searches instead of the
    /// whole network.
    pub fn num_core(&self) -> usize {
        self.core_nodes.len()
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
        // The network's vertices, not the core's: the target and the two climbs
        // are in the numbering the caller uses.
        let vertices = self.core_of.len();
        if to as usize >= vertices || allowed.is_empty() || sources.is_empty() {
            return None;
        }
        let departure = sources.iter().map(|&(_, at)| at).min()?;

        // Walking has to be allowed somewhere for a climb to be usable: in an
        // initial state to walk out of the source, in a final one to walk into
        // the target. The component is one mode, so that is the only question
        // the automaton is asked below the core.
        let leaving = self.any_walks(allowed, allowed.initial);
        let arriving = self.any_walks(allowed, allowed.accepting);

        let (access, climbed) = self.climb(self.hierarchy.upward(), sources, departure, leaving);
        let (egress, _) = self.climb(
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
        for (index, &node) in self.core_nodes.iter().enumerate() {
            let back = egress[node as usize];
            if back == UNREACHABLE {
                continue;
            }
            for state in 0..states {
                if !allowed.accepts(state) {
                    continue;
                }
                let at = crossing.arrivals[index * states + state];
                if at == UNREACHABLE {
                    continue;
                }
                // Only a final state that walks may use a non-empty egress.
                if back > 0 && !self.walks(allowed, state) {
                    continue;
                }
                let arrives = at.saturating_add(back);
                if arrives < best {
                    best = arrives;
                    answer = Some(Found::Rode { exit: index, state });
                }
            }
        }

        let found = answer?;
        let legs = self.tell(network, sources, climbed.as_ref(), &crossing, found, to);
        Some(Itinerary {
            arrives: best,
            legs,
            settled: crossing.settled,
        })
    }

    /// May a journey in this state walk the contracted mode?
    fn walks(&self, allowed: &Modes, state: usize) -> bool {
        allowed.step(state, self.walking) & (1 << state) != 0
    }

    /// May any state in `set`?
    fn any_walks(&self, allowed: &Modes, set: StateSet) -> bool {
        (0..allowed.num_states()).any(|state| set & (1 << state) != 0 && self.walks(allowed, state))
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
    ) -> (Vec<Time>, Option<SearchResult>) {
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
            return (standing, None);
        }
        match dijkstra(graph, &seeds, &SearchOptions::default()) {
            Ok(found) => (found.costs.clone(), Some(found)),
            Err(_) => (vec![UNREACHABLE; graph.num_nodes()], None),
        }
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
        let core = self.core_nodes.len();
        let mut crossing = Crossing {
            states,
            arrivals: vec![UNREACHABLE; core * states],
            parents: vec![None; core * states],
            settled: 0,
        };
        let mut queue: BinaryHeap<Reverse<(Time, usize)>> = BinaryHeap::new();

        // Climb in wherever the source's own search reached the core.
        for (index, &node) in self.core_nodes.iter().enumerate() {
            let there = access[node as usize];
            if there == UNREACHABLE {
                continue;
            }
            let at = departure.saturating_add(there);
            for state in 0..states {
                if allowed.initial & (1 << state) == 0 {
                    continue;
                }
                // A walk to get here needs a state that walks; standing on the
                // source needs nothing.
                if there > 0 && !self.walks(allowed, state) {
                    continue;
                }
                let product = index * states + state;
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
            let index = product / states;
            let vertex = self.core_nodes[index];
            let state = product % states;

            for edge in self.core.out_edges(index as NodeId) {
                let given = self.core.input_index(edge) as usize;
                let symbol = self.core_labels.get(given).copied().unwrap_or(u8::MAX);
                let next = allowed.step(state, symbol);
                if next == 0 {
                    continue;
                }
                let head = self.core.head(edge) as usize;
                let weight = self.core.weight(edge);
                // Told in the network's own vertices, not the core's, because
                // that is what a caller draws and what a walk is expanded
                // against. Only the contracted mode's arcs can be shortcuts; a
                // link arc was never contracted, so it is already an arc.
                let (from, to) = (vertex, self.core_nodes[head]);
                let made = if symbol == self.walking {
                    Move::Walk { from, to }
                } else {
                    Move::Arc { from, to, weight }
                };
                crossing.improve(
                    head,
                    next,
                    now.saturating_add(weight),
                    product,
                    made,
                    &mut queue,
                );
            }

            // The one relaxation a static graph does not have.
            let riding = allowed.step(state, network.riding);
            if riding == 0 {
                continue;
            }
            for (head, connections, first) in network.timetable.edges_from(vertex) {
                let onto_core = self.core_of[head as usize];
                if onto_core == NOT_CORE {
                    continue;
                }
                let boarding = connections.partition_point(|c| c.departs < now);
                if boarding == connections.len() {
                    continue;
                }
                let connection = network.timetable.soonest_from(first + boarding);
                crossing.improve(
                    onto_core as usize,
                    riding,
                    connection.arrives,
                    product,
                    Move::Ride(connection),
                    &mut queue,
                );
            }
        }
        crossing
    }

    /// The journey, as the arcs it actually took.
    fn tell(
        &self,
        network: &Multimodal<'_>,
        sources: &[(NodeId, Time)],
        climbed: Option<&SearchResult>,
        crossing: &Crossing,
        found: Found,
        to: NodeId,
    ) -> Vec<Leg> {
        let states = crossing.states;
        let mut moves: Vec<Move> = Vec::new();
        // When the clock starts: not the earliest of the sources but the one
        // this journey actually left from, since each carries its own head start
        // and a leg departing before its source did is one nobody could take.
        // Both arms set it, so there is nothing sensible to initialise it to.
        let start;
        match found {
            Found::Walked(meeting) => {
                let (origin, at) = self.nearest_source(sources, meeting, climbed);
                start = at;
                moves.push(Move::Walk {
                    from: origin,
                    to: meeting,
                });
                moves.push(Move::Walk { from: meeting, to });
            }
            Found::Rode { exit, state } => {
                // Back through the core, then the climbs either side of it.
                let mut product = exit * states + state;
                let mut crossed: Vec<Move> = Vec::new();
                while let Some((previous, made)) = crossing.parents[product] {
                    crossed.push(made);
                    product = previous;
                }
                crossed.reverse();
                let entry = self.core_nodes[product / states];
                let (origin, at) = self.nearest_source(sources, entry, climbed);
                start = at;
                moves.push(Move::Walk {
                    from: origin,
                    to: entry,
                });
                moves.extend(crossed);
                moves.push(Move::Walk {
                    from: self.core_nodes[exit],
                    to,
                });
            }
        }

        // Now spend the clock on them, expanding every walk into the hops it was
        // made of — a shortcut stands for a path and a caller wants the path.
        let mut legs: Vec<Leg> = Vec::new();
        let mut clock = start;
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

    /// Which source a journey through `entry` actually left from, and when.
    ///
    /// Read off the climb that got there rather than searched for: a
    /// multi-source Dijkstra's path to a vertex begins at whichever source
    /// reached it, so the answer is already in hand. Asking each source how far
    /// it is from `entry` would be one search apiece for something the first
    /// search settled.
    fn nearest_source(
        &self,
        sources: &[(NodeId, Time)],
        entry: NodeId,
        climbed: Option<&SearchResult>,
    ) -> (NodeId, Time) {
        if sources.len() == 1 {
            return sources[0];
        }
        let origin = climbed
            .and_then(|found| found.path(entry))
            .and_then(|path| path.first().copied())
            .unwrap_or(entry);
        sources
            .iter()
            .copied()
            .find(|&(node, _)| node == origin)
            .unwrap_or((origin, departure_of(sources)))
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

/// The earliest any of these sources leaves.
fn departure_of(sources: &[(NodeId, Time)]) -> Time {
    sources.iter().map(|&(_, at)| at).min().unwrap_or(0)
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
    /// It crossed the core, leaving it at core vertex `exit` — a number of the
    /// core's own — in `state`.
    Rode { exit: usize, state: usize },
}

/// What the product search over the core found.
struct Crossing {
    /// How many automaton states a vertex is crossed with — fixed for the whole
    /// crossing, so the tables are indexed state-minor without being told.
    states: usize,
    arrivals: Vec<Time>,
    /// How each product vertex was reached: the move itself, since that is all
    /// telling the journey needs. A `Leg` here would carry times that get spent
    /// again from the clock anyway, and a mode byte only to say which of two
    /// kinds of move it was.
    parents: Vec<Option<(usize, Move)>>,
    settled: usize,
}

impl Crossing {
    /// Settle `made` into every automaton state the arc's label allowed, the
    /// way [`super::relax`] does — the bit walk belongs here rather than at
    /// each call site.
    fn improve(
        &mut self,
        head: usize,
        onto: StateSet,
        arrives: Time,
        from: usize,
        made: Move,
        queue: &mut BinaryHeap<Reverse<(Time, usize)>>,
    ) {
        let mut onto = onto;
        while onto != 0 {
            let state = onto.trailing_zeros() as usize;
            onto &= onto - 1;
            let product = head * self.states + state;
            if arrives < self.arrivals[product] {
                self.arrivals[product] = arrives;
                self.parents[product] = Some((from, made));
                queue.push(Reverse((arrives, product)));
            }
        }
    }
}
