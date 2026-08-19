//! LCSPP: the shortest path whose sequence of transport modes is one you allow.
//!
//! Barrett, Jacob & Marathe, *Formal-Language-Constrained Path Problems* (SIAM
//! Journal on Computing 30(3):809–837, 2000), in the multimodal form Dibbelt,
//! Pajor & Wagner give it in *User-Constrained Multi-Modal Route Planning*
//! (ALENEX 2012) §2.2, where they call it **LCSPP-MS** — MS for modal
//! sequences.
//!
//! ## The problem
//!
//! Merge a network per mode of transport into one graph and the shortest path
//! through it is often one nobody can take: it leaves a car in the middle of a
//! journey, or boards a train from a motorway. The fix is not to score modal
//! changes and hope, but to say which sequences are allowed and search only
//! those. So: label every arc with the mode it belongs to, hand the query a
//! language `L` over those labels, and ask for the shortest path whose labels,
//! read in order, spell a word of `L`. Barrett et al. proved this is solvable
//! in deterministic polynomial time when `L` is regular, which is more than
//! enough for "walk, ride, walk".
//!
//! ## The automaton
//!
//! §2.2 fixes the shape the automaton takes, and [`Modes`] is that shape and no
//! more. A state stands for one or more modes. `(q, σ, q)` is in the transition
//! relation for each mode `σ` the state `q` stands for — travelling *within* a
//! mode is a self-loop. Distinct states are joined only by the **link** label,
//! so a journey may change mode only where the networks were stitched
//! together. States are marked initial or final according to whether their
//! modes may begin or end a journey.
//!
//! The link arcs are the paper's too: *"we link each station node in the
//! railway network to its geographically closest node of the road network …
//! only nodes that are no more than distance δ apart"*, costed by length at
//! walking speed. That is what `Access` builds here, which is why this kernel
//! needs no new model — only a label per arc, which a merged environment
//! already knows.
//!
//! ## The search
//!
//! LCSPP-D, in the survey's words: build the product of the graph and the
//! automaton, take the origin and destination *sets* of product vertices —
//! those pairing the endpoint with an initial or final state — and run Dijkstra
//! between them. A product vertex is `(vertex, state)`; relaxing an arc reads
//! its label, asks the automaton where that label may go, and settles a
//! successor for each answer. Nondeterminism costs nothing but the states.
//!
//! Two relaxations, because a multimodal network has two kinds of arc. A
//! scalar arc has a duration and is relaxed by adding it. A timetable arc has a
//! schedule, so relaxing it is not reading a weight but asking what leaves next
//! along it — a binary search over that arc's connections, exactly as
//! [`crate::kernels::timetable::earliest_arrival`] does. Both are
//! non-decreasing in the time you arrive, which is the FIFO property Dijkstra
//! needs and which §2.1 states these networks have.
//!
//! ## What this is not
//!
//! There is **no preprocessing**. That is the point of having it: the shelf's
//! other multimodal answer, [`crate::Ultra`], spends minutes precomputing so a
//! query costs milliseconds; this spends nothing and pays per query. UCCH, the
//! contribution of the paper this takes §2.2 from, is the speedup that sits
//! between them, and is not here.
//!
//! The survey names two costs, and they are real: to write a constraint you
//! must know what the network's modes are, and a language admits no journeys
//! that combine the modes differently, so this returns one answer rather than a
//! set of alternatives.

#[cfg(test)]
mod tests;
pub mod ucch;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::model::graph::{Graph, NodeId, UNREACHABLE};
use crate::model::technique::{BindError, EarliestArrival, Footprint, Technique};
use crate::model::timetable::{Itinerary, Leg, Time, Timetable, Walk};
use crate::util::progress::Progress;

/// The states an automaton may be in at once, one bit each.
pub(crate) type StateSet = u32;

/// How many states [`Modes`] will hold. The automata §2.2 draws have two to
/// four — one per mode of transport, give or take — so a machine word is room
/// to spare, and it makes the product's inner loop a shift rather than a set.
pub const MAX_STATES: usize = StateSet::BITS as usize;

/// A nondeterministic finite automaton over transport modes.
///
/// Built in the shape §2.2 prescribes — see the module docs — though nothing
/// here enforces that shape: this is a plain NFA, and the caller decides which
/// transitions exist.
#[derive(Debug, Clone)]
pub struct Modes {
    states: usize,
    symbols: usize,
    /// `delta[state * symbols + symbol]`: which states that symbol may lead to.
    delta: Vec<StateSet>,
    initial: StateSet,
    accepting: StateSet,
}

impl Modes {
    /// An automaton over `states` states and `symbols` modes, with no
    /// transitions, no initial state and no final state yet.
    ///
    /// # Panics
    ///
    /// If `states` exceeds [`MAX_STATES`].
    pub fn new(states: usize, symbols: usize) -> Self {
        assert!(
            states <= MAX_STATES,
            "an automaton of {states} states is more than this holds ({MAX_STATES})"
        );
        Modes {
            states,
            symbols,
            delta: vec![0; states * symbols],
            initial: 0,
            accepting: 0,
        }
    }

    /// `(from, symbol, to)`: reading `symbol` in `from` may move to `to`.
    #[must_use]
    pub fn on(mut self, from: usize, symbol: usize, to: usize) -> Self {
        assert!(from < self.states && to < self.states && symbol < self.symbols);
        self.delta[from * self.symbols + symbol] |= 1 << to;
        self
    }

    /// `(q, σ, q)`: travelling within a mode the state stands for.
    #[must_use]
    pub fn within(self, state: usize, symbol: usize) -> Self {
        self.on(state, symbol, state)
    }

    /// A state a journey may begin in.
    #[must_use]
    pub fn starting(mut self, state: usize) -> Self {
        assert!(state < self.states);
        self.initial |= 1 << state;
        self
    }

    /// A state a journey may end in.
    #[must_use]
    pub fn accepting(mut self, state: usize) -> Self {
        assert!(state < self.states);
        self.accepting |= 1 << state;
        self
    }

    pub fn num_states(&self) -> usize {
        self.states
    }

    pub fn num_symbols(&self) -> usize {
        self.symbols
    }

    /// Where reading `symbol` in `state` may lead.
    fn step(&self, state: usize, symbol: u8) -> StateSet {
        let symbol = symbol as usize;
        if symbol >= self.symbols {
            return 0;
        }
        self.delta[state * self.symbols + symbol]
    }

    /// May a journey stop here?
    fn accepts(&self, state: usize) -> bool {
        self.accepting & (1 << state) != 0
    }

    /// Does this automaton accept anything at all? An empty initial or final
    /// set means no path can qualify, which is worth saying rather than
    /// searching for.
    pub fn is_empty(&self) -> bool {
        self.initial == 0 || self.accepting == 0
    }
}

/// A multimodal network: the merged graph, what each of its arcs is, and the
/// schedule the timetable arcs run to.
///
/// `scalar` and `timetable` share one numbering — the merged environment's —
/// which is what makes the two relaxations parts of one search rather than two
/// searches to stitch.
#[derive(Clone, Copy)]
pub struct Multimodal<'a> {
    /// Arcs with a duration: pavements, roads, and the link arcs between
    /// networks.
    pub scalar: &'a Graph,
    /// The mode of each arc of `scalar`, indexed as the edges were *given* to
    /// [`Graph::from_edges`] rather than as they came out of it — a graph
    /// permutes its edges into adjacency order, and everything keyed by the
    /// input, a timetable or a calendar or this, reads through
    /// [`Graph::input_index`].
    pub labels: &'a [u8],
    /// Arcs with a schedule.
    pub timetable: &'a Timetable,
    /// The mode a timetable arc counts as.
    pub riding: u8,
}

impl Multimodal<'_> {
    /// How many vertices the product is built over.
    fn vertices(&self) -> usize {
        self.scalar.num_nodes().max(self.timetable.num_stops())
    }
}

/// The label-constrained search as a configuration: the language it will
/// admit. Binding it to a [`Multimodal`] network builds nothing — the product
/// is searched, never materialised — so the planner borrows the network.
#[derive(Debug, Clone)]
pub struct LabelConstrainedTechnique {
    pub modes: Modes,
}

impl<'a> Technique<'a> for LabelConstrainedTechnique {
    type Inputs = Multimodal<'a>;
    type Planner = LabelConstrained<'a>;

    fn bind(
        &self,
        network: Multimodal<'a>,
        _progress: &Progress,
    ) -> Result<LabelConstrained<'a>, BindError> {
        Ok(LabelConstrained {
            network,
            modes: self.modes.clone(),
        })
    }
}

/// The label-constrained search, bound: a network and the language to
/// search it under.
#[derive(Clone)]
pub struct LabelConstrained<'a> {
    pub network: Multimodal<'a>,
    pub modes: Modes,
}

impl Footprint for LabelConstrained<'_> {
    fn footprint(&self) -> usize {
        0
    }

    /// The product's vertices: every network vertex in every state.
    fn searches(&self) -> (&'static str, usize) {
        ("states", self.network.vertices() * self.modes.num_states())
    }
}

impl EarliestArrival for LabelConstrained<'_> {
    fn earliest_arrival(&self, sources: &[(NodeId, Time)], to: NodeId) -> Option<Itinerary> {
        label_constrained(&self.network, &self.modes, sources, to)
    }
}

/// The earliest arrival at `to` by a journey whose modes `allowed` admits.
///
/// `sources` are `(vertex, time)`: where the journey may begin and when, in the
/// same absolute clock the timetable keeps. Each is entered in every initial
/// state, which is the origin set of product vertices; the answer is the first
/// settled product vertex pairing `to` with a final state.
pub fn label_constrained(
    network: &Multimodal<'_>,
    allowed: &Modes,
    sources: &[(NodeId, Time)],
    to: NodeId,
) -> Option<Itinerary> {
    let vertices = network.vertices();
    if to as usize >= vertices || allowed.is_empty() {
        return None;
    }
    let states = allowed.num_states();

    // One label per product vertex `(vertex, state)`, laid out state-minor so
    // that the states of a vertex sit together — the inner loop walks them.
    let mut earliest = vec![UNREACHABLE; vertices * states];
    let mut arrived_by: Vec<Option<(usize, Leg)>> = vec![None; vertices * states];
    let mut queue: BinaryHeap<Reverse<(Time, usize)>> = BinaryHeap::new();

    for &(from, at) in sources {
        if from as usize >= vertices {
            continue;
        }
        for state in 0..states {
            if allowed.initial & (1 << state) == 0 {
                continue;
            }
            let product = from as usize * states + state;
            if at < earliest[product] {
                earliest[product] = at;
                queue.push(Reverse((at, product)));
            }
        }
    }

    let mut settled = 0usize;
    while let Some(Reverse((now, product))) = queue.pop() {
        // Lazy deletion, as everywhere else in this crate.
        if now > earliest[product] {
            continue;
        }
        settled += 1;
        let vertex = (product / states) as NodeId;
        let state = product % states;
        if vertex == to && allowed.accepts(state) {
            return Some(unwind(&arrived_by, product, now, settled));
        }

        for edge in network.scalar.out_edges(vertex) {
            let given = network.scalar.input_index(edge) as usize;
            let symbol = network.labels.get(given).copied().unwrap_or(u8::MAX);
            let next = allowed.step(state, symbol);
            if next == 0 {
                continue;
            }
            let head = network.scalar.head(edge);
            let arrives = now.saturating_add(network.scalar.weight(edge));
            let leg = Leg::Walk(Walk {
                from: vertex,
                to: head,
                departs: now,
                arrives,
            });
            relax(
                &mut earliest,
                &mut arrived_by,
                &mut queue,
                Reached {
                    head,
                    states: next,
                    arrives,
                    from: product,
                    leg,
                },
                states,
            );
        }

        // The one relaxation a static graph does not have: not "what does this
        // arc cost" but "what is the next thing along it".
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
            relax(
                &mut earliest,
                &mut arrived_by,
                &mut queue,
                Reached {
                    head,
                    states: riding,
                    arrives: connection.arrives,
                    from: product,
                    leg: Leg::Ride(connection),
                },
                states,
            );
        }
    }
    None
}

/// One arc's worth of news: where it lands, in which automaton states, when,
/// and how it got there.
struct Reached {
    head: NodeId,
    states: StateSet,
    arrives: Time,
    from: usize,
    leg: Leg,
}

/// Settle `reached` into every automaton state the arc's label allowed.
fn relax(
    earliest: &mut [Time],
    arrived_by: &mut [Option<(usize, Leg)>],
    queue: &mut BinaryHeap<Reverse<(Time, usize)>>,
    reached: Reached,
    states: usize,
) {
    let mut next = reached.states;
    while next != 0 {
        let state = next.trailing_zeros() as usize;
        next &= next - 1;
        let product = reached.head as usize * states + state;
        if reached.arrives < earliest[product] {
            earliest[product] = reached.arrives;
            arrived_by[product] = Some((reached.from, reached.leg));
            queue.push(Reverse((reached.arrives, product)));
        }
    }
}

/// Read the journey back out of the parent pointers.
fn unwind(
    arrived_by: &[Option<(usize, Leg)>],
    mut product: usize,
    arrives: Time,
    settled: usize,
) -> Itinerary {
    let mut legs = Vec::new();
    while let Some((previous, leg)) = arrived_by[product] {
        legs.push(leg);
        product = previous;
    }
    legs.reverse();
    Itinerary {
        arrives,
        legs,
        settled,
    }
}
