//! LCSPP: the shortest path whose sequence of transport modes a language allows.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::contraction::Ordering as CoreOrdering;
use routelab_core::kernels::lcspp::ucch::Ucch as CoreUcch;
use routelab_core::kernels::lcspp::{label_constrained, Modes as CoreModes, Multimodal};
use routelab_core::model::graph::Graph as CoreGraph;
use routelab_core::model::timetable::Timetable as CoreTimetable;
use routelab_core::NodeId;

use crate::graph::*;
use crate::timetable::*;

/// Which sequences of transport modes a journey may use.
///
/// A nondeterministic finite automaton over the modes, built the way §2.2 of
/// the paper draws it: a state stands for one or more modes, a self-loop is
/// travelling within one, and distinct states are joined only by the link
/// label. Given whole rather than chained, since crossing this boundary once
/// with the transitions in hand is cheaper than crossing it per transition.
#[pyclass(name = "Modes", module = "routelab._routelab", frozen)]
pub struct PyModes {
    pub(crate) inner: Arc<CoreModes>,
}

#[pymethods]
impl PyModes {
    /// `transitions` are `(from_state, symbol, to_state)`; `starting` and
    /// `accepting` are the states a journey may begin and end in.
    #[new]
    #[pyo3(signature = (states, symbols, transitions, starting, accepting))]
    fn new(
        states: usize,
        symbols: usize,
        transitions: Vec<(usize, usize, usize)>,
        starting: Vec<usize>,
        accepting: Vec<usize>,
    ) -> PyResult<Self> {
        if states == 0 || states > routelab_core::kernels::lcspp::MAX_STATES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "an automaton of {states} states; this holds 1 to {}",
                routelab_core::kernels::lcspp::MAX_STATES
            )));
        }
        let mut modes = CoreModes::new(states, symbols);
        for (from, symbol, to) in transitions {
            if from >= states || to >= states || symbol >= symbols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "transition ({from}, {symbol}, {to}) is outside {states} states \
                     and {symbols} symbols"
                )));
            }
            modes = modes.on(from, symbol, to);
        }
        for state in starting {
            if state >= states {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "no state {state} to start in"
                )));
            }
            modes = modes.starting(state);
        }
        for state in accepting {
            if state >= states {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "no state {state} to end in"
                )));
            }
            modes = modes.accepting(state);
        }
        Ok(PyModes {
            inner: Arc::new(modes),
        })
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    #[getter]
    fn num_symbols(&self) -> usize {
        self.inner.num_symbols()
    }

    /// Does this automaton admit any journey at all?
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "Modes({} states, {} symbols)",
            self.inner.num_states(),
            self.inner.num_symbols()
        )
    }
}

/// A multimodal network: the merged graph, the mode of each of its arcs, and
/// the schedule the timetable arcs run to.
///
/// There is nothing to precompute — this is the environment's own graph with a
/// label per arc — so building one costs a copy of the labels and no more.
#[pyclass(name = "Multimodal", module = "routelab._routelab", frozen)]
pub struct PyMultimodal {
    graph: Arc<CoreGraph>,
    /// One per arc, indexed as the edges were given to the graph.
    labels: Vec<u8>,
    timetable: Arc<CoreTimetable>,
    riding: u8,
}

impl PyMultimodal {
    /// The kernel's borrowed view of this network. Both searches take one, and
    /// there is no reason for each to spell it out.
    fn borrow(&self) -> Multimodal<'_> {
        Multimodal {
            scalar: &self.graph,
            labels: &self.labels,
            timetable: &self.timetable,
            riding: self.riding,
        }
    }
}

#[pymethods]
impl PyMultimodal {
    /// `labels` is the mode of each arc, by **position in the graph's input
    /// edge list** rather than by edge id — a graph permutes its edges into
    /// adjacency order, and everything keyed by the input reads through that.
    #[new]
    #[pyo3(signature = (graph, labels, timetable, riding))]
    fn new(graph: &PyGraph, labels: Vec<u8>, timetable: &PyTimetable, riding: u8) -> Self {
        PyMultimodal {
            graph: Arc::clone(&graph.inner),
            labels,
            timetable: Arc::clone(&timetable.inner),
            riding,
        }
    }

    /// The earliest arrival at `target` by a journey `modes` admits.
    ///
    /// `sources` are `(stop, time)` in the timetable's own clock: where the
    /// journey may begin and when.
    #[pyo3(signature = (modes, sources, target))]
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        modes: &PyModes,
        sources: Vec<(NodeId, u32)>,
        target: NodeId,
    ) -> Option<PyItinerary> {
        let network = self.borrow();
        py.detach(|| label_constrained(&network, &modes.inner, &sources, target))
            .map(PyItinerary::from)
    }

    #[getter]
    fn num_arcs(&self) -> usize {
        self.labels.len()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.labels.len() * std::mem::size_of::<u8>()
    }

    fn __repr__(&self) -> String {
        format!("Multimodal({} arcs)", self.labels.len())
    }
}

/// A hierarchy over the walking network, contracted around the vertices where
/// the networks join — UCCH's preprocessing.
///
/// Seconds on a city, against the minutes [`crate::ultra::PyUltra`] wants, and
/// it leaves the language a query input rather than baking it in.
#[pyclass(name = "Ucch", module = "routelab._routelab", frozen)]
pub struct PyUcch {
    inner: Arc<CoreUcch>,
}

#[pymethods]
impl PyUcch {
    /// Contract `walkable` around the endpoints of `links`.
    ///
    /// `graph` and `labels` are the merged network and the mode of each of its
    /// arcs, the same pair [`PyMultimodal`] takes; the arcs labelled `walking`
    /// are the subnetwork contracted and those labelled `link_label` are what
    /// joins the networks. `served` are the vertices a vehicle calls at, which
    /// are never contracted — most are link endpoints already, but a stop the
    /// pavements never reach is joined to nothing and would otherwise be
    /// contracted out from under a trip that rides through it.
    #[staticmethod]
    #[pyo3(signature = (graph, labels, walking, link_label, served, max_degree=20.0, progress=None))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        graph: &PyGraph,
        labels: Vec<u8>,
        walking: u8,
        link_label: u8,
        served: Vec<NodeId>,
        max_degree: f64,
        progress: Option<&crate::progress::PyProgress>,
    ) -> PyResult<Self> {
        let merged = Arc::clone(&graph.inner);
        let counter = crate::progress::counter(progress);
        let built = py.detach(|| {
            // Split the merged network into the subnetwork to contract and the
            // arcs that join networks. Done here rather than by the caller
            // because it is a walk over every arc, and a million and a half of
            // them is not a thing to hand across this boundary twice.
            let mut pavement = Vec::new();
            let mut links = Vec::new();
            for tail in 0..merged.num_nodes() as NodeId {
                for edge in merged.out_edges(tail) {
                    let given = merged.input_index(edge) as usize;
                    let arc = (tail, merged.head(edge), merged.weight(edge));
                    match labels.get(given).copied() {
                        Some(mode) if mode == walking => pavement.push(arc),
                        Some(mode) if mode == link_label => links.push(arc),
                        _ => {}
                    }
                }
            }
            let walkable = CoreGraph::from_edges(merged.num_nodes(), &pavement)?;
            CoreUcch::build_reporting(
                &walkable,
                walking,
                &links,
                link_label,
                &served,
                CoreOrdering::default(),
                max_degree,
                &counter,
            )
        });
        Ok(PyUcch {
            inner: Arc::new(built.map_err(crate::value_err)?),
        })
    }

    /// The earliest arrival at `target` by a journey `modes` admits.
    ///
    /// `network` is the *uncontracted* one, the same
    /// [`PyMultimodal`] plain LCSPP takes: its schedule is what the core rides
    /// and its arcs are what a shortcut is told in.
    #[pyo3(signature = (network, modes, sources, target))]
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        network: &PyMultimodal,
        modes: &PyModes,
        sources: Vec<(NodeId, u32)>,
        target: NodeId,
    ) -> Option<PyItinerary> {
        let borrowed = network.borrow();
        py.detach(|| {
            self.inner
                .earliest_arrival(&borrowed, &modes.inner, &sources, target)
        })
        .map(PyItinerary::from)
    }

    /// Vertices in the core: what a query searches instead of the network.
    #[getter]
    fn num_core(&self) -> usize {
        self.inner.num_core()
    }

    /// Arcs in the core, links included.
    #[getter]
    fn num_arcs(&self) -> usize {
        self.inner.num_arcs()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Did this vertex survive the contraction?
    fn is_core(&self, node: NodeId) -> bool {
        self.inner.is_core(node)
    }

    fn __repr__(&self) -> String {
        format!(
            "Ucch({} core vertices, {} arcs)",
            self.inner.num_core(),
            self.inner.num_arcs()
        )
    }
}
