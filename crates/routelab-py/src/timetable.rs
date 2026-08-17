//! Timetable structures: connections, footpaths, and the two Pyrga models.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::timetable::{
    earliest_arrival as core_earliest_arrival, TimeExpanded as CoreTimeExpanded,
};
use routelab_core::model::timetable::{
    Footpaths as CoreFootpaths, Itinerary as CoreItinerary, Leg, Timetable as CoreTimetable,
    Transfer,
};
use routelab_core::NodeId;

use crate::graph::*;

/// Departures along one edge, as `(trip, departs, arrives)`.
pub(crate) type Departures = Vec<(u32, u32, u32)>;

/// Stop-to-stop walks a rider may take at any time, for a fixed duration.
///
/// The paper's foot-edges, closed under composition — see
/// [`CoreFootpaths`]. Built from positions in a graph's input edge list, each
/// named edge becoming a walk between its endpoints taking its weight.
#[pyclass(name = "Footpaths", module = "routelab._routelab", frozen)]
pub struct PyFootpaths {
    inner: Arc<CoreFootpaths>,
}

#[pymethods]
impl PyFootpaths {
    /// Every edge at the given **positions in the graph's input edge list**
    /// becomes a footpath between its endpoints, taking its weight.
    #[staticmethod]
    fn from_edges(graph: &PyGraph, positions: Vec<u32>) -> Self {
        PyFootpaths {
            inner: Arc::new(CoreFootpaths::from_input_edges(&graph.inner, positions)),
        }
    }

    /// No footpaths at all: the paper's plain model.
    #[staticmethod]
    fn none() -> Self {
        PyFootpaths {
            inner: Arc::new(CoreFootpaths::none()),
        }
    }

    /// How long the walk between two stops takes, if there is one — after
    /// closure, so a two-hop walk answers too.
    fn duration(&self, from: NodeId, to: NodeId) -> Option<u32> {
        self.inner.duration(from, to)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    fn __repr__(&self) -> String {
        format!("Footpaths({} walks)", self.inner.len())
    }
}

/// The footpaths a query walks: the ones given, or none.
pub(crate) fn footpaths_or_none(footpaths: Option<&PyFootpaths>) -> Arc<CoreFootpaths> {
    footpaths.map_or_else(
        || Arc::new(CoreFootpaths::none()),
        |paths| Arc::clone(&paths.inner),
    )
}

/// A day's connections, indexed for a search to read.
///
/// Stops are a graph's nodes, and each connection is filed under the edge it
/// runs along, so nothing here has to pair up parallel columns by hand. See
/// [`CoreTimetable::from_input_connections`].
#[pyclass(name = "Timetable", module = "routelab._routelab", frozen)]
pub struct PyTimetable {
    pub(crate) inner: Arc<CoreTimetable>,
}

#[pymethods]
impl PyTimetable {
    /// Build from connections keyed by **position in the graph's input edge
    /// list** — `{position: [(trip, departs, arrives), ...]}`.
    ///
    /// Not by edge id, for the reason [`PyCalendar::from_windows`] says.
    #[staticmethod]
    #[pyo3(signature = (graph, connections))]
    fn from_connections(graph: &PyGraph, connections: Vec<(u32, Departures)>) -> Self {
        PyTimetable {
            inner: Arc::new(CoreTimetable::from_input_connections(
                &graph.inner,
                connections,
            )),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    #[getter]
    fn num_connections(&self) -> usize {
        self.inner.num_connections()
    }

    /// Stop-to-stop pairs any connection runs along — the edge count of the
    /// time-dependent model, whose node count is `num_stops`.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Earliest arrival at `to` from `sources` — `[(stop, time), ...]`, each
    /// a stop and when you are standing there — walking `footpaths` between
    /// stops if any are given.
    ///
    /// The time-dependent model: a node per stop, and a search that asks each
    /// edge what leaves next.
    #[pyo3(signature = (sources, to, footpaths = None))]
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        to: NodeId,
        footpaths: Option<&PyFootpaths>,
    ) -> Option<PyItinerary> {
        let timetable = Arc::clone(&self.inner);
        let footpaths = footpaths_or_none(footpaths);
        py.detach(|| {
            core_earliest_arrival(&timetable, &sources, to, Transfer::instant(), &footpaths)
        })
        .map(PyItinerary::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "Timetable({} stops, {} connections)",
            self.inner.num_stops(),
            self.inner.num_connections()
        )
    }
}

/// A timetable expanded into a graph with a node per event.
#[pyclass(name = "TimeExpanded", module = "routelab._routelab", frozen)]
pub struct PyTimeExpanded {
    inner: Arc<CoreTimeExpanded>,
}

#[pymethods]
impl PyTimeExpanded {
    /// Expand a timetable, with `footpaths` between its stops if any are
    /// given. Seconds on a city, paid once.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None))]
    fn build(py: Python<'_>, timetable: &PyTimetable, footpaths: Option<&PyFootpaths>) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let expanded =
            py.detach(|| CoreTimeExpanded::build(&timetable, Transfer::instant(), &footpaths));
        PyTimeExpanded {
            inner: Arc::new(expanded),
        }
    }

    #[getter]
    fn num_events(&self) -> usize {
        self.inner.num_events()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Earliest arrival at `to` from `sources` — `[(stop, time), ...]`.
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        to: NodeId,
    ) -> Option<PyItinerary> {
        let expanded = Arc::clone(&self.inner);
        py.detach(|| expanded.earliest_arrival(&sources, to))
            .map(PyItinerary::from)
    }

    fn __repr__(&self) -> String {
        format!("TimeExpanded({} events)", self.inner.num_events())
    }
}

/// When you get there, what you rode, and what the search cost to find it.
#[pyclass(name = "Itinerary", module = "routelab._routelab", frozen)]
pub struct PyItinerary {
    inner: CoreItinerary,
}

impl From<CoreItinerary> for PyItinerary {
    fn from(inner: CoreItinerary) -> Self {
        PyItinerary { inner }
    }
}

#[pymethods]
impl PyItinerary {
    /// Absolute arrival time, in seconds since the service day's midnight.
    #[getter]
    fn arrives(&self) -> u32 {
        self.inner.arrives
    }

    /// Nodes the search settled — stops for one model, events for the other.
    /// The number the two models are actually compared on.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    #[getter]
    fn transfers(&self) -> usize {
        self.inner.transfers()
    }

    /// Each vehicle ridden, as `(trip, from_stop, to_stop, departs, arrives)`,
    /// with the walks between them left out.
    fn rides(&self) -> Vec<(u32, NodeId, NodeId, u32, u32)> {
        self.inner
            .rides()
            .map(|r| (r.trip.0, r.from, r.to, r.departs, r.arrives))
            .collect()
    }

    /// Every stretch of the journey in order, as
    /// `(trip, from_stop, to_stop, departs, arrives)` — `trip` is `None` for
    /// a stretch walked between stops.
    fn legs(&self) -> Vec<(Option<u32>, NodeId, NodeId, u32, u32)> {
        self.inner
            .legs
            .iter()
            .map(|leg| match leg {
                Leg::Ride(r) => (Some(r.trip.0), r.from, r.to, r.departs, r.arrives),
                Leg::Walk(w) => (None, w.from, w.to, w.departs, w.arrives),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Itinerary(arrives={}, {} rides, {} walks, {} transfers)",
            self.inner.arrives,
            self.inner.rides().count(),
            self.inner.walks().count(),
            self.inner.transfers()
        )
    }
}
