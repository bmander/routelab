//! Time-dependent search: a calendar of opening hours, and Dreyfus's query.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use routelab_core::kernels::timedep::{
    time_dependent_dijkstra as core_timedep, Calendar as CoreCalendar, Departure as CoreDeparture,
    Waiting as CoreWaiting, Window as CoreWindow,
};
use routelab_core::{EdgeId, NodeId, Weight};

use crate::graph::*;
use crate::search::*;
use crate::value_err;

/// When each edge may be travelled, on a weekly clock.
///
/// `frozen`, like every other preprocessed structure here, so a search can read
/// it with the GIL released.
#[pyclass(name = "Calendar", module = "routelab._routelab", frozen)]
pub struct PyCalendar {
    inner: Arc<CoreCalendar>,
}

#[pymethods]
impl PyCalendar {
    /// Build from windows keyed by **position in the graph's input edge list**.
    ///
    /// Not by edge id: `Graph` permutes its edges into CSR order, and a calendar
    /// keyed the wrong way shuts the wrong streets without failing. Everything
    /// that produces edges holds input positions, so that is what this takes.
    #[staticmethod]
    #[pyo3(signature = (graph, windows))]
    fn from_windows(graph: &PyGraph, windows: Vec<(u32, Vec<(u32, u32)>)>) -> Self {
        let entries = windows.into_iter().map(|(input, windows)| {
            let windows = windows
                .into_iter()
                .map(|(start, end)| CoreWindow::new(start, end))
                .collect::<Vec<_>>();
            (input, windows)
        });
        PyCalendar {
            inner: Arc::new(CoreCalendar::from_input_windows(&graph.inner, entries)),
        }
    }

    /// A calendar in which nothing is ever shut.
    #[staticmethod]
    fn unrestricted() -> Self {
        PyCalendar {
            inner: Arc::new(CoreCalendar::unrestricted()),
        }
    }

    /// How many edges carry a restriction.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Is `edge` open at `at` seconds past Monday midnight?
    fn is_open(&self, edge: EdgeId, at: u32) -> bool {
        self.inner.is_open(edge, at)
    }

    /// Does `edge` carry a schedule at all? An edge nobody scheduled is open at
    /// every hour, which looks the same as one that happens to be open now.
    fn is_restricted(&self, edge: EdgeId) -> bool {
        self.inner.is_restricted(edge)
    }

    fn __repr__(&self) -> String {
        format!("Calendar({} edges restricted)", self.inner.len())
    }
}

/// Earliest arrival from `sources`, leaving at `departing` on a weekly clock.
///
/// Dreyfus (1969). `waiting` is `"unrestricted"` — wait for a shut edge to open,
/// and pay the wait as travel time — or `"forbidden"`, which treats a shut edge
/// as absent. Costs come back as elapsed seconds, waiting included.
#[pyfunction]
#[pyo3(signature = (graph, calendar, sources, departing, *, waiting="unrestricted", targets=None, max_cost=None))]
#[allow(clippy::too_many_arguments)] // a query with a clock has one more knob
pub(crate) fn time_dependent_dijkstra(
    py: Python<'_>,
    graph: &PyGraph,
    calendar: &PyCalendar,
    sources: Vec<(NodeId, Weight)>,
    departing: u32,
    waiting: &str,
    targets: Option<Vec<NodeId>>,
    max_cost: Option<Weight>,
) -> PyResult<PySearchResult> {
    let waiting = match waiting {
        "unrestricted" => CoreWaiting::Unrestricted,
        "forbidden" => CoreWaiting::Forbidden,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown waiting policy {other:?}; expected 'unrestricted' or 'forbidden'"
            )))
        }
    };
    let departure = CoreDeparture::at(departing).waiting(waiting);
    let graph = Arc::clone(&graph.inner);
    let calendar = Arc::clone(&calendar.inner);
    let options = options(targets, max_cost);
    let result = py
        .detach(|| core_timedep(&graph, &calendar, &sources, &departure, &options))
        .map_err(value_err)?;
    Ok(PySearchResult { inner: result })
}
