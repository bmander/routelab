//! CSA: the connection scan, what an earliest-arrival scan found, and a profile.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::csa::{
    ConnectionScan as CoreConnectionScan, ConnectionScanTechnique, ScanProfile as CoreScanProfile,
    ScanQuery, ScanSearch as CoreScanSearch,
};
use routelab_core::{NodeId, Progress, Technique};

use crate::built;
use crate::timetable::*;

/// A timetable laid out as one array of connections in departure order.
#[pyclass(name = "ConnectionScan", module = "routelab._routelab", frozen)]
pub struct PyConnectionScan {
    inner: Arc<CoreConnectionScan>,
}

#[pymethods]
impl PyConnectionScan {
    /// Sort a timetable's connections into the array, with `footpaths`
    /// between its stops if any are given. Milliseconds on a city.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None))]
    fn build(py: Python<'_>, timetable: &PyTimetable, footpaths: Option<&PyFootpaths>) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let scan = py.detach(|| {
            built(
                ConnectionScanTechnique
                    .bind(transit_network(&timetable, &footpaths), &Progress::new()),
            )
        });
        PyConnectionScan {
            inner: Arc::new(scan),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    #[getter]
    fn num_trips(&self) -> usize {
        self.inner.num_trips()
    }

    /// The length of the array.
    #[getter]
    fn num_connections(&self) -> usize {
        self.inner.num_connections()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Scan from `sources` — `[(stop, time), ...]` — stopping early if
    /// `target` is given, else labelling every stop.
    ///
    /// `departing` is what an elapsed cost is measured from; defaults to the
    /// earliest source.
    #[pyo3(signature = (sources, *, target = None, departing = None))]
    fn search(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        target: Option<NodeId>,
        departing: Option<u32>,
    ) -> PyScanSearch {
        let scan = Arc::clone(&self.inner);
        let query = ScanQuery { target, departing };
        let search = py.detach(|| scan.search(&sources, &query));
        PyScanSearch {
            scan: Arc::clone(&self.inner),
            inner: Arc::new(search),
        }
    }

    /// The profile toward `target` for every stop, over journeys leaving no
    /// earlier than `departing`; `prune` names the one stop the question is
    /// really about, so pairs its own profile dominates are not kept.
    #[pyo3(signature = (target, departing, prune = None))]
    fn profile(
        &self,
        py: Python<'_>,
        target: NodeId,
        departing: u32,
        prune: Option<NodeId>,
    ) -> PyScanProfile {
        let scan = Arc::clone(&self.inner);
        let profile = py.detach(|| scan.profile(target, departing, prune));
        PyScanProfile {
            scan: Arc::clone(&self.inner),
            inner: Arc::new(profile),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConnectionScan({} connections, {} trips)",
            self.inner.num_connections(),
            self.inner.num_trips()
        )
    }
}

/// What an earliest-arrival scan found: every stop's label, read back as any
/// target's itinerary.
#[pyclass(name = "ScanSearch", module = "routelab._routelab", frozen)]
pub struct PyScanSearch {
    scan: Arc<CoreConnectionScan>,
    inner: Arc<CoreScanSearch>,
}

#[pymethods]
impl PyScanSearch {
    /// Distinct stops that received a label.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Connections scanned — the paper's own measure of work.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    /// What an elapsed cost is measured from.
    #[getter]
    fn departing(&self) -> u32 {
        self.inner.departing
    }

    /// Earliest arrival at `stop`, or `None`.
    fn cost(&self, stop: NodeId) -> Option<u32> {
        self.inner.cost(stop)
    }

    /// Every stop reached, as `(stop, earliest arrival)`.
    fn reached(&self) -> Vec<(NodeId, u32)> {
        self.inner.reached()
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    fn path(&self, stop: NodeId) -> Option<Vec<NodeId>> {
        self.scan.path(&self.inner, stop)
    }

    /// The earliest arrival at `stop`, as an itinerary.
    fn itinerary(&self, stop: NodeId) -> Option<PyItinerary> {
        self.scan
            .itinerary(&self.inner, stop)
            .map(PyItinerary::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScanSearch({} stops, {} connections scanned)",
            self.inner.settled, self.inner.scanned
        )
    }
}

/// What a profile scan found: a Pareto profile per stop toward one target.
#[pyclass(name = "ScanProfile", module = "routelab._routelab", frozen)]
pub struct PyScanProfile {
    scan: Arc<CoreConnectionScan>,
    inner: Arc<CoreScanProfile>,
}

#[pymethods]
impl PyScanProfile {
    /// Stops whose profile holds at least one pair.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Connections scanned.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    #[getter]
    fn target(&self) -> NodeId {
        self.inner.target()
    }

    /// The direct walk from `stop` to the target, if there is one.
    fn walk(&self, stop: NodeId) -> Option<u32> {
        self.scan.walk(&self.inner, stop)
    }

    /// The Pareto pairs at `stop` leaving within `[departing, until]`,
    /// earliest first, as `(departs, arrives)`.
    fn pairs(&self, stop: NodeId, departing: u32, until: u32) -> Vec<(u32, u32)> {
        self.scan.pairs(&self.inner, stop, departing, until)
    }

    /// The journey behind each pair, with its departure, earliest first.
    fn journeys(&self, stop: NodeId, departing: u32, until: u32) -> Vec<(u32, PyItinerary)> {
        self.scan
            .journeys(&self.inner, stop, departing, until)
            .into_iter()
            .map(|(dep, itinerary)| (dep, PyItinerary::from(itinerary)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ScanProfile({} stops toward {})",
            self.inner.settled,
            self.inner.target()
        )
    }
}
