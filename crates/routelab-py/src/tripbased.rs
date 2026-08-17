//! Trip-based routing: the transfer set, what a query found, and a profile.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::tripbased::{
    TripBased as CoreTripBased, TripBasedProfile as CoreTripBasedProfile,
    TripBasedSearch as CoreTripBasedSearch,
};
use routelab_core::model::timetable::Transfer;
use routelab_core::NodeId;

use crate::progress::*;
use crate::timetable::*;

/// A timetable as trips and the precomputed transfers between them.
#[pyclass(name = "TripBased", module = "routelab._routelab", frozen)]
pub struct PyTripBased {
    inner: Arc<CoreTripBased>,
}

#[pymethods]
impl PyTripBased {
    /// Compute the transfer set of a timetable, with `footpaths` between its
    /// stops if any are given, and reduce it unless told not to. Seconds on a
    /// city, counted into `progress` trip by trip through both phases.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None, reduce = true, progress = None))]
    fn build(
        py: Python<'_>,
        timetable: &PyTimetable,
        footpaths: Option<&PyFootpaths>,
        reduce: bool,
        progress: Option<&PyProgress>,
    ) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let counter = counter(progress);
        let built = py.detach(|| {
            CoreTripBased::build_reporting(
                &timetable,
                Transfer::instant(),
                &footpaths,
                reduce,
                &counter,
            )
        });
        PyTripBased {
            inner: Arc::new(built),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    /// Lines in the paper's sense — distinct stop sequences whose trips never
    /// overtake — which is more than a feed's own count of routes.
    #[getter]
    fn num_lines(&self) -> usize {
        self.inner.num_lines()
    }

    /// Trips in the paper's sense: one per unbroken chain of connections.
    #[getter]
    fn num_trips(&self) -> usize {
        self.inner.num_trips()
    }

    /// Transfers kept: the set a query scans.
    #[getter]
    fn num_transfers(&self) -> usize {
        self.inner.num_transfers()
    }

    /// Transfers computed before reduction.
    #[getter]
    fn num_initial_transfers(&self) -> usize {
        self.inner.num_initial_transfers()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Run the query from `sources` — `[(stop, time), ...]` — toward
    /// `target`, stopping after `max_transfers` changes if given.
    ///
    /// `departing` is what an elapsed cost is measured from; defaults to the
    /// earliest source.
    #[pyo3(signature = (sources, target, max_transfers = None, departing = None))]
    fn search(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        target: NodeId,
        max_transfers: Option<usize>,
        departing: Option<u32>,
    ) -> PyTripBasedSearch {
        let kernel = Arc::clone(&self.inner);
        let search = py.detach(|| kernel.search(&sources, target, max_transfers, departing));
        PyTripBasedSearch {
            kernel: Arc::clone(&self.inner),
            inner: Arc::new(search),
        }
    }

    /// Every journey worth leaving `source` on for `target` between
    /// `departing` and `until`: the Pareto set over departure, arrival and
    /// changes.
    fn profile(
        &self,
        py: Python<'_>,
        source: NodeId,
        target: NodeId,
        departing: u32,
        until: u32,
    ) -> PyTripBasedProfile {
        let kernel = Arc::clone(&self.inner);
        let profile = py.detach(|| kernel.profile(source, target, departing, until));
        PyTripBasedProfile {
            kernel: Arc::clone(&self.inner),
            inner: Arc::new(profile),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TripBased({} trips, {} transfers)",
            self.inner.num_trips(),
            self.inner.num_transfers()
        )
    }
}

/// What a trip-based query found: the segments it scanned and the journeys
/// that reached its target, one per number of changes worth making.
#[pyclass(name = "TripBasedSearch", module = "routelab._routelab", frozen)]
pub struct PyTripBasedSearch {
    kernel: Arc<CoreTripBased>,
    inner: Arc<CoreTripBasedSearch>,
}

#[pymethods]
impl PyTripBasedSearch {
    /// Distinct trips reached — what this kernel labels.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Trip segments scanned — the paper's own measure of work.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    /// What an elapsed cost is measured from.
    #[getter]
    fn departing(&self) -> u32 {
        self.inner.departing
    }

    /// The stop the query ran toward.
    #[getter]
    fn target(&self) -> NodeId {
        self.inner.target()
    }

    /// Rounds run: one more than the most changes any segment was reached with.
    #[getter]
    fn rounds(&self) -> usize {
        self.inner.rounds()
    }

    /// Earliest arrival at `stop`, or `None` — only the target has one.
    fn cost(&self, stop: NodeId) -> Option<u32> {
        self.inner.cost(stop)
    }

    /// Every segment scanned, as `(changes, trip, [stops...])`.
    fn reached(&self) -> Vec<(usize, u32, Vec<NodeId>)> {
        self.inner
            .reached(&self.kernel)
            .into_iter()
            .map(|(round, trip, stops)| (round, trip.0, stops))
            .collect()
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    fn path(&self, stop: NodeId) -> Option<Vec<NodeId>> {
        self.kernel.path(&self.inner, stop)
    }

    /// The earliest arrival at `stop`, as an itinerary.
    fn itinerary(&self, stop: NodeId) -> Option<PyItinerary> {
        self.kernel
            .itinerary(&self.inner, stop)
            .map(PyItinerary::from)
    }

    /// The Pareto front for `stop`, fewest changes first.
    fn itineraries(&self, stop: NodeId) -> Vec<PyItinerary> {
        self.kernel
            .itineraries(&self.inner, stop)
            .into_iter()
            .map(PyItinerary::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TripBasedSearch({} trips, {} segments scanned)",
            self.inner.settled, self.inner.scanned
        )
    }
}

/// What a trip-based profile found: the Pareto set of departures for one
/// source and target.
#[pyclass(name = "TripBasedProfile", module = "routelab._routelab", frozen)]
pub struct PyTripBasedProfile {
    kernel: Arc<CoreTripBased>,
    inner: Arc<CoreTripBasedProfile>,
}

#[pymethods]
impl PyTripBasedProfile {
    /// Distinct trips reached over every run.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Trip segments scanned over every run.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    /// Departures the source offered in the window: how many times the query
    /// loop ran.
    #[getter]
    fn runs(&self) -> usize {
        self.inner.runs
    }

    #[getter]
    fn source(&self) -> NodeId {
        self.inner.source()
    }

    #[getter]
    fn target(&self) -> NodeId {
        self.inner.target()
    }

    /// The direct walk from source to target, if there is one.
    fn walk(&self) -> Option<u32> {
        self.kernel.walk(&self.inner)
    }

    /// The Pareto set as `(departs, arrives, transfers)`, earliest first.
    fn triples(&self) -> Vec<(u32, u32, usize)> {
        self.kernel.triples(&self.inner)
    }

    /// The journey behind each triple, with its departure, earliest first.
    fn journeys(&self) -> Vec<(u32, PyItinerary)> {
        self.kernel
            .journeys(&self.inner)
            .into_iter()
            .map(|(dep, itinerary)| (dep, PyItinerary::from(itinerary)))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "TripBasedProfile({} journeys over {} departures)",
            self.inner.len(),
            self.inner.runs
        )
    }
}
