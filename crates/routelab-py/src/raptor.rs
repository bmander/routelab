//! RAPTOR: the round-based search and what it found.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::raptor::{
    Raptor as CoreRaptor, RaptorQuery, RaptorSearch as CoreRaptorSearch,
};
use routelab_core::model::timetable::Transfer;
use routelab_core::NodeId;

use crate::timetable::*;

/// A timetable laid out as routes and trips, for round-based search.
#[pyclass(name = "Raptor", module = "routelab._routelab", frozen)]
pub struct PyRaptor {
    inner: Arc<CoreRaptor>,
}

#[pymethods]
impl PyRaptor {
    /// Lay a timetable out as routes and trips, with `footpaths` between its
    /// stops if any are given. Milliseconds on a city.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None))]
    fn build(py: Python<'_>, timetable: &PyTimetable, footpaths: Option<&PyFootpaths>) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let raptor = py.detach(|| CoreRaptor::build(&timetable, Transfer::instant(), &footpaths));
        PyRaptor {
            inner: Arc::new(raptor),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    /// Routes in the paper's sense — distinct stop sequences whose trips
    /// never overtake — which is more than a feed's own count of routes.
    #[getter]
    fn num_routes(&self) -> usize {
        self.inner.num_routes()
    }

    #[getter]
    fn num_trips(&self) -> usize {
        self.inner.num_trips()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Run the rounds from `sources` — `[(stop, time), ...]` — pruning toward
    /// `target` if one is given, and stopping after `max_rounds` rounds (round
    /// `k` allows `k` trips) or when a round improves nothing. Without a
    /// target this is the one-to-all search.
    ///
    /// `departing` is what an elapsed cost is measured from — the moment the
    /// question was asked, which is not the earliest source when every source
    /// carries a head start. Defaults to the earliest source.
    #[pyo3(signature = (sources, *, target = None, max_rounds = None, departing = None))]
    fn search(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        target: Option<NodeId>,
        max_rounds: Option<usize>,
        departing: Option<u32>,
    ) -> PyRaptorSearch {
        let raptor = Arc::clone(&self.inner);
        let query = RaptorQuery {
            target,
            max_rounds,
            departing,
        };
        let search = py.detach(|| raptor.search(&sources, &query));
        PyRaptorSearch {
            raptor: Arc::clone(&self.inner),
            inner: Arc::new(search),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Raptor({} routes, {} trips)",
            self.inner.num_routes(),
            self.inner.num_trips()
        )
    }
}

/// What a round-based search found: every stop's earliest arrival by round,
/// read back as any target's itinerary or its whole Pareto front.
#[pyclass(name = "RaptorSearch", module = "routelab._routelab", frozen)]
pub struct PyRaptorSearch {
    raptor: Arc<CoreRaptor>,
    inner: Arc<CoreRaptorSearch>,
}

#[pymethods]
impl PyRaptorSearch {
    /// Distinct stops that received a label.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Route scans, summed over rounds — the paper's own measure of work.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    /// Rounds run: `k` means journeys of up to `k` trips were considered.
    #[getter]
    fn rounds(&self) -> usize {
        self.inner.rounds()
    }

    /// The earliest source time; what an elapsed cost is measured from.
    #[getter]
    fn departing(&self) -> u32 {
        self.inner.departing
    }

    /// Earliest arrival at `stop` over every round, or `None`.
    fn cost(&self, stop: NodeId) -> Option<u32> {
        self.inner.cost(stop)
    }

    /// The round that first reached `stop`, or `None`.
    fn round_reached(&self, stop: NodeId) -> Option<usize> {
        self.inner.round_reached(stop)
    }

    /// Every stop reached, as `(stop, round first reached, earliest arrival)`.
    fn reached(&self) -> Vec<(NodeId, usize, u32)> {
        self.inner.reached()
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    fn path(&self, stop: NodeId) -> Option<Vec<NodeId>> {
        self.raptor.path(&self.inner, stop)
    }

    /// The earliest arrival at `stop`, however many changes it takes.
    fn itinerary(&self, stop: NodeId) -> Option<PyItinerary> {
        self.raptor
            .itinerary(&self.inner, stop)
            .map(PyItinerary::from)
    }

    /// One itinerary per number of changes that arrives strictly earlier than
    /// any with fewer — the Pareto front, fewest changes first.
    fn itineraries(&self, stop: NodeId) -> Vec<PyItinerary> {
        self.raptor
            .itineraries(&self.inner, stop)
            .into_iter()
            .map(PyItinerary::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RaptorSearch({} stops over {} rounds)",
            self.inner.settled,
            self.inner.rounds()
        )
    }
}
