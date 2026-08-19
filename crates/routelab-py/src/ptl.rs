//! PTL: the labeling over a timetable's events, and its two queries.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::ptl::{PtlTechnique, PublicTransitLabeling as CoreLabeling};
use routelab_core::{Footprint, NodeId, Profiled, Technique};

use crate::progress::*;
use crate::timetable::*;

/// A timetable as its time-expanded event graph and 2-hop reachability
/// labels over it — built once, at some expense, and read by every query.
#[pyclass(name = "PublicTransitLabeling", module = "routelab._routelab", frozen)]
pub struct PyPublicTransitLabeling {
    inner: Arc<CoreLabeling>,
}

#[pymethods]
impl PyPublicTransitLabeling {
    /// Expand a timetable into its events and label them, with `footpaths`
    /// between its stops if any are given. Minutes on a city, paid once;
    /// `progress` counts hubs while `"labeling"` and stops while `"merging"`.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None, progress = None))]
    fn build(
        py: Python<'_>,
        timetable: &PyTimetable,
        footpaths: Option<&PyFootpaths>,
        progress: Option<&PyProgress>,
    ) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let counter = counter(progress);
        let labeling = py.detach(|| {
            PtlTechnique
                .bind(transit_network(&timetable, &footpaths), &counter)
                .expect("a labeling builds from any timetable")
        });
        PyPublicTransitLabeling {
            inner: Arc::new(labeling),
        }
    }

    /// What a query reads and how many there are: `("hubs", n)`.
    #[getter]
    fn searches(&self) -> (&'static str, usize) {
        Footprint::searches(&*self.inner)
    }

    /// [`profile`](Self::profile) under the name every profiled technique
    /// shares.
    fn departures(
        &self,
        py: Python<'_>,
        origin: NodeId,
        target: NodeId,
        departing: u32,
        until: u32,
    ) -> Vec<(u32, PyItinerary)> {
        let labeling = Arc::clone(&self.inner);
        py.detach(|| labeling.departures(origin, target, departing, until))
            .into_iter()
            .map(|(dep, itinerary)| (dep, PyItinerary::from(itinerary)))
            .collect()
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    /// Vertices of the event graph: distinct (stop, time) pairs.
    #[getter]
    fn num_events(&self) -> usize {
        self.inner.num_events()
    }

    /// Entries in the event labels, both directions — the size of the
    /// labeling.
    #[getter]
    fn num_hubs(&self) -> usize {
        self.inner.num_hubs()
    }

    /// Average hubs per event label.
    #[getter]
    fn hubs_per_label(&self) -> f64 {
        self.inner.hubs_per_label()
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
        let labeling = Arc::clone(&self.inner);
        py.detach(|| labeling.earliest_arrival(&sources, to))
            .map(PyItinerary::from)
    }

    /// Every journey worth leaving `origin` on for `target` within
    /// `[departing, until]`, as `(departure, itinerary)`, earliest first.
    fn profile(
        &self,
        py: Python<'_>,
        origin: NodeId,
        target: NodeId,
        departing: u32,
        until: u32,
    ) -> Vec<(u32, PyItinerary)> {
        let labeling = Arc::clone(&self.inner);
        py.detach(|| labeling.profile(origin, target, departing, until))
            .into_iter()
            .map(|(dep, itinerary)| (dep, PyItinerary::from(itinerary)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PublicTransitLabeling({} events, {} hubs)",
            self.inner.num_events(),
            self.inner.num_hubs()
        )
    }
}
