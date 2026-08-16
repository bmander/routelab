//! Reading GTFS feeds.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use routelab_gtfs::{load as gtfs_load, Feed, Pairs};

use crate::value_err;

/// A GTFS feed's connections for one service day.
#[pyclass(name = "Feed", module = "routelab._routelab", frozen)]
pub struct PyFeed {
    inner: Arc<Feed>,
    /// Grouped once at load, because a layer asks for the edges and then for
    /// each edge's connections, and regrouping between the two questions would
    /// sort the whole day twice.
    pairs: Pairs,
}

#[pymethods]
impl PyFeed {
    #[getter]
    fn stop_ids(&self) -> Vec<String> {
        self.inner.stop_ids.clone()
    }

    #[getter]
    fn stop_names(&self) -> Vec<String> {
        self.inner.stop_names.clone()
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    #[getter]
    fn num_connections(&self) -> usize {
        self.inner.num_connections()
    }

    /// The stop pairs any connection runs along, as `(from, to, shortest_ride)`
    /// — the edges a routable layer contributes, weighted by a lower bound.
    fn edges(&self) -> Vec<(u32, u32, u32)> {
        (0..self.pairs.len())
            .map(|p| (self.pairs.from[p], self.pairs.to[p], self.pairs.shortest[p]))
            .collect()
    }

    /// The `(trip, departs, arrives)` triples running along the `index`-th of
    /// [`PyFeed::edges`], in departure order.
    fn connections(&self, index: usize) -> Vec<(u32, u32, u32)> {
        if index >= self.pairs.len() {
            return Vec::new();
        }
        self.pairs.connections(index).collect()
    }

    /// `(lats, lons)` of every stop, in dense index order.
    fn coordinates(&self) -> (Vec<f64>, Vec<f64>) {
        (self.inner.lats.clone(), self.inner.lons.clone())
    }

    /// The GTFS `trip_id` and `route_id` of each dense trip index.
    fn trips(&self) -> (Vec<String>, Vec<String>) {
        (self.inner.trip_ids.clone(), self.inner.trip_routes.clone())
    }

    /// Service this reader could not represent, as a count of trips. See
    /// `routelab_gtfs::Skipped` for what is deliberately excluded from it.
    #[getter]
    fn unimplemented(&self) -> usize {
        self.inner.skipped.unimplemented()
    }

    /// Trips passed over because their service does not run on the date asked
    /// for — the number that says the date filter did something.
    #[getter]
    fn trips_not_running(&self) -> usize {
        self.inner.skipped.trips_not_running
    }

    fn __repr__(&self) -> String {
        format!(
            "Feed({} stops, {} connections)",
            self.inner.num_stops(),
            self.inner.num_connections()
        )
    }
}

/// Read a GTFS feed's connections for one service day.
///
/// `path` may be a `.zip` or a directory of `.txt` files. The date is passed as
/// its parts rather than as a string, so there is no format to get wrong on
/// either side of the boundary.
#[pyfunction]
#[pyo3(signature = (path, year, month, day))]
pub(crate) fn load_gtfs(
    py: Python<'_>,
    path: PathBuf,
    year: i32,
    month: u32,
    day: u32,
) -> PyResult<PyFeed> {
    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| PyValueError::new_err(format!("{year}-{month}-{day} is not a date")))?;
    let (feed, pairs) = py
        .detach(|| {
            gtfs_load(&path, date).map(|feed| {
                let pairs = feed.pairs();
                (feed, pairs)
            })
        })
        .map_err(value_err)?;
    Ok(PyFeed {
        inner: Arc::new(feed),
        pairs,
    })
}
