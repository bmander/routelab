//! Reading OpenStreetMap extracts, and the geometry that comes with them.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;

use routelab_osm::{load as osm_load, OsmNetwork, Profile as OsmProfile};

use crate::{check_index, value_err};

/// A routable network read from an OpenStreetMap extract.
///
/// Holds the graph, the coordinates, and every edge's shape. The shape stays
/// here rather than crossing into Python: a city has hundreds of thousands of
/// edges and a route has a hundred, so points are handed over per edge, when
/// something actually needs to draw one.
#[pyclass(name = "OsmNetwork", module = "routelab._routelab", frozen)]
pub struct PyOsmNetwork {
    inner: Arc<OsmNetwork>,
}

#[pymethods]
impl PyOsmNetwork {
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// OSM node ids, in dense graph order — the labels an environment sees.
    #[getter]
    fn node_ids(&self) -> Vec<i64> {
        self.inner.node_ids.clone()
    }

    /// `(min_lat, min_lon, max_lat, max_lon)`.
    #[getter]
    fn bounds(&self) -> (f64, f64, f64, f64) {
        self.inner.bounds
    }

    /// Every edge as `(tail_index, head_index, seconds)`, over dense node ids.
    fn edges(&self) -> Vec<(u32, u32, u32)> {
        (0..self.inner.num_edges())
            .map(|edge| {
                (
                    self.inner.edge_tails[edge],
                    self.inner.edge_heads[edge],
                    self.inner.edge_weights[edge],
                )
            })
            .collect()
    }

    /// When each restricted edge may be travelled, as `(edge, [(start, end)])`
    /// on the weekly clock. Sparse: unlisted edges are always open.
    ///
    /// Keyed by position in this network's own edge list — the order `edges()`
    /// yields them — not by graph edge id, since a `Graph` permutes.
    fn windows(&self) -> Vec<(u32, Vec<(u32, u32)>)> {
        self.inner.edge_windows.clone()
    }

    /// Conditional tags the reader could not understand, and so ignored.
    #[getter]
    fn unreadable_schedules(&self) -> usize {
        self.inner.unreadable_schedules
    }

    /// Node coordinates projected into local metres, as `(xs, ys)`.
    fn projected(&self) -> (Vec<f64>, Vec<f64>) {
        self.inner.projected()
    }

    /// Node coordinates as they came off the map, as `(lats, lons)` — what you
    /// need to draw a node rather than route through it.
    fn coordinates(&self) -> (Vec<f64>, Vec<f64>) {
        (self.inner.lats.clone(), self.inner.lons.clone())
    }

    /// `(lat, lon)` along an edge, from its tail to its head.
    fn geometry(&self, edge: usize) -> PyResult<Vec<(f64, f64)>> {
        check_index(edge as u32, self.inner.num_edges(), "edge")?;
        Ok(self.inner.geometry(edge))
    }

    /// The dense index of the graph node closest to a coordinate, optionally
    /// limited to the given OSM node ids.
    #[pyo3(signature = (lat, lon, within=None))]
    fn nearest(&self, lat: f64, lon: f64, within: Option<Vec<i64>>) -> Option<usize> {
        match within {
            None => self.inner.nearest(lat, lon, None),
            Some(allowed) => self
                .inner
                .nearest(lat, lon, Some(&allowed.into_iter().collect())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OsmNetwork(num_nodes={}, num_edges={})",
            self.inner.num_nodes(),
            self.inner.num_edges()
        )
    }
}

/// Read an OSM extract into a routable network.
///
/// `speeds` maps a `highway` value to metres per second; a class left out is not
/// routable for this profile.
#[pyfunction]
#[pyo3(signature = (path, speeds, *, respect_oneway=true, use_maxspeed=true, access_keys=vec![]))]
pub(crate) fn load_osm(
    py: Python<'_>,
    path: PathBuf,
    speeds: Vec<(String, f64)>,
    respect_oneway: bool,
    use_maxspeed: bool,
    access_keys: Vec<String>,
) -> PyResult<PyOsmNetwork> {
    let profile = OsmProfile::new(speeds, respect_oneway, use_maxspeed).reading_access(access_keys);
    let network = py.detach(|| osm_load(&path, &profile)).map_err(value_err)?;
    Ok(PyOsmNetwork {
        inner: Arc::new(network),
    })
}

/// Great-circle metres between two `(lat, lon)` points, on the sphere every
/// OSM edge length is measured on — so a layer that prices its own walks
/// agrees with the streets about how long a metre is.
#[pyfunction]
pub(crate) fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    routelab_osm::haversine(lat1, lon1, lat2, lon2)
}
