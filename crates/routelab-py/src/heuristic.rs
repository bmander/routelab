//! Heuristics for goal-directed search.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use routelab_core::{
    Heuristic as _, Landmarks as CoreLandmarks, NodeId, Selection, StandardHeuristic, Weight,
};

use crate::graph::*;
use crate::progress::*;
use crate::value_err;

/// An estimate of the cost remaining to a target, for goal-directed search.
///
/// Built from data the environment gathered — never a Python callable: a callback
/// per settled node would cost more than the search it is meant to accelerate.
#[pyclass(name = "Heuristic", module = "routelab._routelab", frozen)]
pub struct PyHeuristic {
    pub(crate) inner: Arc<StandardHeuristic>,
}

#[pymethods]
impl PyHeuristic {
    /// Estimates nothing, which makes A* into Dijkstra.
    #[staticmethod]
    fn zero() -> Self {
        PyHeuristic {
            inner: Arc::new(StandardHeuristic::Zero),
        }
    }

    /// Straight-line distance between per-node coordinates, priced at
    /// `cost_per_distance` — which must be the cheapest rate any layer charges,
    /// or the estimate stops being a lower bound.
    #[staticmethod]
    fn euclidean(xs: Vec<f64>, ys: Vec<f64>, cost_per_distance: f64) -> PyResult<Self> {
        let heuristic =
            StandardHeuristic::euclidean(xs, ys, cost_per_distance).map_err(value_err)?;
        Ok(PyHeuristic {
            inner: Arc::new(heuristic),
        })
    }

    /// Distances measured from a handful of nodes, combined by the triangle
    /// inequality. Needs no coordinates — only the graph, and the time to walk
    /// it twice per landmark.
    ///
    /// `selection` is `"farthest"` or `"random"`.
    #[staticmethod]
    #[pyo3(signature = (graph, count, selection="farthest", seed=0, progress=None))]
    fn landmarks(
        py: Python<'_>,
        graph: &PyGraph,
        count: usize,
        selection: &str,
        seed: u64,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        let selection = match selection {
            "farthest" => Selection::Farthest,
            "random" => Selection::Random,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown selection {other:?}; expected 'farthest' or 'random'"
                )))
            }
        };
        let graph = Arc::clone(&graph.inner);
        // Two full searches per landmark: seconds on a city, and the reason
        // this is preprocessing rather than something a query can afford.
        let counter = counter(progress);
        let landmarks =
            py.detach(|| CoreLandmarks::build_reporting(&graph, count, selection, seed, &counter));
        Ok(PyHeuristic {
            inner: Arc::new(StandardHeuristic::Landmarks(landmarks)),
        })
    }

    /// How many nodes this heuristic holds data for, or `None` if it needs none.
    #[getter]
    fn coverage(&self) -> Option<usize> {
        self.inner.coverage()
    }

    /// Bytes of precomputed table this heuristic holds, if it holds any.
    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// The estimated cost from `node` to `target`. Exposed so a test can check
    /// admissibility directly, rather than only through its consequences.
    fn estimate(&self, node: NodeId, target: NodeId) -> Weight {
        self.inner.estimate(node, target)
    }

    /// Heuristics describe themselves in core, so this stays a conversion
    /// rather than growing an arm per kind.
    fn __repr__(&self) -> String {
        format!("Heuristic.{}", self.inner)
    }
}
