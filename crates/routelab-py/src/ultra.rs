//! ULTRA: the transfer shortcuts a timetable kernel walks instead of a closure.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::ultra::Ultra as CoreUltra;

use crate::graph::*;
use crate::progress::*;
use crate::timetable::*;

/// The intermediate transfers worth walking, worked out once.
#[pyclass(name = "Ultra", module = "routelab._routelab", frozen)]
pub struct PyUltra {
    inner: Arc<CoreUltra>,
    /// How many vertices the transfer graph numbered, so [`PyUltra::footpaths`]
    /// can hand back a set the kernels index the same way.
    vertices: usize,
}

#[pymethods]
impl PyUltra {
    /// Work out the shortcuts of `timetable` over the transfer graph
    /// `transfers`, counting source stops into `progress`.
    ///
    /// `transfers` is unrestricted: neither bounded by a radius nor
    /// transitively closed, which is the whole point. Its weights are
    /// durations and its vertices are the graph's, so a stop and a place no
    /// vehicle calls at are numbered alike.
    #[staticmethod]
    #[pyo3(signature = (timetable, transfers, progress = None))]
    fn compute(
        py: Python<'_>,
        timetable: &PyTimetable,
        transfers: &PyGraph,
        progress: Option<&PyProgress>,
    ) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let graph = Arc::clone(&transfers.inner);
        let counter = counter(progress);
        let vertices = graph.num_nodes();
        let built = py.detach(|| CoreUltra::compute_reporting(&timetable, &graph, &counter));
        PyUltra {
            inner: Arc::new(built),
            vertices,
        }
    }

    /// Shortcuts kept: the transfer set a query walks.
    #[getter]
    fn num_shortcuts(&self) -> usize {
        self.inner.len()
    }

    /// Candidates found before duplicates were dropped — what the enumeration
    /// produced, against what it kept.
    #[getter]
    fn candidates(&self) -> usize {
        self.inner.candidates()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// The shortcuts as `(from, to, duration)`.
    fn shortcuts(&self) -> Vec<(u32, u32, u32)> {
        self.inner.shortcuts().to_vec()
    }

    /// The shortcuts as the one-hop transfer set a timetable kernel takes —
    /// what it would otherwise be handed a transitive closure of.
    fn footpaths(&self) -> PyFootpaths {
        PyFootpaths::from(self.inner.footpaths(self.vertices))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Ultra({} shortcuts from {} candidates)",
            self.inner.len(),
            self.inner.candidates()
        )
    }
}
