//! ULTRA: the transfer shortcuts a timetable kernel walks instead of a closure.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::ultra::{Buckets as CoreBuckets, Ultra as CoreUltra};
use routelab_core::model::graph::UNREACHABLE;
use routelab_core::NodeId;

use crate::graph::*;
use crate::progress::*;
use crate::timetable::*;
use crate::value_err;

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

    /// The same shortcuts in another numbering — what the wrapped technique
    /// reads when it is bound to the transit network rather than the merged
    /// one. `slot_of` is indexed by this graph's vertices.
    fn footpaths_renumbered(&self, slot_of: Vec<NodeId>, stops: usize) -> PyFootpaths {
        PyFootpaths::from(self.inner.footpaths_renumbered(&slot_of, stops))
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

/// What the walks at either end of a journey cost, and the direct walk they
/// are measured against.
#[pyclass(name = "Endpoints", module = "routelab._routelab", frozen)]
pub struct PyEndpoints {
    /// The direct walk from source to target, if there is one.
    #[pyo3(get)]
    direct: Option<u32>,
    /// `(stop, cost)` for every stop the source can walk to sooner than it can
    /// walk the whole way — the paper's initial transfers, already pruned.
    #[pyo3(get)]
    to_stops: Vec<(NodeId, u32)>,
    /// `(stop, cost)` for every stop that can walk to the target sooner — the
    /// final transfers.
    #[pyo3(get)]
    from_stops: Vec<(NodeId, u32)>,
}

#[pymethods]
impl PyEndpoints {
    fn __repr__(&self) -> String {
        format!(
            "Endpoints({} initial, {} final, direct={})",
            self.to_stops.len(),
            self.from_stops.len(),
            match self.direct {
                Some(cost) => cost.to_string(),
                None => "None".to_string(),
            }
        )
    }
}

/// Bucket-CH over a transfer graph: the one-to-many searches an initial and a
/// final transfer are (the paper's §4.1).
#[pyclass(name = "Buckets", module = "routelab._routelab", frozen)]
pub struct PyBuckets {
    inner: Arc<CoreBuckets>,
}

#[pymethods]
impl PyBuckets {
    /// Contract `graph` and file a bucket entry per vertex each stop reaches.
    ///
    /// Minutes on a city's pavement, and the third of ULTRA's three
    /// preprocessing steps — the other two being the core graph and the
    /// shortcuts computed over it.
    #[staticmethod]
    #[pyo3(signature = (graph, stops, progress = None))]
    fn build(
        py: Python<'_>,
        graph: &PyGraph,
        stops: Vec<NodeId>,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        let graph = Arc::clone(&graph.inner);
        let counter = counter(progress);
        let built = py.detach(|| CoreBuckets::build(&graph, &stops, &counter));
        Ok(PyBuckets {
            inner: Arc::new(built.map_err(value_err)?),
        })
    }

    /// The direct walk from `sources` to `target`, and the transfers at either
    /// end worth having — Algorithm 2, lines 1 to 3.
    #[pyo3(signature = (sources, target))]
    fn endpoints(&self, sources: Vec<(NodeId, u32)>, target: NodeId) -> PyResult<PyEndpoints> {
        let found = self.inner.endpoints(&sources, target).map_err(value_err)?;
        let stops = self.inner.stops();
        let pairs = |costs: &[u32]| -> Vec<(NodeId, u32)> {
            costs
                .iter()
                .enumerate()
                .filter(|(_, &cost)| cost != UNREACHABLE)
                .map(|(index, &cost)| (stops[index], cost))
                .collect()
        };
        Ok(PyEndpoints {
            direct: found.direct,
            to_stops: pairs(&found.to_stop),
            from_stops: pairs(&found.from_stop),
        })
    }

    /// Where the cheapest walk to `to` started and the edges it crossed, for
    /// telling one as legs. `None` where there is no walk.
    fn path(
        &self,
        sources: Vec<(NodeId, u32)>,
        to: NodeId,
    ) -> PyResult<Option<(NodeId, Vec<u32>)>> {
        self.inner.path(&sources, to).map_err(value_err)
    }

    /// Bucket entries filed, both directions.
    #[getter]
    fn num_entries(&self) -> usize {
        self.inner.num_entries()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    fn __repr__(&self) -> String {
        format!(
            "Buckets({} stops, {} entries)",
            self.inner.stops().len(),
            self.inner.num_entries()
        )
    }
}
