//! Core-CH: a network contracted down to the vertices worth keeping.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::kernels::contraction::{
    CoreHierarchy as Core, Ordering as CoreOrdering, Policy,
};
use routelab_core::{Graph as CoreGraph, NodeId};

use crate::graph::*;
use crate::progress::*;
use crate::value_err;

/// A graph contracted around the vertices a caller asked to keep.
#[pyclass(name = "CoreHierarchy", module = "routelab._routelab", frozen)]
pub struct PyCoreHierarchy {
    inner: Arc<Core>,
    core: Arc<CoreGraph>,
}

#[pymethods]
impl PyCoreHierarchy {
    /// Contract `graph` around `keep`, stopping when what is left standing
    /// averages more than `max_degree` arcs a vertex.
    ///
    /// `keep` never gets contracted whatever its priority — the stops, for a
    /// transfer graph. The degree bound is the other half of Core-CH: a core
    /// of stops alone has arcs quadratic in the stops, so contraction stops
    /// early and leaves some street corners standing to keep it sparse.
    ///
    /// Seconds to minutes on a city, paid once.
    #[staticmethod]
    #[pyo3(signature = (graph, keep, max_degree=20.0, max_settled=500, max_hops=5, progress=None))]
    fn build(
        py: Python<'_>,
        graph: &PyGraph,
        keep: Vec<NodeId>,
        max_degree: f64,
        max_settled: usize,
        max_hops: usize,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        let ordering = CoreOrdering {
            policy: Policy::EdgeDifference {
                deleted_neighbours: true,
            },
            max_settled,
            max_hops,
        };
        let graph = Arc::clone(&graph.inner);
        let counter = counter(progress);
        let built =
            py.detach(|| Core::build_reporting(&graph, &keep, ordering, max_degree, &counter));
        let built = built.map_err(value_err)?;
        Ok(PyCoreHierarchy {
            core: Arc::new(built.core().clone()),
            inner: Arc::new(built),
        })
    }

    /// The core: the vertices that survived and the arcs between them, in the
    /// original numbering, so a stop is the node it always was.
    #[getter]
    fn core(&self) -> PyGraph {
        PyGraph {
            inner: Arc::clone(&self.core),
        }
    }

    /// How many vertices are in the core.
    #[getter]
    fn num_core(&self) -> usize {
        self.inner.num_core()
    }

    /// How many were contracted away.
    #[getter]
    fn num_retired(&self) -> usize {
        self.inner.num_retired()
    }

    /// Arcs in the core, against the network it stands in for.
    #[getter]
    fn num_arcs(&self) -> usize {
        self.inner.num_arcs()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Did this vertex survive?
    fn is_core(&self, node: NodeId) -> bool {
        self.inner.is_core(node)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoreHierarchy({} vertices, {} arcs, {} retired)",
            self.inner.num_core(),
            self.inner.num_arcs(),
            self.inner.num_retired()
        )
    }
}
