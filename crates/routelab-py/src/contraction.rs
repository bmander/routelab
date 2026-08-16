//! Contraction hierarchies: the preprocessed graph and its query.

use std::sync::{Arc, OnceLock};

use pyo3::prelude::*;

use routelab_core::util::progress::Progress as CoreProgress;

use routelab_core::kernels::contraction::{
    ContractionHierarchy as CoreHierarchy, MeetingSearch as CoreMeetingSearch,
    Ordering as CoreOrdering, Policy,
};
use routelab_core::{EdgeId, Graph as CoreGraph, NodeId, Weight};

use crate::graph::*;
use crate::progress::*;
use crate::value_err;

/// A contracted graph: the original edges, the shortcuts contraction added, and
/// the ranks that make a query only ever climb.
#[pyclass(name = "ContractionHierarchy", module = "routelab._routelab", frozen)]
pub struct PyContractionHierarchy {
    inner: Arc<CoreHierarchy>,
    graph: Arc<CoreGraph>,
}

#[pymethods]
impl PyContractionHierarchy {
    /// Contract a graph, taking whichever node costs least to lose next.
    ///
    /// One constructor per ordering policy rather than one taking a policy name,
    /// so each takes exactly its own arguments — the same shape as
    /// `Heuristic.euclidean` and `Heuristic.landmarks`. The witness limits are
    /// shared because every policy contracts the same way once the order is
    /// decided; they say how hard contraction looks for an existing path before
    /// giving up and inserting a shortcut it may not have needed.
    ///
    /// Seconds to minutes on a city, paid once.
    #[staticmethod]
    #[pyo3(signature = (graph, deleted_neighbours=true, max_settled=500, max_hops=5, progress=None))]
    fn edge_difference(
        py: Python<'_>,
        graph: &PyGraph,
        deleted_neighbours: bool,
        max_settled: usize,
        max_hops: usize,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            graph,
            Policy::EdgeDifference { deleted_neighbours },
            max_settled,
            max_hops,
            progress,
        )
    }

    /// Contract in a fixed arbitrary order — the control.
    #[staticmethod]
    #[pyo3(signature = (graph, seed=0, max_settled=500, max_hops=5, progress=None))]
    fn random(
        py: Python<'_>,
        graph: &PyGraph,
        seed: u64,
        max_settled: usize,
        max_hops: usize,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            graph,
            Policy::Random { seed },
            max_settled,
            max_hops,
            progress,
        )
    }

    /// Search from `sources` to `target`, upward from both ends.
    fn query(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, Weight)>,
        target: NodeId,
    ) -> PyResult<PyMeetingSearch> {
        let hierarchy = Arc::clone(&self.inner);
        let search = py
            .detach(|| hierarchy.query(&sources, target))
            .map_err(value_err)?;
        Ok(PyMeetingSearch {
            search,
            hierarchy: Arc::clone(&self.inner),
            graph: Arc::clone(&self.graph),
            unpacked: OnceLock::new(),
        })
    }

    #[getter]
    fn num_shortcuts(&self) -> usize {
        self.inner.num_shortcuts()
    }

    #[getter]
    fn num_arcs(&self) -> usize {
        self.inner.num_arcs()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// The contraction rank of every node, highest meaning most important.
    #[getter]
    fn ranks(&self) -> Vec<u32> {
        (0..self.inner.num_nodes() as NodeId)
            .map(|node| self.inner.rank(node))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ContractionHierarchy({} nodes, {} arcs, {} shortcuts)",
            self.inner.num_nodes(),
            self.inner.num_arcs(),
            self.inner.num_shortcuts()
        )
    }
}

/// What a hierarchy query explored: a search from each end and where they met.
///
/// Answers in the caller's edges, not the hierarchy's — `edge_path` unpacks
/// every shortcut back into the original edges it stands for, which is what lets
/// journeys, geometry and provenance work unchanged.
#[pyclass(name = "MeetingSearch", module = "routelab._routelab", frozen)]
pub struct PyMeetingSearch {
    search: CoreMeetingSearch,
    hierarchy: Arc<CoreHierarchy>,
    graph: Arc<CoreGraph>,
    /// The unpacked path, worked out on first ask.
    ///
    /// A caller building a journey wants the edges and the node sequence, and
    /// unpacking a cross-city path expands hundreds of shortcuts — doing it once
    /// per question rather than once per query was most of the cost of asking.
    unpacked: OnceLock<Option<Vec<EdgeId>>>,
}

#[pymethods]
impl PyMeetingSearch {
    /// The distance to `node` — known only for the target the query aimed at.
    fn cost(&self, node: NodeId) -> Option<Weight> {
        self.search.cost(node)
    }

    /// The original edges of the path, source first.
    fn edge_path(&self, node: NodeId) -> Option<Vec<EdgeId>> {
        (node == self.search.target)
            .then(|| self.unpacked().clone())
            .flatten()
    }

    /// The nodes along the path, source first.
    fn path(&self, node: NodeId) -> Option<Vec<NodeId>> {
        let edges = (node == self.search.target)
            .then(|| self.unpacked().as_ref())
            .flatten()?;
        let mut nodes = vec![self.search.origin()?];
        nodes.extend(edges.iter().map(|&edge| self.graph.head(edge)));
        Some(nodes)
    }

    #[getter]
    fn meeting(&self) -> Option<NodeId> {
        self.search.meeting
    }

    /// Nodes settled across both halves — the work the query did.
    #[getter]
    fn settled(&self) -> usize {
        self.search.settled()
    }

    /// Nodes settled by each half, forward first.
    ///
    /// Not `order`, which is what a one-tree search calls its single settle
    /// sequence: there are two here, and a caller that treated this as one list
    /// would find it always two long.
    #[getter]
    fn halves(&self) -> (Vec<NodeId>, Vec<NodeId>) {
        (
            self.search.forward.order.clone(),
            self.search.backward.order.clone(),
        )
    }

    /// Every branch of both search trees, as
    /// `(direction, tail, head, level, original_edges)`.
    ///
    /// `direction` is 0 forward and 1 backward; `level` is the rank of the
    /// branch's higher end; the edges are unpacked, so a branch that is one
    /// shortcut may carry hundreds. That count is what shows the hierarchy: a
    /// long arc is a road the search never had to look at.
    fn branches(&self) -> Vec<(u8, NodeId, NodeId, u32, Vec<EdgeId>)> {
        let mut branches = Vec::new();
        for (direction, half) in [(0u8, &self.search.forward), (1u8, &self.search.backward)] {
            for &node in &half.order {
                let Some((parent, edge)) = half.arrival(node) else {
                    continue;
                };
                let augmented = if direction == 0 {
                    self.hierarchy.upward_edge(edge)
                } else {
                    self.hierarchy.downward_edge(edge)
                };
                let mut edges = Vec::new();
                self.hierarchy.expand_into(augmented, &mut edges);
                let level = self.hierarchy.rank(parent).max(self.hierarchy.rank(node));
                branches.push((direction, parent, node, level, edges));
            }
        }
        branches
    }

    fn __repr__(&self) -> String {
        format!(
            "MeetingSearch(settled={}, distance={:?})",
            self.search.settled(),
            self.search.distance
        )
    }
}

impl PyMeetingSearch {
    /// The unpacked path, computed once.
    fn unpacked(&self) -> &Option<Vec<EdgeId>> {
        self.unpacked
            .get_or_init(|| self.hierarchy.unpack(&self.search))
    }
}

impl PyContractionHierarchy {
    /// Contract, with the GIL released — the shared half of every constructor.
    fn build(
        py: Python<'_>,
        graph: &PyGraph,
        policy: Policy,
        max_settled: usize,
        max_hops: usize,
        progress: Option<&PyProgress>,
    ) -> PyResult<Self> {
        let ordering = CoreOrdering {
            policy,
            max_settled,
            max_hops,
        };
        let graph = Arc::clone(&graph.inner);
        let counter = progress.map_or_else(CoreProgress::new, |p| p.inner.clone());
        let built = py.detach(|| CoreHierarchy::build_reporting(&graph, ordering, &counter));
        Ok(PyContractionHierarchy {
            inner: Arc::new(built.map_err(value_err)?),
            graph,
        })
    }
}
