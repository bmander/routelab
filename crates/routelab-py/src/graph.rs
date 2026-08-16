//! The graph, and what a search over it answers with.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use routelab_core::{
    EdgeId, Graph as CoreGraph, Magnitude, NodeId, SearchResult as CoreResult,
    SearchTree as CoreSearchTree, Weight,
};

use crate::{check_index, value_err};

/// An immutable directed graph with integer weights.
///
/// `frozen` is what lets a search run without the GIL: the graph cannot be
/// mutated from Python while a kernel is reading it.
#[pyclass(name = "Graph", module = "routelab._routelab", frozen, subclass)]
pub struct PyGraph {
    pub(crate) inner: Arc<CoreGraph>,
}

impl PyGraph {
    fn check_node(&self, node: NodeId) -> PyResult<()> {
        check_index(node, self.inner.num_nodes(), "node")
    }

    fn check_edge(&self, edge: EdgeId) -> PyResult<()> {
        check_index(edge, self.inner.num_edges(), "edge")
    }
}

#[pymethods]
impl PyGraph {
    /// Build a graph from `(tail, head, weight)` triples.
    #[new]
    #[pyo3(signature = (num_nodes, edges))]
    fn new(num_nodes: usize, edges: Vec<(NodeId, NodeId, Weight)>) -> PyResult<Self> {
        let graph = CoreGraph::from_edges(num_nodes, &edges).map_err(value_err)?;
        Ok(PyGraph {
            inner: Arc::new(graph),
        })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// `(tail, head, weight)` of `edge`.
    fn edge(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId, Weight)> {
        self.check_edge(edge)?;
        Ok(self.inner.edge(edge))
    }

    /// Position of `edge` in the edge list this graph was built from. Edges are
    /// permuted into CSR order, so this is how per-edge attributes stay attached.
    fn input_index(&self, edge: EdgeId) -> PyResult<u32> {
        self.check_edge(edge)?;
        Ok(self.inner.input_index(edge))
    }

    /// Ids of the out-edges of `node`, in CSR order.
    fn out_edges(&self, node: NodeId) -> PyResult<Vec<EdgeId>> {
        self.check_node(node)?;
        Ok(self.inner.out_edges(node).collect())
    }

    fn out_degree(&self, node: NodeId) -> PyResult<usize> {
        self.check_node(node)?;
        Ok(self.inner.out_degree(node))
    }

    /// `(head, weight, edge_id)` for each out-edge of `node`.
    fn neighbors(&self, node: NodeId) -> PyResult<Vec<(NodeId, Weight, EdgeId)>> {
        self.check_node(node)?;
        Ok(self
            .inner
            .out_edges(node)
            .map(|edge| (self.inner.head(edge), self.inner.weight(edge), edge))
            .collect())
    }

    /// Every edge as `(tail, head, weight)`, in CSR order.
    fn edges(&self) -> Vec<(NodeId, NodeId, Weight)> {
        self.inner.iter_edges().collect()
    }

    /// The same graph with every edge turned around.
    ///
    /// Searching it answers what a forward search cannot: the cost of *reaching*
    /// each node rather than leaving it. Edge ids do not survive — CSR order
    /// follows the tails, and the tails have changed.
    fn reversed(&self) -> PyGraph {
        PyGraph {
            inner: Arc::new(self.inner.reversed()),
        }
    }

    /// Follow `edges` from `start`; return `(end_node, total_weight)`, or raise
    /// if the sequence is not a walk. The independent check on a returned path.
    fn walk(&self, start: NodeId, edges: Vec<EdgeId>) -> PyResult<(NodeId, Weight)> {
        self.inner.walk(start, &edges).ok_or_else(|| {
            PyValueError::new_err(format!("edges do not form a walk starting at node {start}"))
        })
    }

    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(num_nodes={}, num_edges={})",
            self.inner.num_nodes(),
            self.inner.num_edges()
        )
    }
}

/// The shortest-path tree a search produced.
#[pyclass(name = "SearchResult", module = "routelab._routelab", frozen)]
pub struct PySearchResult {
    pub(crate) inner: CoreResult,
}

#[pymethods]
impl PySearchResult {
    /// Cost to `node`, or `None` if it was not reached.
    fn cost(&self, node: NodeId) -> Option<Weight> {
        self.inner.cost(node)
    }

    /// Node ids along the tree path to `node`, source first; `None` if unreached.
    fn path(&self, node: NodeId) -> Option<Vec<NodeId>> {
        self.inner.path(node)
    }

    /// Edge ids along the tree path to `node`; `None` if unreached, `[]` at a source.
    fn edge_path(&self, node: NodeId) -> Option<Vec<EdgeId>> {
        self.inner.edge_path(node)
    }

    /// The node `node` was reached from, or `None` at a source or unreached node.
    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.inner.parent(node)
    }

    /// The edge `node` was reached by, or `None` at a source or unreached node.
    fn parent_edge(&self, node: NodeId) -> Option<EdgeId> {
        self.inner.parent_edge(node)
    }

    /// Cost of every node, with `None` where a node was not reached.
    #[getter]
    fn costs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let costs = (0..self.inner.costs.len() as NodeId).map(|node| self.inner.cost(node));
        PyList::new(py, costs)
    }

    /// Nodes in the order they were settled — the search's trace.
    #[getter]
    fn order(&self) -> Vec<NodeId> {
        self.inner.order.clone()
    }

    /// How many nodes this search settled — the work it did.
    ///
    /// `len(result.order)` says the same thing here, but a search that settles
    /// in more than one direction has no single order to take the length of,
    /// and every result can answer this. It is also the one every comparison
    /// between algorithms is actually about.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.order.len()
    }

    /// Nodes that were reached, in settle order. (An alias for `order` that reads
    /// better when you do not care about the ordering.)
    fn reached(&self) -> Vec<NodeId> {
        self.order()
    }

    /// The shortest-path tree this search grew.
    ///
    /// `magnitude` is `"nodes"` or `"weight"`: what each branch should carry
    /// from the subtree beyond it. See `routelab_core::tree`.
    #[pyo3(signature = (graph, magnitude="weight"))]
    fn tree(&self, graph: &PyGraph, magnitude: &str) -> PyResult<PySearchTree> {
        let magnitude = match magnitude {
            "nodes" => Magnitude::Nodes,
            "weight" => Magnitude::Weight,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown magnitude {other:?}; expected 'nodes' or 'weight'"
                )))
            }
        };
        Ok(PySearchTree {
            inner: self.inner.tree(&graph.inner, magnitude),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(num_nodes={}, settled={})",
            self.inner.costs.len(),
            self.inner.order.len()
        )
    }
}

/// A shortest-path tree, as parallel arrays over its branches.
///
/// Arrays rather than a list of objects: a city-wide search is hundreds of
/// thousands of branches, and most callers filter before they iterate.
#[pyclass(name = "SearchTree", module = "routelab._routelab", frozen)]
pub struct PySearchTree {
    inner: CoreSearchTree,
}

#[pymethods]
impl PySearchTree {
    #[getter]
    fn tails(&self) -> Vec<NodeId> {
        self.inner.tails.clone()
    }

    #[getter]
    fn heads(&self) -> Vec<NodeId> {
        self.inner.heads.clone()
    }

    #[getter]
    fn edges(&self) -> Vec<EdgeId> {
        self.inner.edges.clone()
    }

    /// What each branch carries from the subtree beyond it.
    #[getter]
    fn magnitudes(&self) -> Vec<u64> {
        self.inner.magnitudes.clone()
    }

    /// The largest magnitude — what a renderer scales its widths against.
    #[getter]
    fn peak(&self) -> u64 {
        self.inner.peak()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchTree({} branches, peak={})",
            self.inner.len(),
            self.inner.peak()
        )
    }
}
