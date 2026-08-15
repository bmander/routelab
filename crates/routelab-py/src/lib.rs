//! Python bindings for `routelab-core`.
//!
//! This layer does three things and nothing else: convert Python values into
//! core types, release the GIL while the kernel runs, and turn core errors into
//! Python exceptions. Argument sugar (accepting a bare int for `sources`, and
//! so on) lives in the Python veneer, where it is easier to read and change.

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use routelab_core::{
    bfs as core_bfs, dijkstra as core_dijkstra, EdgeId, Graph as CoreGraph, NodeId, SearchOptions,
    SearchResult as CoreResult, Weight,
};

/// Core errors describe themselves; Python only needs the sentence.
fn value_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Bounds check for the id-taking accessors, phrased the way Python indexing errors are.
fn check_index(id: u32, count: usize, what: &str) -> PyResult<()> {
    if (id as usize) < count {
        Ok(())
    } else {
        Err(PyIndexError::new_err(format!(
            "{what} {id} is out of range for a graph with {count} {what}s"
        )))
    }
}

/// An immutable directed graph with integer weights.
///
/// `frozen` is what lets a search run without the GIL: the graph cannot be
/// mutated from Python while a kernel is reading it.
#[pyclass(name = "Graph", module = "routelab._routelab", frozen, subclass)]
pub struct PyGraph {
    inner: Arc<CoreGraph>,
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
    inner: CoreResult,
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

    /// Nodes that were reached, in settle order. (An alias for `order` that reads
    /// better when you do not care about the ordering.)
    fn reached(&self) -> Vec<NodeId> {
        self.order()
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(num_nodes={}, settled={})",
            self.inner.costs.len(),
            self.inner.order.len()
        )
    }
}

fn options(targets: Option<Vec<NodeId>>, max_cost: Option<Weight>) -> SearchOptions {
    SearchOptions { targets, max_cost }
}

/// Dijkstra's algorithm from `sources`, a list of `(node, initial_cost)` pairs.
#[pyfunction]
#[pyo3(signature = (graph, sources, *, targets=None, max_cost=None))]
fn dijkstra(
    py: Python<'_>,
    graph: &PyGraph,
    sources: Vec<(NodeId, Weight)>,
    targets: Option<Vec<NodeId>>,
    max_cost: Option<Weight>,
) -> PyResult<PySearchResult> {
    let graph = Arc::clone(&graph.inner);
    let options = options(targets, max_cost);
    let result = py
        .detach(|| core_dijkstra(&graph, &sources, &options))
        .map_err(value_err)?;
    Ok(PySearchResult { inner: result })
}

/// Breadth-first search from `sources`, which all start at depth 0.
#[pyfunction]
#[pyo3(signature = (graph, sources, *, targets=None, max_depth=None))]
fn bfs(
    py: Python<'_>,
    graph: &PyGraph,
    sources: Vec<NodeId>,
    targets: Option<Vec<NodeId>>,
    max_depth: Option<Weight>,
) -> PyResult<PySearchResult> {
    let graph = Arc::clone(&graph.inner);
    let options = options(targets, max_depth);
    let result = py
        .detach(|| core_bfs(&graph, &sources, &options))
        .map_err(value_err)?;
    Ok(PySearchResult { inner: result })
}

#[pymodule]
fn _routelab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(bfs, m)?)?;
    Ok(())
}
