//! The searches, called on integer node ids.

use std::sync::Arc;

use pyo3::prelude::*;

use routelab_core::{
    astar as core_astar, bfs as core_bfs, dijkstra as core_dijkstra, NodeId, SearchOptions, Weight,
};

use crate::graph::*;
use crate::heuristic::*;
use crate::value_err;

pub(crate) fn options(targets: Option<Vec<NodeId>>, max_cost: Option<Weight>) -> SearchOptions {
    SearchOptions {
        targets,
        max_cost,
        ..SearchOptions::default()
    }
}

/// Dijkstra's algorithm from `sources`, a list of `(node, initial_cost)` pairs.
#[pyfunction]
#[pyo3(signature = (graph, sources, *, targets=None, max_cost=None))]
pub(crate) fn dijkstra(
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
pub(crate) fn bfs(
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

/// A* from `sources` to `target`, ordered by cost-so-far plus estimate.
#[pyfunction]
#[pyo3(signature = (graph, sources, target, heuristic, *, max_cost=None))]
pub(crate) fn astar(
    py: Python<'_>,
    graph: &PyGraph,
    sources: Vec<(NodeId, Weight)>,
    target: NodeId,
    heuristic: &PyHeuristic,
    max_cost: Option<Weight>,
) -> PyResult<PySearchResult> {
    let graph = Arc::clone(&graph.inner);
    let heuristic = Arc::clone(&heuristic.inner);
    let options = options(None, max_cost);
    let result = py
        .detach(|| core_astar(&graph, &sources, target, heuristic.as_ref(), &options))
        .map_err(value_err)?;
    Ok(PySearchResult { inner: result })
}
