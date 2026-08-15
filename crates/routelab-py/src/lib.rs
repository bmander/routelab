//! Python bindings for `routelab-core`.
//!
//! This layer does three things and nothing else: convert Python values into
//! core types, release the GIL while the kernel runs, and turn core errors into
//! Python exceptions. Argument sugar (accepting a bare int for `sources`, and
//! so on) lives in the Python veneer, where it is easier to read and change.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use routelab_osm::{load as osm_load, OsmNetwork, Profile as OsmProfile};

use routelab_core::contraction::{
    ContractionHierarchy as CoreHierarchy, MeetingSearch as CoreMeetingSearch,
    Ordering as CoreOrdering, Policy,
};
use routelab_core::{
    astar as core_astar, bfs as core_bfs, dijkstra as core_dijkstra, EdgeId, Graph as CoreGraph,
    Heuristic as _, Landmarks as CoreLandmarks, Magnitude, NodeId, SearchOptions,
    SearchResult as CoreResult, SearchTree as CoreSearchTree, Selection, StandardHeuristic, Weight,
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
    #[pyo3(signature = (graph, deleted_neighbours=true, max_settled=500, max_hops=5))]
    fn edge_difference(
        py: Python<'_>,
        graph: &PyGraph,
        deleted_neighbours: bool,
        max_settled: usize,
        max_hops: usize,
    ) -> PyResult<Self> {
        Self::build(
            py,
            graph,
            Policy::EdgeDifference { deleted_neighbours },
            max_settled,
            max_hops,
        )
    }

    /// Contract in a fixed arbitrary order — the control.
    #[staticmethod]
    #[pyo3(signature = (graph, seed=0, max_settled=500, max_hops=5))]
    fn random(
        py: Python<'_>,
        graph: &PyGraph,
        seed: u64,
        max_settled: usize,
        max_hops: usize,
    ) -> PyResult<Self> {
        Self::build(py, graph, Policy::Random { seed }, max_settled, max_hops)
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
    ) -> PyResult<Self> {
        let ordering = CoreOrdering {
            policy,
            max_settled,
            max_hops,
        };
        let graph = Arc::clone(&graph.inner);
        let built = py.detach(|| CoreHierarchy::build(&graph, ordering));
        Ok(PyContractionHierarchy {
            inner: Arc::new(built.map_err(value_err)?),
            graph,
        })
    }
}

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
#[pyo3(signature = (path, speeds, *, respect_oneway=true, use_maxspeed=true))]
fn load_osm(
    py: Python<'_>,
    path: PathBuf,
    speeds: Vec<(String, f64)>,
    respect_oneway: bool,
    use_maxspeed: bool,
) -> PyResult<PyOsmNetwork> {
    let profile = OsmProfile::new(speeds, respect_oneway, use_maxspeed);
    let network = py.detach(|| osm_load(&path, &profile)).map_err(value_err)?;
    Ok(PyOsmNetwork {
        inner: Arc::new(network),
    })
}

/// An estimate of the cost remaining to a target, for goal-directed search.
///
/// Built from data the environment gathered — never a Python callable: a callback
/// per settled node would cost more than the search it is meant to accelerate.
#[pyclass(name = "Heuristic", module = "routelab._routelab", frozen)]
pub struct PyHeuristic {
    inner: Arc<StandardHeuristic>,
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
    #[pyo3(signature = (graph, count, selection="farthest", seed=0))]
    fn landmarks(
        py: Python<'_>,
        graph: &PyGraph,
        count: usize,
        selection: &str,
        seed: u64,
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
        let landmarks = py.detach(|| CoreLandmarks::build(&graph, count, selection, seed));
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

/// A* from `sources` to `target`, ordered by cost-so-far plus estimate.
#[pyfunction]
#[pyo3(signature = (graph, sources, target, heuristic, *, max_cost=None))]
fn astar(
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

#[pymodule]
fn _routelab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyContractionHierarchy>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyHeuristic>()?;
    m.add_class::<PyMeetingSearch>()?;
    m.add_class::<PyOsmNetwork>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PySearchTree>()?;
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    m.add_function(wrap_pyfunction!(load_osm, m)?)?;
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(bfs, m)?)?;
    Ok(())
}
