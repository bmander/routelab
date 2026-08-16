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

use routelab_core::progress::Progress as CoreProgress;
use routelab_gtfs::{load as gtfs_load, Feed, Pairs};
use routelab_osm::{load as osm_load, OsmNetwork, Profile as OsmProfile};

use routelab_core::contraction::{
    ContractionHierarchy as CoreHierarchy, MeetingSearch as CoreMeetingSearch,
    Ordering as CoreOrdering, Policy,
};
use routelab_core::timedep::{
    time_dependent_dijkstra as core_timedep, Calendar as CoreCalendar, Departure as CoreDeparture,
    Waiting as CoreWaiting, Window as CoreWindow,
};
use routelab_core::timetable::{
    earliest_arrival as core_earliest_arrival, Footpaths as CoreFootpaths,
    Itinerary as CoreItinerary, Leg, Raptor as CoreRaptor, RaptorSearch as CoreRaptorSearch,
    TimeExpanded as CoreTimeExpanded, Timetable as CoreTimetable, Transfer,
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
fn load_osm(
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

/// When each edge may be travelled, on a weekly clock.
///
/// `frozen`, like every other preprocessed structure here, so a search can read
/// it with the GIL released.
#[pyclass(name = "Calendar", module = "routelab._routelab", frozen)]
pub struct PyCalendar {
    inner: Arc<CoreCalendar>,
}

#[pymethods]
impl PyCalendar {
    /// Build from windows keyed by **position in the graph's input edge list**.
    ///
    /// Not by edge id: `Graph` permutes its edges into CSR order, and a calendar
    /// keyed the wrong way shuts the wrong streets without failing. Everything
    /// that produces edges holds input positions, so that is what this takes.
    #[staticmethod]
    #[pyo3(signature = (graph, windows))]
    fn from_windows(graph: &PyGraph, windows: Vec<(u32, Vec<(u32, u32)>)>) -> Self {
        let entries = windows.into_iter().map(|(input, windows)| {
            let windows = windows
                .into_iter()
                .map(|(start, end)| CoreWindow::new(start, end))
                .collect::<Vec<_>>();
            (input, windows)
        });
        PyCalendar {
            inner: Arc::new(CoreCalendar::from_input_windows(&graph.inner, entries)),
        }
    }

    /// A calendar in which nothing is ever shut.
    #[staticmethod]
    fn unrestricted() -> Self {
        PyCalendar {
            inner: Arc::new(CoreCalendar::unrestricted()),
        }
    }

    /// How many edges carry a restriction.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Is `edge` open at `at` seconds past Monday midnight?
    fn is_open(&self, edge: EdgeId, at: u32) -> bool {
        self.inner.is_open(edge, at)
    }

    /// Does `edge` carry a schedule at all? An edge nobody scheduled is open at
    /// every hour, which looks the same as one that happens to be open now.
    fn is_restricted(&self, edge: EdgeId) -> bool {
        self.inner.is_restricted(edge)
    }

    fn __repr__(&self) -> String {
        format!("Calendar({} edges restricted)", self.inner.len())
    }
}

/// A GTFS feed's connections for one service day.
#[pyclass(name = "Feed", module = "routelab._routelab", frozen)]
pub struct PyFeed {
    inner: Arc<Feed>,
    /// Grouped once at load, because a layer asks for the edges and then for
    /// each edge's connections, and regrouping between the two questions would
    /// sort the whole day twice.
    pairs: Pairs,
}

#[pymethods]
impl PyFeed {
    #[getter]
    fn stop_ids(&self) -> Vec<String> {
        self.inner.stop_ids.clone()
    }

    #[getter]
    fn stop_names(&self) -> Vec<String> {
        self.inner.stop_names.clone()
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    #[getter]
    fn num_connections(&self) -> usize {
        self.inner.num_connections()
    }

    /// The stop pairs any connection runs along, as `(from, to, shortest_ride)`
    /// — the edges a routable layer contributes, weighted by a lower bound.
    fn edges(&self) -> Vec<(u32, u32, u32)> {
        (0..self.pairs.len())
            .map(|p| (self.pairs.from[p], self.pairs.to[p], self.pairs.shortest[p]))
            .collect()
    }

    /// The `(trip, departs, arrives)` triples running along the `index`-th of
    /// [`PyFeed::edges`], in departure order.
    fn connections(&self, index: usize) -> Vec<(u32, u32, u32)> {
        if index >= self.pairs.len() {
            return Vec::new();
        }
        self.pairs.connections(index).collect()
    }

    /// `(lats, lons)` of every stop, in dense index order.
    fn coordinates(&self) -> (Vec<f64>, Vec<f64>) {
        (self.inner.lats.clone(), self.inner.lons.clone())
    }

    /// The GTFS `trip_id` and `route_id` of each dense trip index.
    fn trips(&self) -> (Vec<String>, Vec<String>) {
        (self.inner.trip_ids.clone(), self.inner.trip_routes.clone())
    }

    /// Service this reader could not represent, as a count of trips. See
    /// `routelab_gtfs::Skipped` for what is deliberately excluded from it.
    #[getter]
    fn unimplemented(&self) -> usize {
        self.inner.skipped.unimplemented()
    }

    /// Trips passed over because their service does not run on the date asked
    /// for — the number that says the date filter did something.
    #[getter]
    fn trips_not_running(&self) -> usize {
        self.inner.skipped.trips_not_running
    }

    fn __repr__(&self) -> String {
        format!(
            "Feed({} stops, {} connections)",
            self.inner.num_stops(),
            self.inner.num_connections()
        )
    }
}

/// Departures along one edge, as `(trip, departs, arrives)`.
type Departures = Vec<(u32, u32, u32)>;

/// Stop-to-stop walks a rider may take at any time, for a fixed duration.
///
/// The paper's foot-edges, closed under composition — see
/// [`CoreFootpaths`]. Built from positions in a graph's input edge list, each
/// named edge becoming a walk between its endpoints taking its weight.
#[pyclass(name = "Footpaths", module = "routelab._routelab", frozen)]
pub struct PyFootpaths {
    inner: Arc<CoreFootpaths>,
}

#[pymethods]
impl PyFootpaths {
    /// Every edge at the given **positions in the graph's input edge list**
    /// becomes a footpath between its endpoints, taking its weight.
    #[staticmethod]
    fn from_edges(graph: &PyGraph, positions: Vec<u32>) -> Self {
        PyFootpaths {
            inner: Arc::new(CoreFootpaths::from_input_edges(&graph.inner, positions)),
        }
    }

    /// No footpaths at all: the paper's plain model.
    #[staticmethod]
    fn none() -> Self {
        PyFootpaths {
            inner: Arc::new(CoreFootpaths::none()),
        }
    }

    /// How long the walk between two stops takes, if there is one — after
    /// closure, so a two-hop walk answers too.
    fn duration(&self, from: NodeId, to: NodeId) -> Option<u32> {
        self.inner.duration(from, to)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    fn __repr__(&self) -> String {
        format!("Footpaths({} walks)", self.inner.len())
    }
}

/// The footpaths a query walks: the ones given, or none.
fn footpaths_or_none(footpaths: Option<&PyFootpaths>) -> Arc<CoreFootpaths> {
    footpaths.map_or_else(
        || Arc::new(CoreFootpaths::none()),
        |paths| Arc::clone(&paths.inner),
    )
}

/// A day's connections, indexed for a search to read.
///
/// Stops are a graph's nodes, and each connection is filed under the edge it
/// runs along, so nothing here has to pair up parallel columns by hand. See
/// [`CoreTimetable::from_input_connections`].
#[pyclass(name = "Timetable", module = "routelab._routelab", frozen)]
pub struct PyTimetable {
    inner: Arc<CoreTimetable>,
}

#[pymethods]
impl PyTimetable {
    /// Build from connections keyed by **position in the graph's input edge
    /// list** — `{position: [(trip, departs, arrives), ...]}`.
    ///
    /// Not by edge id, for the reason [`PyCalendar::from_windows`] says.
    #[staticmethod]
    #[pyo3(signature = (graph, connections))]
    fn from_connections(graph: &PyGraph, connections: Vec<(u32, Departures)>) -> Self {
        PyTimetable {
            inner: Arc::new(CoreTimetable::from_input_connections(
                &graph.inner,
                connections,
            )),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    #[getter]
    fn num_connections(&self) -> usize {
        self.inner.num_connections()
    }

    /// Stop-to-stop pairs any connection runs along — the edge count of the
    /// time-dependent model, whose node count is `num_stops`.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Earliest arrival at `to` from `sources` — `[(stop, time), ...]`, each
    /// a stop and when you are standing there — walking `footpaths` between
    /// stops if any are given.
    ///
    /// The time-dependent model: a node per stop, and a search that asks each
    /// edge what leaves next.
    #[pyo3(signature = (sources, to, footpaths = None))]
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        to: NodeId,
        footpaths: Option<&PyFootpaths>,
    ) -> Option<PyItinerary> {
        let timetable = Arc::clone(&self.inner);
        let footpaths = footpaths_or_none(footpaths);
        py.detach(|| {
            core_earliest_arrival(&timetable, &sources, to, Transfer::instant(), &footpaths)
        })
        .map(PyItinerary::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "Timetable({} stops, {} connections)",
            self.inner.num_stops(),
            self.inner.num_connections()
        )
    }
}

/// A timetable expanded into a graph with a node per event.
#[pyclass(name = "TimeExpanded", module = "routelab._routelab", frozen)]
pub struct PyTimeExpanded {
    inner: Arc<CoreTimeExpanded>,
}

#[pymethods]
impl PyTimeExpanded {
    /// Expand a timetable, with `footpaths` between its stops if any are
    /// given. Seconds on a city, paid once.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None))]
    fn build(py: Python<'_>, timetable: &PyTimetable, footpaths: Option<&PyFootpaths>) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let expanded = py.detach(|| {
            CoreTimeExpanded::build(&timetable, Transfer::instant(), &footpaths)
        });
        PyTimeExpanded {
            inner: Arc::new(expanded),
        }
    }

    #[getter]
    fn num_events(&self) -> usize {
        self.inner.num_events()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Earliest arrival at `to` from `sources` — `[(stop, time), ...]`.
    fn earliest_arrival(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        to: NodeId,
    ) -> Option<PyItinerary> {
        let expanded = Arc::clone(&self.inner);
        py.detach(|| expanded.earliest_arrival(&sources, to))
            .map(PyItinerary::from)
    }

    fn __repr__(&self) -> String {
        format!("TimeExpanded({} events)", self.inner.num_events())
    }
}

/// A timetable laid out as routes and trips, for round-based search.
#[pyclass(name = "Raptor", module = "routelab._routelab", frozen)]
pub struct PyRaptor {
    inner: Arc<CoreRaptor>,
}

#[pymethods]
impl PyRaptor {
    /// Lay a timetable out as routes and trips, with `footpaths` between its
    /// stops if any are given. Milliseconds on a city.
    #[staticmethod]
    #[pyo3(signature = (timetable, footpaths = None))]
    fn build(py: Python<'_>, timetable: &PyTimetable, footpaths: Option<&PyFootpaths>) -> Self {
        let timetable = Arc::clone(&timetable.inner);
        let footpaths = footpaths_or_none(footpaths);
        let raptor = py.detach(|| CoreRaptor::build(&timetable, Transfer::instant(), &footpaths));
        PyRaptor {
            inner: Arc::new(raptor),
        }
    }

    #[getter]
    fn num_stops(&self) -> usize {
        self.inner.num_stops()
    }

    /// Routes in the paper's sense — distinct stop sequences whose trips
    /// never overtake — which is more than a feed's own count of routes.
    #[getter]
    fn num_routes(&self) -> usize {
        self.inner.num_routes()
    }

    #[getter]
    fn num_trips(&self) -> usize {
        self.inner.num_trips()
    }

    #[getter]
    fn footprint(&self) -> usize {
        self.inner.footprint()
    }

    /// Run the rounds from `sources` — `[(stop, time), ...]` — pruning toward
    /// `target` if one is given, and stopping after `max_rounds` rounds (round
    /// `k` allows `k` trips) or when a round improves nothing. Without a
    /// target this is the one-to-all search.
    ///
    /// `departing` is what an elapsed cost is measured from — the moment the
    /// question was asked, which is not the earliest source when every source
    /// carries a head start. Defaults to the earliest source.
    #[pyo3(signature = (sources, target = None, max_rounds = None, departing = None))]
    fn search(
        &self,
        py: Python<'_>,
        sources: Vec<(NodeId, u32)>,
        target: Option<NodeId>,
        max_rounds: Option<usize>,
        departing: Option<u32>,
    ) -> PyRaptorSearch {
        let raptor = Arc::clone(&self.inner);
        let search = py.detach(|| raptor.search(&sources, target, max_rounds, departing));
        PyRaptorSearch {
            raptor: Arc::clone(&self.inner),
            inner: Arc::new(search),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Raptor({} routes, {} trips)",
            self.inner.num_routes(),
            self.inner.num_trips()
        )
    }
}

/// What a round-based search found: every stop's earliest arrival by round,
/// read back as any target's itinerary or its whole Pareto front.
#[pyclass(name = "RaptorSearch", module = "routelab._routelab", frozen)]
pub struct PyRaptorSearch {
    raptor: Arc<CoreRaptor>,
    inner: Arc<CoreRaptorSearch>,
}

#[pymethods]
impl PyRaptorSearch {
    /// Distinct stops that received a label.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    /// Route scans, summed over rounds — the paper's own measure of work.
    #[getter]
    fn scanned(&self) -> usize {
        self.inner.scanned
    }

    /// Rounds run: `k` means journeys of up to `k` trips were considered.
    #[getter]
    fn rounds(&self) -> usize {
        self.inner.rounds()
    }

    /// The earliest source time; what an elapsed cost is measured from.
    #[getter]
    fn departing(&self) -> u32 {
        self.inner.departing
    }

    /// Earliest arrival at `stop` over every round, or `None`.
    fn cost(&self, stop: NodeId) -> Option<u32> {
        self.inner.cost(stop)
    }

    /// The round that first reached `stop`, or `None`.
    fn round_reached(&self, stop: NodeId) -> Option<usize> {
        self.inner.round_reached(stop)
    }

    /// Every stop reached, as `(stop, round first reached, earliest arrival)`.
    fn reached(&self) -> Vec<(NodeId, usize, u32)> {
        self.inner.reached()
    }

    /// The stops along the earliest itinerary to `stop`, sources first.
    fn path(&self, stop: NodeId) -> Option<Vec<NodeId>> {
        self.raptor.path(&self.inner, stop)
    }

    /// The earliest arrival at `stop`, however many changes it takes.
    fn itinerary(&self, stop: NodeId) -> Option<PyItinerary> {
        self.raptor.itinerary(&self.inner, stop).map(PyItinerary::from)
    }

    /// One itinerary per number of changes that arrives strictly earlier than
    /// any with fewer — the Pareto front, fewest changes first.
    fn itineraries(&self, stop: NodeId) -> Vec<PyItinerary> {
        self.raptor
            .itineraries(&self.inner, stop)
            .into_iter()
            .map(PyItinerary::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RaptorSearch({} stops over {} rounds)",
            self.inner.settled,
            self.inner.rounds()
        )
    }
}

/// When you get there, what you rode, and what the search cost to find it.
#[pyclass(name = "Itinerary", module = "routelab._routelab", frozen)]
pub struct PyItinerary {
    inner: CoreItinerary,
}

impl From<CoreItinerary> for PyItinerary {
    fn from(inner: CoreItinerary) -> Self {
        PyItinerary { inner }
    }
}

#[pymethods]
impl PyItinerary {
    /// Absolute arrival time, in seconds since the service day's midnight.
    #[getter]
    fn arrives(&self) -> u32 {
        self.inner.arrives
    }

    /// Nodes the search settled — stops for one model, events for the other.
    /// The number the two models are actually compared on.
    #[getter]
    fn settled(&self) -> usize {
        self.inner.settled
    }

    #[getter]
    fn transfers(&self) -> usize {
        self.inner.transfers()
    }

    /// Each vehicle ridden, as `(trip, from_stop, to_stop, departs, arrives)`,
    /// with the walks between them left out.
    fn rides(&self) -> Vec<(u32, NodeId, NodeId, u32, u32)> {
        self.inner
            .rides()
            .map(|r| (r.trip, r.from, r.to, r.departs, r.arrives))
            .collect()
    }

    /// Every stretch of the journey in order, as
    /// `(trip, from_stop, to_stop, departs, arrives)` — `trip` is `None` for
    /// a stretch walked between stops.
    fn legs(&self) -> Vec<(Option<u32>, NodeId, NodeId, u32, u32)> {
        self.inner
            .legs
            .iter()
            .map(|leg| match leg {
                Leg::Ride(r) => (Some(r.trip), r.from, r.to, r.departs, r.arrives),
                Leg::Walk(w) => (None, w.from, w.to, w.departs, w.arrives),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Itinerary(arrives={}, {} rides, {} walks, {} transfers)",
            self.inner.arrives,
            self.inner.rides().count(),
            self.inner.walks().count(),
            self.inner.transfers()
        )
    }
}

/// How far along a piece of preprocessing is.
///
/// Hand one to a preprocessing call and read it from another thread while that
/// call runs. Reading takes no lock and does not need the GIL back, which is
/// the point: preprocessing releases the GIL precisely so that something else
/// can be doing something, and a progress report that took it back would be
/// asking the work to stop in order to say it had not.
#[pyclass(name = "Progress", module = "routelab._routelab", frozen)]
#[derive(Default)]
pub struct PyProgress {
    inner: CoreProgress,
}

#[pymethods]
impl PyProgress {
    #[new]
    fn new() -> Self {
        PyProgress::default()
    }

    /// `(phase, done, total)`. `total` is 0 when the work cannot say — which is
    /// different from not having started, and worth being able to tell apart.
    fn read(&self) -> (&'static str, u64, u64) {
        let (done, total) = self.inner.read();
        (self.inner.phase(), done, total)
    }

    /// What is happening now — `"contracting"`, `"assembling"`, `"measuring"`.
    ///
    /// A fraction is always within a phase, so this is half the answer: a build
    /// with two unequal halves reports the second from zero rather than letting
    /// the first sit at 100% while it runs.
    #[getter]
    fn phase(&self) -> &'static str {
        self.inner.phase()
    }

    /// How far along, 0 to 1, or `None` from work with no honest measure.
    #[getter]
    fn fraction(&self) -> Option<f64> {
        self.inner.fraction()
    }

    fn __repr__(&self) -> String {
        match self.inner.fraction() {
            Some(fraction) => {
                format!("Progress({} {:.0}%)", self.inner.phase(), fraction * 100.0)
            }
            None => "Progress(unknown)".to_string(),
        }
    }
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
        let counter = progress.map_or_else(CoreProgress::new, |p| p.inner.clone());
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

fn options(targets: Option<Vec<NodeId>>, max_cost: Option<Weight>) -> SearchOptions {
    SearchOptions {
        targets,
        max_cost,
        ..SearchOptions::default()
    }
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

/// Earliest arrival from `sources`, leaving at `departing` on a weekly clock.
///
/// Dreyfus (1969). `waiting` is `"unrestricted"` — wait for a shut edge to open,
/// and pay the wait as travel time — or `"forbidden"`, which treats a shut edge
/// as absent. Costs come back as elapsed seconds, waiting included.
#[pyfunction]
#[pyo3(signature = (graph, calendar, sources, departing, *, waiting="unrestricted", targets=None, max_cost=None))]
#[allow(clippy::too_many_arguments)] // a query with a clock has one more knob
fn time_dependent_dijkstra(
    py: Python<'_>,
    graph: &PyGraph,
    calendar: &PyCalendar,
    sources: Vec<(NodeId, Weight)>,
    departing: u32,
    waiting: &str,
    targets: Option<Vec<NodeId>>,
    max_cost: Option<Weight>,
) -> PyResult<PySearchResult> {
    let waiting = match waiting {
        "unrestricted" => CoreWaiting::Unrestricted,
        "forbidden" => CoreWaiting::Forbidden,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown waiting policy {other:?}; expected 'unrestricted' or 'forbidden'"
            )))
        }
    };
    let departure = CoreDeparture::at(departing).waiting(waiting);
    let graph = Arc::clone(&graph.inner);
    let calendar = Arc::clone(&calendar.inner);
    let options = options(targets, max_cost);
    let result = py
        .detach(|| core_timedep(&graph, &calendar, &sources, &departure, &options))
        .map_err(value_err)?;
    Ok(PySearchResult { inner: result })
}

/// Read a GTFS feed's connections for one service day.
///
/// `path` may be a `.zip` or a directory of `.txt` files. The date is passed as
/// its parts rather than as a string, so there is no format to get wrong on
/// either side of the boundary.
#[pyfunction]
#[pyo3(signature = (path, year, month, day))]
fn load_gtfs(py: Python<'_>, path: PathBuf, year: i32, month: u32, day: u32) -> PyResult<PyFeed> {
    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| PyValueError::new_err(format!("{year}-{month}-{day} is not a date")))?;
    let (feed, pairs) = py
        .detach(|| {
            gtfs_load(&path, date).map(|feed| {
                let pairs = feed.pairs();
                (feed, pairs)
            })
        })
        .map_err(value_err)?;
    Ok(PyFeed {
        inner: Arc::new(feed),
        pairs,
    })
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

/// Great-circle metres between two `(lat, lon)` points, on the sphere every
/// OSM edge length is measured on — so a layer that prices its own walks
/// agrees with the streets about how long a metre is.
#[pyfunction]
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    routelab_osm::haversine(lat1, lon1, lat2, lon2)
}

#[pymodule]
fn _routelab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EARTH_RADIUS", routelab_osm::EARTH_RADIUS)?;
    m.add_function(wrap_pyfunction!(haversine, m)?)?;
    m.add_class::<PyCalendar>()?;
    m.add_class::<PyContractionHierarchy>()?;
    m.add_class::<PyFeed>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyHeuristic>()?;
    m.add_class::<PyMeetingSearch>()?;
    m.add_class::<PyOsmNetwork>()?;
    m.add_class::<PyProgress>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyItinerary>()?;
    m.add_class::<PyFootpaths>()?;
    m.add_class::<PyRaptor>()?;
    m.add_class::<PyRaptorSearch>()?;
    m.add_class::<PySearchTree>()?;
    m.add_class::<PyTimeExpanded>()?;
    m.add_class::<PyTimetable>()?;
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(load_gtfs, m)?)?;
    m.add_function(wrap_pyfunction!(load_osm, m)?)?;
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(bfs, m)?)?;
    Ok(())
}
