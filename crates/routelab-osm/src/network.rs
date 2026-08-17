//! Turning ways into a routable graph.
//!
//! An OSM way is a polyline that may run for kilometres through many
//! intersections; a routing edge runs from one intersection to the next. The
//! work here is finding where ways meet, cutting them there, and keeping what
//! falls in between as the edge's shape rather than as graph nodes.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use rstar::primitives::GeomWithData;
use rstar::RTree;

use crate::conditional::Window;
use crate::profile::{Profile, Travel, WayTags};

/// Metres. The polar radius — the smallest — so distances computed with it are
/// never larger than the ground they cover.
pub const EARTH_RADIUS: f64 = 6_356_752.0;

/// Where an edge's shape lives in the shared point array, and which way round.
///
/// A two-way street is two edges over one polyline, so the points are stored
/// once and the reverse edge reads them backwards.
#[derive(Debug, Clone, Copy)]
pub struct GeometrySpan {
    pub start: u32,
    pub end: u32,
    pub reversed: bool,
}

/// A routable graph built from an extract: dense node ids, edges between them,
/// and the shape of each edge.
#[derive(Debug, Default)]
pub struct OsmNetwork {
    /// OSM node id of each graph node, in dense order.
    pub node_ids: Vec<i64>,
    pub lats: Vec<f64>,
    pub lons: Vec<f64>,

    pub edge_tails: Vec<u32>,
    pub edge_heads: Vec<u32>,
    /// Traversal time in seconds, rounded up.
    pub edge_weights: Vec<u32>,
    pub edge_geometry: Vec<GeometrySpan>,

    /// When each restricted edge may be travelled, as `(edge, windows)` on the
    /// weekly clock. Sparse — a couple of hundred entries out of six hundred
    /// thousand edges on a city, so a dense column would be almost all "always".
    /// Keyed by position in `edge_tails`, which is the order the edges are
    /// handed to the graph builder.
    pub edge_windows: Vec<(u32, Vec<Window>)>,
    /// Schedules that could not be read, and so were ignored. Counted rather
    /// than guessed at: a dropped restriction should be a visible number.
    pub unreadable_schedules: usize,

    /// Interleaved lat/lon of every shape point, shared between edges.
    pub geometry_points: Vec<f64>,

    /// `(min_lat, min_lon, max_lat, max_lon)` over the graph's nodes.
    pub bounds: (f64, f64, f64, f64),

    /// Nodes in an R*-tree, for [`OsmNetwork::nearest`]. Built on first ask
    /// rather than at read time, because most work over an extract never
    /// snaps a coordinate to it, and bulk-loading half a million points is
    /// not free.
    index: OnceLock<RTree<GeomWithData<[f64; 2], usize>>>,
}

impl OsmNetwork {
    pub fn num_nodes(&self) -> usize {
        self.node_ids.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edge_tails.len()
    }

    /// The `(lat, lon)` points along `edge`, from its tail to its head.
    pub fn geometry(&self, edge: usize) -> Vec<(f64, f64)> {
        let span = self.edge_geometry[edge];
        let points: Vec<(f64, f64)> = self.geometry_points
            [span.start as usize * 2..span.end as usize * 2]
            .chunks_exact(2)
            .map(|point| (point[0], point[1]))
            .collect();
        if span.reversed {
            points.into_iter().rev().collect()
        } else {
            points
        }
    }

    /// Where a coordinate sits in the flat local space the index is built in.
    ///
    /// The same projection [`OsmNetwork::projected`] uses — longitude scaled
    /// by the cosine of the extract's own extreme latitude — so that a
    /// straight line in it is a straight line on the ground, near enough for
    /// deciding which of two nodes is closer.
    fn flatten(&self, lat: f64, lon: f64) -> [f64; 2] {
        let (min_lat, _, max_lat, _) = self.bounds;
        let scale = min_lat.abs().max(max_lat.abs()).to_radians().cos();
        [lat, lon * scale]
    }

    /// The nodes in an R*-tree, built the first time anything asks.
    fn index(&self) -> &RTree<GeomWithData<[f64; 2], usize>> {
        self.index.get_or_init(|| {
            RTree::bulk_load(
                (0..self.num_nodes())
                    .map(|node| {
                        GeomWithData::new(self.flatten(self.lats[node], self.lons[node]), node)
                    })
                    .collect(),
            )
        })
    }

    /// The graph node closest to a coordinate, as the crow flies.
    ///
    /// An R*-tree lookup, over an index built on the first call. It used to be
    /// a linear scan, on the grounds that a scan of a few hundred thousand
    /// nodes is cheap next to the routing that follows it — which is true of
    /// one snap and false of a great many. Attaching a feed's six thousand
    /// stops to a city's pavements is six thousand of them, and that is forty
    /// seconds of scanning against under one of asking a tree.
    ///
    /// `within` limits the search to the given OSM node ids. Extracts are full
    /// of stubs that connect to nothing under a given profile, and snapping to
    /// one produces a query with no answer for reasons that have nothing to do
    /// with routing — so a caller that knows which nodes are useful can say so.
    /// The tree is walked in order of distance and the first allowed node
    /// wins, so a permissive set costs no more than no set at all.
    pub fn nearest(&self, lat: f64, lon: f64, within: Option<&HashSet<i64>>) -> Option<usize> {
        let query = self.flatten(lat, lon);
        match within {
            None => self.index().nearest_neighbor(query).map(|found| found.data),
            Some(allowed) => self
                .index()
                .nearest_neighbor_iter(query)
                .find(|found| allowed.contains(&self.node_ids[found.data]))
                .map(|found| found.data),
        }
    }

    /// Project every node into local metres, as `(xs, ys)`.
    ///
    /// A local equirectangular projection, scaled by the cosine of the *highest*
    /// latitude in the extract. Since `cos` falls as you move away from the
    /// equator, using the extreme means the projected distance between any two
    /// nodes is never larger than the ground distance between them — which is
    /// what keeps a straight-line heuristic priced against it admissible. The
    /// error is well under a percent across a city, and always in the direction
    /// that stays correct.
    pub fn projected(&self) -> (Vec<f64>, Vec<f64>) {
        let (min_lat, _, max_lat, _) = self.bounds;
        let extreme = min_lat.abs().max(max_lat.abs());
        let scale = EARTH_RADIUS * extreme.to_radians().cos();
        let xs = self
            .lons
            .iter()
            .map(|lon| scale * lon.to_radians())
            .collect();
        let ys = self
            .lats
            .iter()
            .map(|lat| EARTH_RADIUS * lat.to_radians())
            .collect();
        (xs, ys)
    }
}

/// Great-circle distance in metres, on a sphere of the polar radius.
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (phi1, phi2) = (lat1.to_radians(), lat2.to_radians());
    let dphi = phi2 - phi1;
    let dlambda = (lon2 - lon1).to_radians();
    let a = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlambda / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS * a.sqrt().asin()
}

/// A way this profile can travel, held until the coordinates arrive.
struct KeptWay {
    nodes: Vec<i64>,
    travel: Travel,
}

/// Accumulates ways, then nodes, then cuts one into the other.
pub struct NetworkBuilder<'a> {
    profile: &'a Profile,
    ways: Vec<KeptWay>,
    /// How many times each node is referenced by a kept way. Two or more means
    /// ways meet there, which is the definition of an intersection.
    references: HashMap<i64, u32>,
    coordinates: HashMap<i64, (f64, f64)>,
}

impl<'a> NetworkBuilder<'a> {
    pub fn new(profile: &'a Profile) -> Self {
        NetworkBuilder {
            profile,
            ways: Vec::new(),
            references: HashMap::new(),
            coordinates: HashMap::new(),
        }
    }

    /// Offer a way; it is kept only if the profile can travel it.
    pub fn consider_way(&mut self, nodes: &[i64], tags: &WayTags) {
        let Some(travel) = self.profile.travel(tags) else {
            return;
        };
        if nodes.len() < 2 {
            return;
        }
        for &node in nodes {
            *self.references.entry(node).or_insert(0) += 1;
        }
        self.ways.push(KeptWay {
            nodes: nodes.to_vec(),
            travel,
        });
    }

    /// Offer a node; it is kept only if some kept way referenced it.
    pub fn consider_node(&mut self, id: i64, lat: f64, lon: f64) {
        if self.references.contains_key(&id) {
            self.coordinates.insert(id, (lat, lon));
        }
    }

    /// Whether any way has been kept — the cheap check for "did the profile
    /// match anything at all", before the node pass runs.
    pub fn wants_nodes(&self) -> bool {
        !self.ways.is_empty()
    }

    pub fn finish(self) -> OsmNetwork {
        let mut network = OsmNetwork::default();
        let mut node_index: HashMap<i64, u32> = HashMap::new();

        for way in &self.ways {
            // Cut at every intersection, and at the ends. A run of nodes that
            // nothing else touches is shape, not structure.
            let mut segment_start = 0;
            for position in 1..way.nodes.len() {
                let node = way.nodes[position];
                let is_junction = self.references.get(&node).copied().unwrap_or(0) > 1;
                if is_junction || position == way.nodes.len() - 1 {
                    self.emit_segment(
                        &mut network,
                        &mut node_index,
                        &way.nodes[segment_start..=position],
                        &way.travel,
                    );
                    segment_start = position;
                }
            }
        }

        network.bounds = bounds_of(&network.lats, &network.lons);
        network
    }

    /// Turn one intersection-to-intersection run into edges, if it is usable.
    fn emit_segment(
        &self,
        network: &mut OsmNetwork,
        node_index: &mut HashMap<i64, u32>,
        nodes: &[i64],
        travel: &Travel,
    ) {
        // A node outside the extract has no coordinates, so the segment through
        // it has no length and no shape. Extracts are cut somewhere; this is
        // what the cut looks like from the inside.
        let mut points = Vec::with_capacity(nodes.len() * 2);
        let mut length = 0.0;
        let mut previous: Option<(f64, f64)> = None;
        for &node in nodes {
            let Some(&(lat, lon)) = self.coordinates.get(&node) else {
                return;
            };
            if let Some((prev_lat, prev_lon)) = previous {
                length += haversine(prev_lat, prev_lon, lat, lon);
            }
            previous = Some((lat, lon));
            points.push(lat);
            points.push(lon);
        }

        let (tail, head) = (nodes[0], nodes[nodes.len() - 1]);
        if tail == head || length <= 0.0 {
            return; // a closed loop or a zero-length stub is not an edge
        }

        // Round up: the heuristic prices distance at the profile's top speed, so
        // an edge that came in under the ground it covers would let the estimate
        // exceed the truth.
        let seconds = (length / travel.speed).ceil().min(u32::MAX as f64 - 1.0) as u32;

        let start = (network.geometry_points.len() / 2) as u32;
        network.geometry_points.extend_from_slice(&points);
        let end = (network.geometry_points.len() / 2) as u32;

        let tail_id = self.intern(network, node_index, tail);
        let head_id = self.intern(network, node_index, head);
        // One way splits into many edges, and each of them inherits the way's
        // schedule — a gate part-way along a path shuts the whole path.
        let mut emit = |from, to, reversed, open: &Option<Vec<Window>>| {
            if let Some(windows) = open {
                let edge = network.edge_tails.len() as u32;
                network.edge_windows.push((edge, windows.clone()));
            }
            network.edge_tails.push(from);
            network.edge_heads.push(to);
            network.edge_weights.push(seconds);
            network.edge_geometry.push(GeometrySpan {
                start,
                end,
                reversed,
            });
        };
        if travel.forward {
            emit(tail_id, head_id, false, &travel.forward_open);
        }
        if travel.backward {
            emit(head_id, tail_id, true, &travel.backward_open);
        }
    }

    /// The dense id of an OSM node, assigning one on first sight.
    fn intern(
        &self,
        network: &mut OsmNetwork,
        node_index: &mut HashMap<i64, u32>,
        node: i64,
    ) -> u32 {
        *node_index.entry(node).or_insert_with(|| {
            let (lat, lon) = self.coordinates[&node];
            network.node_ids.push(node);
            network.lats.push(lat);
            network.lons.push(lon);
            (network.node_ids.len() - 1) as u32
        })
    }
}

fn bounds_of(lats: &[f64], lons: &[f64]) -> (f64, f64, f64, f64) {
    // Empty is its own answer rather than a sentinel to fold from: a network
    // with no nodes has no bounds, and zeros say that as well as anything.
    let reduce =
        |values: &[f64], f: fn(f64, f64) -> f64| values.iter().copied().reduce(f).unwrap_or(0.0);
    (
        reduce(lats, f64::min),
        reduce(lons, f64::min),
        reduce(lats, f64::max),
        reduce(lons, f64::max),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walking() -> Profile {
        Profile::new([("residential".to_string(), 1.0)], false, false)
    }

    fn tags(highway: &str) -> WayTags {
        let mut tags = WayTags::default();
        tags.set("highway", highway);
        tags
    }

    /// Nodes 1..5 in a line 0.001 degrees of latitude apart, with node 3 also
    /// used by a second way — so it is an intersection and the line splits there.
    fn crossing() -> OsmNetwork {
        let profile = walking();
        let mut builder = NetworkBuilder::new(&profile);
        builder.consider_way(&[1, 2, 3, 4, 5], &tags("residential"));
        builder.consider_way(&[3, 6], &tags("residential"));
        for (id, lat) in [(1, 0.0), (2, 0.001), (3, 0.002), (4, 0.003), (5, 0.004)] {
            builder.consider_node(id, lat, 0.0);
        }
        builder.consider_node(6, 0.002, 0.001);
        builder.finish()
    }

    #[test]
    fn ways_split_where_they_meet_and_not_elsewhere() {
        let network = crossing();
        // 1->3, 3->5, 3->6, each both ways: node 2 and node 4 are shape, not nodes.
        assert_eq!(network.num_edges(), 6);
        assert_eq!(network.num_nodes(), 4);
        let ids: Vec<i64> = network.node_ids.clone();
        assert!(ids.contains(&1) && ids.contains(&3) && ids.contains(&5) && ids.contains(&6));
        assert!(!ids.contains(&2), "a mid-way node is not an intersection");
    }

    #[test]
    fn the_nodes_between_intersections_become_shape() {
        let network = crossing();
        let edge = (0..network.num_edges())
            .find(|&e| {
                network.node_ids[network.edge_tails[e] as usize] == 1
                    && network.node_ids[network.edge_heads[e] as usize] == 3
            })
            .expect("1 -> 3");
        let shape = network.geometry(edge);
        assert_eq!(shape.len(), 3, "node 2 survives as a shape point");
        assert_eq!(shape[0].0, 0.0);
        assert_eq!(shape[2].0, 0.002);
    }

    #[test]
    fn a_reversed_edge_reads_its_shape_backwards() {
        let network = crossing();
        let forward = (0..network.num_edges())
            .find(|&e| {
                network.node_ids[network.edge_tails[e] as usize] == 1
                    && network.node_ids[network.edge_heads[e] as usize] == 3
            })
            .unwrap();
        let backward = (0..network.num_edges())
            .find(|&e| {
                network.node_ids[network.edge_tails[e] as usize] == 3
                    && network.node_ids[network.edge_heads[e] as usize] == 1
            })
            .unwrap();
        let mut reversed = network.geometry(backward);
        reversed.reverse();
        assert_eq!(network.geometry(forward), reversed);
    }

    #[test]
    fn weights_round_up_so_they_never_undercut_the_distance() {
        let profile = Profile::new([("residential".to_string(), 1.0)], false, false);
        let mut builder = NetworkBuilder::new(&profile);
        builder.consider_way(&[1, 2], &tags("residential"));
        builder.consider_node(1, 0.0, 0.0);
        builder.consider_node(2, 0.001, 0.0);
        let network = builder.finish();

        let metres = haversine(0.0, 0.0, 0.001, 0.0);
        assert!(metres > 110.0 && metres < 112.0, "{metres}");
        assert_eq!(network.edge_weights[0], metres.ceil() as u32);
    }

    #[test]
    fn a_way_whose_nodes_are_missing_is_dropped() {
        let profile = walking();
        let mut builder = NetworkBuilder::new(&profile);
        builder.consider_way(&[1, 2], &tags("residential"));
        builder.consider_node(1, 0.0, 0.0); // node 2 fell outside the extract
        assert_eq!(builder.finish().num_edges(), 0);
    }

    #[test]
    fn projection_never_overstates_the_ground() {
        let network = crossing();
        let (xs, ys) = network.projected();
        for a in 0..network.num_nodes() {
            for b in 0..network.num_nodes() {
                let planar = ((xs[a] - xs[b]).powi(2) + (ys[a] - ys[b]).powi(2)).sqrt();
                let ground = haversine(
                    network.lats[a],
                    network.lons[a],
                    network.lats[b],
                    network.lons[b],
                );
                assert!(planar <= ground + 1e-6, "{planar} > {ground}");
            }
        }
    }
}
