//! End-to-end reads of hand-written extracts.
//!
//! The XML path is the tested one. Everything after parsing — profiles,
//! splitting, weighting, projection — is shared with PBF, so these cover that
//! logic for both; what they cannot cover is PBF's own decoding, which has no
//! fixture because writing a `.pbf` needs tooling this repo does not carry.
//! That gap is real and is closed by running the demo on a real extract.

use std::path::{Path, PathBuf};

use routelab_osm::{load, Profile};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn driving() -> Profile {
    Profile::new(
        [
            ("residential".to_string(), 10.0),
            ("motorway".to_string(), 30.0),
        ],
        true,
        true,
    )
}

fn walking() -> Profile {
    Profile::new(
        [
            ("residential".to_string(), 1.4),
            ("footway".to_string(), 1.4),
        ],
        false,
        true,
    )
}

#[test]
fn a_profile_decides_which_ways_are_roads() {
    let driven = load(&fixture("junction.osm"), &driving()).unwrap();
    let walked = load(&fixture("junction.osm"), &walking()).unwrap();

    // Driving sees Main St and Side St; the footway and the stream are not roads.
    assert_eq!(driven.num_nodes(), 4, "1, 2, 3 and Side St's far end");
    assert!(!driven.node_ids.contains(&5), "footway is not drivable");
    // Walking adds the footway, and with it node 5.
    assert!(walked.node_ids.contains(&5));
    assert!(
        !walked.node_ids.contains(&99),
        "a stream is not a way anyone walks"
    );
}

#[test]
fn oneway_is_honoured_for_drivers_and_ignored_for_walkers() {
    let driven = load(&fixture("junction.osm"), &driving()).unwrap();
    let walked = load(&fixture("junction.osm"), &walking()).unwrap();

    let directed = |network: &routelab_osm::OsmNetwork, from: i64, to: i64| {
        (0..network.num_edges()).any(|e| {
            network.node_ids[network.edge_tails[e] as usize] == from
                && network.node_ids[network.edge_heads[e] as usize] == to
        })
    };

    assert!(directed(&driven, 1, 2), "Main St runs eastbound");
    assert!(!directed(&driven, 2, 1), "and not back");
    assert!(
        directed(&driven, 2, 4) && directed(&driven, 4, 2),
        "Side St"
    );

    assert!(
        directed(&walked, 1, 2) && directed(&walked, 2, 1),
        "on foot"
    );
}

#[test]
fn ways_are_cut_where_they_meet() {
    let driven = load(&fixture("junction.osm"), &driving()).unwrap();
    // Main St is one way in the file but two edges in the graph, because Side St
    // meets it at node 2.
    assert_eq!(driven.num_edges(), 4, "1->2, 2->3, and Side St both ways");
    assert_eq!(driven.node_ids.len(), 4);
}

/// Both streets in the fixture are ~111 m; only the second is posted.
fn weight_from(network: &routelab_osm::OsmNetwork, tail: i64) -> u32 {
    (0..network.num_edges())
        .find(|&e| network.node_ids[network.edge_tails[e] as usize] == tail)
        .map(|e| network.edge_weights[e])
        .unwrap()
}

#[test]
fn a_posted_limit_reaches_the_weight() {
    // residential is 5 m/s here, but the profile tops out at 30, so a posted
    // 36 km/h = 10 m/s is allowed to stand and the posted street is quicker.
    let profile = Profile::new(
        [
            ("residential".to_string(), 5.0),
            ("motorway".to_string(), 30.0),
        ],
        true,
        true,
    );
    let network = load(&fixture("speeds.osm"), &profile).unwrap();
    assert!(weight_from(&network, 1) > weight_from(&network, 3));
}

#[test]
fn a_posted_limit_cannot_exceed_the_profiles_own_top_speed() {
    // Not a style choice: cost_per_distance is 1 / max_speed, so an edge allowed
    // to travel faster than the profile's fastest class would cost less per
    // metre than the heuristic assumes anything can — and A* would stop
    // returning cheapest paths, silently.
    let capped = Profile::new([("residential".to_string(), 5.0)], true, true);
    let network = load(&fixture("speeds.osm"), &capped).unwrap();
    assert_eq!(
        weight_from(&network, 1),
        weight_from(&network, 3),
        "the posted 10 m/s is clamped back to the profile's 5 m/s"
    );
}

#[test]
fn nearest_finds_the_node_you_pointed_at() {
    let network = load(&fixture("junction.osm"), &driving()).unwrap();
    let nearest = network.nearest(0.0001, 0.00195, None).unwrap();
    assert_eq!(network.node_ids[nearest], 3);
}

#[test]
fn an_extract_with_nothing_routable_is_empty_not_an_error() {
    let cycling = Profile::new([("cycleway".to_string(), 4.0)], true, true);
    let network = load(&fixture("junction.osm"), &cycling).unwrap();
    assert_eq!(network.num_edges(), 0);
    assert_eq!(network.num_nodes(), 0);
}

#[test]
fn a_missing_file_says_so() {
    let error = load(&fixture("nowhere.osm"), &driving()).unwrap_err();
    assert!(error.to_string().contains("no such file"), "{error}");
}

#[test]
fn an_unrecognised_extension_says_so() {
    let error = load(Path::new("city.geojson"), &driving()).unwrap_err();
    assert!(
        error.to_string().contains("cannot tell what format"),
        "{error}"
    );
}
