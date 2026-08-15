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

// --- Conditional restrictions ------------------------------------------------

const HOUR: u32 = 3600;

fn walking_with_access() -> Profile {
    walking().reading_access(["foot".to_string(), "access".to_string()])
}

/// The edge from `tail` to `head`, by OSM node id, and its windows if any.
fn scheduled(
    network: &routelab_osm::OsmNetwork,
    tail: i64,
    head: i64,
) -> (usize, Option<Vec<(u32, u32)>>) {
    let edge = (0..network.num_edges())
        .find(|&e| {
            network.node_ids[network.edge_tails[e] as usize] == tail
                && network.node_ids[network.edge_heads[e] as usize] == head
        })
        .unwrap_or_else(|| panic!("no edge {tail} -> {head}"));
    let windows = network
        .edge_windows
        .iter()
        .find(|(index, _)| *index as usize == edge)
        .map(|(_, windows)| windows.clone());
    (edge, windows)
}

fn open_at(windows: &[(u32, u32)], moment: u32) -> bool {
    windows.iter().any(|&(start, end)| {
        if end <= start {
            moment >= start || moment < end
        } else {
            moment >= start && moment < end
        }
    })
}

#[test]
fn a_gate_carries_its_hours_onto_both_of_its_edges() {
    let network = load(&fixture("conditional.osm"), &walking_with_access()).unwrap();
    for (tail, head) in [(2, 3), (3, 2)] {
        let (_, windows) = scheduled(&network, tail, head);
        let windows = windows.unwrap_or_else(|| panic!("{tail} -> {head} should be scheduled"));
        assert_eq!(windows.len(), 7, "one window a day");
        assert!(open_at(&windows, 12 * HOUR), "noon");
        assert!(!open_at(&windows, 3 * HOUR), "3am");
    }
}

#[test]
fn the_way_round_the_gate_carries_no_schedule_at_all() {
    let network = load(&fixture("conditional.osm"), &walking_with_access()).unwrap();
    // Node 4 is touched by one way only, so it is shape rather than a junction
    // and the detour is a single 1 -> 3 edge running through it.
    assert_eq!(scheduled(&network, 1, 3).1, None);
    assert_eq!(scheduled(&network, 3, 1).1, None);
    assert_eq!(
        scheduled(&network, 1, 2).1,
        None,
        "the approach is not gated"
    );
}

#[test]
fn a_nightly_closure_reads_as_open_the_rest_of_the_time() {
    let network = load(&fixture("conditional.osm"), &walking_with_access()).unwrap();
    let (_, windows) = scheduled(&network, 7, 8);
    let windows = windows.expect("the overnight trail is scheduled");
    assert!(open_at(&windows, 12 * HOUR), "open at noon by default");
    assert!(!open_at(&windows, 2 * HOUR), "shut at 2am");
}

#[test]
fn a_profile_that_reads_no_access_keys_sees_no_schedules() {
    // The old behaviour, still available: a profile that was never told which
    // access tags speak for it ignores them rather than guessing.
    let network = load(&fixture("conditional.osm"), &walking()).unwrap();
    assert!(network.edge_windows.is_empty());
}

#[test]
fn a_reversible_lane_runs_each_way_at_its_own_hours() {
    let network = load(&fixture("conditional.osm"), &driving()).unwrap();
    // Drawn 5 -> 6, so `yes @` afternoons is forward and `-1 @` mornings is back.
    let (_, forward) = scheduled(&network, 5, 6);
    let forward = forward.expect("the forward lane is scheduled");
    assert!(
        open_at(&forward, 14 * HOUR),
        "Monday afternoon, with the arrow"
    );
    assert!(!open_at(&forward, 8 * HOUR), "not on a Monday morning");

    let (_, backward) = scheduled(&network, 6, 5);
    let backward = backward.expect("the reverse lane is scheduled");
    assert!(open_at(&backward, 8 * HOUR), "Monday morning, against it");
    assert!(!open_at(&backward, 14 * HOUR), "not in the afternoon");

    // And neither runs at the weekend, which the schedule never mentions.
    let saturday = 5 * 24 * HOUR + 8 * HOUR;
    assert!(!open_at(&forward, saturday) && !open_at(&backward, saturday));
}
