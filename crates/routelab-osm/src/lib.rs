//! Routable graphs from OpenStreetMap extracts.
//!
//! OSM describes the world; a routing graph describes how to move through it,
//! and only for somebody in particular. This crate does the translation: keep
//! the ways a [`Profile`] can travel, cut them where they meet, price each piece
//! at the speed that profile goes, and keep what falls in between as shape.
//!
//! ```no_run
//! use routelab_osm::{load, Profile};
//!
//! let walking = Profile::new([("residential".to_string(), 1.4)], false, true);
//! let network = load("city.osm.pbf".as_ref(), &walking)?;
//! println!("{} nodes, {} edges", network.num_nodes(), network.num_edges());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! It is kept apart from `routelab-core`, which has no dependencies and should
//! keep it that way — file formats are not kernels.

pub mod conditional;
pub mod network;
pub mod profile;
pub mod read;

use std::path::Path;

pub use conditional::{open_windows, Window, WEEK};
pub use network::{haversine, GeometrySpan, NetworkBuilder, OsmNetwork, EARTH_RADIUS};
pub use profile::{Profile, Travel, WayTags};
pub use read::{Format, OsmError};

/// Read an extract and build the graph a profile can travel.
///
/// Two passes over the file: the ways say which nodes matter, then the nodes
/// arrive. The alternative — hold every coordinate in the file, then discard
/// most of them — is the difference between a city's memory and a continent's.
pub fn load(path: &Path, profile: &Profile) -> Result<OsmNetwork, OsmError> {
    let format = Format::detect(path)?;
    // Checked up front so a missing file says so, rather than surfacing as a
    // parse error from whichever reader got there first.
    if !path.is_file() {
        return Err(OsmError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no such file: {}", path.display()),
        )));
    }

    let mut builder = NetworkBuilder::new(profile);
    format.scan_ways(path, &mut |nodes, tags| builder.consider_way(nodes, tags))?;
    if !builder.wants_nodes() {
        // Nothing matched the profile, so there is no reason to read the nodes.
        return Ok(OsmNetwork::default());
    }
    format.scan_nodes(path, &mut |id, lat, lon| {
        builder.consider_node(id, lat, lon)
    })?;
    Ok(builder.finish())
}
