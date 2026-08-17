//! Python bindings for `routelab-core`.
//!
//! This layer does three things and nothing else: convert Python values into
//! core types, release the GIL while the kernel runs, and turn core errors into
//! Python exceptions. Argument sugar (accepting a bare int for `sources`, and
//! so on) lives in the Python veneer, where it is easier to read and change.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// Core errors describe themselves; Python only needs the sentence.
pub(crate) fn value_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Bounds check for the id-taking accessors, phrased the way Python indexing errors are.
pub(crate) fn check_index(id: u32, count: usize, what: &str) -> PyResult<()> {
    if (id as usize) < count {
        Ok(())
    } else {
        Err(PyIndexError::new_err(format!(
            "{what} {id} is out of range for a graph with {count} {what}s"
        )))
    }
}

pub mod contraction;
pub mod csa;
pub mod graph;
pub mod gtfs;
pub mod heuristic;
pub mod osm;
pub mod progress;
pub mod ptl;
pub mod raptor;
pub mod search;
pub mod timedep;
pub mod timetable;
pub mod tripbased;

use crate::contraction::*;
use crate::csa::*;
use crate::graph::*;
use crate::gtfs::*;
use crate::heuristic::*;
use crate::osm::*;
use crate::progress::*;
use crate::ptl::*;
use crate::raptor::*;
use crate::search::*;
use crate::timedep::*;
use crate::timetable::*;
use crate::tripbased::*;

#[pymodule]
fn _routelab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EARTH_RADIUS", routelab_osm::EARTH_RADIUS)?;
    m.add_function(wrap_pyfunction!(haversine, m)?)?;
    m.add_class::<PyCalendar>()?;
    m.add_class::<PyConnectionScan>()?;
    m.add_class::<PyContractionHierarchy>()?;
    m.add_class::<PyFeed>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyHeuristic>()?;
    m.add_class::<PyMeetingSearch>()?;
    m.add_class::<PyOsmNetwork>()?;
    m.add_class::<PyProgress>()?;
    m.add_class::<PyPublicTransitLabeling>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyItinerary>()?;
    m.add_class::<PyFootpaths>()?;
    m.add_class::<PyRaptor>()?;
    m.add_class::<PyRaptorSearch>()?;
    m.add_class::<PyScanProfile>()?;
    m.add_class::<PyScanSearch>()?;
    m.add_class::<PySearchTree>()?;
    m.add_class::<PyTimeExpanded>()?;
    m.add_class::<PyTimetable>()?;
    m.add_class::<PyTripBased>()?;
    m.add_class::<PyTripBasedProfile>()?;
    m.add_class::<PyTripBasedSearch>()?;
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(load_gtfs, m)?)?;
    m.add_function(wrap_pyfunction!(load_osm, m)?)?;
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(bfs, m)?)?;
    Ok(())
}
