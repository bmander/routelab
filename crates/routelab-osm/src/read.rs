//! Reading the two formats an extract arrives in.
//!
//! Both do the same two things — hand over ways, then hand over nodes — and
//! everything downstream is shared, so XML and PBF cannot drift apart in how
//! they interpret a network. Two passes rather than one: the ways say which
//! nodes matter, so the node pass can keep only those, which is the difference
//! between holding a city's coordinates and holding a continent's.

use std::fmt;
use std::path::Path;

use quick_xml::events::Event;
use quick_xml::Reader;

use crate::profile::WayTags;

#[derive(Debug)]
pub enum OsmError {
    Io(std::io::Error),
    /// A file whose extension says neither `.osm`/`.xml` nor `.osm.pbf`.
    UnknownFormat(String),
    Xml(quick_xml::Error),
    Pbf(osmpbf::Error),
}

impl fmt::Display for OsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OsmError::Io(error) => write!(f, "{error}"),
            OsmError::UnknownFormat(path) => write!(
                f,
                "cannot tell what format {path} is; expected .osm.pbf, .osm, or .xml"
            ),
            OsmError::Xml(error) => write!(f, "malformed OSM XML: {error}"),
            OsmError::Pbf(error) => write!(f, "malformed OSM PBF: {error}"),
        }
    }
}

impl std::error::Error for OsmError {}

impl From<std::io::Error> for OsmError {
    fn from(error: std::io::Error) -> Self {
        OsmError::Io(error)
    }
}

impl From<quick_xml::Error> for OsmError {
    fn from(error: quick_xml::Error) -> Self {
        OsmError::Xml(error)
    }
}

impl From<osmpbf::Error> for OsmError {
    fn from(error: osmpbf::Error) -> Self {
        OsmError::Pbf(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Format {
    Xml,
    Pbf,
}

impl Format {
    pub fn detect(path: &Path) -> Result<Self, OsmError> {
        let name = path.to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".pbf") {
            Ok(Format::Pbf)
        } else if name.ends_with(".osm") || name.ends_with(".xml") {
            Ok(Format::Xml)
        } else {
            Err(OsmError::UnknownFormat(path.to_string_lossy().into_owned()))
        }
    }

    pub fn scan_ways(
        self,
        path: &Path,
        visit: &mut dyn FnMut(&[i64], &WayTags),
    ) -> Result<(), OsmError> {
        match self {
            Format::Xml => xml::scan_ways(path, visit),
            Format::Pbf => pbf::scan_ways(path, visit),
        }
    }

    pub fn scan_nodes(
        self,
        path: &Path,
        visit: &mut dyn FnMut(i64, f64, f64),
    ) -> Result<(), OsmError> {
        match self {
            Format::Xml => xml::scan_nodes(path, visit),
            Format::Pbf => pbf::scan_nodes(path, visit),
        }
    }
}

mod xml {
    use super::*;

    /// XML is the format for small files: hand-written fixtures, tiny extracts.
    /// Anything city-sized arrives as PBF.
    pub(super) fn scan_ways(
        path: &Path,
        visit: &mut dyn FnMut(&[i64], &WayTags),
    ) -> Result<(), OsmError> {
        let mut reader = Reader::from_file(path).map_err(map_read_error)?;
        let mut buffer = Vec::new();
        let mut nodes: Vec<i64> = Vec::new();
        let mut tags = WayTags::default();
        let mut inside_way = false;

        loop {
            match reader.read_event_into(&mut buffer) {
                Err(error) => return Err(OsmError::Xml(error)),
                Ok(Event::Eof) => break,
                Ok(Event::Start(element)) | Ok(Event::Empty(element)) => {
                    match element.name().as_ref() {
                        b"way" => {
                            inside_way = true;
                            nodes.clear();
                            tags = WayTags::default();
                        }
                        b"nd" if inside_way => {
                            if let Some(id) =
                                attribute(&element, b"ref").and_then(|v| v.parse().ok())
                            {
                                nodes.push(id);
                            }
                        }
                        b"tag" if inside_way => {
                            let key = attribute(&element, b"k");
                            let value = attribute(&element, b"v");
                            if let (Some(key), Some(value)) = (key, value) {
                                tags.set(&key, &value);
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(element)) if element.name().as_ref() == b"way" => {
                    inside_way = false;
                    visit(&nodes, &tags);
                }
                _ => {}
            }
            buffer.clear();
        }
        Ok(())
    }

    pub(super) fn scan_nodes(
        path: &Path,
        visit: &mut dyn FnMut(i64, f64, f64),
    ) -> Result<(), OsmError> {
        let mut reader = Reader::from_file(path).map_err(map_read_error)?;
        let mut buffer = Vec::new();

        loop {
            match reader.read_event_into(&mut buffer) {
                Err(error) => return Err(OsmError::Xml(error)),
                Ok(Event::Eof) => break,
                Ok(Event::Start(element)) | Ok(Event::Empty(element))
                    if element.name().as_ref() == b"node" =>
                {
                    let id = attribute(&element, b"id").and_then(|v| v.parse().ok());
                    let lat = attribute(&element, b"lat").and_then(|v| v.parse().ok());
                    let lon = attribute(&element, b"lon").and_then(|v| v.parse().ok());
                    if let (Some(id), Some(lat), Some(lon)) = (id, lat, lon) {
                        visit(id, lat, lon);
                    }
                }
                _ => {}
            }
            buffer.clear();
        }
        Ok(())
    }

    /// An attribute's value, or `None` if it is absent or unreadable.
    ///
    /// Unreadable and absent are the same answer here on purpose: every caller
    /// wants a value it can parse, and a `nd` element with a mangled `ref` is
    /// no more usable than one with no `ref` at all.
    fn attribute(element: &quick_xml::events::BytesStart<'_>, name: &[u8]) -> Option<String> {
        let attribute = element.try_get_attribute(name).ok().flatten()?;
        Some(String::from_utf8_lossy(&attribute.value).into_owned())
    }

    /// Readability is not worth unwrapping quick-xml's IO variant here: `load`
    /// checks the file exists first, so anything reaching this point is a
    /// genuine parse failure.
    fn map_read_error(error: quick_xml::Error) -> OsmError {
        OsmError::Xml(error)
    }
}

mod pbf {
    use super::*;
    use osmpbf::{Element, ElementReader};

    pub(super) fn scan_ways(
        path: &Path,
        visit: &mut dyn FnMut(&[i64], &WayTags),
    ) -> Result<(), OsmError> {
        let mut nodes: Vec<i64> = Vec::new();
        let mut tags = WayTags::default();
        ElementReader::from_path(path)?.for_each(|element| {
            if let Element::Way(way) = element {
                nodes.clear();
                nodes.extend(way.refs());
                tags = WayTags::default();
                for (key, value) in way.tags() {
                    tags.set(key, value);
                }
                visit(&nodes, &tags);
            }
        })?;
        Ok(())
    }

    pub(super) fn scan_nodes(
        path: &Path,
        visit: &mut dyn FnMut(i64, f64, f64),
    ) -> Result<(), OsmError> {
        ElementReader::from_path(path)?.for_each(|element| match element {
            Element::Node(node) => visit(node.id(), node.lat(), node.lon()),
            Element::DenseNode(node) => visit(node.id(), node.lat(), node.lon()),
            _ => {}
        })?;
        Ok(())
    }
}
