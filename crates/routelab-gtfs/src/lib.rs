//! Reading a GTFS feed into the connections a timetable search needs.
//!
//! A GTFS feed is a zip of CSV tables describing who runs what, where, and
//! when. What a search wants from it is much smaller: a list of **connections**,
//! each one vehicle leaving one stop at one instant and reaching the next at
//! another, for a single service day.
//!
//! **This crate does not parse GTFS.** It wraps
//! [`gtfs_structures`](https://crates.io/crates/gtfs-structures), because GTFS
//! is a great deal more complicated than it looks — service calendars with
//! per-date exceptions, headway-based service, pickup and drop-off rules,
//! parent stations, times that run past midnight — and a hand-rolled reader
//! would be wrong in ways nobody would notice for months. The work here is
//! turning what that crate produces into connections, and being explicit about
//! what is dropped on the way.
//!
//! It is kept apart from `routelab-core`, which has no dependencies and should
//! keep it that way — file formats are not kernels. Connections come out as
//! plain parallel arrays; whoever holds both ends builds the timetable.

use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDate;
use gtfs_structures::{Gtfs, PickupDropOffType};

/// Seconds since the service day's midnight. May exceed a day: a feed writes
/// `25:30:00` for a vehicle that left before midnight and arrives after it.
pub type Time = u32;

/// What a feed contained that this reader did not use.
///
/// Counted rather than guessed at, in the manner of the OSM reader's
/// `unreadable_schedules`: service quietly dropped should be a number somebody
/// can look at, not a hole discovered later in a route that made no sense.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Skipped {
    /// Trips whose service does not run on the requested date. Not a problem —
    /// this is the number that says the date filter did something.
    pub trips_not_running: usize,
    /// Trips with fewer than two stop times, which describe no movement.
    pub trips_too_short: usize,
    /// Stop times with no departure or arrival time. GTFS allows these to be
    /// interpolated between timepoints; interpolating is guessing, so they are
    /// dropped and the pair around them with them.
    pub stop_times_without_times: usize,
    /// Connections you could not board or alight — `pickup_type` or
    /// `drop_off_type` saying no. These genuinely do not exist as connections
    /// and keeping them would invent service.
    pub connections_not_boardable: usize,
    /// Trips defined by `frequencies.txt` rather than by explicit stop times.
    /// Expanding headway-based service into runs is its own piece of work; for
    /// now those trips are read only for the stop times they do carry, and this
    /// counts how much of the feed is affected.
    pub trips_frequency_based: usize,
}

impl Skipped {
    /// Did anything get dropped that a caller ought to know about?
    pub fn is_clean(&self) -> bool {
        self.trips_too_short == 0
            && self.stop_times_without_times == 0
            && self.trips_frequency_based == 0
    }
}

/// A service day's worth of connections, plus the stops they run between.
///
/// Parallel arrays rather than a `Vec<Connection>`: this crosses into Python,
/// and the kernel that consumes it wants columns.
#[derive(Debug, Default)]
pub struct Feed {
    /// GTFS `stop_id` of each dense stop index.
    pub stop_ids: Vec<String>,
    pub stop_names: Vec<String>,
    pub lats: Vec<f64>,
    pub lons: Vec<f64>,

    /// One entry per connection, all the same length.
    pub from_stop: Vec<u32>,
    pub to_stop: Vec<u32>,
    pub departs: Vec<Time>,
    pub arrives: Vec<Time>,
    /// Dense trip index; `trip_ids[trip[i]]` is the GTFS `trip_id`.
    pub trip: Vec<u32>,
    pub trip_ids: Vec<String>,
    /// The route each dense trip belongs to, by GTFS `route_id`.
    pub trip_routes: Vec<String>,

    pub skipped: Skipped,
}

impl Feed {
    pub fn num_stops(&self) -> usize {
        self.stop_ids.len()
    }

    pub fn num_connections(&self) -> usize {
        self.from_stop.len()
    }

    pub fn num_trips(&self) -> usize {
        self.trip_ids.len()
    }

    /// `(min_lat, min_lon, max_lat, max_lon)` over the stops that have a place.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let reduce = |values: &[f64], f: fn(f64, f64) -> f64| {
            values.iter().copied().reduce(f).unwrap_or(0.0)
        };
        (
            reduce(&self.lats, f64::min),
            reduce(&self.lons, f64::min),
            reduce(&self.lats, f64::max),
            reduce(&self.lons, f64::max),
        )
    }
}

/// What can go wrong before there is a feed to read.
///
/// A date on which nothing runs is deliberately **not** an error: it comes back
/// as a feed with no connections and `Skipped::trips_not_running` saying how
/// many trips were passed over. That distinguishes "you asked for a holiday" from
/// "the file is broken", and a caller can tell the difference.
#[derive(Debug)]
pub enum GtfsError {
    /// The feed could not be opened or parsed.
    Read(gtfs_structures::Error),
}

impl std::fmt::Display for GtfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GtfsError::Read(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for GtfsError {}

impl From<gtfs_structures::Error> for GtfsError {
    fn from(err: gtfs_structures::Error) -> Self {
        GtfsError::Read(err)
    }
}

/// Read the connections that run on `date`.
///
/// `path` may be a `.zip` or a directory of `.txt` files — `gtfs_structures`
/// takes either, which is what lets the test fixture be readable CSV in git
/// rather than a binary nobody can review.
pub fn load(path: &Path, date: NaiveDate) -> Result<Feed, GtfsError> {
    let gtfs = Gtfs::from_path(path.to_string_lossy().as_ref())?;
    Ok(connections_on(&gtfs, date))
}

/// Which `service_id`s run on `date`.
///
/// `calendar.txt` gives a weekday pattern between two dates and
/// `calendar_dates.txt` adds and removes individual days on top — the fiddliest
/// correctness problem in the format, and one this crate does not reimplement.
/// `trip_days` answers it, listing the day offsets a service runs from a start
/// date, so asking about `date` itself means asking whether offset zero is
/// among them.
fn services_on(gtfs: &Gtfs, date: NaiveDate) -> Vec<&str> {
    let named: std::collections::BTreeSet<&str> = gtfs
        .calendar
        .keys()
        .chain(gtfs.calendar_dates.keys())
        .map(String::as_str)
        .collect();
    named
        .into_iter()
        .filter(|service| gtfs.trip_days(service, date).contains(&0))
        .collect()
}

fn connections_on(gtfs: &Gtfs, date: NaiveDate) -> Feed {
    let mut feed = Feed::default();
    let running: std::collections::HashSet<&str> = services_on(gtfs, date).into_iter().collect();

    // Dense indices, assigned on first sight, so the arrays stay small when a
    // feed covers a region and the day covers a corner of it.
    let mut stop_index: HashMap<String, u32> = HashMap::new();
    let mut intern_stop = |stop: &gtfs_structures::Stop, feed: &mut Feed| -> u32 {
        *stop_index.entry(stop.id.clone()).or_insert_with(|| {
            feed.stop_ids.push(stop.id.clone());
            feed.stop_names.push(stop.name.clone().unwrap_or_default());
            feed.lats.push(stop.latitude.unwrap_or(0.0));
            feed.lons.push(stop.longitude.unwrap_or(0.0));
            (feed.stop_ids.len() - 1) as u32
        })
    };

    // Trips in id order, so a feed read twice produces the same indices — a
    // HashMap's order is not one.
    let mut trips: Vec<(&String, &gtfs_structures::Trip)> = gtfs.trips.iter().collect();
    trips.sort_unstable_by_key(|(id, _)| *id);

    for (trip_id, trip) in trips {
        if !running.contains(trip.service_id.as_str()) {
            feed.skipped.trips_not_running += 1;
            continue;
        }
        if !trip.frequencies.is_empty() {
            // Read whatever explicit stop times it has, but say that its
            // headway-based runs were not expanded.
            feed.skipped.trips_frequency_based += 1;
        }
        if trip.stop_times.len() < 2 {
            feed.skipped.trips_too_short += 1;
            continue;
        }

        let dense_trip = feed.trip_ids.len() as u32;
        let mut used = false;

        for pair in trip.stop_times.windows(2) {
            let (leave, reach) = (&pair[0], &pair[1]);
            let (Some(departs), Some(arrives)) = (leave.departure_time, reach.arrival_time) else {
                feed.skipped.stop_times_without_times += 1;
                continue;
            };
            // A stop you may not board, or one you may not get off at, is not a
            // connection you can ride.
            if leave.pickup_type == PickupDropOffType::NotAvailable
                || reach.drop_off_type == PickupDropOffType::NotAvailable
            {
                feed.skipped.connections_not_boardable += 1;
                continue;
            }
            if arrives < departs {
                continue; // a connection that lands before it leaves
            }

            let from = intern_stop(&leave.stop, &mut feed);
            let to = intern_stop(&reach.stop, &mut feed);
            if from == to {
                continue;
            }
            feed.from_stop.push(from);
            feed.to_stop.push(to);
            feed.departs.push(departs);
            feed.arrives.push(arrives);
            feed.trip.push(dense_trip);
            used = true;
        }

        if used {
            feed.trip_ids.push(trip_id.clone());
            feed.trip_routes.push(trip.route_id.clone());
        }
    }

    feed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name)
    }

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    /// The fixture's connections as `(from_stop_id, to_stop_id, departs, arrives)`.
    fn readable(feed: &Feed) -> Vec<(&str, &str, Time, Time)> {
        (0..feed.num_connections())
            .map(|i| {
                (
                    feed.stop_ids[feed.from_stop[i] as usize].as_str(),
                    feed.stop_ids[feed.to_stop[i] as usize].as_str(),
                    feed.departs[i],
                    feed.arrives[i],
                )
            })
            .collect()
    }

    #[test]
    fn a_weekday_reads_the_weekday_service_and_not_the_weekend_one() {
        // 2026-08-17 is a Monday.
        let feed = load(&fixture("tiny"), date(2026, 8, 17)).unwrap();
        let connections = readable(&feed);
        assert!(
            connections.contains(&("A", "B", 28_800, 29_400)),
            "WEEKDAY1 A->B"
        );
        assert!(
            connections.contains(&("B", "C", 29_700, 30_000)),
            "WEEKDAY2 B->C"
        );
        assert!(
            !connections.contains(&("A", "B", 32_400, 33_600)),
            "WEEKEND1 does not run on a Monday"
        );
    }

    #[test]
    fn a_saturday_reads_the_weekend_service_and_not_the_weekday_one() {
        // 2026-08-22 is a Saturday.
        let feed = load(&fixture("tiny"), date(2026, 8, 22)).unwrap();
        let connections = readable(&feed);
        assert!(
            connections.contains(&("A", "B", 32_400, 33_600)),
            "WEEKEND1"
        );
        assert!(
            !connections.iter().any(|c| c.2 == 28_800),
            "no weekday departures on a Saturday"
        );
    }

    #[test]
    fn calendar_dates_override_the_weekday_pattern() {
        // 2026-09-04 is a Friday, but the fixture removes WEEKDAY that day and
        // adds WEEKEND. This is the whole reason not to read calendar.txt alone.
        let friday = load(&fixture("tiny"), date(2026, 9, 4)).unwrap();
        let connections = readable(&friday);
        assert!(
            connections.contains(&("A", "B", 32_400, 33_600)),
            "the weekend service was added to a Friday"
        );
        assert!(
            !connections.iter().any(|c| c.2 == 28_800),
            "and the weekday service was taken off it"
        );

        // The Friday before behaves normally, so it is the exception doing this.
        let normal = load(&fixture("tiny"), date(2026, 8, 28)).unwrap();
        assert!(readable(&normal).iter().any(|c| c.2 == 28_800));
    }

    #[test]
    fn a_time_past_midnight_is_read_past_midnight() {
        // NIGHT1 arrives at 24:10, which is 86_400 + 600. A reader that wrapped
        // it into the next day would put it at 600 and quietly invent a bus
        // running at ten past midnight in the *morning*.
        let feed = load(&fixture("tiny"), date(2026, 8, 17)).unwrap();
        let night = readable(&feed)
            .into_iter()
            .find(|c| c.2 == 85_800)
            .expect("NIGHT1 leaves A at 23:50");
        assert_eq!(night.3, 87_000, "24:10 is a day and ten minutes");
    }

    #[test]
    fn a_date_outside_the_calendar_has_no_service() {
        let feed = load(&fixture("tiny"), date(2020, 1, 1)).unwrap();
        assert_eq!(feed.num_connections(), 0);
        assert!(feed.skipped.trips_not_running > 0, "and says why");
    }

    #[test]
    fn stops_only_appear_when_something_runs_to_them() {
        // Dense indices are assigned on first sight, so a feed covering a region
        // and a day covering a corner of it does not carry the whole region.
        let saturday = load(&fixture("tiny"), date(2026, 8, 22)).unwrap();
        assert_eq!(saturday.num_stops(), 2, "WEEKEND1 only touches A and B");
        let monday = load(&fixture("tiny"), date(2026, 8, 17)).unwrap();
        assert_eq!(monday.num_stops(), 3);
    }

    #[test]
    fn what_was_dropped_is_counted() {
        let feed = load(&fixture("tiny"), date(2026, 8, 17)).unwrap();
        assert!(feed.skipped.is_clean(), "{:?}", feed.skipped);
        assert!(feed.skipped.trips_not_running > 0, "the weekend trip");
    }

    #[test]
    fn a_missing_feed_says_so_rather_than_reading_nothing() {
        assert!(load(&fixture("no-such-feed"), date(2026, 8, 17)).is_err());
    }
}
