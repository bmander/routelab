//! Who is travelling, which decides what counts as a road and how fast it is.
//!
//! The same extract makes three different graphs depending on who is asking. A
//! profile is the answer: which `highway` values are usable, how fast each one
//! goes, and whether one-way restrictions apply. It is data rather than code, so
//! it can be built in Python and applied here.

use std::collections::HashMap;

/// Metres per second, for a way whose class is priced but whose speed is unknown.
const FALLBACK_SPEED: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Profile {
    /// `highway` value to travel speed in metres per second. A class that is
    /// absent is not routable for this profile at all — that is how a footpath
    /// stays out of a driving graph.
    speeds: HashMap<String, f64>,
    /// Whether `oneway` restrictions bind. They do for cars, not for pedestrians.
    respect_oneway: bool,
    /// Whether a way's own `maxspeed` tag overrides its class speed.
    use_maxspeed: bool,
}

impl Profile {
    pub fn new(
        speeds: impl IntoIterator<Item = (String, f64)>,
        respect_oneway: bool,
        use_maxspeed: bool,
    ) -> Self {
        Profile {
            speeds: speeds.into_iter().collect(),
            respect_oneway,
            use_maxspeed,
        }
    }

    /// The fastest anything travels under this profile, which is what makes a
    /// distance-based lower bound admissible.
    pub fn max_speed(&self) -> f64 {
        self.speeds.values().copied().fold(FALLBACK_SPEED, f64::max)
    }

    /// How this profile travels a way, or `None` if it cannot.
    pub fn travel(&self, tags: &WayTags) -> Option<Travel> {
        let class_speed = *self.speeds.get(tags.highway.as_deref()?)?;
        let speed = self
            .use_maxspeed
            .then(|| tags.maxspeed.as_deref().and_then(parse_maxspeed))
            .flatten()
            // A posted limit faster than the profile's own top speed is a limit
            // for the road, not a promise about this traveller: a pedestrian does
            // not walk at 50 km/h because the sign says so.
            .map(|posted| posted.min(self.max_speed()))
            .unwrap_or(class_speed);

        let (forward, backward) = if self.respect_oneway {
            oneway_directions(tags)
        } else {
            (true, true)
        };
        Some(Travel {
            speed,
            forward,
            backward,
        })
    }
}

/// How a particular way is traversed by a particular profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Travel {
    pub speed: f64,
    pub forward: bool,
    pub backward: bool,
}

/// The handful of tags that decide whether and how a way can be travelled.
#[derive(Debug, Default, Clone)]
pub struct WayTags {
    pub highway: Option<String>,
    pub maxspeed: Option<String>,
    pub oneway: Option<String>,
    pub junction: Option<String>,
}

impl WayTags {
    /// Record a tag if it is one of the few that matter, and say whether it was.
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        let field = match key {
            "highway" => &mut self.highway,
            "maxspeed" => &mut self.maxspeed,
            "oneway" => &mut self.oneway,
            "junction" => &mut self.junction,
            _ => return false,
        };
        *field = Some(value.to_string());
        true
    }
}

/// Which directions a way may be travelled, from `oneway` and `junction`.
fn oneway_directions(tags: &WayTags) -> (bool, bool) {
    match tags.oneway.as_deref() {
        Some("yes") | Some("true") | Some("1") => (true, false),
        Some("-1") | Some("reverse") => (false, true),
        Some("no") | Some("false") | Some("0") => (true, true),
        // Roundabouts are one-way by convention, without saying so.
        _ => match tags.junction.as_deref() {
            Some("roundabout") | Some("circular") => (true, false),
            _ => (true, true),
        },
    }
}

/// Parse a `maxspeed` value into metres per second.
///
/// Bare numbers are km/h, which is what the tag means when it says nothing else.
/// Anything unrecognised — "walk", "none", "RO:urban" — yields `None`, and the
/// way falls back to its class speed rather than to a guess.
fn parse_maxspeed(value: &str) -> Option<f64> {
    let value = value.trim();
    let (number, to_metres_per_second) = match value.strip_suffix("mph") {
        Some(number) => (number, 0.447_04),
        None => (value.strip_suffix("km/h").unwrap_or(value), 1.0 / 3.6),
    };
    let parsed: f64 = number.trim().parse().ok()?;
    (parsed > 0.0).then_some(parsed * to_metres_per_second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn driving() -> Profile {
        Profile::new(
            [
                ("motorway".to_string(), 30.0),
                ("residential".to_string(), 8.0),
            ],
            true,
            true,
        )
    }

    fn tags(pairs: &[(&str, &str)]) -> WayTags {
        let mut tags = WayTags::default();
        for (key, value) in pairs {
            tags.set(key, value);
        }
        tags
    }

    #[test]
    fn a_class_the_profile_does_not_price_is_not_routable() {
        assert!(driving().travel(&tags(&[("highway", "footway")])).is_none());
        assert!(driving().travel(&tags(&[("name", "Nowhere")])).is_none());
    }

    #[test]
    fn class_speed_applies_when_nothing_is_posted() {
        let travel = driving()
            .travel(&tags(&[("highway", "residential")]))
            .unwrap();
        assert_eq!(travel.speed, 8.0);
        assert_eq!((travel.forward, travel.backward), (true, true));
    }

    #[test]
    fn a_posted_limit_overrides_the_class() {
        let travel = driving()
            .travel(&tags(&[("highway", "residential"), ("maxspeed", "36")]))
            .unwrap();
        assert!((travel.speed - 10.0).abs() < 1e-9, "36 km/h is 10 m/s");

        let mph = driving()
            .travel(&tags(&[("highway", "residential"), ("maxspeed", "20 mph")]))
            .unwrap();
        assert!((mph.speed - 8.940_8).abs() < 1e-3);
    }

    #[test]
    fn an_unreadable_limit_falls_back_to_the_class() {
        for posted in ["none", "walk", "RO:urban", "", "-5"] {
            let travel = driving()
                .travel(&tags(&[("highway", "residential"), ("maxspeed", posted)]))
                .unwrap();
            assert_eq!(travel.speed, 8.0, "maxspeed={posted:?}");
        }
    }

    #[test]
    fn a_posted_limit_cannot_speed_up_a_pedestrian() {
        let walking = Profile::new([("residential".to_string(), 1.4)], false, true);
        let travel = walking
            .travel(&tags(&[("highway", "residential"), ("maxspeed", "50")]))
            .unwrap();
        assert_eq!(travel.speed, 1.4);
    }

    #[test]
    fn oneway_binds_only_on_profiles_that_respect_it() {
        let oneway = tags(&[("highway", "residential"), ("oneway", "yes")]);
        let driving = driving().travel(&oneway).unwrap();
        assert_eq!((driving.forward, driving.backward), (true, false));

        let walking = Profile::new([("residential".to_string(), 1.4)], false, true);
        let walked = walking.travel(&oneway).unwrap();
        assert_eq!((walked.forward, walked.backward), (true, true));
    }

    #[test]
    fn reversed_and_roundabout_forms_are_understood() {
        let reversed = driving()
            .travel(&tags(&[("highway", "residential"), ("oneway", "-1")]))
            .unwrap();
        assert_eq!((reversed.forward, reversed.backward), (false, true));

        let roundabout = driving()
            .travel(&tags(&[
                ("highway", "residential"),
                ("junction", "roundabout"),
            ]))
            .unwrap();
        assert_eq!((roundabout.forward, roundabout.backward), (true, false));
    }
}
