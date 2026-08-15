//! Who is travelling, which decides what counts as a road and how fast it is.
//!
//! The same extract makes three different graphs depending on who is asking. A
//! profile is the answer: which `highway` values are usable, how fast each one
//! goes, and whether one-way restrictions apply. It is data rather than code, so
//! it can be built in Python and applied here.

use std::collections::HashMap;

use crate::conditional;

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
    /// Which access keys speak for this traveller, most specific first — a
    /// pedestrian reads `foot` then `access`, a driver `motor_vehicle` then
    /// `vehicle` then `access`. Empty means access tags are ignored entirely,
    /// which is what a profile built before this existed gets.
    access_keys: Vec<String>,
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
            access_keys: Vec::new(),
        }
    }

    /// The same profile, reading these access keys — most specific first.
    pub fn reading_access(mut self, keys: impl IntoIterator<Item = String>) -> Self {
        self.access_keys = keys.into_iter().collect();
        self
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

        // When the way may be used at all, from whichever access key speaks for
        // this traveller. `None` is no restriction; an empty list is never.
        let access = self.access_windows(tags);
        if access.as_ref().is_some_and(Vec::is_empty) {
            return None; // shut at every hour there is, so not a way at all
        }

        // Which directions, and — for a reversible lane — when each of them.
        // `oneway=reversible` opens neither direction on its own; the schedule
        // is what reopens it, one way at a time.
        let (forward, backward) = if self.respect_oneway {
            oneway_directions(tags)
        } else {
            (true, true)
        };
        let reversal = self
            .respect_oneway
            .then_some(tags.oneway_conditional.as_deref())
            .flatten()
            .and_then(conditional::oneway_windows);

        // Both restrictions have to hold at once, so they intersect. Either may
        // be absent, and usually is.
        let combine = |direction: Option<Vec<conditional::Window>>| match (&access, direction) {
            (Some(access), Some(direction)) => Some(conditional::intersect(access, &direction)),
            (Some(access), None) => Some(access.clone()),
            (None, direction) => direction,
        };
        let (ahead, behind) = match reversal {
            Some((ahead, behind)) => (combine(Some(ahead)), combine(Some(behind))),
            None => (combine(None), combine(None)),
        };

        // A direction is usable if it was open to begin with, or if a schedule
        // opens it at some hour.
        let usable = |plain: bool, windows: &Option<Vec<conditional::Window>>| {
            plain || windows.as_ref().is_some_and(|w| !w.is_empty())
        };
        let (forward, backward) = (usable(forward, &ahead), usable(backward, &behind));
        if !forward && !backward {
            return None;
        }

        Some(Travel {
            speed,
            forward,
            backward,
            forward_open: ahead,
            backward_open: behind,
        })
    }

    /// The windows during which this traveller may use the way at all.
    ///
    /// **A conditional tag answers on its own.** Where one exists — the most
    /// specific this profile reads — it says both when the way is open and, by
    /// inversion, when it is not, and any plain tag beside it is ignored.
    /// Every Ballard Locks footway carries `foot=yes` next to
    /// `access:conditional=yes @(07:00-21:00)`; reading that `foot=yes` as the
    /// default is what made a whole city's gate disappear.
    ///
    /// Only when nothing states hours does the plain tag decide, and then only
    /// to exclude: `foot=no` keeps a path out of a walking graph.
    fn access_windows(&self, tags: &WayTags) -> Option<Vec<conditional::Window>> {
        if let Some(schedule) = self
            .access_keys
            .iter()
            .find_map(|key| tags.access_for(key).1)
        {
            return conditional::open_windows(schedule);
        }
        match self
            .access_keys
            .iter()
            .find_map(|key| tags.access_for(key).0)
            .and_then(conditional::permits)
        {
            Some(false) => Some(Vec::new()), // never allowed
            _ => None,                       // allowed always, or no opinion
        }
    }
}

/// How a particular way is traversed by a particular profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Travel {
    pub speed: f64,
    pub forward: bool,
    pub backward: bool,
    /// When the forward edge may be used; `None` is always. An empty list means
    /// never, which only arises for a direction a schedule closes entirely.
    pub forward_open: Option<Vec<conditional::Window>>,
    pub backward_open: Option<Vec<conditional::Window>>,
}

/// Access keys, most specific first. A profile names the ones that speak for
/// it, and the first one a way carries is the one that decides.
///
/// `access` is the catch-all every profile falls back to, so it belongs last in
/// every list rather than in this constant.
pub const ACCESS_KEYS: [&str; 6] = [
    "foot",
    "bicycle",
    "motor_vehicle",
    "vehicle",
    "bus",
    "access",
];

/// The handful of tags that decide whether and how a way can be travelled.
///
/// Access tags are kept as `(key, value)` pairs rather than named fields
/// because which of them applies is the profile's business, not this struct's:
/// a pedestrian reads `foot` and falls back to `access`, a driver reads
/// `motor_vehicle` first. Each is stored twice over — plain and `:conditional` —
/// since the plain one is the default the conditional is an exception to.
#[derive(Debug, Default, Clone)]
pub struct WayTags {
    pub highway: Option<String>,
    pub maxspeed: Option<String>,
    pub oneway: Option<String>,
    pub oneway_conditional: Option<String>,
    pub junction: Option<String>,
    /// `(key, plain value)` for each access key the way carries.
    pub access: Vec<(String, String)>,
    /// `(key, conditional value)`, keyed by the same names.
    pub access_conditional: Vec<(String, String)>,
}

impl WayTags {
    /// Record a tag if it is one of the few that matter, and say whether it was.
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        let field = match key {
            "highway" => &mut self.highway,
            "maxspeed" => &mut self.maxspeed,
            "oneway" => &mut self.oneway,
            "oneway:conditional" => &mut self.oneway_conditional,
            "junction" => &mut self.junction,
            other => {
                let (list, name) = match other.strip_suffix(":conditional") {
                    Some(base) => (&mut self.access_conditional, base),
                    None => (&mut self.access, other),
                };
                if !ACCESS_KEYS.contains(&name) {
                    return false;
                }
                list.push((name.to_string(), value.to_string()));
                return true;
            }
        };
        *field = Some(value.to_string());
        true
    }

    fn lookup<'a>(list: &'a [(String, String)], key: &str) -> Option<&'a str> {
        list.iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    /// The plain and conditional access values for `key`.
    pub fn access_for(&self, key: &str) -> (Option<&str>, Option<&str>) {
        (
            Self::lookup(&self.access, key),
            Self::lookup(&self.access_conditional, key),
        )
    }
}

/// Which directions a way may be travelled, from `oneway` and `junction`.
fn oneway_directions(tags: &WayTags) -> (bool, bool) {
    match tags.oneway.as_deref() {
        Some("yes") | Some("true") | Some("1") => (true, false),
        Some("-1") | Some("reverse") => (false, true),
        Some("no") | Some("false") | Some("0") => (true, true),
        // A reversible lane runs one way at a time, and which way is a matter
        // of the clock — `oneway:conditional` says when. Without a schedule to
        // read, the honest reading is that it is not freely traversable in
        // either direction, which is what `travel` turns into a closed edge.
        Some("reversible") | Some("alternating") => (false, false),
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
