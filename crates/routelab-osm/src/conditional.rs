//! Reading OpenStreetMap's `:conditional` tags — the ones that say *when*.
//!
//! A way can carry `access:conditional=yes @(07:00-21:00)` (the Ballard Locks
//! walkway, shut overnight), `access:conditional=no @ (23:00-05:00)` (a trail
//! locked at night), or `oneway:conditional=-1 @ (Mo-Fr 05:00-11:00)` (a
//! reversible expressway lane). This module turns those into the intervals
//! during which the way may actually be travelled.
//!
//! **It parses a corner of `opening_hours` and refuses the rest.** In a Seattle
//! extract, 229 routable ways carry a conditional and the schedules on them are
//! overwhelmingly `HH:MM-HH:MM` or `Mo-Fr HH:MM-HH:MM`. The remainder are not
//! clocks at all — `weight>32000lbs`, `flashing`, `rush_hour`, `dusk-dawn`,
//! `axles=5`, date ranges, `outside school hours` — and a routing library that
//! guessed at those would be inventing restrictions. Anything unreadable leaves
//! the way unrestricted and is counted, so a schedule silently dropped is a
//! number someone can look at rather than a route quietly gone wrong.
//!
//! Windows are half-open `[start, end)` intervals on a weekly clock, seconds
//! since Monday 00:00. `end <= start` wraps. The crate deliberately does not
//! depend on `routelab-core`, so these are plain pairs; whoever holds both ends
//! builds the kernel's calendar from them.

/// Seconds in a week — the whole domain of a schedule this module can express.
pub const WEEK: u32 = 7 * 24 * 60 * 60;

const DAY: u32 = 24 * 60 * 60;

/// A half-open `[start, end)` interval on the weekly clock. `end <= start`
/// wraps past the end of the week.
pub type Window = (u32, u32);

/// What a conditional clause grants during its windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Allow,
    Deny,
}

/// Read an access value — the part before the `@`, or a plain base tag.
///
/// `destination` is treated as passable, which is a simplification worth
/// naming: it really means "no through traffic", and honouring that needs to
/// know where a path is going, which an edge weight cannot express.
fn verdict(value: &str) -> Option<Verdict> {
    match value.trim() {
        "yes" | "permissive" | "designated" | "destination" | "official" => Some(Verdict::Allow),
        "no" | "private" => Some(Verdict::Deny),
        _ => None,
    }
}

/// The weekday a two- or three-letter name refers to; Monday is 0.
fn weekday(name: &str) -> Option<u32> {
    let name = name.trim();
    // `Sun` and `Sat` turn up alongside the standard two-letter forms. Both are
    // unambiguous once the leading pair matches a real day.
    if !(name.len() == 2 || name.len() == 3) {
        return None;
    }
    match &name[..2] {
        "Mo" => Some(0),
        "Tu" => Some(1),
        "We" => Some(2),
        "Th" => Some(3),
        "Fr" => Some(4),
        "Sa" => Some(5),
        "Su" => Some(6),
        _ => None,
    }
}

/// The days a `Mo`, `Mo-Fr` or `Sa-Su` selector names, wrapping through Sunday.
fn weekdays(spec: &str) -> Option<Vec<u32>> {
    let (first, last) = match spec.split_once('-') {
        Some((first, last)) => (weekday(first)?, weekday(last)?),
        None => {
            let day = weekday(spec)?;
            (day, day)
        }
    };
    let span = (last + 7 - first) % 7;
    Some((0..=span).map(|offset| (first + offset) % 7).collect())
}

/// Seconds into the day for `H:MM` or `HH:MM`. `24:00` is the end of the day.
fn time_of_day(text: &str) -> Option<u32> {
    let (hours, minutes) = text.trim().split_once(':')?;
    let hours: u32 = hours.trim().parse().ok()?;
    let minutes: u32 = minutes.trim().parse().ok()?;
    (hours <= 24 && minutes <= 59 && !(hours == 24 && minutes > 0))
        .then_some(hours * 3600 + minutes * 60)
}

/// The weekly intervals a `HH:MM-HH:MM` span covers on the given days.
fn spans(days: &[u32], text: &str) -> Option<Vec<Window>> {
    let (from, to) = text.split_once('-')?;
    let (from, to) = (time_of_day(from)?, time_of_day(to)?);
    // `to <= from` runs past midnight into the next day; `24:00` is all day.
    let length = if to > from {
        to - from
    } else {
        to + DAY - from
    };
    if length == 0 {
        return None; // a zero-length window restricts nothing and says nothing
    }
    Some(
        days.iter()
            .map(|day| {
                let start = (day * DAY + from) % WEEK;
                (start, (start + length) % WEEK)
            })
            .collect(),
    )
}

/// Read a schedule: semicolon-separated rules, each a comma-separated list of
/// time spans optionally prefixed by days.
///
/// Handles `07:00-21:00`, `Mo-Fr 05:00-11:00`, `6:00-9:00, 15:00-18:30`,
/// `Mo-Fr 06:00-09:00,15:00-18:30` — where the days carry across the commas —
/// and `Mo-Fr 05:00-11:00; Sa-Su 08:00-13:30`, where the semicolon starts a
/// fresh rule and the days do *not* carry. Returns `None` for anything else,
/// which is most of what the tag can say.
fn schedule(text: &str) -> Option<Vec<Window>> {
    let mut windows = Vec::new();
    for rule in text.split(';') {
        // Each rule names its own days, or means every day by saying nothing.
        let mut days: Vec<u32> = (0..7).collect();
        for part in rule.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return None;
            }
            // A part is either `<days> <span>` or a bare `<span>` inheriting
            // the days named earlier in the same rule.
            let span = match part.split_once(char::is_whitespace) {
                Some((prefix, rest)) => {
                    days = weekdays(prefix)?;
                    rest.trim()
                }
                None => part,
            };
            windows.extend(spans(&days, span)?);
        }
    }
    (!windows.is_empty()).then_some(windows)
}

/// Split a conditional value into its `<verdict> @ (<schedule>)` clauses.
fn clauses(value: &str) -> Option<Vec<(Verdict, Vec<Window>)>> {
    let mut parsed = Vec::new();
    for clause in split_clauses(value) {
        let (value, condition) = clause.split_once('@')?;
        let condition = condition.trim();
        let condition = condition
            .strip_prefix('(')
            .and_then(|rest| rest.strip_suffix(')'))
            .unwrap_or(condition);
        parsed.push((verdict(value)?, schedule(condition)?));
    }
    (!parsed.is_empty()).then_some(parsed)
}

/// Semicolons separate clauses, but a schedule may contain them too
/// (`Mo-Fr 05:00-11:00; Sa-Su 08:00-13:30` inside one set of brackets), so the
/// split has to respect the parentheses rather than take every semicolon.
fn split_clauses(value: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, character) in value.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ';' if depth == 0 => {
                parts.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(value[start..].trim());
    parts.retain(|part| !part.is_empty());
    parts
}

/// When a way may be travelled, from its conditional tag.
///
/// Returns the windows during which it is open. An empty vector means never;
/// `None` means the schedule could not be read, and the caller should leave the
/// way unrestricted and count the refusal.
///
/// **The conditional stands alone.** Its verdict states one side and the
/// default is the other:
///
/// - `yes @ W` — open during `W`, shut the rest of the time.
/// - `no @ W` — shut during `W`, open the rest of the time.
///
/// A plain `access=`/`foot=` tag beside it does not change that reading, and
/// letting it was a real bug rather than a hypothetical one: every Ballard
/// Locks footway carries `foot=yes` next to `access:conditional=yes @(07:00-21:00)`,
/// and treating that as the default made the gate disappear from a city.
pub fn open_windows(conditional: &str) -> Option<Vec<Window>> {
    let clauses = clauses(conditional)?;
    let open_by_default = clauses[0].0 == Verdict::Deny;

    let (allowed, denied): (Vec<_>, Vec<_>) = clauses
        .into_iter()
        .partition(|(verdict, _)| *verdict == Verdict::Allow);
    let gather = |clauses: Vec<(Verdict, Vec<Window>)>| -> Vec<Window> {
        clauses
            .into_iter()
            .flat_map(|(_, windows)| windows)
            .collect()
    };

    Some(if open_by_default {
        complement(&gather(denied))
    } else {
        merge(&gather(allowed))
    })
}

/// Whether a plain access value permits travel; `None` if it is not a value
/// this module recognises.
pub fn permits(value: &str) -> Option<bool> {
    verdict(value).map(|verdict| verdict == Verdict::Allow)
}

/// When each direction of a reversible way may be travelled.
///
/// `oneway:conditional=-1 @ (Mo-Fr 05:00-11:00); yes @ (Mo-Fr 11:15-23:00)` is
/// the Seattle express lanes: running against the way's drawn direction in the
/// morning and with it in the afternoon. Returns `(forward, backward)` windows,
/// or `None` if the schedule could not be read — in which case the caller is
/// left with `oneway=reversible`, which opens nothing.
pub fn oneway_windows(conditional: &str) -> Option<(Vec<Window>, Vec<Window>)> {
    let (mut forward, mut backward) = (Vec::new(), Vec::new());
    for clause in split_clauses(conditional) {
        let (value, condition) = clause.split_once('@')?;
        let condition = condition.trim();
        let condition = condition
            .strip_prefix('(')
            .and_then(|rest| rest.strip_suffix(')'))
            .unwrap_or(condition);
        let windows = schedule(condition)?;
        // The value is a `oneway` value, not an access one: which way traffic
        // runs during those hours.
        match value.trim() {
            "yes" | "true" | "1" => forward.extend(windows),
            "-1" | "reverse" => backward.extend(windows),
            "no" | "false" | "0" => {
                forward.extend(windows.iter().copied());
                backward.extend(windows);
            }
            _ => return None,
        }
    }
    (!forward.is_empty() || !backward.is_empty()).then(|| (merge(&forward), merge(&backward)))
}

/// The windows both sets cover — a way open only when two restrictions agree.
pub fn intersect(left: &[Window], right: &[Window]) -> Vec<Window> {
    let (left, right) = (merge(left), merge(right));
    let mut both = Vec::new();
    for &(start, end) in &left {
        for &(other_start, other_end) in &right {
            let (from, to) = (start.max(other_start), end.min(other_end));
            if from < to {
                both.push((from, to));
            }
        }
    }
    merge(&both)
}

/// Break windows into non-wrapping pieces, sort them, and fuse any that touch.
fn merge(windows: &[Window]) -> Vec<Window> {
    let mut pieces: Vec<Window> = Vec::with_capacity(windows.len() + 1);
    for &(start, end) in windows {
        if start == end {
            return vec![(0, WEEK)]; // a zero-width wrap is the whole week
        } else if end < start {
            pieces.push((start, WEEK));
            pieces.push((0, end));
        } else {
            pieces.push((start, end));
        }
    }
    pieces.sort_unstable();

    let mut merged: Vec<Window> = Vec::with_capacity(pieces.len());
    for (start, end) in pieces {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// The parts of the week these windows do *not* cover.
fn complement(windows: &[Window]) -> Vec<Window> {
    let covered = merge(windows);
    let mut gaps = Vec::with_capacity(covered.len() + 1);
    let mut cursor = 0;
    for (start, end) in covered {
        if start > cursor {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if cursor < WEEK {
        gaps.push((cursor, WEEK));
    }
    // Midnight Sunday is not a boundary anyone meant, so a gap that runs to the
    // end of the week and another that starts at its beginning are one gap.
    if gaps.len() > 1 && gaps[0].0 == 0 && gaps[gaps.len() - 1].1 == WEEK {
        let (_, first_end) = gaps.remove(0);
        let last = gaps.len() - 1;
        gaps[last].1 = first_end;
    }
    gaps
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u32 = 3600;

    fn at(day: u32, hour: u32) -> u32 {
        day * DAY + hour * HOUR
    }

    /// Is `moment` inside any of these windows? The question the kernel asks.
    fn open_at(windows: &[Window], moment: u32) -> bool {
        windows.iter().any(|&(start, end)| {
            if end <= start {
                moment >= start || moment < end
            } else {
                moment >= start && moment < end
            }
        })
    }

    #[test]
    fn the_locks_are_open_only_during_their_hours() {
        // `access:conditional=yes @(07:00-21:00)` — note the missing space,
        // which is how it is actually written on the ground.
        let windows = open_windows("yes @(07:00-21:00)").unwrap();
        for day in 0..7 {
            assert!(open_at(&windows, at(day, 12)), "day {day} noon");
            assert!(!open_at(&windows, at(day, 3)), "day {day} 3am");
            assert!(!open_at(&windows, at(day, 22)), "day {day} 10pm");
        }
    }

    #[test]
    fn a_nightly_closure_leaves_the_rest_of_the_day_open() {
        // `no @ (23:00-05:00)`, the trail form: the default is open.
        let windows = open_windows("no @ (23:00-05:00)").unwrap();
        for day in 0..7 {
            assert!(open_at(&windows, at(day, 12)), "day {day} noon");
            assert!(!open_at(&windows, at(day, 23)), "day {day} 11pm");
            assert!(!open_at(&windows, at(day, 2)), "day {day} 2am");
            assert!(open_at(&windows, at(day, 6)), "day {day} 6am");
        }
    }

    #[test]
    fn the_verdict_states_one_side_and_the_default_is_the_other() {
        let only_then = open_windows("yes @ (Mo-Fr 05:00-20:00)").unwrap();
        assert!(open_at(&only_then, at(0, 12)), "Monday midday, as stated");
        assert!(!open_at(&only_then, at(5, 12)), "Saturday, by inversion");

        let except_then = open_windows("no @ (Mo-Fr 05:00-20:00)").unwrap();
        assert!(!open_at(&except_then, at(0, 12)));
        assert!(open_at(&except_then, at(5, 12)));
    }

    #[test]
    fn the_verdict_sets_the_default_either_way() {
        let only_then = open_windows("yes @ (Mo-Fr 05:00-20:00)").unwrap();
        assert!(open_at(&only_then, at(0, 12)));
        assert!(
            !open_at(&only_then, at(5, 12)),
            "Saturday is not named, so shut"
        );

        let except_then = open_windows("no @ (Mo-Fr 05:00-20:00)").unwrap();
        assert!(!open_at(&except_then, at(0, 12)));
        assert!(open_at(&except_then, at(5, 12)), "Saturday stays open");
    }

    #[test]
    fn weekday_selectors_bind_only_the_days_they_name() {
        let windows = open_windows("yes @ (Mo-Fr 05:00-11:00)").unwrap();
        assert!(open_at(&windows, at(0, 8)), "Monday");
        assert!(open_at(&windows, at(4, 8)), "Friday");
        assert!(!open_at(&windows, at(5, 8)), "Saturday");
        assert!(!open_at(&windows, at(6, 8)), "Sunday");
    }

    #[test]
    fn a_weekend_selector_wraps_through_sunday() {
        let windows = open_windows("yes @ (Sa-Su 08:00-13:30)").unwrap();
        assert!(open_at(&windows, at(5, 9)), "Saturday");
        assert!(open_at(&windows, at(6, 9)), "Sunday");
        assert!(!open_at(&windows, at(0, 9)), "Monday");
    }

    #[test]
    fn several_clauses_combine() {
        // The express-lane shape, read as access: open in both stated windows.
        let windows = open_windows("yes @ (Mo-Fr 05:00-11:00); yes @ (Sa-Su 08:00-13:30)").unwrap();
        assert!(open_at(&windows, at(0, 8)), "Monday morning");
        assert!(open_at(&windows, at(5, 9)), "Saturday morning");
        assert!(!open_at(&windows, at(0, 20)), "Monday evening");
    }

    #[test]
    fn comma_separated_spans_share_the_days_named_once() {
        let rush = open_windows("yes @ (Mo-Fr 06:00-09:00,15:00-18:30)").unwrap();
        assert!(open_at(&rush, at(0, 7)), "morning peak");
        assert!(open_at(&rush, at(0, 16)), "evening peak");
        assert!(!open_at(&rush, at(0, 12)), "the middle of the day");
        assert!(!open_at(&rush, at(5, 7)), "Saturday is not named");
    }

    #[test]
    fn single_digit_hours_are_read() {
        // `7:00-16:00` occurs in the wild; refusing it would drop a real rule.
        let windows = open_windows("yes @ (7:00-16:00)").unwrap();
        assert!(open_at(&windows, at(0, 8)));
        assert!(!open_at(&windows, at(0, 17)));
    }

    #[test]
    fn all_day_is_spelled_with_an_end_of_24_00() {
        let windows = open_windows("yes @ (00:00-24:00)").unwrap();
        for day in 0..7 {
            assert!(open_at(&windows, at(day, 0)) && open_at(&windows, at(day, 23)));
        }
    }

    #[test]
    fn everything_that_is_not_a_clock_is_refused() {
        // Every one of these is real, and none of them is a schedule. Guessing
        // at any would invent a restriction the map does not state.
        for value in [
            "no @ (weight>32000lbs)",
            "no @ (flashing)",
            "yes @ (rush_hour)",
            "no @ (dusk-dawn)",
            "no @ (sunrise-sunset)",
            "no @ (2018 Aug 09-2018 Oct 19)",
            "yes @ (Aug-May Mo-Fr 07:50-15:30)",
            "no @ (axles=5)",
            "no @ (outside school hours)",
            "yes @ (May 23-Sep 07: Sa-Su)",
            "no @ (variable)",
            "delivery @ (07:00-21:00)",
            "yes @ (25:00-26:00)",
            "yes @ (07:00)",
            "yes",
            "",
        ] {
            assert_eq!(open_windows(value), None, "should refuse {value:?}");
        }
    }

    #[test]
    fn a_nightly_closure_leaves_one_window_a_day_not_two() {
        // 22:00-05:00 shut leaves 05:00-22:00 open — seven windows, one a day,
        // and none of them cut in half at midnight.
        let windows = open_windows("no @ (Mo-Su 22:00-05:00)").unwrap();
        assert_eq!(windows.len(), 7);
        assert_eq!(windows[0], (5 * HOUR, 22 * HOUR));
    }

    #[test]
    fn a_schedule_covering_the_whole_week_leaves_nothing_shut() {
        // `no @ (00:00-24:00)` shuts every hour there is, so nothing is open.
        assert_eq!(open_windows("no @ (00:00-24:00)"), Some(vec![]));
    }

    #[test]
    fn windows_that_meet_across_midnight_are_not_cut_in_two() {
        // A closure of 05:00-07:00 leaves one window running 07:00 to 05:00 the
        // next day, not two ending and starting at midnight.
        let windows = open_windows("no @ (05:00-07:00)").unwrap();
        assert!(open_at(&windows, at(0, 23)) && open_at(&windows, at(1, 2)));
        assert_eq!(windows.len(), 7, "one window per night, not fourteen");
    }

    #[test]
    fn semicolons_inside_brackets_do_not_split_clauses() {
        // The express lanes write both halves inside one condition.
        let windows = open_windows("yes @ (Mo-Fr 05:00-11:00; Sa-Su 08:00-13:30)").unwrap();
        assert!(open_at(&windows, at(0, 8)), "Monday morning");
        assert!(open_at(&windows, at(5, 9)), "Saturday morning");
    }
}
