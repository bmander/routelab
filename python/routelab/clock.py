"""When a journey starts, on the only clock the schedules describe.

Every restriction this library reads from OpenStreetMap repeats weekly — a gate
open 07:00–21:00, a lane reversed `Mo-Fr 05:00-11:00`. None of them names a
date, a month or a public holiday, because schedules that do are refused at
parse time rather than approximated. So the clock is a week: seconds since
Monday 00:00, wrapping at :data:`WEEK`.

That is a real limit and worth knowing rather than discovering. A departure is
*a time of week*, so routing "next Tuesday at nine" and "this Tuesday at nine"
are the same question here, and a holiday timetable cannot be expressed at all.
What it buys is that the schedules which *can* be expressed are exact.
"""

from __future__ import annotations

import datetime as _datetime
from typing import Union

__all__ = ["WEEK", "Departure", "weekly_seconds"]

#: Seconds in a week — the whole domain of a schedule.
WEEK = 7 * 24 * 60 * 60

#: What a caller may hand to ``departing=``. A `datetime` carries the weekday
#: with it; a bare `time` needs one, and Monday is the sensible reading of "at
#: nine" when nothing else is said.
Departure = Union[_datetime.datetime, _datetime.time, int]


def weekly_seconds(departing: Departure) -> int:
    """Seconds since Monday 00:00, from whatever the caller found natural.

        >>> from datetime import datetime, time
        >>> weekly_seconds(time(7, 30))
        27000
        >>> weekly_seconds(datetime(2026, 8, 14, 7, 30))    # a Friday
        372600

    An `int` passes through, so a caller working in the kernel's own units is
    not forced to dress them up.
    """
    if isinstance(departing, bool):
        # `bool` is an `int`, and a boolean departure time is a mistake rather
        # than midnight on Monday.
        raise TypeError(f"a departure time cannot be {departing!r}")
    if isinstance(departing, int):
        return departing % WEEK
    if isinstance(departing, _datetime.datetime):
        day = departing.weekday()
        clock = departing.time()
    elif isinstance(departing, _datetime.time):
        day = 0  # no date given, so the week starts where the clock does
        clock = departing
    else:
        raise TypeError(
            f"a departure time is a datetime, a time, or seconds since Monday "
            f"00:00 — not {type(departing).__name__}"
        )
    seconds = clock.hour * 3600 + clock.minute * 60 + clock.second
    return (day * 24 * 3600 + seconds) % WEEK
