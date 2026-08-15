A hand-written GTFS feed, small enough to check by eye.

Kept as a directory of CSV rather than a zip because `gtfs-structures` reads
either, and a reviewer can see what changed. It is the transit counterpart of
`crates/routelab-osm/tests/data/*.osm`.

Three stops in a line, and a choice worth making at the middle one:

    A --- B --- C

  trip WEEKDAY1  A 08:00 -> B 08:10,  B 08:12 -> C 08:30    (slow through)
  trip WEEKDAY2  B 08:15 -> C 08:20                         (the good change)
  trip WEEKEND1  A 09:00 -> B 09:20                          (Sa-Su only)
  trip NIGHT1    A 23:50 -> B 24:10                          (past midnight)

Leaving A at 08:00 you should reach C at 08:20 by changing at B, not 08:30 by
staying aboard. NIGHT1 exists so a reader that wraps at midnight is caught.
