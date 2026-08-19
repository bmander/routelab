# RAPTOR

> Delling, D., Pajor, T. & Werneck, R. F. *Round-based public transit routing.*
> ALENEX 2012; Transportation Science **49**(3), 591–604 (2015).

`routelab.RAPTOR` · [source](../../python/routelab/kernels/raptor.py) ·
checked against the four other timetable techniques here

## The algorithm

A timetable is **routes** — ordered stop sequences — and **trips**, the vehicles
running along them at different times of day. RAPTOR uses that structure
directly instead of turning the timetable into a graph: round *k* rides one more
vehicle than round *k−1*.

```
τ[k][p] ← ∞                        earliest arrival at stop p using k trips
τ[0][o] ← departure + head start   for every origin o
marked  ← the origins

for round k = 1, 2, … while marked:
    τ[k] ← τ[k-1]
    routes ← every route serving a marked stop, from its earliest marked stop
    clear marked

    for each route r, from stop p:            # one linear scan per route
        t ← no trip
        for each stop q of r, from p onward:
            if t arrives at q before τ[k][q]:
                τ[k][q] ← t's arrival at q;  mark q
            if τ[k-1][q] ≤ t's departure from q:
                t ← earliest trip of r catchable at q

    for each marked p, each footpath p → q of duration w:
        if τ[k][p] + w < τ[k][q]:
            τ[k][q] ← τ[k][p] + w;  mark q
```

The inner loop is why there is no priority queue: each route is touched at most
once per round, and the scan down it is linear. Boarding is checked *after*
alighting at the same stop, which is what lets a rider drop onto an earlier trip
of the same route than the one they rode in on.

`τ[k]` is the earliest arrival using at most `k` trips, so the table is a Pareto
frontier of arrival against changes — built rather than computed afterwards.
`routes` reads it out, and `max_transfers` caps `k`. Every stop holds a label
when the rounds finish, which makes this one-to-all by nature.

## Hello world

A timetable technique needs a source of departures — a GTFS feed. `TINY_GTFS`
below is this repository's fixture feed: three stops in a line, and a change
worth making at the middle one. Substitute the path to any GTFS zip.

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> env = rl.Environment(feed)

>>> planner = rl.RAPTOR().bind(env)
>>> planner.route("A", "C", departing=time(8, 0)).routes[0]
Journey('A' → 'B' → 'C', cost=1200)

```

Twenty minutes. The fixture's slow through-service leaves A at 08:00 and reaches
C at 08:30; the answer rides it as far as B and changes onto the 08:15, arriving
at 08:20. RAPTOR found that in two rounds without a priority queue.

A feed is read for one service day, so the date is as much a part of naming it
as the path is. `bind` indexes the routes and trips; `route` runs one query
against them. Cost is elapsed seconds, and the legs carry the clock:

```python
>>> journey = planner.route("A", "C", departing=time(8, 0)).routes[0]
>>> journey.departs, journey.arrives     # seconds since the service day's midnight
(28800, 30000)
>>> journey.transfers, journey.waiting   # one change, five minutes at B
(1, 300)

```

Times run past 86,400 rather than wrapping at midnight, because GTFS writes
`25:30:00` for a bus that leaves before midnight and arrives after.

## A real feed

A city feed is the same three steps on a bigger layer. Stops are named by their
GTFS `stop_id`, and `nearest` turns a coordinate into one:

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))       # King County Metro, one Monday
env = rl.Environment(feed, rl.Footpaths(feed, within=200))

planner = rl.RAPTOR().bind(env)
journey = planner.route(
    feed.nearest(47.6062, -122.3321),              # downtown Seattle
    feed.nearest(47.7076, -122.2054),              # Juanita, across the lake
    departing=time(8, 30),
).routes[0]
journey.arrives, journey.transfers                 # when you land, and changes
feed.names()[journey.destination]                  # the name on the sign
```

That zip is not in this repository — `data/` is ignored, and everyone fetches
their own feed. Which is also why this block carries no `>>>` prompts: blocks
that have them are run as tests on every commit, and this one could not be.
`rl.GTFS` checks the path the moment you name it, so running this as written
raises `FileNotFoundError` rather than failing later.

## Routes, in the paper's sense

A "route" here is the paper's, not the feed's: a maximal set of trips sharing
one ordered stop sequence, split so no trip overtakes another. That keeps the
scan linear, and it means the count runs higher than `routes.txt` says.

```python
>>> planner.num_routes                   # from a feed with one route_id
3

```

One published bus line, three distinct stop sequences: A→B→C, B→C, and a
late-night A→B. On King County Metro's Monday service the same split turns 139
GTFS routes into 410 RAPTOR routes.

## The Pareto frontier

The rounds are the frontier, so asking for it costs nothing extra — one
journey per round that improved something, fewest changes first.

```python
>>> planner.route("A", "C", departing=time(8, 0)).routes
[Journey('A' → 'B' → 'C', cost=1200), Journey('A' → 'B' → 'C', cost=1800)]

```

Change once and arrive at 08:20, or stay aboard and arrive at 08:30. Both are
real answers; which a rider wants is not the algorithm's business. An answer
leads with the earliest arrival — the journey every other technique here would
have given — and each entry after it buys one fewer change at the price of
arriving later. On a city
feed the spread is wider: downtown Seattle to Juanita is 09:38 with three
changes, 09:41 with two, and 16:20 for a rider who will change only once.
`max_transfers` cuts the same frontier from the other end.

```python
>>> planner.route("A", "C", departing=time(8, 0), max_transfers=0).routes[0]
Journey('A' → 'B' → 'C', cost=1800)

```

## Walking between stops

The last loop relaxes footpaths, and `Footpaths` supplies them: every pair of stops
within a radius, joined by a walk. It sits in the environment beside the feed
and changes what the rounds can do.

```python
>>> env = rl.Environment(feed, rl.Footpaths(feed, within=2000))
>>> journey = rl.RAPTOR().bind(env).route("A", "C", departing=time(8, 0)).routes[0]
>>> journey.transfers, journey.walking
(0, 793)

```

Same 08:20 arrival, and now nobody changes vehicles: A and B are 1.1 km apart,
so a rider who walks it in thirteen minutes catches the 08:15 at B directly.
That is the right answer, and one the plain model cannot express.

## What it refuses

A departure time is required, with no default:

```python
>>> planner.route("A", "C").routes[0]                       # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...route() missing 1 required keyword-only argument: 'departing'

```

And a technique that cannot read a timetable says so, rather than quietly
routing over the feed's edges as though they were streets:

```python
>>> rl.Dijkstra().bind(rl.Environment(feed)).route("A", "C").routes[0]
Traceback (most recent call last):
    ...
TypeError: Dijkstra cannot route over timetable layers; it accepts scalar

```

## Advanced use

### One-to-all, and the picture

Search without a destination and every stop keeps its label.

```python
>>> planner = rl.RAPTOR().bind(rl.Environment(feed))
>>> result = planner.search("A", departing=time(8, 0))
>>> space = planner.explored(result)
>>> space
Rounds(3 stops, out to round 1)
>>> for reach in sorted(space.branches()):
...     print(reach.stop, reach.round, reach.arrives)
A 0 28800
B 1 29400
C 1 30600

```

RAPTOR keeps a label per stop per round, never a parent graph the way Dijkstra
does, so it can honestly report only where each round's frontier lay: stops as
points, coloured by the round that first reached them. That is the picture in
the paper. On a Seattle query it is 6,313 stops out to round six — the origin's
neighbourhood, then everything one bus away, then two.

A reach belongs to **one round**, and its arrival is that round's. Round 1
reaches C at 08:30 by staying aboard; the 08:20 this query answers with is
round 2's, found by changing at B — an improvement to a stop round 2 did not
discover, which is why the space stops at round 1 while the search ran two. The
best arrival is the answer rather than the search, and lives on the result:

```python
>>> result.cost(planner.node_id("C"))             # 08:20, round 2's
30000

```

## See also

- [CSA](csa.md) — the same problem with even the
  routes thrown away: one array, scanned once.
- [Trip-based routing](trip-based.md) —
  labels on trips rather than stops, with the transfers precomputed.
- [The two graph models](pyrga.md)
  RAPTOR is arguing with — a node per departure event, or a node per stop with
  edges that depend on when you arrive.
- [Dijkstra](dijkstra.md) — the same three steps on a road network.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
