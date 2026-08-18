# Trip-based routing

> Witt, S. *Trip-based public transit routing.* ESA 2015, 1025–1036.

`routelab.TripBased` · [source](../../python/routelab/kernels/tripbased.py) ·
checked against [RAPTOR](raptor.md), [CSA](csa.md) and both
[Pyrga models](pyrga.md)

[RAPTOR](raptor.md) labels stops and scans routes. [CSA](csa.md) labels stops
and scans connections. Witt's observation is that once a rider is aboard a trip
at a known stop, everything that can happen next is already written in the
timetable — every stop that trip reaches, and every other trip they could change
onto. So it is **trips** that should carry the labels, and the changes between
them can be worked out once, before any query.

## The algorithm

Preprocessing computes the **transfer set**, then throws most of it away:

```
for each trip t, each stop p it reaches:
    for each stop q reachable from p on foot (or p itself):
        for each line ℓ serving q:
            add transfer t@p → (earliest trip of ℓ catchable at q)@q
    drop transfers that stay on t's own line, and U-turns

reduction:                                        # the paper's §3.2
    walk each trip backwards, keeping the earliest arrival it can reach at
    every stop with and without each transfer; drop every transfer that
    improves nothing
```

A query is then a breadth-first sweep with no priority queue at all:

```
queue[0] ← for each origin stop, the earliest trip of each line there
best ← ∞

for round n = 0, 1, … while queue[n]:
    for each trip segment (t, from, to) in queue[n]:
        if t reaches the target at time a < best:  best ← a; record it
        if t cannot beat best:  prune the segment
        for each transfer out of (t, from..to):
            queue[n+1] ← the segment it lands on
```

Each round is one more change, so the answer set is Pareto over arrival against
changes by construction, as RAPTOR's rounds are. Unlike RAPTOR, this is
point-to-point by construction: the pruning is against the best arrival *at the
target*, so a query needs one.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> env = rl.Environment(feed)

>>> planner = rl.TripBased().bind(env)                # the transfer set is built here
>>> planner.route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

Binding is where this technique spends everything. On King County Metro's
Monday it computes 35,213,140 transfers and keeps 1,216,322 — eleven seconds
and 28 MB — against [CSA](csa.md)'s 0.4 s for the same feed. Ninety-seven
percent of the transfers computed turn out to improve nothing, which is the
reduction doing its job, and it is the difference between a query sweeping a
handful of segments and sweeping thousands.

## The frontier

The rounds are changes, so the frontier falls out the same way it does for
RAPTOR:

```python
>>> planner.frontier("A", "C", departing=time(8, 0))
[Journey('A' → 'B' → 'C', cost=1800), Journey('A' → 'B' → 'C', cost=1200)]

```

Stay aboard and arrive at 08:30; change once and arrive at 08:20. That is
RAPTOR's front on this instance, exactly — the tests hold the two to it.

## What it scanned

The search space is not stops and not a tree: it is the trip segments the sweep
looked at, each from the stop it was boarded at to the last stop whose
transfers were followed.

```python
>>> result = planner.search("A", targets=[planner.node_id("C")], departing=time(8, 0))
>>> result
TripBasedSearch(3 trips, 3 segments scanned)
>>> planner.explored(result)
Segments(3 trip segments, out to round 1)

```

Drawn as lines along the stops, coloured by round the way [RAPTOR](raptor.md)'s
`Rounds` colours its stops: the trips a rider could board first, then everything
one change away, then two. On a Seattle query it is 2,533 segments out to round
four.

## Every departure worth taking

`profile` runs the same sweep once per moment the origin offers a departure,
latest first, keeping labels between runs — the paper's §3.3 — and hands back
one journey per Pareto-optimal combination of departure, arrival and changes.

```python
>>> journeys = planner.profile("A", "C", departing=time(7, 0), until=time(9, 0))
>>> [(j.departs, j.arrives, j.transfers) for j in journeys]
[(28800, 30600, 0), (28800, 30000, 1)]

```

Leave A at 08:00 either way: stay aboard and land at 08:30, or change once and
land at 08:20. Note the third criterion — [CSA](csa.md)'s profile is Pareto
over departure and arrival only, so it reports the one journey; this one keeps
both, because its rounds already know what a change costs.

## The control

`reduce=False` keeps every transfer the first phase computed. It answers
identically and takes longer, which is the paper's own way of showing the
reduction is worth its preprocessing — the same footing `RandomOrder()` has for
a [contraction hierarchy](contraction-hierarchies.md).

```python
>>> rl.TripBased(reduce=False)
TripBased(reduce=False)
>>> rl.TripBased(reduce=False).bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

## What it refuses

The sweep prunes against the best arrival at the target, so there has to be
one:

```python
>>> planner.search("A", departing=time(8, 0))       # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: TripBased, sweeping trips toward a target, searches toward a single target, ...

```

That is a real difference from RAPTOR and CSA, both of which are one-to-all by
nature, and it is why this page has no isochrone to draw.

## What is not here

A minimum change time. Changing vehicles is instantaneous, as it is for the
four techniques this is checked against — though trip-based routing and RAPTOR
are the two whose construction could honour one, since both know which vehicle
a rider is on.

## See also

- [RAPTOR](raptor.md) — labels on stops, scans over routes, and the same
  Pareto front.
- [CSA](csa.md) — labels on stops, one array of connections, and the other
  `profile` here.
- [ULTRA](ultra.md) — `ULTRA(TripBased())`, which removes the footpath radius.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
