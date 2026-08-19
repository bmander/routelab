# Connection scan

> Dibbelt, J., Pajor, T., Strasser, B. & Wagner, D. *Intriguingly simple and
> fast transit routing.* SEA 2013, 43–54.

`routelab.CSA` · [source](../../python/routelab/kernels/csa.py) ·
checked against [RAPTOR](raptor.md), [TripBased](trip-based.md) and both
[Pyrga models](pyrga.md)

[RAPTOR](raptor.md) threw away the graph and kept the routes. CSA's observation
is that even routes are more structure than the question needs. A timetable is
a list of connections — one vehicle, one hop, one departure time, one arrival
time — and if you sort that list by departure, a single pass over it answers the
query.

## The algorithm

```
sort every connection by departure time              # once, at bind
τ[p] ← ∞ for every stop p;  τ[o] ← departure + head start for every origin
aboard[t] ← false for every trip t

for each connection c in departure order, starting at the query's time:
    if τ[target] ≤ c.departs:  stop                  # nothing later can help
    if aboard[c.trip] or τ[c.from] ≤ c.departs:      # reachable?
        aboard[c.trip] ← true
        if c.arrives < τ[c.to]:
            τ[c.to] ← c.arrives
            for each footpath c.to → q of duration w:
                τ[q] ← min(τ[q], c.arrives + w)
```

That is the whole thing: one array, one flag per trip, one label per stop, and
no priority queue, no graph, and no routes.

Why one pass suffices: a timetable is *aperiodic*, so connections can be
totally ordered by departure, and a connection you could have caught is always
earlier in the array than any connection it lets you catch. Being "reachable"
therefore only depends on labels already written — you are aboard the trip
already, or you are standing at its departure stop in time.

The scan starts with a binary search for the first connection leaving after the
query, and with a target it stops as soon as the array passes the target's
label.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> env = rl.Environment(feed)

>>> planner = rl.CSA().bind(env)
>>> planner.route("A", "C", departing=time(8, 0)).routes[0]
Journey('A' → 'B' → 'C', cost=1200)

```

Binding sorts the connections, and that is all it does:

```python
>>> planner.num_connections          # the length of the array a query scans
4
>>> planner.num_trips                # one per unbroken chain of connections
3

```

King County Metro's Monday is 421,604 connections sorted in 0.4 s and 23 MB —
which, next to [trip-based routing](trip-based.md)'s eleven seconds, is what
"intriguingly simple" is buying.

## The sweep, and what it labelled

Without a target the scan runs to the end of the day and every stop holds its
earliest arrival, so unlike the [Pyrga models](pyrga.md) this technique has a
real `search` and something to draw:

```python
>>> result = planner.search("A", departing=time(8, 0))
>>> result
ScanSearch(3 stops, 4 connections scanned)
>>> planner.explored(result)
Scan(3 stops within 20 min)

```

`Scan` stamps every stop with how long after departure the sweep reached it:
the origin's neighbourhood first, the far side of the network last. On a
Seattle query toward one target it labels 4,320 stops and reads 29,426 of the
day's connections — and `result.scanned` is the paper's own measure of work,
which is why it rides on the result rather than being inferred from a clock.

## Reading the array backwards

The same array scanned the other way is a **profile** (the paper's §4, pCSA):
every journey worth leaving on within a window, one per Pareto-optimal pair of
departure and arrival.

```python
>>> journeys = planner.profile("A", "C", departing=time(7, 0), until=time(9, 0))
>>> [(j.departs, j.arrives) for j in journeys]
[(28800, 30000)]

```

One departure worth taking in that two-hour window: leave A at 08:00, reach C
at 08:20. A departure here is the *latest* moment you can leave and still make
that arrival — the paper's profile is a step function and these are its steps —
so `route(origin, target, departing=j.departs)` arrives when `j` does, for
every `j` in the list. The tests hold it to that at every second of a window,
against the time-dependent model.

That is the question a rider with a flexible morning asks, and the one rRAPTOR
would answer for the round-based technique. On a city feed a two-hour window
gives four or five steps rather than one.

## What it refuses

`until` belongs to `profile`, not to `route`, and each verb takes exactly what
it understands:

```python
>>> planner.route("A", "C", departing=time(8, 0), until=time(9, 0)).routes[0]  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...route() got an unexpected keyword argument 'until'

```

A connection scan counts nothing but time, so a cap on changes belongs
elsewhere:

```python
>>> planner.route("A", "C", departing=time(8, 0), max_transfers=1).routes[0]   # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...route() got an unexpected keyword argument 'max_transfers'

```

That is not a gap so much as the model: CSA has no reason to prefer a journey
with fewer changes over another that arrives at the same moment, so on a city
feed it will happily return a four-change itinerary where RAPTOR returns a
three-change one at the same arrival time.

## What is not here

- **The cache-friendlier profile layouts** — pseudoconnections and time
  indexing, pCSA-C and pCSA-CT.
- **The multicriteria profile**, mcpCSA: arrival against trips taken.
- **The minimum expected arrival time problem** (§5), which is the paper's
  answer to unreliable vehicles.
- **A minimum change time.** Changing is instantaneous, as it is for the four
  techniques CSA is checked against.

Journey extraction is not spelled out in the paper; the pointers here are the
obvious ones — the connection that reached each stop, and the connection each
trip was entered with — and every itinerary is validated the same way the other
four are.

## See also

- [RAPTOR](raptor.md) — routes and rounds rather than one array, and Pareto over
  changes by construction.
- [Trip-based routing](trip-based.md) — the other answer to "what should carry
  the label", and the other `profile` here.
- [ULTRA](ultra.md) — `ULTRA(CSA())`, which removes the footpath radius.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
