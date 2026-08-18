# ULTRA

> Baum, M., Buchhold, V., Sauer, J., Wagner, D. & Zündorf, T. *UnLimited
> TRAnsfers for multi-modal route planning: an efficient solution.* ESA 2019,
> 14:1–14:16.

`routelab.ULTRA` · [source](../../python/routelab/kernels/ultra.py) ·
answers exactly as the technique it wraps, and is tested against it

Every timetable technique here relaxes footpaths between stops. Where do those
footpaths come from? In practice: pick a radius, join every pair of stops
within it, and hope. That radius is a lie with consequences at both ends — too
small and a genuinely useful ten-minute walk between two lines is invisible;
too large and the table explodes, because the transfers must be *closed* under
composition and two hundred metres of King County Metro closes to five times
its own size while four hundred closes to a hundred and thirty times.

ULTRA's answer is to compute, once, which walks a journey could ever actually
want — and there turn out to be very few of them.

## The algorithm

The insight: a walking transfer is only ever useful *between two specific
trips*. So enumerate the candidates and keep only the ones that survive.

```
preprocessing, once per network:
    for each trip t, each stop p where a rider could alight:
        run a bounded multi-source Dijkstra over the *street* graph from p
        for each stop q it reaches:
            for each trip u a rider could board at q:
                if (alight t at p, walk p→q, board u at q) is Pareto-optimal
                   among all ways of getting from t to u:
                    keep the shortcut p→q
    everything else is discarded

query:
    run RAPTOR / CSA / trip-based as written, with the kept shortcuts as its
    footpaths
```

The walking search runs over the street network without any radius at all — a
long walk is a path of short hops rather than an edge somebody had to write
down in advance. What comes out is a small set of transfer shortcuts, and the
query is then a stock timetable technique that never knows the difference.

That is why this is a **wrapper**, not a technique: `ULTRA(RAPTOR())`,
`ULTRA(CSA())`, `ULTRA(TripBased())`.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> streets = rl.ScalarEdges(
...     ("A", "B", 60), ("B", "A", 60), ("B", "C", 60), ("C", "B", 60)
... )
>>> env = rl.Environment(feed, streets)

>>> rl.ULTRA(rl.RAPTOR()).bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=120)

```

Two minutes of walking beats waiting for the 08:00 at all. The pavement here is
a street network rather than a table of stop-to-stop footpaths — the same layer
[Dijkstra](dijkstra.md) would route over — and ULTRA is what turns it into
transfers a timetable technique can use.

## It wraps, it does not replace

The same environment, the same question, two different techniques underneath:

```python
>>> rl.ULTRA(rl.RAPTOR())
ULTRA(RAPTOR())
>>> rl.ULTRA(rl.CSA()).bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=120)

```

The wrapped technique is unmodified. That is the paper's claim and the reason
this is worth having: the transfers are a preprocessing problem, solved once,
and every query technique benefits without being rewritten.

## What it refuses

A wrapped technique must keep a label per stop, because ULTRA's query has to
read every stop's arrival to know where a rider could get off. The two that
answer with a journey and nothing else are refused by name — from the shelf,
not from a hand-maintained list:

```python
>>> rl.ULTRA(rl.TimeDependent())                  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ULTRA works the transfers out for a technique that keeps a label per stop, and TimeDependent() keeps none ...

```

## On a real network

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
streets = rl.OSM("seattle.osm.pbf", rl.Walking())
env = rl.Environment(feed, streets, rl.Access(feed, streets))

planner = rl.ULTRA(rl.RAPTOR()).bind(env)     # minutes of preprocessing
planner.route(doorstep, office, departing=time(8, 30))
```

This is the corner of the design space where you pay up front and are rewarded
at query time: minutes of preprocessing, then millisecond queries over a
transfer set with no radius in it anywhere.
[LabelConstrained](label-constrained.md) is the opposite corner — nothing
precomputed, the whole network searched — and [UCCH](ucch.md) is the middle
one.

## What is not here

The paper's *event-based* variants, and its treatment of ULTRA-RAPTOR's
multicriteria form. What is implemented is the core: the shortcut computation,
and the three query techniques it feeds.

## See also

- [RAPTOR](raptor.md), [CSA](csa.md), [trip-based routing](trip-based.md) — the
  three techniques this can wrap.
- [LabelConstrained](label-constrained.md) — the same multimodal question with
  no preprocessing at all.
- [UCCH](ucch.md) — the middle corner: contract the walking, keep the language
  a query input.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
