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

## The walks at either end

Shortcuts cover the walk *between two vehicles*. The walks at the ends of a
journey cannot be precomputed, because one end of each is the query's own
source or target. §4.1 answers them with two one-to-many searches, accelerated
by **Bucket-CH** — which is why ULTRA has three preprocessing steps rather than
one:

```
preprocessing:
    core graph (Core-CH)          → what the shortcut search runs on
    transfer shortcuts            → the intermediate transfers
    Bucket-CH over the whole graph → the initial and final ones

    for each stop w:                        # filing the buckets
        upward search from w; file (w, distance) at every vertex it settles

query (Algorithm 2):
    τ(s,t), Vs, Vt ← one bidirectional CH query for the direct walk
    for v in Vs:  scan v's forward bucket   → τ(s, ·) for every stop
    for v in Vt:  scan v's backward bucket  → τ(·, t) for every stop
    keep only the transfers with τ(s,v) < τ(s,t) and τ(v,t) < τ(s,t)
    run the wrapped technique from those stops
```

The pruning is the paper's, and it is what makes a short query quick: a walk to
a stop that takes longer than walking the whole way to the target can never be
part of a better journey, so it is never considered. Bucket entries are sorted
by distance, so a scan stops at the first one that cannot beat the direct walk.

A hierarchy's search space is a property of its ranks rather than of the
distance, so `Vs` and `Vt` are a few hundred vertices however far apart the two
ends are — which is the difference between reading a few thousand bucket
entries and searching a city.

That is why this is a **wrapper**, not a technique: `ULTRA(RAPTOR())` or
`ULTRA(CSA())`, which are the two the paper names.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> streets = rl.ScalarEdges(
...     ("A", "B", 60), ("B", "A", 60), ("B", "C", 60), ("C", "B", 60)
... )
>>> env = rl.Environment(feed, streets)

>>> rl.ULTRA(rl.RAPTOR()).bind(env).route("A", "C", departing=time(8, 0)).routes[0]
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
>>> rl.ULTRA(rl.CSA()).bind(env).route("A", "C", departing=time(8, 0)).routes[0]
Journey('A' → 'B' → 'C', cost=120)

```

The wrapped technique is unmodified. That is the paper's claim and the reason
this is worth having: the transfers are a preprocessing problem, solved once,
and every query technique benefits without being rewritten.

It is bound to the **transit network** rather than to the environment ULTRA was
given — Algorithm 2's last line runs the black box on `(S ∪ {s,t}, T, R, G̃s)`,
not on the merged graph. The difference is a technique whose tables are a row
per stop against one with a row per street corner: on Seattle's pavement with
King County Metro, 6,313 rows against 560,706, nine rounds of it per query.
ULTRA translates at that seam, so what goes in and comes out is the caller's
own labels.

The shortcuts go in **unclosed**, too. Every other transfer set here is closed
under composition, because a rider who can walk a→b and b→c can walk a→c and a
model that does not know that answers a query dropped on the wrong corner with
nothing. ULTRA's shortcuts need no such repair: each already stands for a whole
walk between two stops, and the paper's guarantee is that every intermediate
transfer a Pareto-optimal journey can need is one of them. Closing them anyway
would rebuild, one composition at a time, exactly the blow-up ULTRA exists to
avoid.

## What it refuses

A wrapped technique must keep a label per stop, because ULTRA's query has to
read every stop's arrival to know where a rider could get off. `RAPTOR` and
`CSA` are those, and the argument's type says so; a caller who is not reading
types runs into the same sentence at the call:

```python
>>> rl.ULTRA(rl.TimeDependent())                  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ULTRA works the transfers out for a technique that keeps a label per stop, and TimeDependent() keeps none ...

```

`TripBased` is not one of them either, for a different reason: its sweep is
point-to-point by construction, and ULTRA's query is one-to-all over the stops
that survived its bucket scans.

## On a real network

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
streets = rl.OSM("seattle.osm.pbf", rl.Walking())
env = rl.Environment(feed, streets, rl.Access(feed, streets))

planner = rl.ULTRA(rl.RAPTOR()).bind(env)     # minutes of preprocessing
planner.route(doorstep, office, departing=time(8, 30)).routes[0]
```

This is the corner of the design space where you pay up front: ten minutes of
preprocessing and 149 MB — a core contraction, the shortcut search over it, and
a second full contraction for the buckets — for a query over a transfer set
with no radius in it anywhere. [LabelConstrained](label-constrained.md) is the
opposite corner, nothing precomputed and the whole network searched, and
[UCCH](ucch.md) is the middle one.

What that buys is on the [trade-offs page](../tradeoffs.md): **11.4 ms** on
this instance, against 34.8 for UCCH and 106.7 for searching the network. The
paper reports 12.5 ms for ULTRA-RAPTOR on Switzerland, which is the right
company to be in.

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
