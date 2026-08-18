# Public transit labeling

> Delling, D., Dibbelt, J., Pajor, T. & Werneck, R. F. *Public transit
> labeling.* SEA 2015, 273–285.

`routelab.PTL` · [source](../../python/routelab/kernels/ptl.py) ·
checked against [RAPTOR](raptor.md), [CSA](csa.md), [TripBased](trip-based.md)
and both [Pyrga models](pyrga.md)

[RAPTOR](raptor.md) and [CSA](csa.md) stopped building a graph. PTL builds the
biggest one of all — and then never searches it. The searching happens once, at
preprocessing, for every possible query at once; what a query does is intersect
two sorted lists.

## The algorithm

The graph is the [time-expanded model](pyrga.md), and the observation is that
every arc in it points *forward in time*. So it is a DAG, and "is there a
journey from this event to that one" is plain reachability.

Reachability in a DAG can be precomputed as **2-hop hub labels**: give every
vertex a forward label of hubs it can reach and a backward label of hubs that
can reach it, such that any reachable pair shares at least one hub.

```
preprocessing:
    build the time-expanded DAG                    # a vertex per event
    for every vertex v:
        L→[v] ← hubs reachable from v              # sorted
        L←[v] ← hubs that reach v                  # sorted
    such that u reaches v  ⟺  L→[u] ∩ L←[v] ≠ ∅

query (earliest arrival, origin p, target q, time t):
    e ← first event at p after t
    for each event f at q, in time order:          # binary search
        if L→[e] ∩ L←[f] ≠ ∅:  return f            # a merge of two sorted lists
```

The intersection is a linear merge of two sorted lists, so a query costs
microseconds and touches nothing that resembles a search. The bill arrives at
bind time instead: minutes and hundreds of megabytes on a city network.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> env = rl.Environment(feed)

>>> planner = rl.PTL().bind(env)                      # the labels are built here
>>> planner.route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

The same twenty minutes the other four techniques return, from a query that did
no searching at all. `footprint` is the number to watch on anything bigger —
this is the technique whose preprocessing you feel.

## Every departure worth taking

`profile` is the paper's stop-label query rather than its event-label one: a
label per stop keeping, per hub, only the latest departure that reaches it and
the earliest arrival from it.

```python
>>> journeys = planner.profile("A", "C", departing=time(7, 0), until=time(9, 0))
>>> [(j.departs, j.arrives) for j in journeys]
[(28800, 30000)]

```

The same question [CSA](csa.md)'s profile answers, and the tests hold the two to
the same answers. Walks at either end are the paper's **superlabels**, assembled
during the query out of the neighbouring stops' labels rather than stored.

## What it refuses

PTL keeps labels, not a table of costs, so — like the two [Pyrga
models](pyrga.md) — it has no search space and says so:

```python
>>> planner.search("A", departing=time(8, 0))     # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
NotImplementedError: PTL answers with a journey rather than a cost per node ...

```

A label answers a *pair* of stops. There is no cheap way to read "every stop's
earliest arrival" out of it, which is exactly the trade: the technique that
knows every answer in advance is the one that cannot enumerate them.

## What is not here

- **RXL**, the paper's own labeling algorithm. The labels here are built by
  pruned labeling (Akiba, Iwata & Yoshida, 2013), which the paper uses as a
  black box. The labels come out larger; every query is the paper's own.
- **Multicriteria labels.** `max_transfers` names [RAPTOR](raptor.md).
- **A minimum change time**, which no timetable technique here honours yet.

## See also

- [Pyrga et al.](pyrga.md) — the time-expanded graph this labels, and the model
  that built it first.
- [CSA](csa.md) — the opposite trade: no preprocessing worth the name, and a
  linear scan per query.
- [Trip-based routing](trip-based.md) — preprocessing in between, spent on
  transfers rather than labels.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
