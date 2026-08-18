# Two models of a timetable

> Pyrga, E., Schulz, F., Wagner, D. & Zaroliagis, C. *Efficient models for
> timetable information in public transportation systems.* ACM Journal of
> Experimental Algorithmics **12**, 2.4 (2007).

`routelab.TimeExpanded`, `routelab.TimeDependent` ·
[source](../../python/routelab/kernels/timetable.py) · each checked against the
other, and against the three techniques that came after

A road network is a graph, and [Dijkstra](dijkstra.md) routes it. A timetable is
not a graph — it is a list of vehicles leaving places at times — so before you
can search one you have to decide what the nodes are. This paper gives the two
answers everything else is a reaction to.

## The two models

**Time-expanded** (§3): a node per *event*. Every departure and every arrival
becomes its own vertex; riding an edge and waiting at a stop become ordinary
edges between them.

```
for each connection (leaves p at t₁, reaches q at t₂):
    node dep(p, t₁),  node arr(q, t₂)
    edge dep(p, t₁) → arr(q, t₂)   of weight t₂ - t₁      # riding
for each stop p, consecutive events e₁ then e₂ at p:
    edge e₁ → e₂                   of weight t(e₂) - t(e₁) # waiting
```

What comes out is an ordinary static graph with non-negative weights, so
`dijkstra` routes it unchanged — and so would A*, or landmarks, or a
contraction hierarchy. That is the model's whole appeal: the timetable problem
becomes a problem you already solved.

**Time-dependent** (§4): a node per *stop*. The graph stays the size of the
network, and the clock moves into the edges.

```
node per stop; edge p → q carrying the sorted departures along it

relax(p → q, arriving at p at t):
    d ← binary search for the first departure after t
    return d.arrival                        # ∞ if none is left today
```

Relaxing an edge is no longer reading a weight but asking what leaves next,
which makes the model small and the search bespoke.

The trade is the paper's point. Time-expanded is a big graph and a stock
search; time-dependent is a small graph and a custom one.

## Hello world

Both are used exactly alike, because the model is an implementation detail of
the same question:

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> env = rl.Environment(feed)

>>> rl.TimeExpanded().bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)
>>> rl.TimeDependent().bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

Twenty minutes either way: ride the 08:00 as far as B, change onto the 08:15,
arrive at 08:20. Two different graphs, one answer — which is the property the
whole shelf is built on, and the reason both models are kept rather than the
better one only.

## What the trade costs

The models disagree about size, and `footprint` and `searches` are where that
shows:

```python
>>> expanded = rl.TimeExpanded().bind(env)
>>> dependent = rl.TimeDependent().bind(env)

>>> expanded.num_events                      # a node per departure and arrival
8
>>> expanded.searches, dependent.searches
(('events', 8), ('stops', 3))
>>> expanded.footprint > dependent.footprint
True

```

Three stops become eight event nodes on a feed with four connections. On a
city's weekday that ratio is what decides the model: a few thousand stops
against hundreds of thousands of events, so the expanded graph is built once at
bind time and its footprint is worth reading before you build one.

The same ratio shows in the work a query does. Both answer identically; the
expanded model settles more, because it is settling events rather than places:

```python
>>> a = expanded.route("A", "C", departing=time(8, 0))
>>> b = dependent.route("A", "C", departing=time(8, 0))
>>> a.cost == b.cost, (a.settled, b.settled)
(True, (5, 3))

```

## What they refuse

Neither model keeps a table of costs — each answers with a journey and nothing
else — so neither has a search space to hand over, and both say so rather than
inventing one:

```python
>>> result = None
>>> expanded.explored(result)                  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
NotImplementedError: TimeExpanded answers with a journey and keeps no search space, ...

```

The refusal names the techniques that *do* keep one, which is how the shelf is
meant to be navigated: [CSA](csa.md), [RAPTOR](raptor.md) or
[TripBased](trip-based.md).

## What is not here

Changing vehicles is instantaneous in both, which is the paper's **simple**
model. Its **realistic** one charges a minimum change time, and that is not
expressible with one label per stop: staying in your seat must not be charged,
so the cost of boarding depends on which vehicle you arrived on. The paper's
own answer is more nodes (§4.2). [RAPTOR](raptor.md)'s rounds are another,
[CSA](csa.md)'s trip flags a third, and [trip-based routing](trip-based.md)'s
labels on trips a fourth — which is a fair summary of what the next decade of
this literature was about.

## See also

- [RAPTOR](raptor.md) — the first technique here to stop building a graph at all.
- [CSA](csa.md) — one array of connections, scanned once.
- [Public transit labeling](ptl.md) — the biggest graph of all, searched once at
  bind time and never again.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
