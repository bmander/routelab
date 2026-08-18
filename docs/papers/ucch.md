# UCCH

> Dibbelt, J., Pajor, T. & Wagner, D. *User-constrained multi-modal route
> planning.* ALENEX 2012 §3.

`routelab.UCCH` · [source](../../python/routelab/kernels/lcspp.py) ·
answers exactly as [LabelConstrained](label-constrained.md), and is tested
against it

[LabelConstrained](label-constrained.md) searches the whole network on every
query, and most of that network is pavement. A [contraction
hierarchy](contraction-hierarchies.md) is the standard cure for too much
pavement — but contracting a *multimodal* network naively breaks the whole
point of the language, and this paper is about how to do it anyway.

## The problem with contracting

Contract a merged network and a shortcut can span two modes. Such a shortcut
carries a modal transfer *inside* it, invisible to the query — so a query
forbidding that transfer cannot use the shortcut, and the path that avoided the
transfer may already have been discarded when the shortcut replaced it. Bake
the language into the preprocessing and it stops being a query input, which is
the one thing label-constrained routing is for.

## The algorithm

Contract each mode's subnetwork alone, and never contract a vertex where the
networks join:

```
core ← every vertex where two modes meet          # the link endpoints: stops
for each mode m separately:
    contract m's own subnetwork, least important first,
    stopping when the core averages more than max_degree arcs a vertex
    — never contracting a vertex in the core

query:
    climb out of the origin through its own mode's hierarchy
    climb out of the destination the same way
    run the label-constrained search on the core, with the automaton intact
```

No shortcut ever crosses a mode boundary, because no contracted vertex was ever
on one. So the automaton stays a query input, and the search that runs on the
core is the same product-graph search — just on two percent of the vertices.

## Hello world

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> pavement = rl.ScalarEdges(
...     ("A", "B", 900), ("B", "A", 900), ("B", "C", 900), ("C", "B", 900)
... )
>>> env = rl.Environment(feed, pavement)

>>> rl.UCCH().bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

The same twenty minutes [LabelConstrained](label-constrained.md) returns, which
is the claim: a speedup, not a different answer.

## The language is still an input

That is the whole reason this exists rather than an ordinary hierarchy, so it
is worth demonstrating that a contracted network still honours a language it
was not built for:

```python
>>> on_foot = rl.Modes(states={"foot": ["foot"]}, start=["foot"], end=["foot"])
>>> rl.UCCH(on_foot).bind(env).route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1800)

```

Thirty minutes on foot, from a planner whose preprocessing knew nothing about
this constraint. Bake the language in and this query would have been unanswerable
or wrong.

Where nothing joins two networks — a feed and its footpaths, with no street
layer, which is the environment above — there is no core distinct from the
network, so this is `LabelConstrained` with a hierarchy built for nothing. That
is the plain model rather than a refusal, and the answers are identical.

## What it costs

The honest measurement, on King County Metro and Seattle's pavements: about
**three and a half times faster** than
[LabelConstrained](label-constrained.md), for a few minutes of contraction.

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
streets = rl.OSM("seattle.osm.pbf", rl.Walking())
env = rl.Environment(feed, streets, rl.Access(feed, streets))

planner = rl.UCCH().bind(env)        # a few minutes: contracting the pavement
```

Most of what is left after that is the transit search *inside* the core, which
UCCH does not touch — the paper says as much — so 3.5× is close to its ceiling
here rather than a disappointing result. `max_degree` is the dial: stop
contracting once the core averages more than this many arcs per vertex. Lower
leaves more standing and contracts faster.

The three multimodal techniques are three corners of one trade:

| | preprocessing | query |
|---|---|---|
| [`LabelConstrained`](label-constrained.md) | none | the whole network |
| `UCCH` | minutes | a core of ~2% of it |
| [`ULTRA`](ultra.md) | minutes | milliseconds, over precomputed transfers |

## See also

- [LabelConstrained](label-constrained.md) — the search this speeds up, and
  where the language and the modes are explained.
- [Contraction hierarchies](contraction-hierarchies.md) — the same contraction
  idea on a network with only one mode in it.
- [ULTRA](ultra.md) — the third corner.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
