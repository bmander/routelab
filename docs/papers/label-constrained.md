# Label-constrained routing

> Barrett, C., Jacob, R. & Marathe, M. *Formal-language-constrained path
> problems.* SIAM Journal on Computing **30**(3), 809–837 (2000).
>
> In the multimodal form of Dibbelt, J., Pajor, T. & Wagner, D.
> *User-constrained multi-modal route planning.* ALENEX 2012 §2.2.

`routelab.LabelConstrained`, `routelab.Modes` ·
[source](../../python/routelab/kernels/lcspp.py) · checked against every other
timetable technique here

"Get me there, but I will not cycle on a motorway." "Bus or train, no walking
between them." A multimodal query is not only about cost — it is about which
*sequences* of transport modes are acceptable. Barrett, Jacob and Marathe's
answer is to label every arc with the mode it represents and let the traveller
supply a formal language over those labels. A path is admissible when the word
it spells is in the language.

## The algorithm

Label the arcs, write the language as an automaton, and search the product:

```
each arc carries a mode symbol:  foot, transit, link, …
the language is an automaton A over those symbols

search the product graph V × states(A):
    a state is (vertex v, automaton state s)
    relaxing arc v → w with symbol σ moves to (w, A.step(s, σ))
      — and is forbidden outright if A has no such step
    origins start in A's initial states; a journey may end only in a final one
```

The paper's result is that for a *regular* language this costs no more than
searching the product graph, which is `|states|` times the network — and that
regular is enough for the modal constraints anyone actually writes. Nothing is
precomputed: the automaton is a query input, so changing your mind about
cycling on motorways costs nothing.

`Modes` builds the shape Dibbelt, Pajor & Wagner's §2.2 uses, and no more than
that shape: a state stands for one or more modes, travelling within a mode is a
self-loop, and distinct states are joined **only** by the link label. So a
journey may change mode exactly where two networks were stitched together, and
nowhere else.

## Hello world

Left alone, `Modes` reads the environment and builds the automaton that fits
it:

```python
>>> import routelab as rl
>>> from datetime import date, time

>>> feed = rl.GTFS(TINY_GTFS, date(2026, 9, 7))       # a Monday
>>> pavement = rl.ScalarEdges(
...     ("A", "B", 900), ("B", "A", 900), ("B", "C", 900), ("C", "B", 900)
... )
>>> env = rl.Environment(feed, pavement)

>>> planner = rl.LabelConstrained().bind(env)         # nothing precomputed
>>> planner.route("A", "C", departing=time(8, 0))
Journey('A' → 'B' → 'C', cost=1200)

```

Twenty minutes by bus, against thirty on foot. Binding builds no index, no
transfer set and no hierarchy — it compiles the language and stops, which is
what makes this the baseline the other two multimodal techniques are measured
against.

## The language decides

Write the automaton yourself and the same query answers differently, because it
is a different question:

```python
>>> on_foot = rl.Modes(states={"foot": ["foot"]}, start=["foot"], end=["foot"])
>>> aboard = rl.Modes(states={"aboard": ["transit"]}, start=["aboard"], end=["aboard"])

>>> walked = rl.LabelConstrained(on_foot).bind(env).route("A", "C", departing=time(8, 0))
>>> ridden = rl.LabelConstrained(aboard).bind(env).route("A", "C", departing=time(8, 0))
>>> (walked.cost, walked.walking), (ridden.cost, ridden.walking)
((1800, 1800), (1200, 0))

```

Thirty minutes entirely on foot, or twenty entirely aboard. The cost is not the
only thing that changed — the *kind* of journey did, which is the whole point of
constraining the language rather than reweighting the graph.

A language must say where a journey may begin and end, and refuses rather than
guessing:

```python
>>> rl.Modes(states={"foot": ["foot"]})
Modes({'foot': ['foot']}, link='link')
>>> rl.LabelConstrained(rl.Modes(states={"foot": ["foot"]})).bind(env)  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: a language needs somewhere to begin and end: pass start= and end= ...

```

## Where the modes come from

An arc's mode comes from the layer that emitted it. A timetable layer is
ridden; everything else is walked, unless it says otherwise — and
`routelab.Access`, the layer that joins a feed to a street network, says
`"link"`, because joining two networks is exactly what the paper's link arcs
are.

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
streets = rl.OSM("seattle.osm.pbf", rl.Walking())
env = rl.Environment(feed, streets, rl.Access(feed, streets))

rl.LabelConstrained().bind(env)      # reads Figure 1(a): foot ⇄ link ⇄ aboard
```

With a link layer present, `Modes()` derives the paper's Figure 1(a): a walking
state and a riding state, joined both ways by the link. Without one — a feed
and its footpaths, no streets — the timetable's stops *are* the walking
network's nodes, so one state stands for both modes and a rider changes
vehicles standing still, which is what every other timetable technique here
assumes. That is the plain model rather than a refusal, and it is the case the
hello world above is in.

## What it costs

Nothing at bind, and the whole network at query time. That is the trade, and it
is the reason the other two multimodal techniques exist:

| | preprocessing | query |
|---|---|---|
| `LabelConstrained` | none | searches the whole network |
| [`UCCH`](ucch.md) | minutes | searches a core of ~2% of it |
| [`ULTRA`](ultra.md) | minutes | milliseconds, over precomputed transfers |

## See also

- [UCCH](ucch.md) — this search with the walking contracted first, and the
  language still a query input.
- [ULTRA](ultra.md) — the other end: precompute the transfers, then run a stock
  timetable technique.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
