# ALT landmarks

> Goldberg, A. V. & Harrelson, C. *Computing the shortest path: A\* search meets
> graph theory.* SODA 2005, 156–165.

`routelab.Landmarks` · [source](../../python/routelab/kernels/heuristics.py) ·
a heuristic for [`AStar`](astar.md), tested against [Dijkstra](dijkstra.md)

A straight-line heuristic *assumes*: that the ground is flat, that coordinates
mean something, and that nothing crosses a metre faster than the fastest thing
in the network. This paper measures instead. Pick a handful of nodes, compute
the distance to and from each one ahead of time, and the triangle inequality
turns those measurements into a bound — one that already knows about one-way
streets, dead ends, rivers, and the fact that most roads are not motorways.

## The algorithm

Preprocessing, once per network:

```
choose k landmarks L ⊂ V
for each ℓ in L:
    to[ℓ]   ← dijkstra from ℓ on the graph            # dist(ℓ, ·)
    from[ℓ] ← dijkstra from ℓ on the reversed graph   # dist(·, ℓ)
```

Then the bound, for a query toward `t`:

```
h(v) = max over ℓ in L of:
           from[ℓ][v] - from[ℓ][t]      # ℓ behind you:  dist(v,ℓ) - dist(t,ℓ)
           to[ℓ][t]   - to[ℓ][v]        # ℓ beyond you:  dist(ℓ,t) - dist(ℓ,v)
```

Each line is the triangle inequality rearranged, so each is a lower bound on
`dist(v, t)`, and the largest of them is the sharpest bound available. It is
admissible by construction, which is what lets [A*](astar.md) use it and still
return cheapest paths. A landmark informs when it lies roughly *beyond* one end
of the trip; one sitting off to the side contributes nothing, which is why they
are pushed to the edges of the network.

Cost: two full searches per landmark at bind time, and two integers per
landmark per node held for as long as the planner lives. Sixteen landmarks over
a 250,000-node city is a few seconds and about 32 MB.

## Hello world

```python
>>> import routelab as rl

>>> corridor = [(f"m{i}", f"m{i + 1}", 100) for i in range(5)]
>>> spur = [("m0", "s1", 100)] + [(f"s{i}", f"s{i + 1}", 100) for i in range(1, 4)]
>>> env = rl.Environment(rl.ScalarEdges(*corridor, *spur, bidirectional=True))

>>> rl.AStar(rl.Landmarks(2)).bind(env).route("m0", "m5").routes[0]
Journey('m0' → 'm1' → 'm2' → 'm3' → 'm4' → 'm5', cost=500)

```

Note what this environment does *not* have: no coordinates, and no declared
rate per metre. `Euclidean` cannot run here at all, and says so by name:

```python
>>> compiled = env.compile()
>>> sorted(rl.AStar(rl.Euclidean()).missing_from(compiled))
['cost_per_distance', 'positions']
>>> sorted(rl.AStar(rl.Landmarks(2)).missing_from(compiled))
[]

```

That is the second thing the paper buys, and on some networks the more useful
one: a landmark bound asks the environment for nothing but its edges. Networks
whose geometry is unknown, distorted, or beside the point — a transfer graph, a
schedule, an abstract instance — can still be searched with guidance.

## The dial, and its control

`count` trades memory for sharpness with the usual diminishing returns.
`selection` is a policy: `"farthest"` spreads landmarks to the edges of the
network, where they inform, and `"random"` is the control that shows the
spreading is doing real work.

```python
>>> rl.Landmarks(16)
Landmarks(16, selection='farthest')
>>> rl.Landmarks(4, selection="random", seed=7)
Landmarks(4, selection='random')

```

A landmark set that cannot beat random is not earning its memory. On Seattle's
driving network, spreading beats scattering by about 2× for the same memory,
which is why `"farthest"` is the default and `"random"` is kept rather than
deleted.

Both are refused where they are written rather than at bind time, because a
technique is a value you can put in a list and compare, and a value that is
wrong should say so when it is made:

```python
>>> rl.Landmarks(0)
Traceback (most recent call last):
    ...
ValueError: a landmark heuristic needs at least one landmark, got 0

```

## On a real map

Seattle's driving network, 258,029 nodes, corner to corner — the same
11.2-minute route from every row:

| | settled | of graph | query | preprocessing |
|---|---:|---:|---:|---|
| [Dijkstra](dijkstra.md) | 41,162 | 16.0% | 5.0 ms | — |
| [A* (euclidean)](astar.md) | 12,173 | 4.7% | 2.7 ms | — |
| A* (16 landmarks) | 1,749 | 0.7% | 0.7 ms | 1.0 s, 33 MB |

```python
env = rl.Environment(rl.OSM("seattle.osm.pbf", rl.Driving()))
planner = rl.AStar(rl.Landmarks(16)).bind(env)     # a second, and 33 MB
```

The count is the dial: 2 landmarks (4 MB) already beat Euclidean on this
network, and 32 (66 MB) settle 443 nodes. The reason the gap is this wide is
the one [A*](astar.md) ends on — a straight-line bound priced at motorway speed
is roughly three times too optimistic on every street that is not a motorway,
and a measured bound has no such problem.

## See also

- [A*](astar.md) — the search this is a heuristic for, and the straight-line
  bound it is arguing with.
- [Contraction hierarchies](contraction-hierarchies.md) — more preprocessing
  again, and a query that stops searching the city altogether.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
