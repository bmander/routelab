# A*

> Hart, P. E., Nilsson, N. J. & Raphael, B. *A formal basis for the heuristic
> determination of minimum cost paths.* IEEE Transactions on Systems Science
> and Cybernetics **4**(2), 100–107 (1968).

`routelab.AStar` · [source](../../python/routelab/kernels/astar.py) ·
returns what [Dijkstra](dijkstra.md) returns, and is tested against it

[Dijkstra](dijkstra.md) settles nodes in order of what they cost to reach,
which means it spreads in every direction equally — including away from where
you are going. A* orders the queue by cost-so-far *plus an estimate of the cost
still to go*, so the search leans toward the destination.

## The algorithm

Dijkstra's, with one line changed: the queue is keyed on `dist[u] + h(u)`
rather than `dist[u]`.

```
dist[v] ← ∞                       for every node v
dist[o] ← its head start          for every origin o
queue  ← the origins, keyed by dist + h

while queue:
    u ← pop lowest dist[u] + h(u)
    if u = destination:  stop
    for each edge u → v of cost w:
        if dist[u] + w < dist[v]:
            dist[v]   ← dist[u] + w
            parent[v] ← the edge u → v
            push v at dist[v] + h(v)
```

`h(v)` estimates what remains from `v` to the destination. The paper's result
is that if `h` never *overestimates* — if it is **admissible** — then A* still
returns a cheapest path, and the closer `h` is to the truth the fewer nodes it
settles. `h = 0` is admissible and gives Dijkstra back exactly.

Two consequences worth stating plainly. The costs recorded are real costs, not
the estimates the queue was sorted by. And A* is goal-directed by nature: an
estimate is an estimate *to somewhere*, so a query needs exactly one
destination.

## Hello world

A heuristic has to come from somewhere, and here it comes from the layers. This
network is a corridor with a spur running the wrong way, and every node has a
position:

```python
>>> import routelab as rl

>>> corridor = [(f"m{i}", f"m{i + 1}", 100) for i in range(5)]
>>> spur = [("m0", "s1", 100)] + [(f"s{i}", f"s{i + 1}", 100) for i in range(1, 4)]
>>> streets = rl.ScalarEdges(
...     *corridor, *spur, bidirectional=True, cost_per_distance=1.0
... )
>>> places = rl.Positions(
...     {f"m{i}": (i * 100.0, 0.0) for i in range(6)}
...     | {f"s{i}": (-i * 100.0, 0.0) for i in range(1, 5)}
... )
>>> env = rl.Environment(streets, places)

>>> rl.AStar(rl.Euclidean()).bind(env).route("m0", "m5").routes[0]
Journey('m0' → 'm1' → 'm2' → 'm3' → 'm4' → 'm5', cost=500)

```

`Euclidean()` is the straight-line bound: distance to the destination, priced
at the fastest rate any layer charges per metre. `cost_per_distance=1.0` on the
layer is what declares that rate — a hundred seconds per hundred metres.

## What the guidance buys

The same answer, from fewer settled nodes. That difference is the whole point
of the paper, and `result.settled` is how you measure it:

```python
>>> def settled(technique, aim):
...     planner = technique.bind(env)
...     result = planner.search("m0", **{aim: planner.node_id("m5")})
...     return result.settled, planner.journey(result, "m5").cost

>>> settled(rl.Dijkstra(), "targets")     # a set of targets to stop at
(10, 500)
>>> settled(rl.AStar(rl.Euclidean()), "target")   # the one it aims at
(6, 500)

```

Six nodes rather than ten, and the same 500-second journey. The four A* never
looked at are the spur: it heads away from `m5`, so the estimate makes every
node on it look expensive long before the search would have reached the end of
it. Dijkstra has no way to know that and walks the whole thing.

## Zero, asked for out loud

`Zero()` is the admissible heuristic that estimates nothing, which turns A*
back into Dijkstra. It is the control every guided-search measurement needs.

```python
>>> settled(rl.AStar(rl.Zero()), "target")
(10, 500)

```

The heuristic is a required argument, and this is why. An A* whose heuristic
quietly fell back to zero is Dijkstra wearing A*'s name — the one thing a
benchmark must never fail to notice — so the degenerate case has to be asked
for by name.

## What it refuses

`Euclidean` needs coordinates for every node and a rate to price them at. An
environment that supplies neither is refused before anything is built, and the
refusal names what is missing rather than guessing:

```python
>>> plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
>>> sorted(rl.AStar(rl.Euclidean()).missing_from(plain.compile()))
['cost_per_distance', 'positions']

```

`missing_from` is how a study skips the datasets a technique cannot run on —
ask before binding, and no exception is needed. Binding anyway raises, and
names the nodes nothing placed.

A* needs exactly one destination, since there is no single thing an estimate
could be a bound on otherwise — so `target=` is required, and there is no
spelling of the call that hands it two:

```python
>>> rl.AStar(rl.Euclidean()).bind(env).search("m0")    # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...search() missing 1 required keyword-only argument: 'target'

```

## On a real map

Same three steps, a bigger layer — and the honest caveat about straight-line
bounds:

```python
env = rl.Environment(rl.OSM("seattle.osm.pbf", rl.Driving()))
planner = rl.AStar(rl.Euclidean()).bind(env)
```

A profile's top speed becomes the layer's rate, so the bound must assume
*everything* moves at motorway speed. The wider a network's range of speeds,
the weaker the bound — guidance helps least exactly where the network is most
varied. Same code, same heuristic, Liechtenstein:

| profile | speeds | short route | long route |
|---|---|---:|---:|
| Walking | 0.5–1.4 m/s | 15% of Dijkstra's nodes | 46% |
| Driving | 5–31 m/s | 56% | 98% |

That is the argument for measuring the network instead of assuming things about
it, which is what [landmarks](landmarks.md) do.

## See also

- [Landmarks](landmarks.md) — a heuristic measured through the network rather
  than assumed from its geometry.
- [Dijkstra](dijkstra.md) — the same search with `h = 0`.
- [Contraction hierarchies](contraction-hierarchies.md) — not guiding the
  search but removing most of it.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
