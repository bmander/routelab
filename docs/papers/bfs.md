# Breadth-first search

> Moore, E. F. *The shortest path through a maze.* Proceedings of an
> International Symposium on the Theory of Switching, 285–292 (1959).

`routelab.BFS` · [source](../../python/routelab/kernels/bfs.py) ·
checked against the same oracle [Dijkstra](dijkstra.md) is

Moore's paper asks a different question from Dijkstra's, published the same
year: not what a path *costs* but how many steps it takes. When every edge is
worth the same, that is the same question — and the priority queue collapses
into a plain FIFO.

## The algorithm

```
depth[v] ← ∞           for every node v
depth[o] ← 0           for every origin o
queue   ← the origins             # FIFO, not a heap

while queue:
    u ← pop front                 # settle u: depth[u] is now final
    if every destination settled:  stop
    if depth[u] = max_depth:       stop expanding u
    for each edge u → v:
        if depth[v] = ∞:
            depth[v] ← depth[u] + 1
            parent[v] ← the edge u → v
            push v at the back
```

A queue settles nodes in nondecreasing depth for the same reason a heap settles
them in nondecreasing cost, but for free: every edge adds exactly one, so
arriving later can never mean arriving shallower. `O(n + m)`, with no
comparisons at all.

That is also the constraint. Every origin starts at depth 0, because a FIFO is
only correct when the frontier enters at a single depth — so unlike Dijkstra,
origins here cannot carry a head start.

## Hello world

```python
>>> import routelab as rl

>>> streets = rl.ScalarEdges(
...     ("home", "a", 300),
...     ("a", "b", 60),
...     ("b", "work", 240),
...     ("home", "work", 900),
... )
>>> env = rl.Environment(streets)

>>> rl.BFS().bind(env).route("home", "work")
Journey('home' → 'work', cost=1)

```

One hop, and `cost` is that hop count. [Dijkstra](dijkstra.md) on the same
network answers `Journey('home' → 'a' → 'b' → 'work', cost=600)` — three
edges totalling ten minutes, against one edge totalling fifteen. Neither is
wrong; they are answers to different questions, which is the reason both are on
the shelf.

A journey's legs still carry their real weights, so a hop-counted answer can
still be priced:

```python
>>> journey = rl.BFS().bind(env).route("home", "work")
>>> journey.cost, [leg.weight for leg in journey.legs]
(1, [900])

```

## Bounding the depth

`max_depth` is BFS's only query option: expand no further than this many hops.
On a one-to-all search that is "everything within three changes", which is the
shape a reachability question takes.

```python
>>> planner = rl.BFS().bind(env)
>>> planner.search("home").settled
4
>>> planner.search("home", max_depth=1).settled
3

```

Two hops from home reaches everything; one hop reaches `a` and `work` and stops.

## What it refuses

An origin cannot carry a head start, because a hop count has nowhere to put
one:

```python
>>> planner.route({"home": 0, "b": 120}, "work")
Traceback (most recent call last):
    ...
ValueError: BFS counts hops, so origins cannot carry an initial cost: 'b'

```

That is a refusal rather than a rounding, and it is the same shape every
technique here uses: say what cannot be honoured, and name the thing that
cannot honour it.

## See also

- [Dijkstra](dijkstra.md) — the same search when edges have costs, and the
  control every technique here is measured against.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
