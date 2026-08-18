# Dijkstra's algorithm

> Dijkstra, E. W. *A note on two problems in connexion with graphs.*
> Numerische Mathematik **1**, 269–271 (1959).

`routelab.Dijkstra` · [source](../../python/routelab/kernels/dijkstra.py) ·
checked against a pure-Python reference, and both against Bellman–Ford

## The algorithm

The paper's second problem: "find the path of minimum total length between two
given nodes".

```
dist[v] ← ∞                       for every node v
dist[o] ← its head start          for every origin o
queue  ← the origins

while queue:
    u ← pop cheapest              # settle u: dist[u] is now final
    if every destination settled:  stop
    if dist[u] > bound:            stop
    for each edge u → v of cost w:
        if dist[u] + w < dist[v]:
            dist[v]   ← dist[u] + w
            parent[v] ← the edge u → v
            push v at dist[v]
```

A settled node is final only because no edge costs less than nothing: every
path still under construction already costs at least `dist[u]` before it
reaches `u`. That is why costs here are non-negative integers.

`parent` is a shortest-path tree rooted at the origins, and a journey is read
off it backwards from the destination. The loop runs to exhaustion unless one
of those two `stop`s fires: the search is one-to-all by nature, and a
destination or a bound only cuts it short. `O((n + m) log n)` per query, and
nothing to precompute.

## Hello world

Three steps: describe a network, bind a technique to it, ask a question.

```python
>>> import routelab as rl

>>> streets = rl.ScalarEdges(          # each tuple: from, to, cost in seconds
...     ("home", "a", 300),
...     ("a", "b", 60),
...     ("b", "work", 240),
...     ("home", "work", 900),
... )
>>> env = rl.Environment(streets)

>>> planner = rl.Dijkstra().bind(env)
>>> planner.route("home", "work")
Journey('home' → 'a' → 'b' → 'work', cost=600)

```

Edges are directed, and nodes take whatever names you already use — strings
here, transit stop ids or OpenStreetMap node ids elsewhere. Ten minutes the
long way round beats fifteen straight down the arterial, and the answer carries
its parts:

```python
>>> journey = planner.route("home", "work")
>>> journey.cost, journey.nodes
(600, ['home', 'a', 'b', 'work'])

```

Those three steps are the whole API. An `Environment` holds the network — a
list of edges here, a GTFS feed or an OpenStreetMap extract elsewhere.
`rl.Dijkstra()` is a free-to-make configuration; `bind` does any
precomputation; `route` asks one question.

## A real map

An OpenStreetMap extract is a layer like the hand-written edges above, so
nothing about the query changes — only what it runs on. Nodes keep their OSM
ids, which is how a route traces back to the map it came from.

```python
streets = rl.OSM("seattle.osm.pbf", rl.Driving())
env = rl.Environment(streets)

planner = rl.Dijkstra().bind(env)
journey = planner.route(
    streets.nearest(47.6062, -122.3321),      # snap a coordinate to a node id
    streets.nearest(47.6740, -122.1215),
)
journey.cost                                  # seconds of driving
journey.geometry                              # the whole route, as (lat, lon)
```

Each leg carries the shape of the street it followed — `leg.geometry` is the
polyline, not just the two endpoints — and a journey stitches them into one.

That extract is not in this repository — `data/` is ignored, and everyone
fetches their own from [Geofabrik](https://download.geofabrik.de) or
[BBBike](https://download.bbbike.org/osm/bbbike/). Which is also why this block
carries no `>>>` prompts: blocks that have them are run as tests on every commit,
and this one could not be. `rl.OSM` checks the path the moment you name it, so
running this as written raises `FileNotFoundError` rather than failing later.

The profile decides who is travelling, because one file is three networks:
`Walking()`, `Cycling()` and `Driving()` disagree about which ways count, how
fast each goes, and whether `oneway` binds.

## Asking different questions

A query may have several origins, each carrying the cost already spent getting
there. This is ordinary multi-source Dijkstra — only the initial queue differs
— and it is the shape a multimodal query needs: every transit stop within
walking distance is an origin, with its walk as the head start.

```python
>>> planner.route({"home": 0, "b": 120}, "work")
Journey('b' → 'work', cost=360)

```

`max_cost` stops the search once the cheapest node left exceeds the bound. On
a one-to-all search that gives an isochrone; on a point-to-point one it says
"don't bother if it is further than this".

```python
>>> print(planner.route("home", "work", max_cost=500))
None

```

A technique refuses options it does not take, and names the technique they
belong to:

```python
>>> planner.route("home", "work", max_transfers=1)      # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: Dijkstra takes no max_transfers; a cap on changes belongs to RAPTOR(), ...

```

## Advanced use

### Looking at the search

The route is the answer, but the search is the work, and the work separates
techniques — so every planner can hand over what it explored.

```python
>>> result = planner.search("home")             # no destination: one-to-all
>>> result.settled
4
>>> tree = planner.explored(result)
>>> tree
ShortestPathTree(3 branches, magnitude='weight', peak=600)

```

Dijkstra grows and returns a shortest-path tree. Each branch carries the total
of everything hanging off it, so a city-sized search draws as a river network
rather than a hundred thousand identical lines.

```python
for branch in tree.branches(min_magnitude=1000):
    branch.tail, branch.head, branch.magnitude

tree.geojson()          # drop into QGIS, geojson.io, Leaflet
```

### The kernel directly

Underneath the Python sits a Rust kernel on dense integer ids, which is what
every search above actually ran on. Reach for it directly to implement or
benchmark an algorithm, not to route over a real network.

```python
>>> graph = rl.Graph.from_edges([(0, 1, 60), (1, 3, 120), (0, 2, 90), (2, 3, 30)])
>>> result = rl.dijkstra(graph, 0)
>>> result.cost(3), result.path(3)
(120, [0, 2, 3])

```


## See also

- [A*](astar.md) — the same search, guided by an estimate
  of the cost still to go.
- [Contraction hierarchies](contraction-hierarchies.md) — the
  same answer without searching the city.
- `rl.BFS()` — Moore's breadth-first search: the same question by hop count,
  when every edge costs the same.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
