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
Answer(Journey('home' → 'a' → 'b' → 'work', cost=600), SearchResult(num_nodes=4, settled=4))

```

A query answers with three things, and the journey is one of them:

```python
>>> answer = planner.route("home", "work")
>>> answer.routes                      # every journey worth having, best first
[Journey('home' → 'a' → 'b' → 'work', cost=600)]
>>> answer.searchspace()               # what the search looked at
ShortestPathTree(3 branches, magnitude='weight', peak=600)
>>> answer.raw.settled                 # the kernel's own table
4

```

`routes` is a list because a technique that tells journeys apart by more than
one criterion has more than one answer, and one that does not has a Pareto set
of exactly one — a front of one rather than a lesser kind of answer. Nothing
above searched twice: the space and the table were read off the search the
routes came from.

Edges are directed, and nodes take whatever names you already use — strings
here, transit stop ids or OpenStreetMap node ids elsewhere. Ten minutes the
long way round beats fifteen straight down the arterial, and the answer carries
its parts:

```python
>>> journey = planner.route("home", "work").routes[0]
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
).routes[0]
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
>>> planner.route({"home": 0, "b": 120}, "work").routes[0]
Journey('b' → 'work', cost=360)

```

`max_cost` stops the search once the cheapest node left exceeds the bound. On
a one-to-all search that gives an isochrone; on a point-to-point one it says
"don't bother if it is further than this".

```python
>>> planner.route("home", "work", max_cost=500).routes
[]

```

Each planner's `route` declares exactly the options its own algorithm
understands, so an option it does not take is refused by name — by Python
itself, and by a type checker before that:

```python
>>> planner.route("home", "work", max_transfers=1).routes[0]      # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...route() got an unexpected keyword argument 'max_transfers'

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

`searchspace()` is a method rather than an attribute for two reasons: it takes
options — `magnitude="nodes"` counts settled nodes where the default
accumulates travel time — and building it is real work over everything
settled, which a caller who only wanted a route should not pay for.

Every technique answers `route` the same way. The ones that keep no table hold
their routes and nothing else: `raw` is `None`, and `searchspace()` says so and
names the techniques that do keep one.

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
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
