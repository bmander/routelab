# routelab

Reference implementations of routing algorithms — Python veneer, Rust kernels.

Routing research has a transfer problem. On one side are papers with bespoke C++
harnesses, each with its own binary formats and console apps, benchmarked on
whichever network the authors had handy. On the other are production engines
(OTP2, MOTIS, Valhalla) that are excellent at serving traffic and poor at
answering "what does this new algorithm actually buy?" There is no lobby between
the two — no place where two implementations of the same problem can be swapped,
run on the same instance, and compared.

routelab is that lobby. It is not a production routing engine and does not want
to be. It is a place to implement published algorithms honestly — fast enough
that their constant factors mean something, behind an API shared across
implementations, checked against something independent that is obviously
correct.

**Status: early.** Today it has a static graph, the searches everything else
builds on (Dijkstra, BFS, A*), two heuristics, contraction hierarchies,
time-dependent search over OpenStreetMap's scheduled restrictions, and both of
Pyrga et al.'s timetable models over GTFS. The path from here runs on through
schedule-based search — RAPTOR, CSA — and then the multimodal and multicriteria
layers above them.

## Install

Requires Python 3.9+ and a [Rust toolchain](https://rustup.rs).

```bash
git clone https://github.com/bmander/routelab && cd routelab
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # builds the Rust kernel via maturin
```

## Use

Describe a world, then ask an algorithm to route over it.

```python
import routelab as rl

env = rl.Environment()
env.register(rl.ScalarEdges(("a", "b", 1), ("b", "c", 15)))

technique = rl.Dijkstra()           # a configuration, costing nothing
planner = technique.bind(env)       # preprocessing, if the technique has any
planner.route("a", "c")             # Journey('a' → 'b' → 'c', cost=16)
```

An environment is assembled from layers, and nodes are labelled by whatever you
already call them — stop ids, OSM node ids, `("bike", 42)`. Each leg of a journey
remembers which layer it came from, which is what makes a multimodal answer
readable rather than just a number:

```python
streets = rl.ScalarEdges(("home", "stop_a", 300), bidirectional=True)
transit = rl.ScalarEdges(("stop_a", "stop_b", 120))
env = rl.Environment(streets, transit)

journey = rl.Dijkstra().bind(env).route("home", "stop_b")
[(leg.head, leg.source) for leg in journey.legs]
# [('stop_a', ScalarEdges(2 edges)), ('stop_b', ScalarEdges(1 edges))]
```

Queries take the arguments the problem actually needs — several origins, each
already carrying the cost of an access walk, and a bound on how far to look:

```python
rl.Dijkstra().bind(env).route({"stop_a": 0, "stop_b": 45}, "home", max_cost=600)
```

Configuring and binding are separate because the middle step gets expensive:
sixteen landmarks over a city is a second and 33 MB, and that belongs to a verb
rather than to a constructor. It also makes a technique a *value* — something you
can name, put in a list, and point at more than one dataset:

```python
study = {
    "dijkstra": rl.Dijkstra(),
    "astar": rl.AStar(rl.Euclidean()),
    "alt-16": rl.AStar(rl.Landmarks(16)),
}
for name, technique in study.items():
    if technique.missing_from(env.compile()):
        continue                                  # this dataset cannot support it
    planner = technique.bind(env)
```

There is no registry of names in the library: a dictionary of techniques is a
line you write, and no fixed set could serve both a demo's dropdown and a
parameter sweep. `rl.route(rl.Dijkstra(), env, "a", "c")` does the whole thing in
one call when you have exactly one question to ask.

### Guided search

A* takes a heuristic — an estimate of the cost still to go — and settles fewer
nodes for the same answer. Layers supply what the estimate is built from:

```python
env = rl.Environment(
    rl.ScalarEdges(streets, cost_per_distance=0.71),   # walking, ~1.4 m/s
    rl.ScalarEdges(transit, cost_per_distance=0.04),   # fastest mode, ~25 m/s
    rl.Positions({"home": (0, 0), "stop_a": (300, 40)}),
)

rl.AStar(rl.Euclidean()).bind(env).route("home", "stop_b")
```

`cost_per_distance` is the least a layer can charge to cover one unit of ground.
`Euclidean` takes the **minimum** across layers, because a path may ride the
fastest one the whole way — add a train to a walking network and a bound priced
at walking speed starts overestimating, which turns an admissible heuristic into
one that quietly returns paths that are not the cheapest. A layer that declares
no rate disables the heuristic rather than being assumed slow; a heuristic that
cannot be built says so instead of falling back:

```python
rl.AStar(rl.Euclidean()).bind(env)
# ValueError: Euclidean needs a position for every node; 2 have none ...
```

The environment does not keep a coordinates table or a rate. `Euclidean` derives
both when it binds — `rl.Plane` and `rl.Pace`, two small specifications with the
same `missing_from`/`bind` shape as a heuristic — and refuses if the layers
cannot supply them. That is what keeps the environment a merge of labels, edges
and provenance rather than a place every technique's needs accumulate.

There is no default heuristic. A* whose guidance silently became zero is Dijkstra
wearing its name — the one failure a benchmark cannot see — so `rl.Zero()` has to
be asked for out loud.

Whether guidance pays depends on how tight the bound is, which is a thing to
measure rather than assume. On a 200×200 grid, corner to corner
(`benchmarks/bench_astar.py`):

| | settled | of graph | ms |
|---|---:|---:|---:|
| Dijkstra | 40,000 | 100% | 3.3 |
| A* (zero) | 40,000 | 100% | 3.3 |
| A* (euclidean), 8-connected | 794 | 2% | 0.3 |
| A* (euclidean), 4-connected | 40,000 | 100% | 3.9 |

Same code, same heuristic, same answers. The last row moves in L1 while the
estimate measures L2, so the bound is loose by up to √2 everywhere and the
guidance buys exactly nothing.

### Real maps

An OpenStreetMap extract is a layer like any other:

```python
env = rl.Environment().register(rl.OSM("liechtenstein.osm.pbf", rl.Driving()))
planner = rl.AStar(rl.Euclidean()).bind(env)
planner.route(env.sources[0].nearest(47.226, 9.523), ...)
```

Nodes keep their OSM ids, so a route traces back to the map it came from, and
each leg can produce the shape of the street it followed — `compiled.geometry(leg.edge)`
gives the polyline, not just the endpoints.

A `Profile` decides who is travelling, because the same file is three different
networks: which `highway` classes count, how fast each goes, whether `oneway`
binds. `Walking()`, `Cycling()`, and `Driving()` ship as defaults, and any of them
can be adjusted — `Driving().but(respect_oneway=False)`.

One consequence worth knowing, because it is counter-intuitive and the demo
measures it. A profile's top speed becomes the layer's `cost_per_distance`, so a
straight-line bound must assume *everything* moves at motorway speed. The wider a
network's range of speeds, the weaker that bound — guidance helps least exactly
where the network is most varied. Same code, same heuristic, Liechtenstein:

| profile | speeds | short route | long route |
|---|---|---:|---:|
| Walking | 0.5–1.4 m/s | 15% of Dijkstra's nodes | 46% |
| Driving | 5–31 m/s | 56% | 98% |

That is the argument for the other heuristic in the box.

### Measuring instead of assuming

`Euclidean` assumes one thing about a network and must assume the worst of it:
priced at the fastest layer, it is three times too optimistic on every street
that is not a motorway. `Landmarks` measures instead — precomputing the distance
to and from a handful of fixed nodes, and bounding by the triangle inequality:

```python
planner = rl.AStar(rl.Landmarks(16)).bind(env)   # a few seconds, ~32 MB
```

The bound it produces already knows about one-way streets, dead ends, bridges,
and speed limits, because it was measured through them. It also asks nothing of
the environment — no coordinates, no declared speeds — which makes it the
heuristic for networks whose geometry is unknown or beside the point.

Seattle, driving, corner to corner (258,029 nodes):

| | settled | of graph | query | preprocessing |
|---|---:|---:|---:|---|
| Dijkstra | 41,162 | 16.0% | 5.0 ms | — |
| A* (euclidean) | 12,173 | 4.7% | 2.7 ms | — |
| A* (16 landmarks) | 1,749 | 0.7% | 0.7 ms | 1.0 s, 33 MB |

Same 11.2-minute route from all three. The landmark count is the dial: 2
landmarks (4 MB) already beat Euclidean, 32 (66 MB) settle 443 nodes. Spreading
them to the edges of the network beats scattering them at random by about 2× for
the same memory, which is why `selection="farthest"` is the default and
`"random"` is kept as the control.

```bash
python demos/route_city.py liechtenstein.osm.pbf \
    --from 47.2260,9.5230 --to 47.1300,9.5210 --profile walking
```

writes a map of the route with every settled node behind it — blue where both
searches went, orange for the 8,330 nodes A* never had to look at.

For poking at it by hand, `demos/serve.py` puts the same thing behind a local
page: click an origin, click a destination, and the route comes back with the
search drawn underneath it.

```bash
python demos/serve.py data/Seattle.osm.pbf     # then open http://localhost:8000
python demos/serve.py data/Seattle.osm.pbf --gtfs data/kcm.zip --date 2026-08-17
```

Under the map is a **node board**, and it is these three steps drawn as a graph:
layers feed an `Environment`, a technique binds to one and becomes a planner, a
`Query` asks it something. Drag the nodes, pull the wires out, plug them in
somewhere else. There is no second representation of what the controls mean —
the board *is* the query, so unplugging a wire really does remove an argument
and the page says which one.

The map is a node in it, and the graph runs out through the map and back: it
gives the query an `origin` and a `destination` and takes a `route` and a
`space` in return. Cross the two points and the trip reverses. Unplug `space`
and no search space gets built — ten megabytes of GeoJSON that nothing was
listening for. Unplug `route` and the query still runs and still reports what it
cost, with nothing drawing it.

A node whose configuration the board has never had an answer for greys out and
spins until it does, and because a node's identity includes everything upstream
of it, the spinners spread exactly as far as the rebuild does and no further. A
change that costs nothing shows nothing. Where the work can count itself the
node shows how far along: contraction reports nodes retired, then arcs
assembled; a landmark table reports searches run. Where it cannot — a file
parser that yields no counts — it says so rather than inventing a bar.

Watch that bar on a big network and it will teach you something the timings
table hides. Contracting Seattle's walking graph settles 99.5% of its 554,393
nodes in the first minute and spends two more on the rest: the last nodes left
are the most connected, and they are where the shortcut search does its real
work.

Which is what makes the refusals worth causing on purpose. Wire a GTFS layer
into `Dijkstra` and the node turns red with the library's own sentence:

```
Dijkstra cannot route over timetable layers; it accepts scalar
```

A dropdown in the board's toolbar loads a starting point — road with landmarks,
road with a contraction hierarchy, plain Dijkstra as the control, walking that
reads the clock, and either timetable model when the demo was given a feed. They
are places to begin rather than modes: load one and then pull it apart. The pins
stay put across a change, which is the point of having more than one.

Rewiring is cheap because everything is cached by a canonical spelling of the
node and everything upstream of it, so going back to a shape you had before is
free. That matters because the expensive things are exactly the ones worth
comparing. Seattle is 258,029 nodes and 590,671 edges from a 65 MB extract, read
in about five seconds; swapping the technique node re-runs the same query the
other way:

| wired up as | settled | query |
|---|---:|---:|
| `Dijkstra` | 16,250 | 3.3 ms |
| `AStar` ← `Euclidean` | 5,941 | 1.5 ms |
| `AStar` ← `Landmarks(16)` | 2,496 | 0.9 ms |
| `ContractionHierarchy` ← `EdgeDifference` | 221 | 0.6 ms |
| `TimeDependent` ← GTFS | 217 stops | 1.7 ms |
| `TimeExpanded` ← GTFS | 893 events | 5.0 ms |

Same two pins throughout; the first four return the same route.

Country extracts come from [Geofabrik](https://download.geofabrik.de), city ones
from [BBBike](https://download.bbbike.org/osm/bbbike/). Neither is committed —
`data/` is ignored.

### Looking at the search

The route is the answer; the search is the work, and most of what separates one
algorithm from another lives there. Every planner can hand over what it explored:

```python
result = planner.search(origin, targets=[planner.node_id(destination)])
tree = planner.explored(result)          # ShortestPathTree(19529 branches, ...)
tree.geojson()                           # drop into QGIS, geojson.io, Leaflet
```

Dijkstra and A* explore by growing a shortest-path tree, so that is what they
report. Each branch carries the total of everything hanging off it, which is what
makes a hundred thousand identical lines into a picture: magnitudes accumulate
toward the root, so the tree renders like a river network — thick where the whole
search flowed, thinning to capillaries where it stopped.

```python
for branch in tree.branches(min_magnitude=1000):
    branch.tail, branch.head, branch.magnitude
```

`magnitude="weight"` accumulates travel time beyond each branch; `"nodes"` counts
settled nodes instead. `SearchSpace` is the general promise — an algorithm that
explores differently reports something else, and a schedule-based search will
report a decision graph rather than a tree — but whatever it explored, you can
draw it.

The interactive demo draws exactly this, which is the quickest way to see what a
heuristic buys: the same 11.2-minute Seattle route settles 41,161 branches under
Dijkstra, reaching across Lake Washington to Bellevue, and 12,172 under A*, which
never leaves the corridor.

### Not searching the city at all

Everything above narrows the search. A contraction hierarchy (Geisberger,
Sanders, Schultes and Delling, *Exact Routing in Large Road Networks Using
Contraction Hierarchies*) stops doing one. Preprocessing removes nodes one at a
time, least important first, adding a **shortcut** wherever removing a node
would otherwise have lengthened a shortest path. What comes out is the original
graph plus shortcuts and a rank per node — and a query that climbs from the
source, climbs from the target, and meets above the trip:

```python
planner = rl.ContractionHierarchy().bind(env)    # ~6 s on Seattle, 19 MB
planner.route(origin, destination)               # a fifth of a millisecond
```

Seattle, driving, 25 random trips (`benchmarks/bench_contraction.py`):

| | preprocessing | memory | settled | query |
|---|---|---:|---:|---:|
| Dijkstra | — | — | 129,992 | 16.920 ms |
| A* (euclidean) | — | 4 MB | 65,523 | 12.751 ms |
| A* (16 landmarks) | 1.3 s | 33 MB | 10,509 | 2.769 ms |
| Contraction hierarchy | 6.0 s | 19 MB | **251** | **0.152 ms** |

Identical costs on every trip — the benchmark fails if any row disagrees. Not
necessarily the identical *path*: where two routes tie, a search climbing from
both ends cannot honour the `(cost, node)` tie-break a one-directional search
does, so it may return a different equally-cheap way. The claim is exact
distances, and that is what the tests hold it to.
Preprocessing adds 504,022 shortcuts to Seattle's 590,671 edges, an 85% larger
graph in exchange for touching 0.1% of it per query.

This is the first technique here that searches a graph the environment has never
seen, which is the constraint worth stating out loud: **a technique may search
whatever graph it likes, but it answers in the caller's terms.** Every shortcut
is unpacked back into the original edges it stands for before anything leaves
the planner, so journeys, legs, geometry and provenance work exactly as they do
under Dijkstra.

It is also the first search that is not one tree, so `explored()` reports a
different `SearchSpace`:

```python
space = planner.explored(result)         # MeetingTrees(..., longest span=289)
for direction, tail, head, level, edges in space.branches(min_span=50):
    ...                                  # 'forward' or 'backward', and what it leapt
```

Branches are drawn with their unpacked geometry, so every line is real road
rather than a straight cut through the buildings a shortcut jumped over. The
hierarchy shows up in two properties instead: `span`, how many original edges a
branch stands for, and `level`, the rank of its higher end. The longest branch
of a cross-town Seattle query stands for a couple of hundred edges: that is the
query crossing the city in one step, which is the whole trick. The demo colours
the two halves differently, which makes the shape
plain — two small fans climbing away from either end of the trip, joined by a
few enormous arcs, and nothing in between.

Ordering is a policy and never a correctness choice:

```python
rl.ContractionHierarchy(rl.EdgeDifference())      # the paper's recipe (default)
rl.ContractionHierarchy(rl.RandomOrder(seed=0))   # the control
```

Both answer identical distances; the random one just builds a far bigger
hierarchy to do it. The same holds for the witness-search limits — a witness
search that gives up early adds a shortcut that was not needed, costing space
and query time and never accuracy, which is why `max_settled` and `max_hops` are
tuning knobs rather than correctness ones.

### When the network is not always open

OpenStreetMap records *when* a way may be used, not only whether. The walkway
across Seattle's Hiram M. Chittenden Locks is tagged
`foot:conditional=yes @ (07:00-21:00)`; the I-5 express lanes are
`oneway=reversible` with hours. `Dijkstra` ignores every one of those and routes
the always-open network, honestly and knowingly. `TimeDependentDijkstra` reads
the clock — Dreyfus, *An Appraisal of Some Shortest-Path Algorithms* (1969),
whose observation is that Dijkstra generalises unchanged provided arrival is
non-decreasing in departure.

```python
env = rl.Environment(rl.OSM("seattle.osm.pbf", rl.Walking()))
planner = rl.TimeDependentDijkstra().bind(env)
planner.route(ballard, magnolia, departing=time(8, 0))    # 30.0 min
planner.route(ballard, magnolia, departing=time(22, 0))   # 61.6 min
```

Binding is where the calendar is built: `rl.Schedule` walks the layers' opening
hours into one kernel calendar, and the planner keeps it — `planner.calendar`
— rather than the environment. Bind to a network with no schedule and it says
so, in the same breath a heuristic would.

`python demos/route_by_clock.py seattle.osm.pbf` prints the same trip at every
hour. 354 of Seattle's 1,480,122 walking edges carry a schedule — a rounding
error that changes the answer completely for the trips that meet one:

| depart | moving | legs | scheduled edges used |
|---|---|---|---|
| 06:00 | 61.6 min | 153 | 0 |
| 07:00 | **30.0 min** | 106 | 20 |
| 20:00 | 30.0 min | 106 | 20 |
| 21:00 | 61.6 min | 153 | 0 |

The gate opens at seven and shuts at nine, and the alternative is the long way
around the ship canal. Waiting is a policy rather than an assumption:
`TimeDependentDijkstra("forbidden")` treats a shut edge as absent, which is the
control that shows waiting doing real work.

Two things are stated rather than discovered. The clock is **weekly** — seconds
since Monday 00:00 — because every restriction OSM can express repeats weekly;
a schedule naming a date or a holiday is refused at parse time and counted in
`unreadable_schedules` rather than approximated. And a conditional tag is read as
the whole story for its edge: `yes @ (hours)` means shut outside them, `no @
(hours)` means open outside them, and the base `access` tag does not get a vote.

### Timetables: two models of the same departures

Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable Information
in Public Transportation Systems* (ACM JEA 12, Article 2.4, 2007) is not a paper
about an algorithm. A timetable is not a network with weights on it — it is a
set of **connections**, each one a vehicle leaving one stop at one instant and
reaching another at another — and the paper is about the two ways to make that
into something a shortest-path algorithm can read.

```python
env = rl.Environment(rl.GTFS("kcm.zip", date(2026, 8, 17)))
journey = rl.TimeDependent().bind(env).route(origin, target, departing=time(8, 30))
journey.arrives, journey.transfers, journey.waiting   # 33600, 4, 1200
```

Both models answer with the same verb, `route(..., departing=)`, because
comparing them is the point:

| model | nodes | bind | memory | query | settled | arrives |
|---|---|---|---|---|---|---|
| Time-dependent | 6,313 stops | — | — | 0.1 ms | 187 | 09:20 |
| Time-expanded | 837,924 events | 0.2 s | 37 MB | 2.5 ms | 740 | 09:20 |

King County Metro plus Sound Transit on a Monday: 6,313 stops, 421,604
connections, 7,038 stop pairs, 0 trips the reader could not represent. The
time-expanded model builds a node per departure and per arrival, and what comes
out is an **ordinary static graph** — `dijkstra` routes it unchanged, and so
would A*, or landmarks, or a contraction hierarchy. That generality is the whole
appeal, and 133× the nodes is what it costs. The time-dependent model keeps one
node per stop and pays in the search instead: relaxing an edge is not reading a
weight but a binary search for what leaves next.

Neither is the reference implementation. **Each is the other's** — they must
agree on every query, which is the paper's thesis and this module's main test,
checked over random timetables, against a brute-force oracle, and on the real
feed.

The cost model is what makes the seam load-bearing rather than decorative:

```python
rl.Dijkstra().bind(env)
# TypeError: Dijkstra cannot route over timetable layers; it accepts scalar
```

A GTFS layer contributes one edge per pair of adjacent stops weighted by the
*shortest ride anyone makes along it* — a genuine lower bound, enough to give
every stop a label and to snap a coordinate to one, and not enough to route on.
Routing it as though those weights told the whole story is a wrong answer that
looks like a right one, so `Dijkstra` refuses it and `missing_from` says so
before anything is built. The departures themselves are what the two timetable
techniques derive from the layer at bind — `rl.Departures`, kept as
`planner.timetable` — which is the pattern every clock-reading technique here
follows, and the one the next one should.

#### Crossing the street

A feed says where its stops are and what leaves them. It rarely says that the
northbound stop and the southbound one across the street are, for a rider, the
same place — King County Metro's has no `transfers.txt` and no parent stations
at all — and a timetable technique that cannot cross the street cannot make
most real trips. Downtown to the U District at 08:30 came back at 09:20 with
four changes because every change had to happen at one pole; 39 of 200 random
stop pairs had no answer at all.

The paper's *foot-edges* are the fix: a walk between two stops that a rider may
take at any time, for a fixed duration. Here they are a layer, and an ordinary
`"scalar"` one:

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
env = rl.Environment(feed, rl.Footpaths(feed, within=200))     # 13,908 walks
journey = rl.TimeDependent().bind(env).route(origin, target, departing=time(8, 30))
journey.arrives, journey.transfers                          # 09:04, 1
[leg for leg in journey.legs if leg.trip is None]           # the walk across 3rd Ave
```

A timetable technique always accepted a scalar layer beside its timetable; now
it reads one, as the links a rider may walk — `rl.Walks`, derived at bind like
the departures are. Nothing about the environment changed to carry them, and a
walk leg has provenance and geometry through the same machinery as every other
leg. How far a rider will walk is a modelling choice, which is why it is a
knob on a layer you register and not something the environment does for you.
Two hundred metres joins the two sides of a street and the bays of a transit
centre: 197 of the same 200 pairs now route, every one of them using a walk,
and the models still agree on all of them.

Two things worth knowing about how they are carried. The kernel **closes the
set under composition** — walk A→B and B→C and you may walk A→C — because the
time-dependent search chains walks on its own and the time-expanded graph must
not, or a pair of opposite footpaths would spawn events for ever; closing the
set is what lets one model take a walk one hop at a time and the other in one
and still agree. Seattle's 13,908 walks close to 74,636. And in the
time-expanded graph a walk is not a new event but an edge from the arrival at
one stop to the first departure you could catch at the other, which keeps the
event count where it was; the walk edges themselves are 5.2 million, and the
model's memory goes from 37 MB to 136 MB. That is the paper's trade again, and
worth having a number for.

Two limits worth stating. Changing vehicles is **instantaneous**, which is the
paper's *simple* model; its *realistic* one charges a minimum change time and is
not expressible with one label per stop, because staying in your seat must not be
charged and a stop label cannot say which vehicle you arrived on. The paper's
answer is more nodes (§4.2), and that is its own increment — so `Transfer` offers
only `Transfer::instant()` rather than a parameter that would be quietly ignored.
And a timetable is on a **linear service day** — GTFS writes `25:30:00`, so
`service_seconds` does not wrap the way `weekly_seconds` does. The two clocks are
deliberately not interchangeable, which means walking and transit cannot yet
share one environment. That is the multimodal problem, and the conversion needs a
service-day-to-calendar anchor: exactly the thing that must not be implicit.

`python demos/route_transit.py kcm.zip --date 2026-08-17` runs both models and
prints the itinerary, walks included; `--walk 0` is the plain model, and the
quickest way to see what the footpaths buy.

### Underneath

The kernels are also callable directly, on dense integer ids, as the papers
state them. This is the layer to implement or benchmark an algorithm against:

```python
# (tail, head, weight) triples; weights are non-negative ints, conventionally seconds.
graph = rl.Graph.from_edges([(0, 1, 60), (1, 3, 120), (0, 2, 90), (2, 3, 30)])

result = rl.dijkstra(graph, 0)
result.cost(3)        # 120
result.path(3)        # [0, 2, 3]
result.edge_path(3)   # edge ids, for getting back to your own per-edge data

rl.dijkstra(graph, {0: 0, 2: 45})     # many sources, each with an initial cost
rl.dijkstra(graph, 0, targets=[3])    # stop as soon as these are settled
rl.dijkstra(graph, 0, max_cost=90)    # or bound the search: an isochrone
rl.bfs(graph, 0, max_depth=2)         # hop counts, ignoring weights

# A* wants a compiled heuristic, which is what Heuristic.bind produces — bound
# to the same environment the graph came from, since it is indexed by node id.
compiled = env.compile()
rl.astar(compiled.graph, 0, compiled.node_id("stop_b"), rl.Euclidean().bind(compiled))
```

Every search records the order it settled nodes in (`result.order`), which is how
you compare algorithms by work done rather than by wall-clock alone.

## The contract

Three commitments hold the project together, and every algorithm added later has
to keep them.

**One API per problem.** A search takes a graph, sources, and bounds, and returns
a result you can ask for costs, paths, and the order nodes were settled in. Two
algorithms solving the same problem are drop-in substitutes, so comparing them is
a one-line change rather than a porting project.

**Something independent to check every kernel against.** The static searches
have a twin in [`routelab.reference`](python/routelab/reference.py): the same
algorithm written the obvious way in pure Python, slow and legible, diffed over
random graphs on every result field rather than on distances alone. Where a
second implementation would only be a second copy of the same reasoning, the
check is something else independent — a brute-force oracle over tiny instances,
or two models of one problem that must agree. The timetable kernels have both,
and neither of the two is the reference: each is the other's. "Returns the right
answer" stays a falsifiable claim either way, which is what makes contributing a
new kernel tractable — write it, diff it against something that cannot be wrong
in the same direction, and the diff is the review.

**Results are checkable, not merely reported.** `Graph.walk` follows a returned
edge path and reports where it lands and what it cost. A path is only correct if
walking it arrives at the target at the reported cost, and the tests check exactly
that rather than trusting the search's own bookkeeping.

## Layout

```
crates/routelab-core/   Rust: CSR graph, searches, heuristics, contraction, timetables. No Python.
crates/routelab-osm/    Reading OpenStreetMap extracts. Kernel-free; wraps `osmpbf`.
crates/routelab-gtfs/   Reading GTFS feeds. Kernel-free; wraps `gtfs-structures`.
crates/routelab-py/     PyO3 bindings. Conversion and GIL release, nothing else.
python/routelab/        The veneer: constructors, argument sugar, reference implementations.
tests/                  Behaviour tests and differential tests against the reference.
```

Kernel work goes in `routelab-core`, where it is usable from Rust and testable
without Python. The bindings layer stays thin on purpose: sugar is easier to read,
change, and document in Python.

## Design notes

**Integer weights.** Costs are `u32`, conventionally seconds. The schedule-based
literature is integer-valued throughout, and float comparison is a poor foundation
for the Pareto dominance checks the multicriteria algorithms are built on.

**CSR, immutable.** A graph is built once from an edge list and never mutated,
which is what lets searches run with the GIL released. Edges are permuted into CSR
order, so edge ids are not positions in your input list — `Graph.input_index`
maps back, which is how per-edge attributes (mode, trip, route) stay attached.
Preprocessing that needs to *grow* a graph — contraction inserting shortcuts —
builds its own mutable adjacency and hands back finished CSR graphs, rather than
making every search pay for an edge list that might change underneath it.

**Deterministic tie-breaking.** Nodes settle in order of `(cost, node)`, and
out-edges relax in CSR order. Equal-cost paths are common in transit networks, and
without a rule for them two correct implementations disagree constantly and
usefully diffing them becomes impossible. Every search is deterministic; not all
of them agree with each other, because a rule about settle order says nothing
across algorithms that do not settle in one order — a bidirectional search picks
its own equally-cheap winner among ties.

**The environment is a merge, not a bag.** Compiling layers produces exactly
three things: a numbering of labels, one graph, and which layer each edge came
from. A calendar, a timetable, a coordinates table, a rate — anything a
technique reads beyond the graph — is derived by that technique at bind time
from the compiled layers (`rl.Schedule`, `rl.Departures`, `rl.Plane`,
`rl.Pace`), and refused there if the layers cannot supply it. The rule that
falls out is worth stating: a thing is an *argument* — a constructor parameter,
a wire on the demo's board — if and only if it is a choice. A heuristic is a
choice; a calendar assembled from the layers has one possible construction and
no knobs, so it is derived rather than passed. This is what lets the next
schedule-based algorithm arrive without touching `environment.py`: it brings
its own derivation, and the environment need not know what it is for.

**Multi-source with initial costs.** The one-to-all search takes
`(node, initial_cost)` pairs rather than a single origin. Every multimodal
algorithm needs this — the transit search starts from a set of stops each already
reached at some cost — so it belongs in the primitive rather than in a wrapper.

## Development

```bash
cargo test                             # Rust: core kernels and bindings
maturin develop && pytest              # Python: veneer and differential tests
cargo fmt && cargo clippy --all-targets
```

## License

MIT.
