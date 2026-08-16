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
time-dependent search over OpenStreetMap's scheduled restrictions, both of
Pyrga et al.'s timetable models over GTFS, RAPTOR, CSA, and trip-based
routing. The path from here runs on through the multimodal and multicriteria
layers above them.

## What is implemented

Each technique is a published algorithm, implemented as its paper states it,
and named after it in the API. This is the shelf; the sections below say what
each one buys and what it costs.

| Paper | Technique | Section |
|---|---|---|
| Dijkstra, *A note on two problems in connexion with graphs* (1959) | `Dijkstra()` | [Use](#use) |
| Moore, *The shortest path through a maze* (1959) — breadth-first search | `BFS()` | [Use](#use) |
| Hart, Nilsson & Raphael, *A formal basis for the heuristic determination of minimum cost paths* (1968) — A* | `AStar(Euclidean())`, `AStar(Zero())` | [Guided search](#guided-search) |
| Goldberg & Harrelson, *Computing the shortest path: A\* search meets graph theory* (2005) — ALT landmarks | `AStar(Landmarks(16))` | [Measuring instead of assuming](#measuring-instead-of-assuming) |
| Geisberger, Sanders, Schultes & Delling, *Contraction hierarchies: faster and simpler hierarchical routing in road networks* (2008) | `ContractionHierarchy(EdgeDifference())`, `ContractionHierarchy(RandomOrder())` | [Not searching the city at all](#not-searching-the-city-at-all) |
| Dreyfus, *An appraisal of some shortest-path algorithms* (1969) — time-dependent Dijkstra | `TimeDependentDijkstra()` | [When the network is not always open](#when-the-network-is-not-always-open) |
| Pyrga, Schulz, Wagner & Zaroliagis, *Efficient models for timetable information in public transportation systems* (2007) — time-expanded and time-dependent models, with foot-edges | `TimeExpanded()`, `TimeDependent()`, `Footpaths(feed, within=)` | [Timetables](#timetables-two-models-of-the-same-departures) |
| Delling, Pajor & Wagner, *Engineering time-expanded graphs for faster timetable information* (2009) | *not yet implemented* | |
| Geisberger, *Contraction of timetable networks with realistic transfers* (2010) | *not yet implemented* | |
| Bast, Carlsson, Eigenwillig, Geisberger, Harrelson, Raychev & Viger, *Fast routing in very large public transportation networks using transfer patterns* (2010) | *not yet implemented* | |
| Delling, Pajor & Werneck, *Round-based public transit routing* (2012) — RAPTOR | `RAPTOR()` | [Not building a graph at all](#not-building-a-graph-at-all) |
| Dibbelt, Pajor, Strasser & Wagner, *Intriguingly simple and fast transit routing* (2013) — CSA | `CSA()`, `CSA().bind(env).profile(...)` | [One array, scanned once](#one-array-scanned-once) |
| Witt, *Trip-based public transit routing* (2015) | `TripBased()`, `TripBased().bind(env).profile(...)` | [Trips, and the transfers between them](#trips-and-the-transfers-between-them) |
| Delling, Dibbelt, Pajor & Werneck, *Public transit labeling* (2015) | *not yet implemented* | |
| Baum, Buchhold, Sauer, Wagner & Zündorf, *UnLimited TRAnsfers for multi-modal route planning: an efficient solution* (2019) — ULTRA | *not yet implemented* | |

The rows marked *not yet implemented* are the shelf's gaps, in the order the
literature filled them: after Pyrga et al. the timetable graphs were engineered
harder, then RAPTOR and CSA — both here — stopped building a graph at all,
Witt's trip-based search — here too — stopped labelling stops, and the rest
are what came after. Every kernel that is here is checked against something that
cannot be wrong in the same direction — a pure-Python reference, a brute-force
oracle, or the paper's own second model — see [The contract](#the-contract).

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

The same shape holds for every technique, a timetable included:
`RAPTOR().bind(env).route({"stop_a": 0, "stop_b": 45}, "stop_z", departing=time(8, 30))`
stands at stop_b forty-five seconds after departing. Options are per technique
and declared as data — `Dijkstra.options` is `{"max_cost"}` — and an option a
technique does not take is refused by name, with the technique it belongs to.

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
| `RAPTOR` ← GTFS + `Footpaths` | 1,532 stops, 5 rounds | 1.1 ms |

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
explores differently reports something else: a contraction hierarchy reports two
`MeetingTrees`, RAPTOR reports `Rounds` (every stop, by the round that first
reached it), and the two Pyrga models, which keep no search space, refuse
`explored()` in so many words — but whatever it explored, you can draw it.
The identity underneath is the same for every technique that keeps a table:
`planner.route(a, b) == planner.journey(planner.search(a, targets=[...]), b)`.

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

| depart | moving | waiting | legs | scheduled |
|---|---|---|---|---|
| 06:00 | 61.6 min | 0 min | 153 | 0 |
| 07:00 | **30.0 min** | 0 min | 106 | 20 |
| 20:00 | 30.0 min | 0 min | 106 | 20 |
| 21:00 | 61.6 min | 0 min | 153 | 0 |

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
comparing them is the point — and so do RAPTOR, CSA and TripBased, below.
Twenty-five random stop pairs at 08:30 with 200 m footpaths
(`benchmarks/bench_transit.py`), all five agreeing on every one:

| technique | nodes | bind | memory | settled | query |
|---|---|---|---|---|---|
| `TimeDependent` | 6,313 stops | 0.3 s | 11 MB | 3,256 | 1.3 ms |
| `TimeExpanded` | 837,924 events | 1.1 s | 150 MB | 66,447 | 14.9 ms |
| `RAPTOR` | 6,313 stops | 0.4 s | 16 MB | 4,126 | 1.5 ms |
| `CSA` | 6,313 stops | 0.4 s | 23 MB | 3,411 | 0.6 ms |
| `TripBased` | 12,482 trips | 10.5 s | 28 MB | 8,455 | 1.0 ms |

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
before anything is built. The departures themselves are what the timetable
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

`python demos/route_transit.py kcm.zip --date 2026-08-17` runs all five and
prints the itinerary, walks included; `--walk 0` is the plain model, and the
quickest way to see what the footpaths buy.

### Not building a graph at all

Delling, Pajor & Werneck, *Round-Based Public Transit Routing* (2012). Both of
Pyrga et al.'s models make a timetable into a graph and hand it to a
shortest-path search; the priority queue and the graph are the cost. RAPTOR's
observation is that a timetable has structure a graph throws away — **routes**
(ordered stop sequences) and the **trips** along them — and that with it the
search is a few array scans and no heap. Round *k* scans, once each, every
route touched in round *k−1* and rides the earliest trip that can be caught,
so after *k* rounds every stop holds its earliest arrival with at most *k−1*
changes; footpaths are relaxed one hop from whatever a round improved.

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
env = rl.Environment(feed, rl.Footpaths(feed, within=200))

planner = rl.RAPTOR().bind(env)                          # routes and trips indexed at bind: 0.3 s, 16 MB
planner.route(downtown, juanita, departing=time(8, 30))  # Journey(... cost=4130): 09:38, 3 changes
planner.route(downtown, juanita, departing=time(8, 30), max_transfers=1)   # 16:20 — or None
planner.frontier(downtown, juanita, departing=time(8, 30))
# [Journey(1 change, arrives 16:20), Journey(2 changes, 09:41), Journey(3 changes, 09:38)]

result = planner.search(downtown, departing=time(8, 30))  # every stop, by the round that first reached it
planner.explored(result)                                  # Rounds(6,313 stops over 6 rounds)
```

Downtown Seattle to Juanita, across the lake: three changes gets there at
09:38, two at 09:41, and a rider who will change only once waits until 16:20 —
three answers a rider might reasonably want, and the graph models return one.

Two things fall out of the construction that the graph models cannot give.
RAPTOR is **one-to-all**: after the rounds every stop holds its label, so it
has a real `search()` and something to draw — `Rounds`, each stop coloured by
the round that first got there, which is the picture in the paper and what the
demo draws. And it is **Pareto** by
construction: arrival against changes, one incomparable journey per round that
improved something, which is what `frontier` hands back and `max_transfers`
cuts.

The refusals are the library's, as everywhere:

```python
planner.route(origin, target)
# ValueError: RAPTOR needs a departure time: pass departing=time(8, 30), ...
rl.TimeDependent().bind(env).route(origin, target, departing=time(8, 30), max_transfers=1)
# ValueError: TimeDependent takes no max_transfers; a cap on changes belongs to RAPTOR(), which searches by round.
rl.TimeDependent().bind(env).explored(result)
# NotImplementedError: TimeDependent answers with a journey and keeps no search space, so there is nothing to draw. The techniques that keep a table report one: ask CSA() or RAPTOR().
```

Routes here are the paper's — distinct stop sequences, split so that no trip
overtakes another — so `planner.num_routes` is larger than the feed's
`routes.txt`; on the KCM feed's Monday, 139 GTFS routes become 410. What
is not here: McRAPTOR (more criteria than changes), rRAPTOR (a range of
departure times — the question CSA's `profile`, next, answers), and a minimum
change time — a new kernel `Transfer` constructor that all five timetable
techniques would take at once, and this and TripBased are the two that could
honour it.

### One array, scanned once

Dibbelt, Pajor, Strasser & Wagner, *Intriguingly Simple and Fast Transit
Routing* (2013). RAPTOR threw away the graph and kept the routes; CSA's
observation is that even routes are more structure than the question needs. A
connection is *reachable* if you are already aboard its trip or standing at
its departure stop in time — and because a timetable is aperiodic, that can be
checked in one pass over the connections sorted by departure. So: **one
array**, a label per stop, a flag per trip, and a linear scan. A reachable
connection flags its trip and, if it improves its arrival stop, walks that
stop's footpaths; the scan starts at the first connection leaving after the
query (a binary search) and, with a target, stops at the first one leaving
after the target's label.

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
env = rl.Environment(feed, rl.Footpaths(feed, within=200))

planner = rl.CSA().bind(env)                             # 421,604 connections sorted at bind: 0.4 s, 23 MB
planner.route(downtown, juanita, departing=time(8, 30))  # Journey(... cost=4143): 09:39, 3 changes, 0.4 ms
result = planner.search(downtown, departing=time(8, 30)) # ScanSearch(6313 stops, 345498 connections scanned)
planner.explored(result)                                 # Scan(6,313 stops within 1124 min)
```

Without a target the scan runs to the end of the day and every stop holds
its earliest arrival, so like RAPTOR this technique has a real `search()` and
a space to draw: `Scan`, every stop stamped with how long after departure the
sweep reached it — the origin's neighbourhood first, the far side of the
network last, and with a target the sweep stopping the moment it passes the
target's label. Toward Juanita it labels 4,320 stops and reads 29,426 of the
day's connections; `settled` counts stops, and `result.scanned` is the
paper's own measure of work.

The same array read the other way is a **profile** (the paper's §4, pCSA):
scan latest departure first, keep at every stop the Pareto pairs of (leave
here at, arrive at the target by), and each connection learns the earliest it
can deliver a rider — by getting off, staying aboard, or changing into its
arrival stop's profile — and offers that as a pair at its departure stop and
at every stop that can walk to it. When the scan reaches the start of the
window the origin's profile is the answer: one journey per moment worth
leaving.

```python
planner.profile(downtown, juanita, departing=time(8, 30), until=time(10, 30))
# [Journey(leave 08:35, arrive 09:39), Journey(09:07 → 10:05), Journey(09:37 → 10:35), Journey(10:07 → 11:06)]   38 ms
```

A departure is the *latest* moment you can leave and still make that
arrival — the profile is a step function and these are its steps — so
`route(origin, target, departing=j.departs)` arrives when `j` does, for every
`j` in the list; the tests hold it to that at every second of a window,
against the time-dependent model. It is the question rRAPTOR would answer for
the round-based technique, and the first place here that a technique answers
something other than "the one best journey from now".

The refusals are the library's: `route(..., until=)` is refused with a
pointer at `profile()`, and `max_transfers` names RAPTOR, since a connection
scan counts nothing but time — which is also why the 08:35 journey above
changes four times where RAPTOR's 09:39 changes three: same arrival, and CSA
has no reason to prefer one over the other. What is not here: the paper's cache-friendlier
profile layouts (pseudoconnections and time indexing, pCSA-C/CT), its
multi-criteria profile (mcpCSA, arrival against trips taken), and its minimum
expected arrival time problem (§5) — each its own increment. Journey
extraction is not spelled out in the paper; the pointers here are the obvious
ones (the connection that reached each stop, the connection each trip was
entered with) and every itinerary is checked against the same
`is_valid` the other four answer to.

### Trips, and the transfers between them

Witt, *Trip-Based Public Transit Routing* (2015). RAPTOR labels stops and
scans routes; CSA labels stops and scans connections. Witt's observation is
that once a rider is aboard a trip at a known stop, everything that can
happen next is already written in the timetable — every stop the trip
reaches, and every other trip they could change onto — so it is **trips**
that should carry labels, and the changes between them can be worked out
once, ahead of any query. Binding computes that **transfer set**: for every
trip and every stop it reaches, the earliest trip of every line at that stop
or a footpath away, minus the ones that stay on your own line and the
U-turns; then a *reduction* walks each trip backwards keeping the earliest it
can reach every stop with and without each transfer, and drops the transfers
that improve nothing. A query is then a breadth-first sweep with no priority
queue: round *n* scans every *trip segment* reached with *n* changes, checks
whether the trip reaches the target, prunes it if it cannot beat the best
arrival so far, and follows its transfers into round *n+1*.

```python
feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
env = rl.Environment(feed, rl.Footpaths(feed, within=200))

planner = rl.TripBased().bind(env)                       # 35,213,140 transfers computed, 1,216,322 kept: 11 s, 28 MB
planner.route(downtown, juanita, departing=time(8, 30))  # Journey(... cost=4116): 09:38, 2 changes, 0.9 ms
planner.frontier(downtown, juanita, departing=time(8, 30))
# [Journey(1 change, arrives 16:30), Journey(2 changes, 09:38)]     — RAPTOR's front, exactly

result = planner.search(downtown, targets=[planner.node_id(juanita)], departing=time(8, 30))
result.settled, result.scanned                           # 9,807 trips labelled, 2,533 segments scanned
planner.explored(result)                                 # Segments(2,533 trip segments over 4 rounds)
```

Two things are different in kind here, and both show in the numbers. Binding
is **the expensive half** — seconds where the others take a fraction of one,
which is why it reports progress as it works, the way contraction and a
landmark table do, and why the transfer set is the thing worth measuring:
KCM's 12,482 trips make 35 million transfers and reduction keeps 3.5% of
them (the paper keeps 16% of London's). Almost all of that time is the
reduction, which is why both algorithms are parallel over trips — a thread
per core drawing blocks until there are none. `TripBased(reduce=False)` is the
paper's own control (its Table 3): identical answers, 300 MB instead of 28,
and 8.9 ms a query instead of 0.9. Reduction is a policy and never a
correctness choice, on the same footing as `RandomOrder()` for a hierarchy —
the tests hold the reduced and unreduced sets to the same front on every
query. And `settled` counts **trips**, since that is what this kernel labels
(the paper's own footnote on how to compare it): 9,807 of 12,482 sounds like
most of the network until you see that only 2,533 segments were scanned. The
rest were labelled to say *don't bother* — reaching a trip at a stop settles
every later trip of its line at once, which is the trick, and the segments
the sweep actually read are what `Segments` draws: each vehicle from the stop
it was boarded at, coloured by the changes it took to get aboard.

The same sweep, run once per moment the origin offers a departure in a
window — latest first, keeping the labels between runs, since whatever an
earlier departure would find on a trip a later one already reached is
dominated — is the paper's **profile** (§3.3):

```python
planner.profile(downtown, juanita, departing=time(8, 30), until=time(10, 30))
# [Journey(leave 08:35, arrive 09:38, 2 changes), Journey(09:07 → 10:32, 3), Journey(09:37 → 10:38, 2), ...]   107 ms
```

Nine journeys where CSA's profile of the same window holds four, because
this one is Pareto over three criteria — departure, arrival *and* changes —
and a journey that leaves later, or changes less, for the same arrival is a
different answer. A departure is the latest moment you can leave and still
make that journey, walk included, and a window holds a journey by when it
leaves: the tests hold every triple to RAPTOR's front asked at every second
of a window, and every itinerary to `is_valid`.

The refusals are the library's. `search()` without a target is refused the
way A\* and a hierarchy refuse it — the paper's query is point-to-point, its
target is what the lines are checked against and what prunes the rest, and
there is no one-to-all form to fall back on. `max_transfers` names RAPTOR and
TripBased; `until` names `profile()` on CSA and TripBased. What is not here:
the paper's SIMD and three-loop query layout (§3.4) and its transfer
preferences — each its own increment. Preprocessing is parallel over trips,
which is the one place the paper's "trivially parallelized" is taken up: both
algorithms judge each trip on its own, so a thread per core does. Changing vehicles is instantaneous, and this is the
second kernel — a transfer knows both trips — that a minimum change time could
land in.

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
a one-line change rather than a porting project. Every technique declares the
query options it takes and refuses the rest by name, so the substitution is
never silent.

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

The same three words divide both halves. **Kernels** are the papers, one entry
each. **Model** is the vocabulary they all speak — a type earns a place there
when more than one technique reads it, which is the whole test. **Util** is
plumbing with no routing content.

```
crates/routelab-core/     Rust. No Python.
  kernels/                Dijkstra, BFS, A*, ALT landmarks, contraction,
                          time-dependent, Pyrga's two timetable models, RAPTOR, CSA,
                          trip-based.
  model/                  CSR graph, search options and results, search trees,
                          the heuristic trait, the timetable structures, and the
                          lines-and-trips layout RAPTOR and trip-based both read.
                          Names nothing above it: the dependency runs one way.
  util/                   Progress counters, seeded RNG.
crates/routelab-osm/      Reading OpenStreetMap extracts. Kernel-free; wraps `osmpbf`.
crates/routelab-gtfs/     Reading GTFS feeds. Kernel-free; wraps `gtfs-structures`.
crates/routelab-py/       PyO3 bindings, one module per wrapped subsystem.
                          Conversion and GIL release, nothing else.
python/routelab/          The veneer: constructors, argument sugar, docstrings.
  kernels/                One module per paper, holding both roads to it and the
                          spec only it reads — an ordering, a calendar, a heuristic.
  model/                  Graph, Environment, Journey, results, search spaces.
  data/                   The OSM, GTFS and footpath layers.
  util/                   Clocks and argument coercion.
  reference.py            Pure-Python twins of the static kernels — the oracle.
tests/                    Mirrors the veneer. The differential suite sits at the
                          top, because it belongs to no single technique.
demos/                    Runnable examples, and the node board behind `serve.py`.
benchmarks/               What each technique costs, on a real city.
```

Kernel work goes in `routelab-core`, where it is usable from Rust and testable
without Python. The bindings layer stays thin on purpose: sugar is easier to read,
change, and document in Python.

A new technique is a new file in both `kernels/` directories and nothing else
moved: it reuses the model, declares the query options it takes, brings its own
derivation of whatever it needs beyond the graph, and is checked against
something that cannot be wrong in the same direction.

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
its own derivation, and the environment need not know what it is for. The
corollary for knobs: a bound on one question is a query option, a property of
the technique is a constructor argument — `max_transfers` is the former,
`waiting` the latter.

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
