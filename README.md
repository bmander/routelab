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
implementations, checked against a reference implementation that is obviously
correct.

**Status: early.** Today it has a static graph and the two searches everything
else builds on. The path from here runs through time-dependent and schedule-based
search: RAPTOR, CSA, then the multimodal and multicriteria layers above them.

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

planner = rl.Dijkstra(env)          # preprocess once
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

journey = rl.Dijkstra(env).route("home", "stop_b")
[(leg.head, leg.source) for leg in journey.legs]
# [('stop_a', ScalarEdges(2 edges)), ('stop_b', ScalarEdges(1 edges))]
```

Queries take the arguments the problem actually needs — several origins, each
already carrying the cost of an access walk, and a bound on how far to look:

```python
rl.Dijkstra(env).route({"stop_a": 0, "stop_b": 45}, "home", max_cost=600)
```

The planner is the unit of comparison: swap `rl.Dijkstra` for `rl.BFS` and the
rest of the line is unchanged. `rl.route(rl.Dijkstra, env, "a", "c")` does the
whole thing in one call when you only have one question to ask — but building a
planner and keeping it is what makes preprocessing worth doing, which is the
whole game for the algorithms this project is heading toward.

### Guided search

A* takes a heuristic — an estimate of the cost still to go — and settles fewer
nodes for the same answer. Layers supply what the estimate is built from:

```python
env = rl.Environment(
    rl.ScalarEdges(streets, cost_per_distance=0.71),   # walking, ~1.4 m/s
    rl.ScalarEdges(transit, cost_per_distance=0.04),   # fastest mode, ~25 m/s
    rl.Positions({"home": (0, 0), "stop_a": (300, 40)}),
)

rl.AStar(env, rl.Euclidean()).route("home", "stop_b")
```

`cost_per_distance` is the least a layer can charge to cover one unit of ground.
The environment takes the **minimum** across layers, because a path may ride the
fastest one the whole way — add a train to a walking network and a bound priced
at walking speed starts overestimating, which turns an admissible heuristic into
one that quietly returns paths that are not the cheapest. A layer that declares
no rate disables the heuristic rather than being assumed slow; a heuristic that
cannot be built says so instead of falling back:

```python
rl.AStar(env, rl.Euclidean())
# ValueError: Euclidean needs a position for every node; 2 have none ...
```

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
planner = rl.AStar(env, rl.Euclidean())
planner.route(planner.environment.sources[0].nearest(47.226, 9.523), ...)
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

That is the argument for landmark heuristics, which do not price distance at all.

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
```

The extract is read once at startup and the planners are built once, so a click
costs a query and not a load — Seattle is 258,029 nodes and 590,671 edges from a
65 MB extract, read in about five seconds, and a cross-town route answers in
four milliseconds. Switching the algorithm dropdown re-runs the same query the
other way: 19,530 nodes settled under A*, 44,557 under Dijkstra, for the same
12.9-minute answer.

Country extracts come from [Geofabrik](https://download.geofabrik.de), city ones
from [BBBike](https://download.bbbike.org/osm/bbbike/). Neither is committed —
`data/` is ignored.

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

**A reference implementation for every kernel.** Each kernel has a twin in
[`routelab.reference`](python/routelab/reference.py): the same algorithm written
the obvious way in pure Python. It is slow and legible, and it is what the fast
one is checked against — over random graphs, on every result field, not just
distances. "Returns the right answer" is a falsifiable claim in this domain, which
is what makes contributing a new kernel tractable: write it, diff it against the
reference, and the diff is the review.

**Results are checkable, not merely reported.** `Graph.walk` follows a returned
edge path and reports where it lands and what it cost. A path is only correct if
walking it arrives at the target at the reported cost, and the tests check exactly
that rather than trusting the search's own bookkeeping.

## Layout

```
crates/routelab-core/   Rust: CSR graph, Dijkstra, BFS. No Python.
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

**Deterministic tie-breaking.** Nodes settle in order of `(cost, node)`, and
out-edges relax in CSR order. Equal-cost paths are common in transit networks, and
without a rule for them two correct implementations disagree constantly and
usefully diffing them becomes impossible.

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
