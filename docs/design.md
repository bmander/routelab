# How it is built

What routelab is for, what every kernel here is checked against, where the code
lives, and the handful of decisions the whole shelf rests on.

## Why this exists

Routing research has a transfer problem. On one side sit papers with bespoke
C++ harnesses, each with its own binary formats and console apps, benchmarked
on whichever network the authors had handy. On the other sit production engines
(OTP2, MOTIS, Valhalla), excellent at serving traffic and poor at answering
"what does this new algorithm actually buy?" There is no lobby between the two
— no place to swap two implementations of the same problem, run them on the
same instance, and compare.

routelab is that lobby. It is not a production routing engine and does not want
to be. It is a place to implement published algorithms honestly — fast enough
that their constant factors mean something, behind an API shared across
implementations, checked against something independent and obviously correct.

**Status: early.** The shelf is on [the index](index.md); the papers it has not
reached yet are listed in the [README](../README.md). The path from here runs
through the multicriteria layers above what is implemented.

## The contract

Three commitments hold the project together, and every algorithm added later
must keep them.

**One API per problem.** A search takes a graph, sources, and bounds, and returns
a result you can ask for costs, paths, and the order nodes were settled in. Two
algorithms solving the same problem are drop-in substitutes, so comparing them is
a one-line change rather than a porting project. Every technique declares the
query options it takes and refuses the rest by name, so the substitution is
never silent.

**Something independent to check every kernel against.** The static searches
have a twin in [`routelab.reference`](../python/routelab/reference.py): the same
algorithm written the obvious way in pure Python, slow and legible, diffed over
random graphs on every result field rather than on distances alone. Where a
second implementation would only be a second copy of the same reasoning, the
check is something else independent — a brute-force oracle over tiny instances,
or two models of one problem that must agree. The timetable kernels have both,
and neither is the reference: each is the other's. "Returns the right answer"
stays a falsifiable claim either way, which makes contributing a new kernel
tractable — write it, diff it against something that cannot be wrong in the
same direction, and the diff is the review.

**Results are checkable, not merely reported.** `Graph.walk` follows a returned
edge path and reports where it lands and what it cost. A path is correct only
if walking it arrives at the target at the reported cost, and the tests check
exactly that rather than trusting the search's own bookkeeping.

## Layout

The same three words divide both halves. **Kernels** are the papers, one entry
each. **Model** is the vocabulary they all speak — a type earns a place there
when more than one technique reads it; that is the whole test. **Util** is
plumbing with no routing content.

```
crates/routelab-core/     Rust. No Python.
  kernels/                Dijkstra, BFS, A*, ALT landmarks, contraction and its
                          core variant, time-dependent, Pyrga's two timetable
                          models, RAPTOR, CSA, trip-based, PTL, ULTRA.
  model/                  CSR graph, search options and results, search trees,
                          the heuristic trait, the timetable structures, and the
                          lines-and-trips layout RAPTOR and trip-based both read.
                          Names nothing above it: the dependency runs one way.
  util/                   Progress counters, seeded RNG.
crates/routelab-osm/      Reading OpenStreetMap extracts. Kernel-free; wraps
                          `osmpbf`, and `rstar` for the index a snap reads.
crates/routelab-gtfs/     Reading GTFS feeds. Kernel-free; wraps `gtfs-structures`.
crates/routelab-py/       PyO3 bindings, one module per wrapped subsystem.
                          Conversion and GIL release, nothing else.
python/routelab/          The veneer: constructors, argument sugar, docstrings.
  kernels/                One module per paper, holding both roads to it and the
                          spec only it reads — an ordering, a calendar, a
                          heuristic, a transfer graph.
  model/                  Graph, Environment, Journey, results, search spaces.
  data/                   The OSM, GTFS and footpath layers.
  util/                   Clocks and argument coercion.
  reference.py            Pure-Python twins of the static kernels — the oracle.
tests/                    Mirrors the veneer. The differential suite sits at the
                          top, because it belongs to no single technique.
demos/                    Runnable examples, and the node board behind `serve.py`.
benchmarks/               What each technique costs, on a real city.
docs/                     One page per paper, run as tests by `tests/test_docs.py`.
```

Kernel work goes in `routelab-core`, where it is usable from Rust and testable
without Python. The bindings layer stays thin on purpose: sugar is easier to read,
change, and document in Python.

A new technique is a new file in both `kernels/` directories and nothing else
moved: it reuses the model, declares the query options it takes on its planner's
own signature, brings its own derivation of whatever it needs beyond the graph,
and is checked against something that cannot be wrong in the same direction.

## Design notes

**Integer weights.** Costs are `u32`, conventionally seconds. The schedule-based
literature is integer-valued throughout, and float comparison is a poor foundation
for the Pareto dominance checks the multicriteria algorithms are built on.

**CSR, immutable.** A graph is built once from an edge list and never mutated,
which lets searches run with the GIL released. Edges are permuted into CSR
order, so edge ids are not positions in your input list — `Graph.input_index`
maps back, which keeps per-edge attributes (mode, trip, route) attached.
Preprocessing that needs to *grow* a graph — contraction inserting shortcuts —
builds its own mutable adjacency and hands back finished CSR graphs, rather
than making every search pay for an edge list that might change underneath it.

**Deterministic tie-breaking.** Nodes settle in order of `(cost, node)`, and
out-edges relax in CSR order. Equal-cost paths are common in transit networks,
and without a rule for them two correct implementations disagree constantly and
usefully diffing them becomes impossible. Every search is deterministic; not
all of them agree with each other, because a rule about settle order says
nothing across algorithms that do not settle in one order — a bidirectional
search picks its own equally-cheap winner among ties.

**The environment is a merge, not a bag.** Compiling layers produces exactly
three things: a numbering of labels, one graph, and which layer each edge came
from. A calendar, a timetable, a coordinates table, a rate — anything a
technique reads beyond the graph — is derived by that technique at bind time
from the compiled layers (`rl.Schedule`, `rl.Departures`, `rl.Plane`,
`rl.Pace`), and refused there if the layers cannot supply it. The rule that
follows is worth stating: a thing is an *argument* — a constructor parameter,
a wire on the demo's board — if and only if it is a choice. A heuristic is a
choice; a calendar assembled from the layers has one possible construction and
no knobs, so it is derived rather than passed. This lets the next
schedule-based algorithm arrive without touching `environment.py`: it brings
its own derivation, and the environment need not know what it is for. The
corollary for knobs: a bound on one question is a query option, a property of
the technique is a constructor argument — `max_transfers` is the former,
`waiting` the latter.

**A technique takes the options its problem needs**, on the signature of its
own bound planner. `DijkstraPlanner.route` takes `max_cost`;
`RAPTORPlanner.route` takes `departing` and `max_transfers`; asking either for
the other's is a `TypeError` from Python, and an error a type checker reports
before anything runs. There is no registry of option names, so nothing can go
stale as the shelf grows.

**A technique may search whatever graph it likes, but it answers in the
caller's terms.** A contraction hierarchy searches a graph the environment has
never seen; the planner unpacks every shortcut back into the original edges it
stands for before anything leaves, so journeys, legs, geometry and provenance
work exactly as they do under Dijkstra.

**Multi-source with initial costs.** The one-to-all search takes
`(node, initial_cost)` pairs rather than a single origin. Every multimodal
algorithm needs this — the transit search starts from a set of stops, each
already reached at some cost — so it belongs in the primitive rather than in a
wrapper.

## Development

```bash
cargo test                             # Rust: core kernels and bindings
maturin develop && pytest              # Python: veneer and differential tests
cargo fmt && cargo clippy --all-targets
python docs/build.py --out site        # render these pages to a static site
```

## See also

- [The shelf](index.md) — every paper implemented here.
- [What preprocessing buys](tradeoffs.md) — the whole shelf measured side by side.
- [Seeing it run](demos.md) — the command-line demos and the node board.
