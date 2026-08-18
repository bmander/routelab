# The shelf

One page per paper implemented in routelab. Each says what the paper observed,
sketches its algorithm, then runs it — set up an environment, bind the
technique, ask a question — with a section per variant or refinement the paper
adds.

Start with [Dijkstra](papers/dijkstra.md) for the shape; every other page
assumes it.

## Road networks

| Paper | Technique | Page |
|---|---|---|
| Dijkstra, *A note on two problems in connexion with graphs* (1959) | `Dijkstra()` | [Dijkstra's algorithm](papers/dijkstra.md) |
| Moore, *The shortest path through a maze* (1959) | `BFS()` | [Breadth-first search](papers/bfs.md) |
| Hart, Nilsson & Raphael, *A formal basis for the heuristic determination of minimum cost paths* (1968) | `AStar(Euclidean())`, `AStar(Zero())` | [A*](papers/astar.md) |
| Goldberg & Harrelson, *Computing the shortest path: A\* search meets graph theory* (2005) | `AStar(Landmarks(16))` | [ALT landmarks](papers/landmarks.md) |
| Geisberger, Sanders, Schultes & Delling, *Contraction hierarchies* (2008) | `ContractionHierarchy(EdgeDifference())` | [Contraction hierarchies](papers/contraction-hierarchies.md) |
| Dreyfus, *An appraisal of some shortest-path algorithms* (1969) | `TimeDependentDijkstra()` | [Time-dependent Dijkstra](papers/time-dependent.md) |

## Timetables

| Paper | Technique | Page |
|---|---|---|
| Pyrga, Schulz, Wagner & Zaroliagis, *Efficient models for timetable information in public transportation systems* (2007) | `TimeExpanded()`, `TimeDependent()` | [Two models of a timetable](papers/pyrga.md) |
| Delling, Pajor & Werneck, *Round-based public transit routing* (2012) | `RAPTOR()` | [RAPTOR](papers/raptor.md) |
| Dibbelt, Pajor, Strasser & Wagner, *Intriguingly simple and fast transit routing* (2013) | `CSA()` | [Connection scan](papers/csa.md) |
| Witt, *Trip-based public transit routing* (2015) | `TripBased()` | [Trip-based routing](papers/trip-based.md) |
| Delling, Dibbelt, Pajor & Werneck, *Public transit labeling* (2015) | `PTL()` | [Public transit labeling](papers/ptl.md) |

## Multimodal

| Paper | Technique | Page |
|---|---|---|
| Baum, Buchhold, Sauer, Wagner & Zündorf, *UnLimited TRAnsfers for multi-modal route planning* (2019) | `ULTRA(RAPTOR())`, `ULTRA(CSA())` | [ULTRA](papers/ultra.md) |
| Barrett, Jacob & Marathe, *Formal-language-constrained path problems* (2000) | `LabelConstrained()`, `Modes(...)` | [Label-constrained routing](papers/label-constrained.md) |
| Dibbelt, Pajor & Wagner, *User-constrained multi-modal route planning* (2012) §3 — UCCH | `UCCH()` | [UCCH](papers/ucch.md) |

The shelf's *to do* list — Delling, Pajor & Wagner (2009), Geisberger (2010),
transfer patterns (2010) — appears in the [README](../README.md); those papers
have no page because they have no implementation.

## Side by side

- [What preprocessing buys](tradeoffs.md) — every technique in each class as one
  point: what it paid at bind time against what a query cost, measured on the
  same instance and checked to agree before it was timed.

## The shape every page shares

Three steps, the same for a road network and a timetable.

```python
>>> import routelab as rl

>>> env = rl.Environment()                              # describe a world
>>> env.register(rl.ScalarEdges(("a", "b", 1), ("b", "c", 15)))
Environment(1 layer)

>>> technique = rl.Dijkstra()                           # a configuration, costing nothing
>>> planner = technique.bind(env)                       # preprocessing, if any
>>> planner.route("a", "c")                             # the question
Journey('a' → 'b' → 'c', cost=16)

```

**An environment is layers.** A GTFS feed, an OSM extract, a table of walks, a
hand-written list of edges — each is a layer, and nodes take whatever names
you already use: stop ids, OSM node ids, `("bike", 42)`. Every leg of a journey
remembers its layer, which makes a multimodal answer readable rather than just
a number.

**Configuring and binding are separate** because the middle step gets expensive.
Sixteen landmarks over a city cost a second and 33 MB; that belongs to a verb,
not a constructor. The split also makes a technique a *value* — something you
can name, put in a dictionary, and point at more than one dataset.

**A technique takes the options its problem needs**, declared as data, and
refuses the rest by name, saying which technique they belong to. `Dijkstra`
takes `max_cost`; `RAPTOR` takes `departing` and `max_transfers`; asking either
for the other's is an error that tells you where to look.

## Reading the code blocks

Two kinds appear on these pages; the prompt tells them apart.

A block with `>>>` prompts **is a test**. `tests/test_docs.py` runs every one
as part of the ordinary suite, so a page describing an API that has since moved
fails the suite rather than misleading you. They run against fixtures the
repository ships: a handful of hand-written edges, or the three-stop GTFS feed
under `crates/routelab-gtfs/tests/data/tiny`, which the timetable pages name
`TINY_GTFS`.

A block without prompts is **written to be read**. Those run on a real city —
Seattle's 258,029 nodes, King County Metro's 421,604 connections — where the
numbers are the point and no test suite should download a 65 MB extract to
check them. Their outputs are pasted as comments from the benchmarks under
`benchmarks/`.

To run them — or any snippet on these pages — install routelab first. It builds
a Rust kernel, so you need a [Rust toolchain](https://rustup.rs) alongside
Python 3.9+:

```bash
git clone https://github.com/bmander/routelab && cd routelab
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'        # builds the Rust kernel via maturin

pytest tests/test_docs.py      # run every prompted block on this shelf
```

## Where else to look

- [README](../README.md) — what routelab is for, how to install it, and a tour
  of the whole shelf in one narrative.
- [The contract](../README.md#the-contract) — what every kernel here is checked
  against, and why an independent oracle rather than a golden file.
- `demos/serve.py` — the node board: layers, a technique, a query, wired up on a
  page, with the search drawn under a map.
- `benchmarks/` — the source of the real-network numbers on these pages.
