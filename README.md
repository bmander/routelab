# routelab

`routelab` provides reference implementations of published routing algorithms; kernels in Rust with
Python bindings with a common API for easy comparison.

## Quickstart

Requires Python 3.9+ and a [Rust toolchain](https://rustup.rs).

```bash
git clone https://github.com/bmander/routelab && cd routelab
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # builds the Rust kernel via maturin
```

Describe a world as layers, bind a technique to it, plan a route:

```python
import routelab as rl

env = rl.Environment()
env.register(rl.ScalarEdges(("a", "b", 1), ("b", "c", 15)))

technique = rl.Dijkstra()           # a configuration, costing nothing
planner = technique.bind(env)       # preprocessing, if the technique has any
planner.route("a", "c").routes[0]   # Journey('a' → 'b' → 'c', cost=16)
```

The same three steps hold for every technique on the shelf, a real city and a
real timetable included:

```python
from datetime import date, time

feed = rl.GTFS("kcm.zip", date(2026, 8, 17))
env = rl.Environment(feed, rl.Footpaths(feed, within=200))

answer = rl.RAPTOR().bind(env).route(downtown, juanita, departing=time(8, 30))
answer.routes 
answer.searchspace()
```

Then [**bmander.github.io/routelab**](https://bmander.github.io/routelab/) — or
[`docs/`](docs/index.md), the same pages as Markdown. There is one per paper:
what it observed, its algorithm as pseudocode, and a runnable hello-world that
the test suite runs, so a page cannot drift from the code.

## What is implemented

| Paper | Technique | Page |
|---|---|---|
| Dijkstra, *A note on two problems in connexion with graphs* (1959) | `Dijkstra()` | [Dijkstra's algorithm](docs/papers/dijkstra.md) |
| Moore, *The shortest path through a maze* (1959) | `BFS()` | [Breadth-first search](docs/papers/bfs.md) |
| Hart, Nilsson & Raphael, *A formal basis for the heuristic determination of minimum cost paths* (1968) | `AStar(Euclidean())`, `AStar(Zero())` | [A*](docs/papers/astar.md) |
| Goldberg & Harrelson, *Computing the shortest path: A\* search meets graph theory* (2005) | `AStar(Landmarks(16))` | [ALT landmarks](docs/papers/landmarks.md) |
| Geisberger, Sanders, Schultes & Delling, *Contraction hierarchies* (2008) | `ContractionHierarchy(EdgeDifference())` | [Contraction hierarchies](docs/papers/contraction-hierarchies.md) |
| Dreyfus, *An appraisal of some shortest-path algorithms* (1969) | `TimeDependentDijkstra()` | [Time-dependent Dijkstra](docs/papers/time-dependent.md) |
| Pyrga, Schulz, Wagner & Zaroliagis, *Efficient models for timetable information in public transportation systems* (2007) | `TimeExpanded()`, `TimeDependent()`, `Footpaths(feed, within=)` | [Two models of a timetable](docs/papers/pyrga.md) |
| Delling, Pajor & Werneck, *Round-based public transit routing* (2012) | `RAPTOR()` | [RAPTOR](docs/papers/raptor.md) |
| Dibbelt, Pajor, Strasser & Wagner, *Intriguingly simple and fast transit routing* (2013) | `CSA()` | [Connection scan](docs/papers/csa.md) |
| Witt, *Trip-based public transit routing* (2015) | `TripBased()` | [Trip-based routing](docs/papers/trip-based.md) |
| Delling, Dibbelt, Pajor & Werneck, *Public transit labeling* (2015) | `PTL()` | [Public transit labeling](docs/papers/ptl.md) |
| Baum, Buchhold, Sauer, Wagner & Zündorf, *UnLimited TRAnsfers for multi-modal route planning* (2019) | `ULTRA(RAPTOR())`, `ULTRA(CSA())` | [ULTRA](docs/papers/ultra.md) |
| Barrett, Jacob & Marathe, *Formal-language-constrained path problems* (2000) | `LabelConstrained()`, `Modes(...)` | [Label-constrained routing](docs/papers/label-constrained.md) |
| Dibbelt, Pajor & Wagner, *User-constrained multi-modal route planning* (2012) §3 | `UCCH()` | [UCCH](docs/papers/ucch.md) |

Not yet implemented, in the order the literature filled them: Delling, Pajor &
Wagner, *Engineering time-expanded graphs for faster timetable information*
(2009); Geisberger, *Contraction of timetable networks with realistic
transfers* (2010); and Bast et al., *Fast routing in very large public
transportation networks using transfer patterns* (2010).

Every kernel here is checked against something that cannot be wrong in the same
direction — a pure-Python reference, a brute-force oracle, or the paper's own
second model. See [the contract](docs/design.md#the-contract).

## Elsewhere

- [What preprocessing buys](docs/tradeoffs.md) — the whole shelf as two axes:
  what each technique paid at bind time against what a query cost.
- [Seeing it run](docs/demos.md) — the command-line demos, and the node board
  behind `demos/serve.py`.
- [How it is built](docs/design.md) — what this is for, the contract, the
  layout, and the decisions underneath.

## License

MIT.
