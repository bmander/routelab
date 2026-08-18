# What preprocessing buys

Every technique on this shelf answers the same question. What separates them is
*when* they spend: some do nothing until asked and then search hard, others
spend minutes rewriting the network so that a query barely searches at all.

These are the two axes. **Preprocessing** is paid once per network, at `bind`.
**Query** is paid once per question, and is the median over a set of random
trips. Both are logarithmic, because the classes span four orders of magnitude
in each direction, and a linear axis would put half the shelf on top of the
origin.

**Down and to the left is better.** A point down and left of another is faster
in both senses, and the technique it beats is *dominated* — which does happen,
and is worth naming when it does. The corner nobody occupies is the bottom
left: a fast query with no preprocessing. That empty space is the whole subject
of the literature these pages describe.

Every row within a chart comes from one run on one instance, and each benchmark
checks that all of its techniques return the same answers before it reports a
time — a row that got faster by getting a different answer fails the benchmark
rather than appearing here.

## Road networks

![Preprocessing against query time for the road-network techniques](plots/road.svg)

| technique | preprocessing | memory | settled | median query |
|---|---:|---:|---:|---:|
| [Dijkstra](papers/dijkstra.md) | none | — | 115,698 | 15.931 ms |
| [A* euclidean](papers/astar.md) | 0.2 s | 4 MB | 55,250 | 11.261 ms |
| [A* 16 landmarks](papers/landmarks.md) | 1.3 s | 29 MB | 9,370 | 2.626 ms |
| [Contraction hierarchy](papers/contraction-hierarchies.md) | 6.4 s | 17 MB | 247 | 0.171 ms |

A clean frontier, and the tidiest result on this page: each step costs more
preprocessing than the last and returns a faster query, in the order the
literature produced them. Nothing here is dominated.

Dijkstra sits in the "none" slot at the left rather than at zero, because a log
axis has no zero and pretending its preprocessing is very small rather than
absent would be a quiet lie about a real difference in kind — it has no
preprocessing step at all.

The span is the thing to look at: six seconds of contraction buys a query
**93× faster** than the one it started from. The settled column says why — 247
nodes touched instead of 115,698, on a network of 230,104.

## Timetables

![Preprocessing against query time for the timetable techniques](plots/timetable.svg)

| technique | preprocessing | memory | settled | median query |
|---|---:|---:|---:|---:|
| [TimeDependent](papers/pyrga.md) | 0.6 s | 11 MB | 3,256 stops | 3.874 ms |
| [TimeExpanded](papers/pyrga.md) | 1.4 s | 150 MB | 66,447 events | 14.785 ms |
| [RAPTOR](papers/raptor.md) | 0.4 s | 16 MB | 4,126 stops | 1.192 ms |
| [CSA](papers/csa.md) | 0.4 s | 25 MB | 3,411 stops | 0.668 ms |
| [TripBased](papers/trip-based.md) | 14.5 s | 28 MB | 8,455 trips | 1.050 ms |
| [PTL](papers/ptl.md) | 61.3 s | 1,165 MB | 12,973 hubs | 0.495 ms |
| [ULTRA(RAPTOR)](papers/ultra.md) | 29.2 s | 15 MB | 6,179 stops | 1.917 ms |

Not a frontier — a scatter, and the interesting one on this page.

**TimeExpanded is dominated outright.** It costs three times TimeDependent's
preprocessing, a hundred and fifty megabytes against eleven, and answers seven
times slower. That is not a failure of the implementation; it is the paper's own
trade, stated honestly. A node per event turns the timetable into an ordinary
graph that any stock shortest-path search can route — and on a city's weekday
that means 837,924 nodes, which is what a query then has to search. You buy
simplicity and generality, and pay in size.

**CSA is the surprise.** It spends the same 0.4 s the graph models spend and
answers faster than TripBased, which spent twelve seconds precomputing
transfers. Sorting 421,604 connections into one array and scanning them turns
out to be extremely hard to beat on a city-sized instance — "intriguingly
simple", as the title has it.

**PTL is at the far corner, and its axis is the wrong one.** Its query is the
fastest here, but the number that decides whether you can use it is not on this
chart: 1,165 MB, against 11–28 MB for everything else. The scatter shows time
against time; on memory PTL is forty times its nearest neighbour.

**ULTRA is not comparable, and is drawn anyway.** It appears here because it
was measured in the same run, but it is solving a harder problem than the rows
around it: unlimited walking transfers rather than transfers within a fixed
radius. Twenty-nine seconds of preprocessing to answer in 1.9 ms what the
RAPTOR underneath answers in 1.2 ms is not a bad showing — it is a technique
paying for a capability this instance does not need, because a transitively
closed 200 m footpath set is already the thing ULTRA exists to avoid needing.
The chart where it earns that is the next one.

## Multimodal

![Preprocessing against query time for the multimodal techniques](plots/multimodal.svg)

| technique | preprocessing | memory | median query |
|---|---:|---:|---:|
| [LabelConstrained](papers/label-constrained.md) | 0.4 s | 14 MB | 106.724 ms |
| [UCCH](papers/ucch.md) | 189.3 s | 66 MB | 34.791 ms |
| [ULTRA(RAPTOR)](papers/ultra.md) | 612.6 s | 149 MB | 11.445 ms |

A harder instance than the two above: 560,706 nodes of pavement and timetable
joined by link arcs, and every trip starts and ends on the street rather than
at a stop. All three return the same arrivals on all ten trips.

This is the page's cleanest frontier, and the three corners the multimodal
pages describe. Precompute nothing and search the whole network: 107 ms.
Contract the pavement for three minutes and search a core: 35 ms. Work out
every transfer a journey could want, for ten minutes, and answer in 11 ms.
Each step buys almost exactly what it pays for.

**UCCH does what it claims** — 3.1× faster than searching the whole network,
close to the ~3.5× the [UCCH page](papers/ucch.md) quotes and, as that page
says, near its ceiling: what remains is the transit search inside the core,
which UCCH does not touch.

**ULTRA lands where the paper says it should.** The published ULTRA-RAPTOR
answers Switzerland in 12.5 ms; this answers Seattle with King County Metro in
11.4. It did not always: this chart used to show 849 ms, and the three things
that were wrong are worth naming, because each was an implementation detail
rather than anything the paper left vague.

1. The walks at either end were two unbounded Dijkstras over the whole street
   graph. §4.1 answers them with **Bucket-CH**, and now so does this.
2. The wrapped RAPTOR was bound to the merged network, so its tables held a row
   per street corner — 560,706 of them, nine rounds a query — where the paper
   runs the black box on the transit network, 6,313 stops.
3. The shortcuts were being closed under composition before the technique read
   them. A radius's foot-edges are a scatter of small components and closing
   them is cheap; ULTRA's shortcuts span the network, and closing 24,554 of
   them made 230 MB of walks no journey takes. They are already the set a query
   needs — that is the paper's whole guarantee — so they go in as they are.

Together those took the query from 849 ms to 11.4 and the footprint from 374 MB
to 149. Nothing about the answers changed: the three techniques agreed before
and agree now, which is the only reason the changes could be made with any
confidence.

What ULTRA still costs is the ten minutes up front — a core contraction, the
shortcut search over it, and a second full contraction for the buckets — and
149 MB to hold the result. That is the trade, and on this instance it is worth
it three times over.

## Reproducing these

Each chart is one command, and every number on this page came from these three
on one machine — an Intel Core i9-8950HK, macOS 15.7.4, routelab 0.1.0:

```bash
python benchmarks/bench_contraction.py data/Seattle.osm.pbf --trips 25
python benchmarks/bench_transit.py data/kcm.zip --date 2026-08-17 --pairs 25
python benchmarks/bench_multimodal.py data/Seattle.osm.pbf data/kcm.zip \
    --date 2026-08-17 --pairs 10
```

The extracts are not in the repository — `data/` is ignored, and everyone
fetches their own from [Geofabrik](https://download.geofabrik.de) or
[BBBike](https://download.bbbike.org/osm/bbbike/) and their own feed.

**The shape is the point, not the absolute values.** Another machine will move
every point by some constant factor, which on a log axis slides the whole cloud
without changing what it says. What would change the shape is a different
network: a smaller city compresses the preprocessing axis, and a feed with
denser service moves the timetable techniques relative to each other. The
numbers live in [`docs/measurements.json`](measurements.json) with the commands
that produced them; [`docs/plot.py`](plot.py) redraws the charts from it.

## See also

- [The shelf](index.md) — every paper implemented here, with a page each.
- `benchmarks/` — the scripts, which check agreement before they report a time.
