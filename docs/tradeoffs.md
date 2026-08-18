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
| [Dijkstra](papers/dijkstra.md) | none | — | 115,698 | 19.777 ms |
| [A* euclidean](papers/astar.md) | 0.2 s | 4 MB | 55,250 | 10.887 ms |
| [A* 16 landmarks](papers/landmarks.md) | 1.2 s | 29 MB | 9,370 | 3.629 ms |
| [Contraction hierarchy](papers/contraction-hierarchies.md) | 7.3 s | 17 MB | 247 | 0.253 ms |

A clean frontier, and the tidiest result on this page: each step costs more
preprocessing than the last and returns a faster query, in the order the
literature produced them. Nothing here is dominated.

Dijkstra sits in the "none" slot at the left rather than at zero, because a log
axis has no zero and pretending its preprocessing is very small rather than
absent would be a quiet lie about a real difference in kind — it has no
preprocessing step at all.

The span is the thing to look at: seven seconds of contraction buys a query
**78× faster** than the one it started from. The settled column says why — 247
nodes touched instead of 115,698, on a network of 230,104.

## Timetables

![Preprocessing against query time for the timetable techniques](plots/timetable.svg)

| technique | preprocessing | memory | settled | median query |
|---|---:|---:|---:|---:|
| [TimeDependent](papers/pyrga.md) | 0.4 s | 11 MB | 3,256 stops | 2.099 ms |
| [TimeExpanded](papers/pyrga.md) | 1.2 s | 150 MB | 66,447 events | 15.264 ms |
| [RAPTOR](papers/raptor.md) | 0.4 s | 16 MB | 4,126 stops | 1.515 ms |
| [CSA](papers/csa.md) | 0.4 s | 25 MB | 3,411 stops | 0.602 ms |
| [TripBased](papers/trip-based.md) | 12.2 s | 28 MB | 8,455 trips | 0.940 ms |
| [PTL](papers/ptl.md) | 58.1 s | 1,165 MB | 12,973 hubs | 0.343 ms |
| [ULTRA(RAPTOR)](papers/ultra.md) | 15.6 s | 15 MB | 6,184 stops | 2.259 ms |

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
radius. Read it as the cost of a capability, not as a slower RAPTOR — the
comparison it belongs in is the multimodal one below.

## Multimodal

![Preprocessing against query time for the multimodal techniques](plots/multimodal.svg)

| technique | preprocessing | memory | median query |
|---|---:|---:|---:|
| [LabelConstrained](papers/label-constrained.md) | 0.4 s | 14 MB | 109.790 ms |
| [UCCH](papers/ucch.md) | 202.8 s | 66 MB | 36.797 ms |
| [ULTRA(RAPTOR)](papers/ultra.md) | 617.5 s | 239 MB | 849.041 ms |

A harder instance than the two above: 560,706 nodes of pavement and timetable
joined by link arcs, and every trip starts and ends on the street rather than
at a stop. All three return the same arrivals on all ten trips.

**UCCH does what it claims.** Three and a half minutes of contracting the
pavement buys a query **3.0× faster** than searching the whole network — close
to the ~3.5× the [UCCH page](papers/ucch.md) quotes, and, as that page says,
near this technique's ceiling: what remains is the transit search inside the
core, which UCCH does not touch.

**ULTRA is dominated here, and the reason is on the x-axis of the wrong
chart.** Ten minutes of preprocessing and a query eight times slower than doing
no preprocessing at all. This is not the result the paper reports, and it is
worth being precise about why: on the [timetable chart](#timetables) above,
`ULTRA(RAPTOR)` answers in 2.3 ms — because there its walking network is the
feed's own footpaths, a few thousand edges. Here it is Seattle's entire
pavement, and ULTRA's query has to search that street graph outward from the
origin and inward to the destination *without a radius* — which is precisely the
capability it exists to provide. The shortcut set makes the transit half fast;
the walking legs at either end are what 849 ms is made of. A bound on that
initial search is what the paper uses and what this implementation does not yet
expose, so read this point as the cost of an unbounded walk on a city-sized
street graph rather than as a verdict on the technique.

That caveat is the reason to keep the chart rather than quietly drop the row.
A trade-off plot is only honest if it also shows the trades that went badly.

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
