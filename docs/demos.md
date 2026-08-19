# Seeing it run

Four runnable programs under [`demos/`](../demos). Each takes a real extract or
feed; `data/` is ignored by git, so bring your own — country extracts from
[Geofabrik](https://download.geofabrik.de), city ones from
[BBBike](https://download.bbbike.org/osm/bbbike/), and any GTFS zip.

## A route, and the search behind it

```bash
python demos/route_city.py liechtenstein.osm.pbf \
    --from 47.2260,9.5230 --to 47.1300,9.5210 --profile walking
```

writes a map of the route with every settled node behind it — blue where both
searches went, orange for the nodes A\* never had to look at. It is the quickest
way to see what a heuristic buys: the same 11.2-minute Seattle route settles
41,161 branches under Dijkstra, reaching across Lake Washington to Bellevue, and
12,172 under A\*, which never leaves the corridor.

```bash
python demos/route_by_clock.py seattle.osm.pbf     # the same trip at every hour
python demos/route_transit.py kcm.zip --date 2026-08-17   # every timetable technique, one itinerary each
```

`route_transit.py --walk 0` drops the footpaths, which is the quickest way to
see what crossing the street buys.

## The node board

```bash
python demos/serve.py data/Seattle.osm.pbf     # then open http://localhost:8000
python demos/serve.py data/Seattle.osm.pbf --gtfs data/kcm.zip --date 2026-08-17
```

Click an origin, click a destination, and the route comes back with the search
drawn underneath. Under the map sits a **node board**: the three steps drawn as
a graph. Layers feed an `Environment`, a technique binds to one and becomes a
planner, a `Query` asks it something. Drag the nodes, pull the wires out, plug
them in somewhere else. There is no second representation of what the controls
mean — the board *is* the query, so unplugging a wire really does remove an
argument, and the page says which one.

The map is a node too, and the graph runs out through the map and back: the map
gives the query an `origin` and a `destination` and takes a `route` and a
`space` in return. Cross the two points and the trip reverses. Unplug `space`
and no search space gets built — ten megabytes of GeoJSON nothing was listening
for. Unplug `route` and the query still runs and still reports what it cost,
with nothing drawing it.

A node whose configuration the board has never answered greys out and spins
until it does, and because a node's identity includes everything upstream of
it, the spinners spread exactly as far as the rebuild does and no further. A
change that costs nothing shows nothing. Where the work can count itself, the
node shows how far along it is: contraction reports nodes retired, then arcs
assembled; a landmark table reports searches run. Where it cannot — a file
parser that yields no counts — the node says so rather than inventing a bar.

Watch that bar on a big network and it teaches you something the timings table
hides. Contracting Seattle's walking graph settles 99.5% of its 554,393 nodes in
the first minute and spends two more on the rest: the last nodes left are the
most connected, and that is where the shortcut search does its real work.

That is what makes the refusals worth causing on purpose. Wire a GTFS layer
into `Dijkstra` and the node turns red with the library's own sentence:

```
Dijkstra cannot route over timetable layers; it accepts scalar
```

A dropdown in the board's toolbar loads a starting point — road with landmarks,
road with a contraction hierarchy, plain Dijkstra as the control, walking that
reads the clock, and either timetable model when the demo was given a feed.
These are places to begin, not modes: load one and pull it apart. The pins stay
put across a change, which is the point of having more than one.

Rewiring is cheap because everything is cached by a canonical spelling of the
node and everything upstream of it, so returning to a shape you had before is
free. That matters because the expensive things are exactly the ones worth
comparing. Seattle is 258,029 nodes and 590,671 edges from a 65 MB extract,
read in about five seconds; swapping the technique node re-runs the same query
the other way:

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

## See also

- [The shelf](index.md) — every paper implemented here.
- [What preprocessing buys](tradeoffs.md) — the same comparison, measured properly.
- [How it is built](design.md) — the contract, the layout, and the decisions underneath.
