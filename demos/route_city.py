"""Route across a real city, and draw what each algorithm had to look at.

    python demos/route_city.py seattle.osm.pbf --from 47.6615,-122.3446 --to 47.5707,-122.3117

Prints the cost and the work — nodes settled, milliseconds — for Dijkstra and
A*, then writes `route.html`: the route drawn on a map, with every node each
search settled shaded behind it. Dijkstra's disc against A*'s ellipse is the
whole argument for a heuristic, in one picture.

Extracts come from https://download.geofabrik.de — any `.osm.pbf` will do. The
page needs the network only for its basemap tiles; the data is inlined.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}


def coordinate(text: str) -> "tuple[float, float]":
    lat, _, lon = text.partition(",")
    return float(lat), float(lon)


def timed(fn, *args, **kwargs):
    """Run something and report how long it took, in milliseconds."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an .osm.pbf (or .osm) file")
    parser.add_argument("--from", dest="origin", type=coordinate, required=True, metavar="LAT,LON")
    parser.add_argument("--to", dest="destination", type=coordinate, required=True, metavar="LAT,LON")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="driving")
    parser.add_argument("--out", type=Path, default=Path("route.html"))
    args = parser.parse_args()

    if not args.extract.is_file():
        print(f"no extract at {args.extract}", file=sys.stderr)
        print("download one from https://download.geofabrik.de", file=sys.stderr)
        return 1

    layer = rl.OSM(args.extract, PROFILES[args.profile]())
    env = rl.Environment(layer)
    compiled, load_ms = timed(env.compile)
    print(
        f"{compiled.graph.num_nodes:,} nodes, {compiled.graph.num_edges:,} edges "
        f"({args.profile}, {load_ms / 1000:.1f}s)"
    )

    # Snap to somewhere the origin can actually get to. Extracts are full of
    # stubs — a driveway reachable only on foot, a fragment cut off at the
    # extract's edge — and landing on one produces a query with no answer for
    # reasons that have nothing to do with routing.
    origin = layer.nearest(*args.origin)
    reachable = rl.Dijkstra().bind(env).search(origin)
    connected = [compiled.label(node) for node in reachable.order]
    destination = layer.nearest(*args.destination, within=connected)
    print(f"routing over {len(connected):,} nodes reachable from the origin")

    planners = [("dijkstra", rl.Dijkstra().bind(env)), ("astar", rl.AStar(rl.Euclidean()).bind(env))]
    searches = {}
    journey = None
    for name, planner in planners:
        target = planner.node_id(destination)
        result, ms = timed(planner.search, origin, targets=[target])
        searches[name] = result
        cost = result.cost(target)
        if cost is None:
            print(f"{name:<10} no route found")
            continue
        print(f"{name:<10} {len(result.order):>8,} settled  {ms:>7.0f} ms  {cost / 60:>6.1f} min")
        journey = journey or planner.route(origin, destination).routes[0]

    if journey is None:
        print("\nnothing to draw: those points are not connected in this profile")
        return 1

    write_map(args.out, compiled, layer, journey, searches)
    print(f"\nwrote {args.out}  ({len(journey.legs)} legs, {journey.cost / 60:.1f} min)")
    return 0


def write_map(path, compiled, layer, journey, searches):
    """A self-contained Leaflet page: the route, over the work each search did.

    Drawing both settled sets on top of each other would hide the smaller one
    under the larger, which is exactly backwards — the interesting quantity is
    what guidance let A* *skip*. So the layers are the difference and the
    overlap, not the two sets.
    """
    route = journey.geometry
    coordinates = layer.coordinates()
    as_labels = lambda result: {compiled.label(node) for node in result.order}

    dijkstra = as_labels(searches["dijkstra"])
    astar = as_labels(searches["astar"])
    layers = {
        "settled by both": sorted(dijkstra & astar),
        "skipped by A*": sorted(dijkstra - astar),
    }
    points = {
        name: [coordinates[label] for label in labels] for name, labels in layers.items()
    }

    payload = json.dumps(
        {"route": route, "settled": points, "center": route[len(route) // 2]}
    )
    path.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>routelab — route and search</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html, body, #map { height: 100%; margin: 0; }
  .legend { background: #fff; padding: 8px 10px; font: 13px system-ui, sans-serif;
            line-height: 1.6; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.3); }
  .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px;
            border-radius: 50%; vertical-align: middle; }
</style>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = __DATA__;
const map = L.map('map').setView(data.center, 13);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '© OpenStreetMap'}).addTo(map);

// Settled nodes first, so the route draws over them.
const shades = {'settled by both': '#9dc6e0', 'skipped by A*': '#e8a33d'};
for (const [name, points] of Object.entries(data.settled)) {
  const group = L.layerGroup(points.map(p =>
    L.circleMarker(p, {radius: 1.5, stroke: false, fillOpacity: 0.35,
                       fillColor: shades[name] || '#bbb'})));
  group.addTo(map);
}

const route = L.polyline(data.route, {color: '#d1495b', weight: 5}).addTo(map);
map.fitBounds(route.getBounds(), {padding: [30, 30]});

const legend = L.control({position: 'bottomright'});
legend.onAdd = () => {
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = Object.entries(data.settled).map(([name, points]) =>
    `<span class="swatch" style="background:${shades[name] || '#bbb'}"></span>` +
    `${name}: ${points.length.toLocaleString()}`).join('<br>') +
    '<br><span class="swatch" style="background:#d1495b"></span>route';
  return div;
};
legend.addTo(map);
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
