"""An interactive routing demo: click two points on a map, watch the search.

    python demos/serve.py data/Seattle.osm.pbf

Then open http://localhost:8000. Click once to drop an origin, again for a
destination, and the page draws the route over the search tree that found it —
the part a routing engine normally throws away. Branch widths follow the total
hanging off each one, so the tree reads like a river network: thick where the
whole search flowed, thinning to capillaries where it gave up.

The extract is read once at startup and the planners are built once; each click
is a query against them, which is the whole point of a planner being an object
you keep. Switching profile or algorithm rebuilds only what has to change.

Nothing here is a production server: it is the standard library's `http.server`,
single-threaded, bound to localhost, with no cache and no rate limiting. It is a
way to look at what the algorithms do.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}
ALGORITHMS = ("astar", "dijkstra")


class Router:
    """One extract, its environments, and the planners built over them.

    Environments are built per profile on demand and kept: reading the extract
    is seconds, so doing it per request would make the demo feel like the
    algorithms are slow when it is the loading that is.
    """

    def __init__(self, path: Path):
        self.path = path
        self._environments: "dict[str, tuple[rl.Environment, rl.OSM]]" = {}
        self._planners: "dict[tuple[str, str], rl.Planner]" = {}

    def environment(self, profile: str) -> "tuple[rl.Environment, rl.OSM]":
        if profile not in self._environments:
            started = time.perf_counter()
            layer = rl.OSM(self.path, PROFILES[profile]())
            environment = rl.Environment(layer)
            compiled = environment.compile()
            print(
                f"  {profile}: {compiled.graph.num_nodes:,} nodes, "
                f"{compiled.graph.num_edges:,} edges "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
            self._environments[profile] = (environment, layer)
        return self._environments[profile]

    def planner(self, profile: str, algorithm: str) -> rl.Planner:
        if (profile, algorithm) not in self._planners:
            environment, _ = self.environment(profile)
            self._planners[(profile, algorithm)] = (
                rl.AStar(environment, rl.Euclidean())
                if algorithm == "astar"
                else rl.Dijkstra(environment)
            )
        return self._planners[(profile, algorithm)]

    @lru_cache(maxsize=8)
    def _reachable(self, profile: str, origin: int) -> "tuple[int, ...]":
        """Labels reachable from `origin` — what a destination may snap to.

        Extracts are full of stubs that connect to nothing under a given
        profile, and snapping a click to one produces "no route" for reasons
        that have nothing to do with routing.
        """
        environment, _ = self.environment(profile)
        compiled = environment.compile()
        result = rl.Dijkstra(environment).search(origin)
        return tuple(compiled.label(node) for node in result.order)

    def route(
        self,
        profile: str,
        algorithm: str,
        origin,
        destination,
        branches: "int | None" = None,
    ) -> dict:
        """Route between two `(lat, lon)` clicks, and report the search too."""
        environment, layer = self.environment(profile)
        compiled = environment.compile()
        planner = self.planner(profile, algorithm)

        start = layer.nearest(*origin)
        end = layer.nearest(*destination, within=self._reachable(profile, start))

        began = time.perf_counter()
        result = planner.search(start, targets=[planner.node_id(end)])
        elapsed = (time.perf_counter() - began) * 1000

        journey = planner.route(start, end)
        if journey is None:
            return {"error": "no route between those points"}

        coordinates = layer.coordinates()
        # The search space, as the planner reports it. Every branch is drawn by
        # default — keeping only the heaviest keeps the trunk and throws away
        # the crown, which is exactly the part that shows how far the search
        # reached. A city-wide Dijkstra is ~40k branches and about 10 MB of
        # GeoJSON, which localhost and a canvas renderer both take in stride.
        tree = planner.explored(result)
        return {
            "route": [point for leg in journey.legs for point in compiled.geometry(leg.edge)],
            "tree": tree.geojson(limit=branches),
            "peak": tree.peak,
            "branch_count": len(tree),
            "snapped": [coordinates[start], coordinates[end]],
            "seconds": journey.cost,
            "legs": len(journey.legs),
            "settled_count": len(result.order),
            "graph_nodes": compiled.graph.num_nodes,
            "ms": round(elapsed, 1),
        }


class Handler(BaseHTTPRequestHandler):
    router: Router
    center: "tuple[float, float]"

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        url = urlparse(self.path)
        if url.path == "/":
            self.respond(200, "text/html; charset=utf-8", self.page().encode("utf-8"))
        elif url.path == "/route":
            self.respond(200, "application/json", self.route(parse_qs(url.query)))
        else:
            self.respond(404, "text/plain", b"not found")

    def route(self, query: "dict[str, list[str]]") -> bytes:
        def point(name: str) -> "tuple[float, float]":
            lat, _, lon = query[name][0].partition(",")
            return float(lat), float(lon)

        try:
            branches = query.get("branches", [""])[0]
            payload = self.router.route(
                query.get("profile", ["driving"])[0],
                query.get("algorithm", ["astar"])[0],
                point("from"),
                point("to"),
                int(branches) if branches else None,
            )
        except (KeyError, ValueError) as error:
            payload = {"error": str(error)}
        # Compact separators: the payload is mostly numbers, and the default
        # ", " / ": " padding is about a tenth of ten megabytes.
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def page(self) -> str:
        return PAGE.replace("__CENTER__", json.dumps(list(self.center)))

    def respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Quieter than the default, which logs every tile-less request."""
        if "/route" in str(args[0] if args else ""):
            print(f"  {args[0]}", flush=True)


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>routelab</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html, body, #map { height: 100%; margin: 0; font: 14px system-ui, sans-serif; }
  #panel { position: absolute; top: 12px; left: 12px; z-index: 1000; background: #fff;
           padding: 12px 14px; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,.3);
           min-width: 240px; }
  h1 { font-size: 15px; margin: 0 0 8px; }
  label { display: block; margin: 6px 0 2px; color: #555; font-size: 12px; }
  select { width: 100%; padding: 4px; }
  #status { margin-top: 10px; color: #333; line-height: 1.5; }
  #status b { font-variant-numeric: tabular-nums; }
  .hint { color: #777; font-size: 12px; }
</style>
<div id="panel">
  <h1>routelab</h1>
  <label for="profile">profile</label>
  <select id="profile">
    <option>driving</option><option>cycling</option><option>walking</option>
  </select>
  <label for="algorithm">algorithm</label>
  <select id="algorithm">
    <option value="astar">A* (euclidean)</option>
    <option value="dijkstra">Dijkstra</option>
  </select>
  <div id="status" class="hint">Click the map to set an origin.</div>
</div>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// Zoom buttons move aside: the panel wants the top-left corner.
// Canvas, not SVG: a search tree is thousands of lines, and thousands of DOM
// nodes is where a map stops being interactive.
const map = L.map('map', {zoomControl: false, preferCanvas: true}).setView(__CENTER__, 12);
L.control.zoom({position: 'topright'}).addTo(map);
// A grey basemap, not the standard one: standard OSM tiles draw roads in the
// same oranges and yellows the search tree needs, and the tree disappears into
// the streets it is drawn over.
L.tileLayer('https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, attribution: '© OpenStreetMap, © CARTO'}).addTo(map);

const status = document.getElementById('status');
const drawn = L.layerGroup().addTo(map);
let clicks = [];

map.on('click', event => {
  const point = [event.latlng.lat, event.latlng.lng];
  if (clicks.length === 2) { clicks = []; drawn.clearLayers(); }
  clicks.push(point);
  L.circleMarker(point, {radius: 6, color: '#1d3557', weight: 2,
                         fillColor: '#fff', fillOpacity: 1}).addTo(drawn);
  if (clicks.length === 1) {
    status.className = 'hint';
    status.textContent = 'Now click a destination.';
  } else {
    request();
  }
});

for (const id of ['profile', 'algorithm']) {
  document.getElementById(id).addEventListener('change', () => {
    if (clicks.length === 2) { redraw(); }
  });
}

function redraw() {
  drawn.clearLayers();
  clicks.forEach(point => L.circleMarker(point, {radius: 6, color: '#1d3557',
    weight: 2, fillColor: '#fff', fillOpacity: 1}).addTo(drawn));
  request();
}

async function request() {
  const profile = document.getElementById('profile').value;
  const algorithm = document.getElementById('algorithm').value;
  status.className = 'hint';
  status.textContent = 'routing…';

  const query = new URLSearchParams({
    from: clicks[0].join(','), to: clicks[1].join(','), profile, algorithm});
  const answer = await (await fetch('/route?' + query)).json();
  if (answer.error) {
    status.textContent = answer.error;
    return;
  }

  // The search tree first, so the route draws over it. Widths follow the
  // square root of each branch's share of the peak: the trunk carries the whole
  // search and the twigs almost none of it, and a linear scale would render
  // everything but the trunk invisible.
  L.geoJSON(answer.tree, {
    style: feature => ({
      color: '#1d6fa5',
      weight: 0.6 + 9 * Math.sqrt(feature.properties.share),
      opacity: 0.8,
    }),
  }).addTo(drawn);
  L.polyline(answer.route, {color: '#d1495b', weight: 4}).addTo(drawn);
  answer.snapped.forEach(p => L.circleMarker(p, {radius: 4, stroke: false,
    fillOpacity: 1, fillColor: '#1d3557'}).addTo(drawn));

  const minutes = (answer.seconds / 60).toFixed(1);
  const share = (100 * answer.settled_count / answer.graph_nodes).toFixed(1);
  status.className = '';
  status.innerHTML =
    `<b>${minutes}</b> min over <b>${answer.legs}</b> legs<br>` +
    `settled <b>${answer.settled_count.toLocaleString()}</b> nodes ` +
    `(${share}% of ${answer.graph_nodes.toLocaleString()}) in <b>${answer.ms}</b> ms<br>` +
    `<span class="hint">tree: ${answer.branch_count.toLocaleString()} branches, ` +
    `${answer.tree.features.length.toLocaleString()} drawn</span>`;
}
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path, help="an .osm.pbf file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="driving")
    args = parser.parse_args()

    if not args.extract.is_file():
        print(f"no extract at {args.extract}")
        print("city extracts: https://download.bbbike.org/osm/bbbike/")
        return 1

    print(f"reading {args.extract.name}")
    router = Router(args.extract)
    _, layer = router.environment(args.profile)
    min_lat, min_lon, max_lat, max_lon = layer.bounds

    Handler.router = router
    Handler.center = ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nhttp://localhost:{args.port}  — click two points on the map\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
