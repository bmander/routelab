"""An interactive routing demo: click two points on a map, watch the search.

    python demos/serve.py data/Seattle.osm.pbf

Then open http://localhost:8000. Click once to drop an origin, again for a
destination, and the page draws the route over the search tree that found it —
the part a routing engine normally throws away. Branch widths follow the total
hanging off each one, so the tree reads like a river network: thick where the
whole search flowed, thinning to capillaries where it gave up.

Either endpoint can then be dragged, and the route follows the cursor. The
search space does not: it is up to sixty thousand branches, so it is drawn once
when the drag stops. Dragging is also the quickest way to feel the difference
between the algorithms — the route keeps up under a contraction hierarchy and
visibly does not under Dijkstra.

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
#: The techniques on offer, as values — configured here, bound to an
#: environment when someone first asks for one. Adding an algorithm to the demo
#: is adding a line here; everything downstream asks the planner what it found
#: and what it explored, and neither answer depends on which one it is.
ALGORITHMS: "dict[str, rl.Planner]" = {
    "dijkstra": rl.Dijkstra(),
    "astar": rl.AStar(rl.Euclidean()),
    "landmarks": rl.AStar(rl.Landmarks(16)),
    "ch": rl.ContractionHierarchy(),
    "timedep": rl.TimeDependentDijkstra(),
}

def wants_time(technique: rl.Planner) -> bool:
    """Does this technique want a departure time?

    Asked of the technique rather than hardcoded here: a planner that needs a
    schedule declares it, and that is the same declaration `missing_from` uses
    to say a dataset cannot support it.
    """
    return "schedule" in getattr(technique, "requires", frozenset())


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
        """The planner for a profile and algorithm, built once and kept.

        Building is where preprocessing happens, and landmarks make that a real
        cost — a couple of seconds and tens of megabytes for a city. Paying it
        per click would be absurd; paying it once is the entire argument for a
        planner being an object you hold on to.
        """
        if (profile, algorithm) not in self._planners:
            environment, _ = self.environment(profile)
            started = time.perf_counter()
            self._planners[(profile, algorithm)] = ALGORITHMS[algorithm].bind(environment)
            elapsed = time.perf_counter() - started
            if elapsed > 0.1:
                print(f"  {algorithm} on {profile}: ready in {elapsed:.1f}s", flush=True)
        return self._planners[(profile, algorithm)]

    @lru_cache(maxsize=8)
    def _reachable(self, profile: str, origin: int) -> "tuple[int, ...]":
        """Labels reachable from `origin` — what a destination may snap to.

        Extracts are full of stubs that connect to nothing under a given
        profile, and snapping a click to one produces "no route" for reasons
        that have nothing to do with routing.

        A quarter of a second on a city: a whole Dijkstra, and a label for every
        node it settled. Which is why nothing calls this until a route has
        already failed — see `route`.
        """
        environment, _ = self.environment(profile)
        compiled = environment.compile()
        result = rl.Dijkstra().bind(environment).search(origin)
        return tuple(compiled.label(node) for node in result.order)

    def route(
        self,
        profile: str,
        algorithm: str,
        origin,
        destination,
        branches: "int | None" = None,
        explore: bool = True,
        departing: int = 0,
    ) -> dict:
        """Route between two `(lat, lon)` points, and report the search too.

        `explore` is what makes dragging an endpoint feel live. The route is a
        few hundred points; the search space behind it is up to sixty thousand
        branches and ten megabytes of GeoJSON, which is worth building once when
        the drag stops and not sixty times a second while it is moving.
        """
        environment, layer = self.environment(profile)
        compiled = environment.compile()
        planner = self.planner(profile, algorithm)

        start = layer.nearest(*origin)
        end = layer.nearest(*destination)

        # Snap plainly, and work out what is actually connected only if that
        # turns out to have failed. Restricting the snap up front costs a
        # quarter of a second per request — more than any of these algorithms
        # spends routing, and the same quarter-second for all of them, which
        # would flatten the difference this demo exists to show. Paying it on
        # the rare failure keeps a drag answering at the speed of the search.
        #
        # Only the time-dependent technique understands a departure, and the
        # others rightly refuse one, so it is passed to whoever asked for it.
        when = {"departing": departing} if wants_time(ALGORITHMS[algorithm]) else {}

        began = time.perf_counter()
        result = planner.search(start, targets=[planner.node_id(end)], **when)
        elapsed = (time.perf_counter() - began) * 1000

        if result.cost(planner.node_id(end)) is None:
            end = layer.nearest(*destination, within=self._reachable(profile, start))
            began = time.perf_counter()
            result = planner.search(start, targets=[planner.node_id(end)], **when)
            elapsed = (time.perf_counter() - began) * 1000

        target = planner.node_id(end)
        if result.cost(target) is None:
            return {"error": "no route between those points"}
        # Built from the result already in hand rather than by asking the
        # planner to route again — the same query twice is the one thing a drag
        # cannot afford.
        journey = rl.Journey.from_result(compiled, result, end)

        coordinates = layer.coordinates()
        answer = {
            "route": journey.geometry,
            "snapped": [coordinates[start], coordinates[end]],
            "seconds": journey.cost,
            "waiting": journey.waiting,
            "legs": len(journey.legs),
            # How many edges this profile has a schedule for, and whether the
            # chosen technique is reading it. A schedule quietly ignored is the
            # failure nobody can see, so the page is told and says so.
            "scheduled_edges": 0 if compiled.calendar is None else len(compiled.calendar),
            "reads_clock": bool(when),
            # How many legs of *this* route are scheduled. The number that
            # answers "why did changing the hour do nothing": a route over
            # edges nobody scheduled cannot care what time it is.
            "scheduled_legs": 0
            if compiled.calendar is None
            else sum(1 for leg in journey.legs if compiled.calendar.is_restricted(leg.edge)),
            "settled_count": result.settled,
            "graph_nodes": compiled.graph.num_nodes,
            "ms": round(elapsed, 1),
        }
        if explore:
            # The search space, as the planner reports it. Every branch is drawn
            # by default — keeping only the heaviest keeps the trunk and throws
            # away the crown, which is exactly the part that shows how far the
            # search reached. A city-wide Dijkstra is ~58k branches and about
            # 10 MB of GeoJSON, which localhost and a canvas renderer both take
            # in stride.
            space = planner.explored(result)
            answer["tree"] = space.geojson(limit=branches)
            answer["peak"] = space.peak
            answer["branch_count"] = len(space)
        return answer


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
                explore=query.get("explore", ["1"])[0] != "0",
                departing=int(query.get("departing", ["0"])[0]),
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
  /* Fixed, not min-width: the status text changes length with every answer,
     and a panel that resizes under the cursor is its own distraction. */
  #panel { position: absolute; top: 12px; left: 12px; z-index: 1000; background: #fff;
           padding: 12px 14px; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,.3);
           width: 280px; box-sizing: border-box; }
  h1 { font-size: 15px; margin: 0 0 8px; }
  label { display: block; margin: 6px 0 2px; color: #555; font-size: 12px; }
  select { width: 100%; padding: 4px; }
  #status { margin-top: 10px; color: #333; line-height: 1.5; }
  #status b { font-variant-numeric: tabular-nums; }
  .hint { color: #777; font-size: 12px; }
  /* A div, not Leaflet's default pin: it keeps the endpoint markers looking
     like the circles they were before they became draggable, and asks the CDN
     for no images. */
  .pin { border-radius: 50%; background: #fff; border: 2px solid #1d3557;
         box-sizing: border-box; cursor: grab; }
  .leaflet-drag-target .pin { cursor: grabbing; }
  .row { display: flex; align-items: center; gap: 8px; }
  .row select { width: auto; }
  #clock { font-variant-numeric: tabular-nums; font-size: 15px; }
  #minute { width: 100%; margin: 6px 0 0; }
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
    <option value="landmarks">A* (16 landmarks)</option>
    <option value="dijkstra">Dijkstra</option>
    <option value="ch">Contraction hierarchy</option>
    <option value="timedep">Time-dependent Dijkstra</option>
  </select>
  <div id="when" hidden>
    <label for="day">departing</label>
    <div class="row">
      <select id="day">
        <option value="0">Mon</option><option value="1">Tue</option>
        <option value="2">Wed</option><option value="3">Thu</option>
        <option value="4">Fri</option><option value="5">Sat</option>
        <option value="6">Sun</option>
      </select>
      <b id="clock">08:00</b>
    </div>
    <input id="minute" type="range" min="0" max="1435" step="5" value="480">
  </div>
  <div id="status" class="hint">Click the map to set an origin.</div>
  <div id="note" class="hint" style="margin-top:6px">Drag either endpoint to re-route.</div>
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
// Two groups, because a drag invalidates them at different rates. The route
// follows the cursor; the search space behind it is stale the moment the
// endpoint moves, so it is cleared for the duration and drawn again at the end.
const space = L.layerGroup().addTo(map);
const route = L.layerGroup().addTo(map);
const PIN = L.divIcon({className: 'pin', iconSize: [14, 14], iconAnchor: [7, 7]});
let pins = [];

// A pin's position as the query string spells it. Shared, because both the
// request and the URL it is recorded in have to agree about the wording.
function at(pin) {
  const point = pin.getLatLng();
  return `${point.lat},${point.lng}`;
}

function dropPin(latlng) {
  const pin = L.marker(latlng, {icon: PIN, draggable: true}).addTo(map);
  // Live while moving, and the whole picture once it stops.
  pin.on('dragstart', () => space.clearLayers());
  pin.on('drag', () => trace());
  pin.on('dragend', () => request(true));
  pins.push(pin);
  return pin;
}

map.on('click', event => {
  if (pins.length === 2) {
    pins.forEach(pin => pin.remove());
    pins = [];
    space.clearLayers();
    route.clearLayers();
  }
  dropPin(event.latlng);

  if (pins.length === 1) {
    status.className = 'hint';
    status.textContent = 'Now click a destination.';
  } else {
    request(true);
  }
});

// Everything that decides a query lives in the URL, so a result can be copied
// and handed to someone else. `replaceState`, not `pushState`: dragging a pin
// would otherwise fill the back button with a hundred near-identical steps.
function syncUrl() {
  const query = new URLSearchParams({
    profile: document.getElementById('profile').value,
    algorithm: document.getElementById('algorithm').value,
  });
  if (pins.length === 2) {
    query.set('from', at(pins[0]));
    query.set('to', at(pins[1]));
  }
  if (!document.getElementById('when').hidden) {
    query.set('day', document.getElementById('day').value);
    query.set('minute', document.getElementById('minute').value);
  }
  history.replaceState(null, '', '?' + query);
}

// And read back, so a pasted URL reproduces exactly what it recorded.
function restoreFromUrl() {
  const query = new URLSearchParams(location.search);
  for (const id of ['profile', 'algorithm', 'day', 'minute']) {
    const value = query.get(id);
    if (value !== null) { document.getElementById(id).value = value; }
  }
  showWhen();
  showClock();
  const points = ['from', 'to']
    .map(name => query.get(name))
    .filter(Boolean)
    .map(pair => L.latLng(...pair.split(',').map(Number)));
  points.forEach(dropPin);
  if (points.length) {
    map.setView(points[0], points.length === 2 ? 14 : 16);
  }
  if (pins.length === 2) { request(true); }
}

const when = document.getElementById('when');
const day = document.getElementById('day');
const minute = document.getElementById('minute');
const clock = document.getElementById('clock');

// Seconds since Monday 00:00 — the weekly clock the schedules are written on.
function departing() {
  return day.value * 86400 + minute.value * 60;
}

function showClock() {
  const total = Number(minute.value);
  const pad = n => String(n).padStart(2, '0');
  clock.textContent = `${pad(Math.floor(total / 60))}:${pad(total % 60)}`;
}
showClock();

// Only the time-dependent technique reads a departure, so the control appears
// when it is chosen and not before — an input that changes nothing is worse
// than no input at all.
function showWhen() {
  when.hidden = document.getElementById('algorithm').value !== 'timedep';
}
showWhen();

for (const id of ['profile', 'algorithm', 'day']) {
  document.getElementById(id).addEventListener('change', () => {
    showWhen();
    if (pins.length === 2) { request(true); }
  });
}

// Dragging the slider re-routes live, exactly as dragging a pin does, and for
// the same reason: the route is cheap and the search space is not.
minute.addEventListener('input', () => {
  showClock();
  if (pins.length === 2) { space.clearLayers(); trace(); }
});
minute.addEventListener('change', () => {
  if (pins.length === 2) { request(true); }
});

// Last, because it drives everything above it.
restoreFromUrl();

// One request in flight at a time, with the last drag position remembered.
// Leaflet fires `drag` on every mousemove, and queueing sixty of those a second
// would make the route lag further behind the cursor the longer you dragged.
// This instead runs as fast as the server can answer and no faster.
let busy = false, missed = false;
async function trace() {
  if (busy) { missed = true; return; }
  busy = true;
  try {
    await request(false);
  } finally {
    busy = false;
    if (missed) { missed = false; trace(); }
  }
}

async function request(explore) {
  const profile = document.getElementById('profile').value;
  const algorithm = document.getElementById('algorithm').value;
  if (explore) {
    status.className = 'hint';
    status.textContent = 'routing…';
  }

  const query = new URLSearchParams({
    from: at(pins[0]), to: at(pins[1]), profile, algorithm,
    departing: departing()});
  if (!explore) { query.set('explore', '0'); }

  syncUrl();
  const answer = await (await fetch('/route?' + query)).json();
  if (answer.error) {
    status.className = 'hint';
    status.textContent = answer.error;
    route.clearLayers();
    return;
  }

  // The search space first, so the route draws over it. Widths follow the
  // square root of each branch's share of the peak: the trunk carries the whole
  // search and the twigs almost none of it, and a linear scale would render
  // everything but the trunk invisible.
  //
  // `direction` is only present on spaces made of more than one search — a
  // hierarchy's two halves, climbing away from either end — and colouring by it
  // is what makes them tell apart. One colour when it is absent.
  const halves = {forward: '#1d6fa5', backward: '#7a3b9c'};
  if (answer.tree) {
    space.clearLayers();
    L.geoJSON(answer.tree, {
      style: feature => ({
        color: halves[feature.properties.direction] || '#1d6fa5',
        weight: 0.6 + 9 * Math.sqrt(feature.properties.share),
        opacity: 0.8,
      }),
    }).addTo(space);
  }

  route.clearLayers();
  L.polyline(answer.route, {color: '#d1495b', weight: 4}).addTo(route);
  answer.snapped.forEach(p => L.circleMarker(p, {radius: 4, stroke: false,
    fillOpacity: 1, fillColor: '#1d3557'}).addTo(route));

  const minutes = (answer.moving_minutes = ((answer.seconds - answer.waiting) / 60)).toFixed(1);
  const share = (100 * answer.settled_count / answer.graph_nodes).toFixed(1);
  // A ten-hour wait for a gate is not a ten-hour walk, and reporting one total
  // makes it look like one.
  const waited = answer.waiting > 0
    ? ` + <b>${humanise(answer.waiting)}</b> waiting`
    : '';
  status.className = '';
  status.innerHTML =
    `<b>${minutes}</b> min walking${waited} over <b>${answer.legs}</b> legs<br>` +
    `settled <b>${answer.settled_count.toLocaleString()}</b> nodes ` +
    `(${share}% of ${answer.graph_nodes.toLocaleString()}) in <b>${answer.ms}</b> ms` +
    (answer.tree
      ? `<br><span class="hint">tree: ${answer.branch_count.toLocaleString()} branches, ` +
        `${answer.tree.features.length.toLocaleString()} drawn</span>`
      : `<br><span class="hint">drop the pin to draw the search</span>`) +
    scheduleNote(answer);
}

function humanise(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes} min`;
}

// Say when this profile has a schedule that the chosen algorithm is not
// reading. Otherwise the map looks the same at every hour and there is nothing
// to tell you the clock was never consulted.
function scheduleNote(answer) {
  if (!answer.scheduled_edges) { return ''; }
  if (!answer.reads_clock) {
    return `<br><span class="hint">${answer.scheduled_edges} edges here are ` +
           `scheduled; this algorithm ignores them — pick ` +
           `<b>Time-dependent Dijkstra</b> to route with the clock</span>`;
  }
  // Reading the clock and still seeing no change is the confusing case, and it
  // has an ordinary cause: this particular route never touches a scheduled
  // edge, so there is nothing for the hour to change.
  if (!answer.scheduled_legs) {
    return `<br><span class="hint">none of this route is scheduled, so the ` +
           `hour cannot change it (${answer.scheduled_edges} edges in this ` +
           `profile are)</span>`;
  }
  return `<br><span class="hint"><b>${answer.scheduled_legs}</b> of ` +
         `${answer.legs} legs are scheduled</span>`;
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
