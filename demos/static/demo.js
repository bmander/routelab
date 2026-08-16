// The map, the pins, and the board that decides what happens between them.
//
// Everything here is glue. The interesting parts are `board.js`, which is the
// graph, and the server, which evaluates it — this file only carries a query
// from one to the other and draws what comes back.

// Zoom buttons move aside: the readout wants the top-left corner. Canvas, not
// SVG: a search tree is thousands of lines, and thousands of DOM nodes is where
// a map stops being interactive.
const map = L.map('map', {zoomControl: false, preferCanvas: true})
  .setView(SETUP.center, 12);
L.control.zoom({position: 'topright'}).addTo(map);
// A grey basemap, not the standard one: standard OSM tiles draw roads in the
// same oranges and yellows the search tree needs, and the tree disappears into
// the streets it is drawn over.
L.tileLayer('https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, attribution: '© OpenStreetMap, © CARTO'}).addTo(map);

const readout = document.getElementById('readout');
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
  pin.on('drag', () => ask(false));
  pin.on('dragend', () => ask(true));
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
    readout.className = 'hint';
    readout.textContent = 'Now click a destination.';
  } else {
    ask(true);
  }
});

// --- the board ---------------------------------------------------------

const board = new Board(document.getElementById('board'), explore => {
  syncUrl();
  if (pins.length === 2) {
    // A slider still moving asks for the route alone; anything else — a wire,
    // a dropdown, a node let go of — asks for the whole picture.
    if (explore === false) { space.clearLayers(); ask(false); } else { ask(true); }
  }
});

// Wirings worth starting from. Each is one layer, one technique, and whatever
// that technique needs configuring with — the smallest graph that demonstrates
// something, rather than a gallery.
//
// They are starting points and not modes: load one and then pull it apart. The
// pins stay where they are across a change, which is the point of having more
// than one — the same two points, routed six different ways.
const PRESETS = [
  {
    id: 'road-astar',
    label: 'Road · A* with landmarks',
    layer: ['OSM', {profile: 'driving'}],
    technique: ['AStar', {}],
    // A second input, and the node that fills it.
    config: ['heuristic', 'Landmarks', {count: 16}],
  },
  {
    id: 'road-dijkstra',
    label: 'Road · Dijkstra',
    layer: ['OSM', {profile: 'driving'}],
    technique: ['Dijkstra', {}],
  },
  {
    id: 'road-ch',
    label: 'Road · contraction hierarchy',
    layer: ['OSM', {profile: 'driving'}],
    technique: ['ContractionHierarchy', {}],
    config: ['ordering', 'EdgeDifference', {}],
  },
  {
    id: 'walking-clock',
    label: 'Walking · reads the clock',
    layer: ['OSM', {profile: 'walking'}],
    technique: ['TimeDependentDijkstra', {waiting: 'unrestricted'}],
  },
  {
    id: 'transit-dependent',
    label: 'Transit · time-dependent',
    layer: ['GTFS', {}],
    technique: ['TimeDependent', {}],
  },
  {
    id: 'transit-expanded',
    label: 'Transit · time-expanded',
    layer: ['GTFS', {}],
    technique: ['TimeExpanded', {}],
  },
];

/** Lay a preset out on the board, replacing whatever was there.
 *
 * Assembled as a spec and handed to `Board.load`, which is the same door a
 * pasted URL comes through: the reset, the port back-filling and the
 * compatibility check all happen once, in one place, and a preset that names a
 * port wrongly is refused here rather than sent to the server.
 */
function load(preset) {
  const out = type => ports(type, 'out')[0][0];
  const [layer, layerParams] = preset.layer;
  const [technique, techniqueParams] = preset.technique;
  const spec = {
    nodes: [
      {id: 'map', type: 'Map', x: 20, y: 10},
      {id: 'layer', type: layer, x: 232, y: 10, params: layerParams},
      {id: 'env', type: 'Environment', x: 444, y: 10},
      {id: 'technique', type: technique, x: 656, y: 10, params: techniqueParams},
      {id: 'query', type: 'Query', x: 868, y: 10},
    ],
    links: [
      {from: 'layer', fromPort: out(layer), to: 'env', toPort: 'layers'},
      {from: 'env', fromPort: 'environment', to: 'technique', toPort: 'environment'},
      {from: 'technique', fromPort: 'planner', to: 'query', toPort: 'planner'},
      // The map, on both sides of it.
      {from: 'map', fromPort: 'origin', to: 'query', toPort: 'origin'},
      {from: 'map', fromPort: 'destination', to: 'query', toPort: 'destination'},
      {from: 'query', fromPort: 'route', to: 'map', toPort: 'route'},
      {from: 'query', fromPort: 'space', to: 'map', toPort: 'space'},
    ],
  };
  if (preset.config) {
    const [port, type, params] = preset.config;
    spec.nodes.push({id: 'config', type, x: 232, y: 148, params});
    spec.links.push({from: 'config', fromPort: out(type), to: 'technique', toPort: port});
  }
  board.load(spec);
}

/** Can this demo offer this preset, given what it was started with?
 *
 * Asked of the node types it uses, which already declare it — one answer, and
 * a preset cannot claim to work when a node it needs is greyed out.
 */
function available(preset) {
  return [preset.layer[0], preset.technique[0], preset.config && preset.config[1]]
    .filter(Boolean)
    .every(type => !TYPES[type].available || TYPES[type].available());
}

/** What the demo opens with, and what `reset` goes back to. */
function defaultGraph() {
  load(PRESETS[0]);
}

// --- add-node menu -----------------------------------------------------

const menu = document.getElementById('menu');
// Cascades new nodes so a second one does not land on the first.
let dropped = 0;
document.getElementById('add').addEventListener('click', event => {
  event.stopPropagation();
  menu.textContent = '';
  const groups = {};
  for (const [type, spec] of Object.entries(TYPES)) {
    (groups[spec.group] ||= []).push([type, spec]);
  }
  for (const [group, entries] of Object.entries(groups)) {
    const heading = document.createElement('div');
    heading.className = 'group';
    heading.textContent = group;
    menu.append(heading);
    for (const [type, spec] of entries) {
      const button = document.createElement('button');
      button.textContent = spec.title;
      if (spec.available && !spec.available()) {
        button.disabled = true;
        button.title = 'start the demo with --gtfs to offer this';
      }
      button.addEventListener('click', () => {
        menu.hidden = true;
        // Into the middle of whatever is in view, cascaded so that adding two
        // in a row does not stack the second exactly on the first.
        const box = board.viewport.getBoundingClientRect();
        board.add(type,
          (box.width / 2 - board.pan.x) / board.zoom - 88 + dropped * 26,
          (box.height / 2 - board.pan.y) / board.zoom - 40 + dropped * 22);
        dropped = (dropped + 1) % 5;
        syncUrl();
      });
      menu.append(button);
    }
  }
  const anchor = event.target.getBoundingClientRect();
  const board_box = document.getElementById('board').getBoundingClientRect();
  menu.style.left = (anchor.left - board_box.left - 60) + 'px';
  menu.style.top = (anchor.bottom - board_box.top + 4) + 'px';
  menu.hidden = false;
});
document.addEventListener('mousedown', event => {
  if (!menu.hidden && !menu.contains(event.target)) { menu.hidden = true; }
});

document.getElementById('reset').addEventListener('click', () => {
  defaultGraph();
  syncUrl();
  if (pins.length === 2) { ask(true); }
});

// A menu of starting points, not a statement about what is on the board — pull
// a preset apart and the board is no longer that preset, and a dropdown still
// naming it would be lying. So it reads "load a preset" and goes back to saying
// that as soon as it has loaded one.
const presets = document.getElementById('preset');
for (const preset of PRESETS) {
  if (!available(preset)) { continue; }
  presets.append(new Option(preset.label, preset.id));
}
presets.addEventListener('change', () => {
  const preset = PRESETS.find(entry => entry.id === presets.value);
  presets.value = '';
  if (!preset) { return; }
  load(preset);
  syncUrl();
  if (pins.length === 2) { ask(true); }
});

// A node is deleted with the keyboard, the way it is in every editor that has
// nodes. Selection is the last node whose header was pressed.
let selected = null;
document.getElementById('nodes').addEventListener('mousedown', event => {
  const node = event.target.closest('.node');
  document.querySelectorAll('.node.selected').forEach(n => n.classList.remove('selected'));
  selected = node ? node.dataset.id : null;
  if (node) { node.classList.add('selected'); }
});
window.addEventListener('keydown', event => {
  if ((event.key === 'Delete' || event.key === 'Backspace') && selected
      && !['INPUT', 'SELECT'].includes(document.activeElement.tagName)) {
    board.remove(selected);
    selected = null;
  }
});

// --- resizing the board ------------------------------------------------

const grip = document.getElementById('grip');
grip.addEventListener('mousedown', event => {
  event.preventDefault();
  const panel = document.getElementById('board');
  const start = event.clientY, height = panel.getBoundingClientRect().height;
  const move = move_event => {
    panel.style.flexBasis =
      Math.max(60, Math.min(window.innerHeight - 120, height - (move_event.clientY - start)))
      + 'px';
    map.invalidateSize();
  };
  const stop = () => {
    window.removeEventListener('mousemove', move);
    window.removeEventListener('mouseup', stop);
  };
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', stop);
});

// --- URL state ---------------------------------------------------------

// Everything that decides a query lives in the URL, so a result can be copied
// and handed to someone else — the graph included, which is what makes "here is
// the wiring that reproduces it" a link rather than a paragraph.
// `replaceState`, not `pushState`: dragging a pin would otherwise fill the back
// button with a hundred near-identical steps.
function syncUrl(spec) {
  const query = new URLSearchParams({board: spec || JSON.stringify(board.serialise())});
  if (pins.length === 2) {
    query.set('from', at(pins[0]));
    query.set('to', at(pins[1]));
  }
  history.replaceState(null, '', '?' + query);
}

function restoreFromUrl() {
  const query = new URLSearchParams(location.search);
  const spec = query.get('board');
  let loaded = false;
  if (spec) {
    try { board.load(JSON.parse(spec)); loaded = board.nodes.size > 0; }
    catch (error) { console.warn('could not read the board from the URL', error); }
  }
  if (!loaded) { defaultGraph(); }

  const points = ['from', 'to']
    .map(name => query.get(name))
    .filter(Boolean)
    .map(pair => L.latLng(...pair.split(',').map(Number)));
  points.forEach(dropPin);
  if (points.length) {
    map.setView(points[0], points.length === 2 ? 14 : 16);
  }
  if (pins.length === 2) { ask(true); }
}

// --- what is already built ---------------------------------------------

// Signatures the server has answered for. It caches what it builds under the
// same idea of what a node *is*, so anything in here costs nothing to ask for
// again and anything absent may cost seconds — which is exactly the difference
// worth showing on the board.
//
// It starts empty on every load, and the server fills it in: each answer names
// the nodes it evaluated, cache hit or not, so a reloaded page learns on its
// first query that the city it is looking at was read some time ago.
const built = new Set();

// Only shown after a moment. Most rebuilds are cache hits that come back in
// single-digit milliseconds, and a spinner that appears and vanishes inside one
// frame reads as a flicker rather than as work.
const SETTLE = 150;
let pending = null;

/** What every node upstream of the query is, right now. The board's record of
 * what has been built is keyed by these, and a request carries the ones it was
 * sent with so the answer is filed under the right board. */
function signatures() {
  const query = [...board.nodes.values()].find(node => node.type === 'Query');
  const planner = query && board.sources(query.id, 'planner')[0];
  const taken = new Map();
  if (planner) {
    for (const id of board.upstream(planner)) { taken.set(id, board.signature(id)); }
  }
  return taken;
}

/** Spin whatever this request will have to build, once it is clear it is slow.
 *
 * Two different reasons to spin, and both are true. A node upstream of the
 * query spins when this board has never seen an answer for what it currently
 * is — that is the rebuild, and it may cost seconds. The query itself spins
 * whenever it is waiting, because unlike everything else it is never cached:
 * the search runs again every time.
 */
function markStale(sent) {
  const query = [...board.nodes.values()].find(node => node.type === 'Query');
  if (!query) { return; }
  const needed = [...sent].filter(([, spelling]) => !built.has(spelling)).map(([id]) => id);
  clearTimeout(pending);
  pending = setTimeout(() => {
    [...needed, query.id].forEach(id => board.busy(id, true));
    watch();
  }, SETTLE);
}

/** Record what came back, and let the board go still.
 *
 * `sent` is what each node's signature was when the request left, not what it
 * is now. Rewire while an answer is in flight and the two differ — and stamping
 * the server's "I built these" onto today's board would record a signature for
 * something nobody built, so the next genuinely expensive rebuild would show no
 * spinner. Which is the one thing the spinners exist to rule out.
 */
function settle(answer, sent) {
  clearTimeout(pending);
  clearInterval(watching);
  watching = null;
  for (const id of Object.keys(answer.built || {})) {
    if (sent.has(id)) { built.add(sent.get(id)); }
  }
  board.idle();
}

// While something is building, ask what it is doing. The server answers from a
// different request entirely — it can, because what the work writes into is an
// atomic counter rather than something it has to stop to report.
let watching = null;
function watch() {
  clearInterval(watching);
  watching = setInterval(async () => {
    let jobs;
    try { jobs = await (await fetch('/progress')).json(); }
    catch (error) { return; }
    for (const node of board.layer.querySelectorAll('.node.busy')) {
      board.report(node.dataset.id, jobs[node.dataset.id] || null);
    }
    // Builds nest — an AStar is waiting on an Environment which is waiting on
    // an OSM read — so the one that has been running longest is the outermost
    // and the least informative. Name the innermost instead: the one that can
    // count itself, or failing that the one that started most recently.
    // Started last means innermost, and it is the only exact test: the three
    // of them begin in the same instant, so elapsed cannot tell them apart.
    const running = Object.values(jobs).sort((a, b) => b.seq - a.seq);
    const doing = running.find(job => job.fraction !== null) || running[0];
    if (doing) { readout.innerHTML = building(doing); }
  }, 400);
}

/** What to say about the longest-running job, in the readout. */
function building(job) {
  // The count, not just the percentage. Contraction spends a third of its time
  // in the last half-percent of nodes, and a bar apparently stuck at 100% is
  // the exact thing this was added to rule out — the number underneath it is
  // visibly still climbing.
  const detail = job.total
    ? `${job.phase} <b>${job.done.toLocaleString()}</b> of ` +
      `${job.total.toLocaleString()}`
    : 'no count to report — this step cannot say how far along it is';
  return `building <b>${job.what}</b><br>${detail}` +
         `<br><span class="hint">${job.elapsed.toFixed(0)}s so far</span>`;
}

// --- asking ------------------------------------------------------------

// One request in flight at a time, and one waiting behind it at most.
//
// Leaflet fires `drag` on every mousemove, and queueing sixty of those a second
// would make the route lag further behind the cursor the longer you dragged. So
// only the newest pending ask survives — but **exploring is sticky**, and that
// is the part worth stating, because getting it wrong is what made the search
// tree stop appearing.
//
// A drag that lands while a request is in flight leaves an ask waiting. Let go
// of the pin and `dragend` asks for the whole picture, tree included. If the
// waiting route-only ask were allowed to replace that, or to run after it, the
// tree would be requested and then immediately thrown away — the route would
// keep working perfectly, which is exactly what made it puzzling.
let inFlight = false, waiting = null;

function ask(explore) {
  waiting = {explore: explore || (waiting !== null && waiting.explore)};
  if (!inFlight) { drain(); }
}

async function drain() {
  inFlight = true;
  try {
    while (waiting !== null) {
      const {explore} = waiting;
      waiting = null;
      await request(explore);
    }
  } finally {
    inFlight = false;
  }
}

async function request(explore) {
  if (pins.length !== 2) { return; }
  if (explore) {
    readout.className = 'hint';
    readout.textContent = 'routing…';
  }
  const sent = signatures();
  markStale(sent);

  const spec = JSON.stringify(board.serialise());
  const query = new URLSearchParams({from: at(pins[0]), to: at(pins[1]), board: spec});
  if (!explore) { query.set('explore', '0'); }

  // Recorded when the drag stops, not on every frame of it. Safari throws past
  // roughly a hundred `replaceState` calls in thirty seconds, and a hierarchy
  // answering in a millisecond makes a hundred of them in two seconds.
  if (explore) { syncUrl(spec); }
  let answer;
  try {
    answer = await (await fetch('/route?' + query)).json();
  } catch (error) {
    // Anything thrown here used to leave the board spinning for ever with no
    // word of why, which is precisely the state the spinners exist to rule out.
    settle({}, sent);
    readout.className = 'hint';
    readout.textContent = `the server did not answer: ${error}`;
    return;
  }
  settle(answer, sent);
  board.blame(answer.node);
  if (answer.error) {
    readout.className = 'hint';
    readout.textContent = answer.error;
    route.clearLayers();
    // Where it snapped, even though it found nothing: "no journey from this
    // stop at this hour" is only useful if you can see which stop it means.
    (answer.snapped || []).forEach(p => L.circleMarker(p, {radius: 4, stroke: false,
      fillOpacity: 1, fillColor: '#1d3557'}).addTo(route));
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
  //
  // Cleared only when this request asked for a tree: an answer with none is
  // either a route-only refresh mid-drag, which must leave what is drawn
  // alone, or a query whose `space` output goes nowhere, which must not.
  const halves = {forward: '#1d6fa5', backward: '#7a3b9c'};
  if (explore) { space.clearLayers(); }
  if (answer.tree) {
    L.geoJSON(answer.tree, {
      style: feature => ({
        color: halves[feature.properties.direction] || '#1d6fa5',
        weight: 0.6 + 9 * Math.sqrt(feature.properties.share),
        opacity: 0.8,
      }),
    }).addTo(space);
  }

  // Only if something is listening for it. A wire that changes nothing is not
  // a wire, so unplugging `route` from the map really does stop drawing it —
  // and the readout still reports what the search cost, because the query was
  // asked either way.
  route.clearLayers();
  if (answer.drawn) {
    L.polyline(answer.route, {color: '#d1495b', weight: 4}).addTo(route);
  }
  answer.snapped.forEach(p => L.circleMarker(p, {radius: 4, stroke: false,
    fillOpacity: 1, fillColor: '#1d3557'}).addTo(route));

  const minutes = ((answer.seconds - answer.waiting) / 60).toFixed(1);
  const share = (100 * answer.settled_count / answer.graph_nodes).toFixed(1);
  // A ten-hour wait for a gate is not a ten-hour walk, and reporting one total
  // makes it look like one.
  const waited = answer.waiting > 0
    ? ` + <b>${humanise(answer.waiting)}</b> waiting`
    : '';
  // What the first line says depends on what was actually asked. A transit
  // answer's headline is when you get there and how many times you change; a
  // street answer's is how long it takes. The second line is the same question
  // for both — how much of the graph the search had to settle — which is the
  // number the two timetable models exist to be compared on.
  const headline = answer.transit
    ? `arrive <b>${hhmm(answer.arrives)}</b>, <b>${minutes}</b> min riding${waited}<br>` +
      `<b>${answer.transfers}</b> transfer${answer.transfers === 1 ? '' : 's'} ` +
      `over <b>${answer.legs}</b> stops`
    : `<b>${minutes}</b> min moving${waited} over <b>${answer.legs}</b> legs`;

  readout.className = '';
  readout.innerHTML =
    `${headline}<br>` +
    `settled <b>${answer.settled_count.toLocaleString()}</b> ${answer.nodes_are || 'nodes'} ` +
    `(${share}% of ${answer.graph_nodes.toLocaleString()}) in <b>${answer.ms}</b> ms` +
    spaceNote(answer) + scheduleNote(answer);
}

// What the query's second output did, or why it did nothing.
function spaceNote(answer) {
  if (answer.transit) {
    return `<br><span class="hint">a timetable query answers with an itinerary, ` +
           `so there is no search space to draw</span>`;
  }
  if (answer.tree) {
    return `<br><span class="hint">tree: ${answer.branch_count.toLocaleString()} ` +
           `branches, ${answer.tree.features.length.toLocaleString()} drawn</span>`;
  }
  if (!board.links.some(l => l.fromPort === 'space')) {
    return `<br><span class="hint">nothing is wired to the query's ` +
           `<b>space</b> output, so no search space was built</span>`;
  }
  return `<br><span class="hint">drop the pin to draw the search</span>`;
}

function humanise(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes} min`;
}

// A service-day clock, which runs past 24:00 rather than wrapping: a bus that
// leaves at 23:50 and arrives at 24:10 made one trip, not one that landed
// before it left.
function hhmm(seconds) {
  const pad = n => String(n).padStart(2, '0');
  return `${pad(Math.floor(seconds / 3600))}:${pad(Math.floor(seconds % 3600 / 60))}`;
}

// Say when this layer has a schedule that the wired-up technique is not
// reading. Otherwise the map looks the same at every hour and there is nothing
// to tell you the clock was never consulted.
function scheduleNote(answer) {
  if (answer.transit || !answer.scheduled_edges) { return ''; }
  if (!answer.reads_clock) {
    return `<br><span class="hint">${answer.scheduled_edges} edges here are ` +
           `scheduled; this technique ignores them — wire in ` +
           `<b>TimeDependentDijkstra</b> to route with the clock</span>`;
  }
  // Reading the clock and still seeing no change is the confusing case, and it
  // has an ordinary cause: this particular route never touches a scheduled
  // edge, so there is nothing for the hour to change.
  if (!answer.scheduled_legs) {
    return `<br><span class="hint">none of this route is scheduled, so the ` +
           `hour cannot change it (${answer.scheduled_edges} edges in this ` +
           `layer are)</span>`;
  }
  return `<br><span class="hint"><b>${answer.scheduled_legs}</b> of ` +
         `${answer.legs} legs are scheduled</span>`;
}

// Last, because it drives everything above it.
restoreFromUrl();
