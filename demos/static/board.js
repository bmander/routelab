// A node board, in the manner of a shader editor.
//
// The graph here is not a picture of the query — it *is* the query. It
// serialises to the JSON the server evaluates, so unplugging a wire really does
// remove an argument, and the error you get back is the one the library would
// have raised if you had written the call by hand.
//
// The vocabulary is the library's own three steps and nothing else: layers make
// an `Environment`, a technique binds to one and becomes a planner, and a
// `Query` asks it something. `TYPES` below is the whole language.

// What each node looks like and what it produces. `inputs` and `outputs` name
// each socket and the kind of value it carries, and those two tables are all
// the wiring rules there are — a port takes what it says it takes, which is the
// same contract `Planner.accepts` states one level down. `kind` is what a node
// is *for*, which is only used to colour its header.
//
// A node with one output does not spell it out: it produces one value of its
// own kind, on a socket named after that kind. Only `Query` and `Map` have more
// than one, and they say so.
// How each node looks. The *wiring* — what each port accepts, what a node
// produces, what depends on what — is not here: it arrives from the server in
// `SETUP.nodes`, because it is one fact and the board's whole claim is that
// what it draws is what the server does. A second copy could drift, and a
// drifted copy fails silently: the client would predict rebuilds the server
// never does, or miss ones it does.
const LOOKS = {
  OSM: {
    title: 'OSM', group: 'layers',
    sub: () => SETUP.extract,
    fields: [{id: 'profile', label: 'profile', type: 'select',
              options: () => SETUP.profiles, value: 'driving'}],
  },
  GTFS: {
    title: 'GTFS', group: 'layers',
    sub: () => SETUP.feed,
    available: () => Boolean(SETUP.feed),
    fields: [{id: '_date', label: 'service day', type: 'note',
              text: () => SETUP.date}],
  },
  Footpaths: {
    title: 'Footpaths', group: 'layers',
    sub: () => 'walks between stops',
    available: () => Boolean(SETUP.feed),
    fields: [{id: 'within', label: 'within (m)', type: 'number',
              value: 200, min: 10, max: 2000}],
  },
  Environment: {title: 'Environment', group: 'environment'},
  Euclidean: {title: 'Euclidean', group: 'heuristics'},
  Landmarks: {
    title: 'Landmarks', group: 'heuristics',
    fields: [{id: 'count', label: 'landmarks', type: 'number',
              value: 16, min: 1, max: 64}],
  },
  EdgeDifference: {title: 'EdgeDifference', group: 'orderings'},
  RandomOrder: {
    title: 'RandomOrder', group: 'orderings',
    fields: [{id: 'seed', label: 'seed', type: 'number', value: 0, min: 0, max: 999}],
  },
  Dijkstra: {title: 'Dijkstra', group: 'techniques'},
  BFS: {title: 'BFS', group: 'techniques', sub: () => 'fewest hops'},
  AStar: {title: 'AStar', group: 'techniques'},
  ContractionHierarchy: {title: 'ContractionHierarchy', group: 'techniques'},
  TimeDependentDijkstra: {
    title: 'TimeDependentDijkstra', group: 'techniques',
    fields: [{id: 'waiting', label: 'waiting', type: 'select',
              options: () => ['unrestricted', 'forbidden'], value: 'unrestricted'}],
  },
  TimeDependent: {
    title: 'TimeDependent', group: 'techniques',
    sub: () => 'a node per stop',
    available: () => Boolean(SETUP.feed),
  },
  TimeExpanded: {
    title: 'TimeExpanded', group: 'techniques',
    sub: () => 'a node per event',
    available: () => Boolean(SETUP.feed),
  },
  Query: {
    title: 'Query', group: 'query', sub: () => 'route(a, b)',
    fields: [
      {id: 'day', label: 'weekday', type: 'select', value: '0',
       options: () => [['0', 'Mon'], ['1', 'Tue'], ['2', 'Wed'], ['3', 'Thu'],
                       ['4', 'Fri'], ['5', 'Sat'], ['6', 'Sun']]},
      {id: 'minute', label: 'departing', type: 'time', value: 480},
    ],
  },
  Map: {title: 'Map', group: 'map', sub: () => 'click to place'},
};

// Wiring from the server, looks from here. One entry per node type, and a type
// the server does not know about cannot be drawn — which is the point.
const TYPES = Object.fromEntries(
  Object.entries(SETUP.nodes).map(([name, wiring]) => [name, {...wiring, ...LOOKS[name]}]));

/** What a node's settings start as, before anything has been chosen. */
function defaults(type) {
  return Object.fromEntries((TYPES[type].fields || [])
    .filter(field => field.type !== 'note')
    .map(field => [field.id, field.value]));
}

/** A node's sockets, in the order they are drawn: inputs first, then outputs. */
function ports(type, direction) {
  const spec = TYPES[type];
  if (direction === 'in') { return Object.entries(spec.inputs || {}); }
  return Object.entries(spec.outputs || {[spec.kind]: spec.kind});
}

// One colour per kind, shared by a port's dot and the wire leaving it — which
// is what lets you see at a glance that a heuristic cannot go where an
// environment belongs.
// `route` and `space` are deliberately the colours those things are drawn in on
// the map above, so a wire and what comes out of it are the same colour.
const COLOURS = {
  layer: '#8ec07c', environment: '#83a9d1', heuristic: '#d5a05a',
  ordering: '#d5a05a', planner: '#c08ac4',
  point: '#7ec8c8', route: '#d1495b', space: '#1d6fa5',
};

// Node geometry. Wires are drawn from these, and the stylesheet lays nodes out
// from them too — handed over as custom properties rather than restated there,
// because a padding changed in one place and not the other detaches every wire
// from its dot and nothing catches it.
const HEADER = 26, ROW = 22, WIDTH = 176, BODY_TOP = 4;

class Board {
  constructor(host, onchange) {
    this.onchange = onchange;
    this.nodes = new Map();
    this.links = [];
    this.pan = {x: 40, y: 16};
    this.zoom = 1;
    this.next = 1;

    host.style.setProperty('--node-width', `${WIDTH}px`);
    host.style.setProperty('--node-header', `${HEADER}px`);
    host.style.setProperty('--port-row', `${ROW}px`);
    host.style.setProperty('--body-top', `${BODY_TOP}px`);

    this.layer = host.querySelector('#nodes');
    this.wires = host.querySelector('#wiregroup');
    this.viewport = host.querySelector('#viewport');
    this.dragging = null;

    this.viewport.addEventListener('mousedown', event => this.onDown(event));
    window.addEventListener('mousemove', event => this.onMove(event));
    window.addEventListener('mouseup', event => this.onUp(event));
    this.viewport.addEventListener('wheel', event => this.onWheel(event), {passive: false});
  }

  // --- the model ------------------------------------------------------

  add(type, x, y) {
    const id = 'n' + this.next++;
    this.nodes.set(id, {id, type, params: defaults(type), x, y});
    this.draw();
    return id;
  }

  remove(id) {
    this.nodes.delete(id);
    this.links = this.links.filter(l => l.from !== id && l.to !== id);
    this.draw();
    this.onchange(true);
  }

  connect(from, fromPort, to, toPort) {
    if (!this.compatible(from, fromPort, to, toPort)) { return false; }
    // One wire per socket, except a bag of layers: an `Environment` takes as
    // many as you give it, and everything else takes one argument — including
    // a layer that takes a layer, like `Footpaths`. Even then a second wire
    // from the same source replaces rather than duplicates.
    const target = TYPES[this.nodes.get(to).type];
    const many = target.kind === 'environment' && target.inputs[toPort] === 'layer';
    this.links = this.links.filter(l =>
      !(l.to === to && l.toPort === toPort && (!many || l.from === from)));
    this.links.push({from, fromPort, to, toPort});
    this.draw();
    this.onchange(true);
    return true;
  }

  compatible(from, fromPort, to, toPort) {
    const source = this.nodes.get(from), target = this.nodes.get(to);
    if (!source || !target || from === to) { return false; }
    const gives = Object.fromEntries(ports(source.type, 'out'))[fromPort];
    const wants = Object.fromEntries(ports(target.type, 'in'))[toPort];
    return Boolean(gives) && gives === wants && !this.reaches(to, from);
  }

  /** Is `goal` downstream of `start`? The check that keeps a graph a graph.
   *
   * Stops at a terminal node, whose outputs owe nothing to its inputs. Without
   * that the obvious wiring — the map hands the query two points and gets a
   * route back — would look like a loop and be refused.
   */
  reaches(start, goal) {
    if (start === goal) { return true; }
    if (TYPES[this.nodes.get(start).type].terminal) { return false; }
    return this.links.filter(l => l.from === start)
      .some(l => this.reaches(l.to, goal));
  }

  serialise() {
    return {
      nodes: [...this.nodes.values()].map(n =>
        ({id: n.id, type: n.type, params: n.params, x: Math.round(n.x), y: Math.round(n.y)})),
      links: this.links.map(l =>
        ({from: l.from, fromPort: l.fromPort, to: l.to, toPort: l.toPort})),
    };
  }

  load(spec) {
    this.nodes = new Map();
    this.links = [];
    for (const node of spec.nodes || []) {
      if (!TYPES[node.type]) { continue; }
      // Defaults under whatever was saved, so a board written before a field
      // existed still opens with that field set to something.
      this.nodes.set(node.id, {...node, params: {...defaults(node.type), ...node.params}});
      this.next = Math.max(this.next, Number(String(node.id).slice(1)) + 1 || 1);
    }
    this.links = (spec.links || [])
      .filter(l => this.nodes.has(l.from) && this.nodes.has(l.to))
      // A link saved before nodes had more than one output named only its
      // target socket. There was one source socket then, so it is not
      // ambiguous — fill it in rather than drop the wire.
      .map(l => ({
        from: l.from,
        fromPort: l.fromPort || ports(this.nodes.get(l.from).type, 'out')[0][0],
        to: l.to,
        toPort: l.toPort || l.port,
      }))
      .filter(l => this.compatible(l.from, l.fromPort, l.to, l.toPort));
    this.draw();
  }

  // --- identity -------------------------------------------------------

  /** What a node is, as a string: its type, its settings, and everything
   * upstream of it.
   *
   * The server caches what it builds under the same idea, so two boards that
   * would produce the same object spell the same and the second costs nothing.
   * The spelling itself does not have to match the server's — this is only ever
   * compared against other strings from here — but the *rule* does, or the
   * board would promise a rebuild that never happens, or hide one that does.
   *
   * Only argument ports count. A node with more than one output is never
   * upstream of anything, so which socket a wire left by cannot change what
   * gets built. And a terminal node's inputs are not arguments at all — what
   * comes out of the map does not depend on what is drawn on it — so they are
   * skipped, which is also what stops Map -> Query -> Map recurring forever.
   */
  signature(id) {
    const node = this.nodes.get(id);
    const params = Object.keys(node.params || {}).sort()
      .map(key => `${key}=${JSON.stringify(node.params[key])}`);
    const wired = TYPES[node.type].terminal ? [] :
      ports(node.type, 'in').map(([port]) => port).sort()
        .map(port =>
          `${port}=[${this.sources(id, port).map(s => this.signature(s)).join(',')}]`);
    return `${node.type}(${[...params, ...wired].join(', ')})`;
  }

  /** Ids feeding one input port, in the order they were wired. */
  sources(id, port) {
    return this.links.filter(l => l.to === id && l.toPort === port).map(l => l.from);
  }

  /** Every node upstream of `id` through argument ports, `id` included.
   *
   * What a query actually depends on — which is not the whole board. A
   * technique parked to one side with nothing plugged into the query is never
   * built, and should not be shown waiting to be.
   */
  upstream(id, found = new Set()) {
    if (found.has(id)) { return found; }
    found.add(id);
    if (TYPES[this.nodes.get(id).type].terminal) { return found; }
    for (const [port] of ports(this.nodes.get(id).type, 'in')) {
      for (const source of this.sources(id, port)) { this.upstream(source, found); }
    }
    return found;
  }

  /** Show a node as working, and stop it being changed while it is.
   *
   * Both halves matter: a spinner says the answer on screen is not this node's
   * yet, and a live dropdown would invite a change whose result would arrive
   * out of order with the one already in flight.
   */
  busy(id, working) {
    const element = this.layer.querySelector(`[data-id="${id}"]`);
    if (!element) { return; }
    element.classList.toggle('busy', working);
    element.querySelectorAll('select, input').forEach(field => {
      field.disabled = working;
    });
    if (!working) { this.report(id, null); }
  }

  /** Say how a working node is getting on, or clear it.
   *
   * The seconds tick whether or not there is a fraction, and they are the part
   * that actually answers "is this hung" — contraction's percentage can sit at
   * 99.9% for a minute while the count behind it climbs, because the last
   * nodes it contracts are its most connected.
   */
  report(id, job) {
    const element = this.layer.querySelector(`[data-id="${id}"]`);
    if (!element) { return; }
    const tick = element.querySelector('.tick');
    const strip = element.querySelector('.progress');
    if (!job) {
      tick.textContent = '';
      element.classList.remove('measured');
      return;
    }
    tick.textContent = `${job.elapsed.toFixed(0)}s`;
    if (job.fraction === null || job.fraction === undefined) {
      element.classList.remove('measured');
      return;
    }
    element.classList.add('measured');
    strip.querySelector('i').style.width = `${(job.fraction * 100).toFixed(2)}%`;
    // Two decimals near the end, because one would read as a stuck 100%.
    const percent = job.fraction > 0.99
      ? (job.fraction * 100).toFixed(2) : (job.fraction * 100).toFixed(0);
    strip.querySelector('.phase').textContent = `${job.phase} ${percent}%`;
  }

  /** Nothing is working any more. */
  idle() {
    this.layer.querySelectorAll('.busy')
      .forEach(node => this.busy(node.dataset.id, false));
  }

  // --- geometry -------------------------------------------------------

  /** Where a port's dot sits, in board coordinates. */
  socket(id, direction, port) {
    const node = this.nodes.get(id);
    return {
      x: node.x + (direction === 'in' ? 0 : WIDTH),
      y: node.y + HEADER + BODY_TOP
         + this.portIndex(node.type, direction, port) * ROW + ROW / 2,
    };
  }

  /** A port's row within its node — inputs first, then outputs. */
  portIndex(type, direction, port) {
    const inputs = ports(type, 'in');
    if (direction === 'in') { return inputs.findIndex(([name]) => name === port); }
    return inputs.length + ports(type, 'out').findIndex(([name]) => name === port);
  }

  // --- rendering ------------------------------------------------------

  draw() {
    this.layer.style.transform =
      `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`;
    this.wires.setAttribute('transform',
      `translate(${this.pan.x}, ${this.pan.y}) scale(${this.zoom})`);
    this.drawNodes();
    this.drawWires();
  }

  drawNodes() {
    const seen = new Set();
    for (const node of this.nodes.values()) {
      seen.add(node.id);
      let element = this.layer.querySelector(`[data-id="${node.id}"]`);
      if (!element) {
        element = this.renderNode(node);
        this.layer.append(element);
      }
      element.style.left = node.x + 'px';
      element.style.top = node.y + 'px';
    }
    for (const element of [...this.layer.children]) {
      if (!seen.has(element.dataset.id)) { element.remove(); }
    }
  }

  renderNode(node) {
    const spec = TYPES[node.type];
    const element = document.createElement('div');
    element.className = 'node';
    element.dataset.id = node.id;
    element.dataset.kind = spec.kind;

    const header = document.createElement('header');
    header.innerHTML = `<span>${spec.title}</span>` +
      (spec.sub ? `<span class="sub">${spec.sub()}</span>` : '') +
      `<span class="tick"></span>`;
    element.append(header);

    // Shown only while this node is working and only when the work can say how
    // far it has got. A bar that appears for everything would have to invent a
    // number for the things that cannot count themselves.
    const strip = document.createElement('div');
    strip.className = 'progress';
    strip.innerHTML = `<div class="bar"><i></i></div><span class="phase"></span>`;
    element.append(strip);

    const body = document.createElement('div');
    body.className = 'body';
    for (const [port, kind] of ports(node.type, 'in')) {
      body.append(this.renderPort(node.id, 'in', port, kind));
    }
    for (const [port, kind] of ports(node.type, 'out')) {
      body.append(this.renderPort(node.id, 'out', port, kind));
    }
    for (const field of spec.fields || []) {
      body.append(this.renderField(node, field));
    }
    element.append(body);
    return element;
  }

  renderPort(id, direction, port, kind) {
    const row = document.createElement('div');
    row.className = `port ${direction}`;
    row.dataset.port = port;
    row.dataset.direction = direction;
    row.dataset.kind = kind;
    row.dataset.node = id;
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = COLOURS[kind];
    // Named, not typed. For a node with one output the two are the same word,
    // which is why it read fine until a node had two of the same kind and the
    // map grew a pair of sockets both labelled "point".
    row.append(dot, document.createTextNode(port));
    return row;
  }

  renderField(node, field) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.dataset.field = field.id;
    if (field.type === 'note') {
      wrap.innerHTML = `<label>${field.label}</label><span class="note">${field.text()}</span>`;
      return wrap;
    }

    const label = document.createElement('label');
    label.textContent = field.label;
    wrap.append(label);

    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      for (const option of field.options()) {
        const [value, text] = Array.isArray(option) ? option : [option, option];
        input.append(new Option(text, value));
      }
      input.value = node.params[field.id];
    } else if (field.type === 'time') {
      // A clock, because 08:30 is what somebody means and 510 is not.
      const clock = document.createElement('span');
      clock.className = 'clock';
      input = document.createElement('input');
      Object.assign(input, {type: 'range', min: 0, max: 1435, step: 5});
      input.value = node.params[field.id];
      const show = () => {
        const pad = n => String(n).padStart(2, '0');
        clock.textContent = `${pad(Math.floor(input.value / 60))}:${pad(input.value % 60)}`;
      };
      show();
      input.addEventListener('input', show);
      label.append(clock);
    } else {
      input = document.createElement('input');
      Object.assign(input, {type: 'number', min: field.min, max: field.max});
      input.value = node.params[field.id];
    }

    const read = () => {
      node.params[field.id] = field.type === 'number' || field.type === 'time'
        ? Number(input.value) : input.value;
    };
    // A slider wants both events: `input` fires continuously and re-routes
    // live, `change` fires when you let go and is when the expensive picture is
    // worth redrawing. Nothing else does. A dropdown answers `input` and
    // `change` for one choice, and when that choice is a profile the first of
    // them starts a six-second read of an extract the second immediately
    // starts again.
    if (field.type === 'time') {
      input.addEventListener('input', () => { read(); this.onchange(false); });
    }
    input.addEventListener('change', () => { read(); this.onchange(true); });
    // Otherwise a click on a select starts dragging the node under it.
    input.addEventListener('mousedown', event => event.stopPropagation());
    wrap.append(input);
    return wrap;
  }

  drawWires() {
    this.wires.textContent = '';
    for (const link of this.links) {
      const from = this.nodes.get(link.from);
      const a = this.socket(link.from, 'out', link.fromPort);
      const b = this.socket(link.to, 'in', link.toPort);
      const kind = Object.fromEntries(ports(from.type, 'out'))[link.fromPort];
      this.wires.append(...this.wire(a, b, COLOURS[kind], link));
    }
    if (this.dragging && this.dragging.what === 'wire') {
      const {from, kind, cursor, origin} = this.dragging;
      // Drawn the way it will look once it lands: a wire dragged leftwards from
      // an output is a backward wire even before it has anywhere to go.
      const [head, tail] = origin.direction === 'out' ? [from, cursor] : [cursor, from];
      this.wires.append(...this.wire(head, tail, COLOURS[kind]));
    }
  }

  /** A bezier, plus a fat invisible copy of it that is easy to click. */
  wire(a, b, colour, link) {
    const d = this.path(a, b);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.setAttribute('d', d);
    line.setAttribute('class', link ? 'wire' : 'wire dangling');
    line.setAttribute('stroke', colour);

    if (!link) { return [line]; }
    const hit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hit.setAttribute('d', d);
    hit.setAttribute('class', 'wire hit');
    hit.addEventListener('mousedown', event => {
      event.stopPropagation();
      this.links = this.links.filter(l => l !== link);
      this.draw();
      this.onchange(true);
    });
    return [hit, line];
  }

  /** The curve between two sockets.
   *
   * Forward is the ordinary S-bend. Backward — an answer returning to the map
   * it will be drawn on — sags underneath the row instead, because the same
   * S-bend run in reverse throws its control points far off to either side and
   * the wire disappears past the edge of the board. Nodes paint over the wires,
   * so a sagging wire passes behind them.
   */
  path(a, b) {
    const dx = b.x - a.x;
    if (dx > -20) {
      const bend = Math.max(30, dx * 0.45);
      return `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`;
    }
    const sag = Math.min(190, 70 + Math.abs(dx) * 0.1);
    return `M ${a.x} ${a.y} C ${a.x + 70} ${a.y + sag}, ${b.x - 70} ${b.y + sag}, ${b.x} ${b.y}`;
  }

  // --- interaction ----------------------------------------------------

  /** A page point in board coordinates. */
  where(event) {
    const box = this.viewport.getBoundingClientRect();
    return {
      x: (event.clientX - box.left - this.pan.x) / this.zoom,
      y: (event.clientY - box.top - this.pan.y) / this.zoom,
    };
  }

  onDown(event) {
    const port = event.target.closest('.port');
    if (port) {
      event.preventDefault();
      // Grabbing a connected input takes the wire off it rather than adding a
      // second, which is how you re-plug something without cutting it first —
      // the far end comes with you.
      let origin = {node: port.dataset.node, port: port.dataset.port,
                    direction: port.dataset.direction};
      if (port.dataset.direction === 'in') {
        const held = this.links.find(l =>
          l.to === port.dataset.node && l.toPort === port.dataset.port);
        if (held) {
          this.links = this.links.filter(l => l !== held);
          origin = {node: held.from, port: held.fromPort, direction: 'out'};
        }
      }
      this.dragging = {
        what: 'wire', origin, kind: port.dataset.kind,
        from: this.socket(origin.node, origin.direction, origin.port),
        cursor: this.where(event),
      };
      this.highlight(this.dragging.kind, origin.direction);
      this.draw();
      return;
    }

    const node = event.target.closest('.node');
    if (node && event.target.closest('header')) {
      event.preventDefault();
      const model = this.nodes.get(node.dataset.id);
      const at = this.where(event);
      this.dragging = {what: 'node', id: model.id,
                       dx: model.x - at.x, dy: model.y - at.y};
      node.classList.add('dragging');
      return;
    }
    if (node) { return; }

    this.dragging = {what: 'pan', x: event.clientX - this.pan.x, y: event.clientY - this.pan.y};
    this.viewport.classList.add('panning');
  }

  onMove(event) {
    if (!this.dragging) { return; }
    if (this.dragging.what === 'pan') {
      this.pan = {x: event.clientX - this.dragging.x, y: event.clientY - this.dragging.y};
      this.draw();
    } else if (this.dragging.what === 'node') {
      const at = this.where(event);
      const node = this.nodes.get(this.dragging.id);
      node.x = at.x + this.dragging.dx;
      node.y = at.y + this.dragging.dy;
      this.draw();
    } else {
      this.dragging.cursor = this.where(event);
      this.drawWires();
    }
  }

  onUp(event) {
    if (!this.dragging) { return; }
    const drag = this.dragging;
    this.dragging = null;
    this.viewport.classList.remove('panning');
    this.layer.querySelectorAll('.dragging').forEach(n => n.classList.remove('dragging'));
    this.highlight(null);

    if (drag.what === 'wire') {
      const port = event.target.closest('.port');
      let landed = false;
      if (port && port.dataset.direction !== drag.origin.direction) {
        const [source, target] = drag.origin.direction === 'out'
          ? [drag.origin, {node: port.dataset.node, port: port.dataset.port}]
          : [{node: port.dataset.node, port: port.dataset.port}, drag.origin];
        landed = this.connect(source.node, source.port, target.node, target.port);
      }
      // `connect` has already drawn and told, so only the miss is left to
      // report — and a wire pulled off and dropped on nothing really is
      // disconnected. Telling twice ran the whole query twice, which on this
      // board is two searches and two ten-megabyte search spaces.
      if (!landed) {
        this.draw();
        this.onchange(true);
      }
    } else if (drag.what === 'node') {
      this.onchange(true);
    }
  }

  /** Light up every socket a dragged wire could legally land on. */
  highlight(kind, from) {
    this.layer.querySelectorAll('.port').forEach(port => {
      const fits = kind && port.dataset.kind === kind &&
        port.dataset.direction !== from;
      port.classList.toggle('eligible', Boolean(fits));
    });
  }

  onWheel(event) {
    event.preventDefault();
    const box = this.viewport.getBoundingClientRect();
    const at = {x: event.clientX - box.left, y: event.clientY - box.top};
    const factor = Math.exp(-event.deltaY * 0.0015);
    const zoom = Math.min(1.8, Math.max(0.35, this.zoom * factor));
    // Keep whatever is under the cursor under the cursor.
    this.pan = {
      x: at.x - (at.x - this.pan.x) * (zoom / this.zoom),
      y: at.y - (at.y - this.pan.y) * (zoom / this.zoom),
    };
    this.zoom = zoom;
    this.draw();
  }

  /** Mark the node the server could not build, if it named one. */
  blame(id) {
    this.layer.querySelectorAll('.failed').forEach(n => n.classList.remove('failed'));
    if (id) {
      const element = this.layer.querySelector(`[data-id="${id}"]`);
      if (element) { element.classList.add('failed'); }
    }
  }
}
