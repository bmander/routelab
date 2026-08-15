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

// What each node looks like and what it produces. `kind` is the type of value
// on its output; `inputs` names each socket and the kind it accepts. Those two
// facts are all the wiring rules there are — a port takes what it says it
// takes, which is the same contract `Planner.accepts` states one level down.
const TYPES = {
  OSM: {
    kind: 'layer', title: 'OSM', group: 'layers',
    sub: () => SETUP.extract,
    fields: [{id: 'profile', label: 'profile', type: 'select',
              options: () => SETUP.profiles, value: 'driving'}],
  },
  GTFS: {
    kind: 'layer', title: 'GTFS', group: 'layers',
    sub: () => SETUP.feed,
    available: () => Boolean(SETUP.feed),
    fields: [{id: '_date', label: 'service day', type: 'note',
              text: () => SETUP.date}],
  },
  Environment: {
    kind: 'environment', title: 'Environment', group: 'environment',
    inputs: {layers: 'layer'},
  },
  Euclidean: {kind: 'heuristic', title: 'Euclidean', group: 'heuristics'},
  Landmarks: {
    kind: 'heuristic', title: 'Landmarks', group: 'heuristics',
    fields: [{id: 'count', label: 'landmarks', type: 'number',
              value: 16, min: 1, max: 64}],
  },
  EdgeDifference: {kind: 'ordering', title: 'EdgeDifference', group: 'orderings'},
  RandomOrder: {
    kind: 'ordering', title: 'RandomOrder', group: 'orderings',
    fields: [{id: 'seed', label: 'seed', type: 'number', value: 0, min: 0, max: 999}],
  },
  Dijkstra: {
    kind: 'planner', title: 'Dijkstra', group: 'techniques',
    inputs: {environment: 'environment'},
  },
  BFS: {
    kind: 'planner', title: 'BFS', group: 'techniques', sub: () => 'fewest hops',
    inputs: {environment: 'environment'},
  },
  AStar: {
    kind: 'planner', title: 'AStar', group: 'techniques',
    inputs: {environment: 'environment', heuristic: 'heuristic'},
  },
  ContractionHierarchy: {
    kind: 'planner', title: 'ContractionHierarchy', group: 'techniques',
    inputs: {environment: 'environment', ordering: 'ordering'},
  },
  TimeDependentDijkstra: {
    kind: 'planner', title: 'TimeDependentDijkstra', group: 'techniques',
    inputs: {environment: 'environment'},
    fields: [{id: 'waiting', label: 'waiting', type: 'select',
              options: () => ['unrestricted', 'forbidden'], value: 'unrestricted'}],
  },
  TimeDependent: {
    kind: 'planner', title: 'TimeDependent', group: 'techniques',
    sub: () => 'a node per stop',
    available: () => Boolean(SETUP.feed),
    inputs: {environment: 'environment'},
  },
  TimeExpanded: {
    kind: 'planner', title: 'TimeExpanded', group: 'techniques',
    sub: () => 'a node per event',
    available: () => Boolean(SETUP.feed),
    inputs: {environment: 'environment'},
  },
  Query: {
    kind: 'query', title: 'Query', group: 'query',
    sub: () => 'route(a, b)',
    inputs: {planner: 'planner'},
    fields: [
      {id: 'day', label: 'weekday', type: 'select', value: '0',
       options: () => [['0', 'Mon'], ['1', 'Tue'], ['2', 'Wed'], ['3', 'Thu'],
                       ['4', 'Fri'], ['5', 'Sat'], ['6', 'Sun']]},
      {id: 'minute', label: 'departing', type: 'time', value: 480},
    ],
  },
};

// One colour per kind, shared by a port's dot and the wire leaving it — which
// is what lets you see at a glance that a heuristic cannot go where an
// environment belongs.
const COLOURS = {
  layer: '#8ec07c', environment: '#83a9d1', heuristic: '#d5a05a',
  ordering: '#d5a05a', planner: '#c08ac4', query: '#d1495b',
};

const HEADER = 26, ROW = 22, WIDTH = 176;

class Board {
  constructor(host, onchange) {
    this.host = host;
    this.onchange = onchange;
    this.nodes = new Map();
    this.links = [];
    this.pan = {x: 40, y: 16};
    this.zoom = 1;
    this.next = 1;

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
    const spec = TYPES[type];
    const params = {};
    for (const field of spec.fields || []) {
      if (field.type !== 'note') { params[field.id] = field.value; }
    }
    const id = 'n' + this.next++;
    this.nodes.set(id, {id, type, params, x, y});
    this.draw();
    return id;
  }

  remove(id) {
    this.nodes.delete(id);
    this.links = this.links.filter(l => l.from !== id && l.to !== id);
    this.draw();
    this.onchange();
  }

  connect(from, to, port) {
    if (from === to || !this.compatible(from, to, port)) { return false; }
    // One wire per socket, except a bag of layers: an `Environment` takes as
    // many as you give it, and everything else takes one argument.
    const many = to === null ? false : TYPES[this.nodes.get(to).type].inputs[port] === 'layer'
      && this.nodes.get(to).type === 'Environment';
    this.links = this.links.filter(l =>
      !(l.to === to && l.port === port && (!many || l.from === from)));
    this.links.push({from, to, port});
    this.draw();
    this.onchange();
    return true;
  }

  compatible(from, to, port) {
    const source = this.nodes.get(from), target = this.nodes.get(to);
    if (!source || !target) { return false; }
    const wants = (TYPES[target.type].inputs || {})[port];
    return wants === TYPES[source.type].kind && !this.reaches(to, from);
  }

  /** Is `goal` downstream of `start`? The check that keeps a graph a graph. */
  reaches(start, goal) {
    if (start === goal) { return true; }
    return this.links.filter(l => l.from === start)
      .some(l => this.reaches(l.to, goal));
  }

  serialise() {
    return {
      nodes: [...this.nodes.values()].map(n =>
        ({id: n.id, type: n.type, params: n.params, x: Math.round(n.x), y: Math.round(n.y)})),
      links: this.links.map(l => ({from: l.from, to: l.to, port: l.port})),
    };
  }

  load(spec) {
    this.nodes = new Map();
    this.links = [];
    for (const node of spec.nodes || []) {
      if (!TYPES[node.type]) { continue; }
      this.nodes.set(node.id, {...node, params: {...node.params}});
      this.next = Math.max(this.next, Number(String(node.id).slice(1)) + 1 || 1);
    }
    this.links = (spec.links || []).filter(l =>
      this.nodes.has(l.from) && this.nodes.has(l.to));
    this.draw();
  }

  // --- geometry -------------------------------------------------------

  /** Where a port's dot sits, in board coordinates. */
  socket(id, direction, index) {
    const node = this.nodes.get(id);
    return {
      x: node.x + (direction === 'in' ? 0 : WIDTH),
      y: node.y + HEADER + index * ROW + ROW / 2,
    };
  }

  inputIndex(type, port) {
    return Object.keys(TYPES[type].inputs || {}).indexOf(port);
  }

  outputIndex(type) {
    return Object.keys(TYPES[type].inputs || {}).length;
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
      (spec.sub ? `<span class="sub">${spec.sub()}</span>` : '');
    element.append(header);

    const body = document.createElement('div');
    body.className = 'body';
    for (const [port, kind] of Object.entries(spec.inputs || {})) {
      body.append(this.renderPort(node.id, 'in', port, kind));
    }
    // A query is where the graph ends, so it has nothing on its right edge. A
    // port that could never be connected to is a promise the board cannot keep.
    if (spec.kind !== 'query') {
      body.append(this.renderPort(node.id, 'out', node.type, spec.kind));
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
    row.append(dot, document.createTextNode(direction === 'in' ? port : kind));
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

    // Two events, because they mean different things while dragging a slider:
    // `input` fires continuously and re-routes live, `change` fires when you
    // let go and is when the expensive picture is worth redrawing.
    input.addEventListener('input', () => {
      node.params[field.id] = field.type === 'number' || field.type === 'time'
        ? Number(input.value) : input.value;
      this.onchange(false);
    });
    input.addEventListener('change', () => this.onchange(true));
    // Otherwise a click on a select starts dragging the node under it.
    input.addEventListener('mousedown', event => event.stopPropagation());
    wrap.append(input);
    return wrap;
  }

  drawWires() {
    this.wires.textContent = '';
    for (const link of this.links) {
      const from = this.nodes.get(link.from), to = this.nodes.get(link.to);
      const a = this.socket(link.from, 'out', this.outputIndex(from.type));
      const b = this.socket(link.to, 'in', this.inputIndex(to.type, link.port));
      this.wires.append(...this.wire(a, b, COLOURS[TYPES[from.type].kind], link));
    }
    if (this.dragging && this.dragging.what === 'wire') {
      const {from, kind, cursor} = this.dragging;
      this.wires.append(this.wire(from, cursor, COLOURS[kind])[1]);
    }
  }

  /** A bezier, plus a fat invisible copy of it that is easy to click. */
  wire(a, b, colour, link) {
    const bend = Math.max(30, Math.abs(b.x - a.x) * 0.45);
    const d = `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`;

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.setAttribute('d', d);
    line.setAttribute('class', link ? 'wire' : 'wire dangling');
    line.setAttribute('stroke', colour);

    if (!link) { return [line, line]; }
    const hit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hit.setAttribute('d', d);
    hit.setAttribute('class', 'wire hit');
    hit.addEventListener('mousedown', event => {
      event.stopPropagation();
      this.links = this.links.filter(l => l !== link);
      this.draw();
      this.onchange();
    });
    return [hit, line];
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
      const index = port.dataset.direction === 'in'
        ? this.inputIndex(this.nodes.get(port.dataset.node).type, port.dataset.port)
        : this.outputIndex(this.nodes.get(port.dataset.node).type);
      const anchor = this.socket(port.dataset.node, port.dataset.direction, index);
      // Grabbing a connected input takes the wire off it rather than adding a
      // second, which is how you re-plug something without cutting it first.
      let origin = {node: port.dataset.node, direction: port.dataset.direction};
      if (port.dataset.direction === 'in') {
        const held = this.links.find(l =>
          l.to === port.dataset.node && l.port === port.dataset.port);
        if (held) {
          this.links = this.links.filter(l => l !== held);
          origin = {node: held.from, direction: 'out'};
        }
      }
      this.dragging = {
        what: 'wire', origin, kind: port.dataset.kind,
        from: origin.direction === 'out'
          ? this.socket(origin.node, 'out', this.outputIndex(this.nodes.get(origin.node).type))
          : anchor,
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
      if (port && port.dataset.direction !== drag.origin.direction) {
        if (drag.origin.direction === 'out') {
          this.connect(drag.origin.node, port.dataset.node, port.dataset.port);
        } else {
          this.connect(port.dataset.node, drag.origin.node, drag.origin.port);
        }
      }
      this.draw();
      // A wire pulled off and dropped on nothing really is disconnected.
      this.onchange();
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
