"""What nodes the board offers, and what each one takes.

This is the board's whole vocabulary, as data. Adding a node is adding an entry
here and its shape in ``static/board.js`` — the two files are a contract, and
neither half works alone.
"""

from __future__ import annotations

import routelab as rl

PROFILES = {"walking": rl.Walking, "cycling": rl.Cycling, "driving": rl.Driving}

#: What each node type builds, and what has to be plugged into it first.
#:
#: This is the whole vocabulary of the board, and it is deliberately a
#: transcription of the library's own three steps rather than a menu of
#: features: layers make an environment, a technique binds to one and becomes a
#: planner, a planner takes a query. Adding a node is adding an entry here and
#: its shape in `static/board.js` — nothing between the two knows what the nodes
#: mean.
#:
#: `inputs` maps a port name to the kind of value it accepts. `kind` is what a
#: node is for, which decides how it is built; the two together are all the
#: wiring rules there are, and a port takes what it says it takes.
#:
#: What is *not* a port is as deliberate as what is. A thing is a wire iff it
#: is a choice: a heuristic or an ordering has alternatives and knobs, so it is
#: wired in. The calendar a `TimeDependentDijkstra` reads, or the coordinates
#: `Euclidean` prices, have one possible construction from the layers already
#: plugged into the environment, so the technique derives them at bind and no
#: wire is drawn — a wire that could not change anything would be decoration.
#:
#: Shipped to the page as-is (see `Router.catalogue`), which is why it is data
#: rather than code: `static/board.js` adds titles, fields and colours to these
#: entries and never restates the wiring.
NODES: "dict[str, dict]" = {
    "OSM": {"kind": "layer", "inputs": {}},
    "GTFS": {"kind": "layer", "inputs": {}},
    "Environment": {"kind": "environment", "inputs": {"layers": "layer"}},
    "Euclidean": {"kind": "heuristic", "inputs": {}},
    "Landmarks": {"kind": "heuristic", "inputs": {}},
    "EdgeDifference": {"kind": "ordering", "inputs": {}},
    "RandomOrder": {"kind": "ordering", "inputs": {}},
    "Dijkstra": {"kind": "planner", "inputs": {"environment": "environment"}},
    "BFS": {"kind": "planner", "inputs": {"environment": "environment"}},
    "AStar": {
        "kind": "planner",
        "inputs": {"environment": "environment", "heuristic": "heuristic"},
    },
    "ContractionHierarchy": {
        "kind": "planner",
        "inputs": {"environment": "environment", "ordering": "ordering"},
    },
    # `clock` names the clock a technique's departure time is read on, for
    # the ones that take one: "week" repeats (a street schedule), "day" is the
    # service day a feed was read for. Absent for a technique that has no hour.
    "TimeDependentDijkstra": {
        "kind": "planner",
        "inputs": {"environment": "environment"},
        "clock": "week",
    },
    "TimeDependent": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "TimeExpanded": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "RAPTOR": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "CSA": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "TripBased": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    "PTL": {"kind": "planner", "inputs": {"environment": "environment"}, "clock": "day"},
    # Multimodality with nothing precomputed: the modes each arc belongs to are
    # already in the environment, and the language decides which sequences of
    # them a journey may use.
    "LabelConstrained": {
        "kind": "planner",
        "inputs": {"environment": "environment"},
        "clock": "day",
    },
    # A technique that wraps a technique: ULTRA works out the transfers and
    # the one wired into it does the routing, which is the paper's own shape.
    "ULTRA": {
        "kind": "planner",
        "inputs": {"environment": "environment", "technique": "planner"},
        "clock": "day",
    },
    # Walks between nearby stops, made from another layer's coordinates. A
    # layer that takes a layer: its output is edges like any other layer's,
    # and how far a rider will walk is the knob — which is why it is a node
    # and not something the environment does behind your back.
    "Footpaths": {"kind": "layer", "inputs": {"stops": "layer"}},
    # The step from a stop onto the pavement outside it. Two layers in,
    # because it is the only thing that knows they are the same place, and
    # edges out like any other layer.
    "Access": {"kind": "layer", "inputs": {"stops": "layer", "streets": "layer"}},
    # A query takes its endpoints from wires like everything else, so "where
    # from" is an argument rather than an ambient fact about the page. Its two
    # answers are separate outputs because they cost different amounts: the
    # route is a few hundred points, the search space up to ten megabytes.
    "Query": {
        "kind": "query",
        "inputs": {"planner": "planner", "origin": "point", "destination": "point"},
        "outputs": {"route": "route", "space": "space"},
    },
    # The map, as a node. A source of two points and a sink for two answers —
    # and not a cycle, because where the pins are owes nothing to what is drawn.
    "Map": {
        "kind": "map",
        "inputs": {"route": "route", "space": "space"},
        "outputs": {"origin": "point", "destination": "point"},
        # Its outputs owe nothing to its inputs, so its inputs are not
        # arguments and nothing upstream of it is upstream of anything. Which
        # is what stops a signature walking Map -> Query -> Map for ever.
        "terminal": True,
    },
}


#: What each technique node stands for in the library — the class whose
#: `options` say which query knobs the board should show under it. Declared
#: by the technique, read here; the board learns what a technique takes from
#: the library rather than from a second table.
TECHNIQUES: "dict[str, type]" = {
    "Dijkstra": rl.Dijkstra,
    "BFS": rl.BFS,
    "AStar": rl.AStar,
    "ContractionHierarchy": rl.ContractionHierarchy,
    "TimeDependentDijkstra": rl.TimeDependentDijkstra,
    "TimeDependent": rl.TimeDependent,
    "TimeExpanded": rl.TimeExpanded,
    "RAPTOR": rl.RAPTOR,
    "CSA": rl.CSA,
    "TripBased": rl.TripBased,
    "PTL": rl.PTL,
    "LabelConstrained": rl.LabelConstrained,
    "ULTRA": rl.ULTRA,
}
for _kind, _cls in TECHNIQUES.items():
    # `departing` is covered by `clock`; the rest are the knobs a Query node
    # shows when this technique is wired into it.
    NODES[_kind]["options"] = sorted(_cls.options - {"departing"})
