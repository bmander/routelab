"""routelab — reference implementations of routing algorithms.

The Python layer is the showroom: friendly constructors, flexible arguments,
docstrings, and a pure-Python reference implementation of every kernel to check
the fast one against. The Rust layer (``routelab._routelab``) is the kernel:
CSR graphs and searches whose constant factors are the point.

There are two ways in. The high road describes a world and asks it questions:

    >>> import routelab as rl
    >>> env = rl.Environment()
    >>> env.register(rl.ScalarEdges(("a", "b", 1), ("b", "c", 15)))
    Environment(1 layer)
    >>> rl.Dijkstra().bind(env).route("a", "c")
    Journey('a' → 'b' → 'c', cost=16)

The low road is the kernel itself, on dense integer ids, as the papers state it:

    >>> graph = rl.Graph.from_edges([(0, 1, 60), (1, 3, 120), (0, 2, 90), (2, 3, 30)])
    >>> rl.dijkstra(graph, 0).cost(3)
    120

Take the high road to route over a real network, the low road to implement or
benchmark an algorithm.
"""

from __future__ import annotations

from . import _routelab, data, kernels, reference
from .kernels import heuristics, orderings
from .util._args import Nodes, Sources
from .kernels.departures import Departures, Walks
from .model.environment import (
    CompiledEnvironment,
    EdgeSource,
    Environment,
    Positions,
    ScalarEdges,
)
from .model.graph import Graph
from .kernels.heuristics import Euclidean, Heuristic, Landmarks, Pace, Plane, Zero
from .model.journey import Journey, Leg
from .kernels.orderings import EdgeDifference, Ordering, RandomOrder
from .util.clock import WEEK, service_seconds, weekly_seconds
from .kernels.schedule import Schedule
from .kernels import (
    BFS,
    CSA,
    RAPTOR,
    AStar,
    ContractionHierarchy,
    Dijkstra,
    Planner,
    TimeDependent,
    TimeDependentDijkstra,
    TimeExpanded,
    TimetablePlanner,
    astar,
    bfs,
    clock_readers,
    dijkstra,
    route,
)
from .model.search import EdgeResult, Result, SearchResult
from .model.searchspace import (
    Arrival,
    Branch,
    Leap,
    MeetingTrees,
    Reach,
    Rounds,
    Scan,
    SearchSpace,
    ShortestPathTree,
)
from .data import GTFS, OSM, Cycling, Driving, Footpaths, Profile, Walking

__all__ = [
    "AStar",
    "Arrival",
    "BFS",
    "Branch",
    "CSA",
    "CompiledEnvironment",
    "ContractionHierarchy",
    "Cycling",
    "Departures",
    "Dijkstra",
    "Driving",
    "EdgeSource",
    "EdgeDifference",
    "Environment",
    "Euclidean",
    "Footpaths",
    "Graph",
    "Heuristic",
    "Journey",
    "Landmarks",
    "Leg",
    "MeetingTrees",
    "Nodes",
    "GTFS",
    "OSM",
    "Ordering",
    "Pace",
    "Planner",
    "Plane",
    "Positions",
    "Profile",
    "RandomOrder",
    "Result",
    "ScalarEdges",
    "Schedule",
    "SearchResult",
    "SearchSpace",
    "ShortestPathTree",
    "TimeDependent",
    "TimeDependentDijkstra",
    "TimeExpanded",
    "Sources",
    "WEEK",
    "Walks",
    "Walking",
    "Zero",
    "astar",
    "bfs",
    "dijkstra",
    "heuristics",
    "orderings",
    "reference",
    "route",
    "data",
    "weekly_seconds",
    "RAPTOR",
    "TimetablePlanner",
    "EdgeResult",
    "Leap",
    "Reach",
    "Rounds",
    "Scan",
    "clock_readers",
    "kernels",
    "service_seconds",
]

__version__ = _routelab.__version__
