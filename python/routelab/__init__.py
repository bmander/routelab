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

from . import _routelab, heuristics, orderings, reference, sources
from ._args import Nodes, Sources
from .environment import (
    CompiledEnvironment,
    EdgeSource,
    Environment,
    Positions,
    ScalarEdges,
)
from .graph import Graph
from .heuristics import Euclidean, Heuristic, Landmarks, Zero
from .journey import Journey, Leg
from .orderings import EdgeDifference, Ordering, RandomOrder
from .planners import BFS, AStar, ContractionHierarchy, Dijkstra, Planner, route
from .search import Result, SearchResult, astar, bfs, dijkstra
from .searchspace import Branch, MeetingTrees, SearchSpace, ShortestPathTree
from .sources import OSM, Cycling, Driving, Profile, Walking

__all__ = [
    "AStar",
    "BFS",
    "Branch",
    "CompiledEnvironment",
    "ContractionHierarchy",
    "Cycling",
    "Dijkstra",
    "Driving",
    "EdgeSource",
    "EdgeDifference",
    "Environment",
    "Euclidean",
    "Graph",
    "Heuristic",
    "Journey",
    "Landmarks",
    "Leg",
    "MeetingTrees",
    "Nodes",
    "OSM",
    "Ordering",
    "Planner",
    "Positions",
    "Profile",
    "RandomOrder",
    "Result",
    "ScalarEdges",
    "SearchResult",
    "SearchSpace",
    "ShortestPathTree",
    "Sources",
    "Walking",
    "Zero",
    "astar",
    "bfs",
    "dijkstra",
    "heuristics",
    "orderings",
    "reference",
    "route",
    "sources",
]

__version__ = _routelab.__version__
