"""One module per paper.

Each technique here is a published algorithm, named after it, holding both
roads to it: the kernel called on integer node ids, and the pair of classes
that binds it to an environment — a :class:`Technique` holding the
configuration and an ``XPlanner`` holding the data and the verbs. Beside them
live the specs only they read — a heuristic, a contraction ordering, a
calendar, a timetable.
"""

from .planner import (
    Front,
    GraphPlanner,
    Origins,
    Planner,
    Tabled,
    Technique,
    TimetablePlanner,
    TimetableTechnique,
    TreePlanner,
    route,
)
from . import departures, heuristics, orderings, schedule
from .astar import AStar, AStarPlanner, astar
from .bfs import BFS, BFSPlanner, bfs
from .contraction import ContractionHierarchy, ContractionHierarchyPlanner
from .csa import CSA, CSAPlanner
from .dijkstra import Dijkstra, DijkstraPlanner, dijkstra
from .ptl import PTL, PTLPlanner
from .raptor import RAPTOR, RAPTORPlanner
from .timedep import TimeDependentDijkstra, TimeDependentDijkstraPlanner
from .timetable import (
    TimeDependent,
    TimeDependentPlanner,
    TimeExpanded,
    TimeExpandedPlanner,
)
from .tripbased import TripBased, TripBasedPlanner
from .lcspp import (
    UCCH,
    LabelConstrained,
    LabelConstrainedPlanner,
    Modes,
    UCCHPlanner,
)
from .ultra import ULTRA, Transfers, ULTRAPlanner

__all__ = [
    "AStar",
    "AStarPlanner",
    "BFS",
    "BFSPlanner",
    "CSA",
    "CSAPlanner",
    "ContractionHierarchy",
    "ContractionHierarchyPlanner",
    "Dijkstra",
    "DijkstraPlanner",
    "Front",
    "GraphPlanner",
    "Origins",
    "PTL",
    "PTLPlanner",
    "Planner",
    "Tabled",
    "RAPTOR",
    "RAPTORPlanner",
    "Technique",
    "TimeDependent",
    "TimeDependentDijkstra",
    "TimeDependentDijkstraPlanner",
    "TimeDependentPlanner",
    "TimeExpanded",
    "TimeExpandedPlanner",
    "TimetablePlanner",
    "TimetableTechnique",
    "TreePlanner",
    "TripBased",
    "TripBasedPlanner",
    "Transfers",
    "LabelConstrained",
    "LabelConstrainedPlanner",
    "Modes",
    "UCCH",
    "UCCHPlanner",
    "ULTRA",
    "ULTRAPlanner",
    "astar",
    "bfs",
    "departures",
    "dijkstra",
    "heuristics",
    "orderings",
    "route",
    "schedule",
]
