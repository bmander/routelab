"""One module per paper.

Each technique here is a published algorithm, named after it, holding both
roads to it: the kernel called on integer node ids, and the :class:`Planner`
that binds it to an environment. Beside them live the specs only they read — a
heuristic, a contraction ordering, a calendar, a timetable.

Every module is imported here, and that is load-bearing rather than tidy:
:func:`techniques` finds what the library ships by walking
``Planner.__subclasses__()``, so a kernel that nobody imported does not exist.
"""

from .planner import (
    OPTIONS,
    Origins,
    Planner,
    TimetablePlanner,
    clock_readers,
    names,
    owners,
    route,
    techniques,
)
from . import departures, heuristics, orderings, schedule
from .astar import AStar, astar
from .bfs import BFS, bfs
from .contraction import ContractionHierarchy
from .csa import CSA
from .dijkstra import Dijkstra, dijkstra
from .ptl import PTL
from .raptor import RAPTOR
from .timedep import TimeDependentDijkstra
from .timetable import TimeDependent, TimeExpanded
from .tripbased import TripBased

__all__ = [
    "AStar",
    "BFS",
    "CSA",
    "ContractionHierarchy",
    "Dijkstra",
    "OPTIONS",
    "Origins",
    "PTL",
    "Planner",
    "RAPTOR",
    "TimeDependent",
    "TimeDependentDijkstra",
    "TimeExpanded",
    "TripBased",
    "TimetablePlanner",
    "astar",
    "bfs",
    "clock_readers",
    "departures",
    "dijkstra",
    "heuristics",
    "names",
    "orderings",
    "owners",
    "route",
    "schedule",
    "techniques",
]
