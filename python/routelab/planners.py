"""Planners: a technique you configure, then bind to an environment.

    technique = AStar(Landmarks(16))     # configuration, costing nothing
    planner = technique.bind(env)        # preprocessing, costing seconds
    planner.route("a", "c")              # queries, costing milliseconds

Three steps rather than two, and the middle one is why. Every algorithm this
project is heading toward — ULTRA's transfer shortcuts, transfer patterns,
contraction hierarchies — earns its query speed with preprocessing paid once and
amortized. Sixteen landmarks over a city is already a second and 33 MB. Spending
that inside a constructor hides it; spending it in :meth:`Planner.bind` gives it
a verb.

Configuring separately from binding is also what makes a technique a *value*.
Layers and heuristics have always worked this way — ``ScalarEdges(...)``,
``Landmarks(16)`` — and planners were the one exception, which is why every
consumer that wanted to name and compare configured algorithms had to invent its
own table of lambdas. Now:

    study = [Dijkstra(), AStar(Euclidean()), AStar(Landmarks(16))]
    for technique in study:
        if technique.missing_from(compiled):
            continue                     # this dataset cannot support it
        technique.bind(env).route(origin, destination)

A bound planner holds the environment as it was when it was bound. Register
another layer and you have a different world; bind the technique again.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple, Union

from . import _routelab
from .util.clock import service_seconds, weekly_seconds
from .departures import Departures, Walks
from .model.environment import CompiledEnvironment, Environment
from .heuristics import Heuristic
from .schedule import Schedule
from .model.journey import Journey
from .orderings import EdgeDifference, Ordering
from .model.search import Result, SearchResult, astar, bfs, dijkstra
from .model.searchspace import MeetingTrees, Rounds, SearchSpace, ShortestPathTree

__all__ = [
    "AStar",
    "BFS",
    "ContractionHierarchy",
    "Dijkstra",
    "Origins",
    "Planner",
    "RAPTOR",
    "TimeDependent",
    "TimeDependentDijkstra",
    "TimeExpanded",
    "TimetablePlanner",
    "clock_readers",
    "names",
    "owners",
    "route",
    "techniques",
]

#: How a caller names where to start: one label, a list of labels, or a mapping
#: of labels to the cost of already being there. Tuples and strings are single
#: labels — a label may itself be a tuple, so iterability cannot decide this.
Origins = Union[Hashable, Iterable[Hashable], Mapping[Hashable, int]]



def techniques() -> "List[type]":
    """Every technique in the library, most general first.

    A list of the classes rather than a registry of names: what it is for is
    answering "which of these takes that?" when a refusal has to name someone,
    so nothing here is a lookup table anyone routes through.
    """
    found: "List[type]" = []
    stack = [Planner]
    while stack:
        technique = stack.pop()
        stack.extend(technique.__subclasses__())
        # What this library ships, and what a caller could actually write
        # down: not the abstract middle of the hierarchy, and not somebody
        # else's subclass — a test's or an application's — which would be
        # named in a refusal nobody could act on.
        if technique in (Planner, TimetablePlanner):
            continue
        if technique.__module__.split(".")[0] == __name__.split(".")[0]:
            found.append(technique)
    return sorted(found, key=lambda cls: cls.__name__)


def owners(option: str) -> "List[type]":
    """The techniques that take ``option`` — who a refusal should point at."""
    return [cls for cls in techniques() if option in cls.options]


def names(classes: "Iterable[type]") -> str:
    """``"A(), B() or C()"`` — techniques, as a caller would write them."""
    written = [f"{cls.__name__}()" for cls in classes]
    if not written:
        return ""
    if len(written) == 1:
        return written[0]
    return f"{', '.join(written[:-1])} or {written[-1]}"


def clock_readers(compiled: CompiledEnvironment) -> Optional[str]:
    """The technique or techniques that would read this environment's clock.

    A refusal that says "use TimeDependentDijkstra()" on a GTFS feed is wrong
    advice, so the sentence is assembled from what the layers actually hold
    and from which techniques exist: departures want the timetable techniques,
    opening hours want the ones that read a schedule, and an environment with
    neither has no clock for anyone to read — ``None``. A new kernel joins the
    sentence by existing rather than by being added to a list.
    """
    if not Departures.missing_from(compiled):
        return names(cls for cls in techniques() if issubclass(cls, TimetablePlanner))
    if not Schedule.missing_from(compiled):
        return names(
            cls
            for cls in techniques()
            if "departing" in cls.options and not issubclass(cls, TimetablePlanner)
        )
    return None


#: What each query option *is*, for the first half of a refusal. Who it belongs
#: to is not written down: :func:`owners` asks the techniques, so the sentence
#: cannot go stale as the shelf grows. ``departing`` is missing because what it
#: is depends on the environment — see :func:`clock_readers`.
OPTIONS = {
    "max_cost": "a cost bound",
    "max_depth": "a hop bound",
    "max_transfers": "a cap on changes",
    "magnitude": "a magnitude, which belongs to explored() on a technique that grows a tree",
}


class Planner:
    """A routing technique: configured on construction, bound to data later.

    Every technique answers the same three verbs — :meth:`bind`, :meth:`route`,
    :meth:`search` — and declares, as data, which query options it takes
    (:attr:`options`) and which it insists on (:attr:`required`). An option a
    technique does not take is refused by name, with the technique it belongs
    to; that is one sentence written once, here, rather than one per planner.
    """

    #: Cost models this algorithm knows how to route over.
    accepts: "frozenset[str]" = frozenset({"scalar"})

    #: The query options this technique understands, and the ones it cannot do
    #: without. Declared rather than discovered so a caller — or a board — can
    #: know what a technique takes before asking it anything. Empty here on
    #: purpose: a technique inherits no knobs it did not ask for, so a refusal
    #: naming who takes an option can never name someone who merely forgot to
    #: unset it.
    options: "frozenset[str]" = frozenset()
    required: "frozenset[str]" = frozenset()

    #: The environment this was bound to, or ``None`` while it is still just a
    #: configuration.
    environment: Optional[Environment] = None
    compiled: Optional[CompiledEnvironment] = None

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "Planner":
        """Attach this technique to an environment and do its preprocessing.

        Returns a *new* planner and leaves this one as it was, so a technique can
        be bound to several environments — which is the whole point of writing
        one down: the same configuration, measured across datasets.

        Args:
            progress: A counter to write into while preprocessing runs, for
                anything watching from another thread. Six seconds of
                contraction is long enough that "still working" and "hung" want
                telling apart. Techniques with no honest measure of their own
                progress leave it alone — see :mod:`routelab._routelab` and
                `Progress.fraction` returning ``None``.
        """
        unsupported = environment.cost_models - self.accepts
        if unsupported:
            raise TypeError(
                f"{type(self).__name__} cannot route over "
                f"{', '.join(sorted(unsupported))} layers; it accepts "
                f"{', '.join(sorted(self.accepts))}"
            )
        bound = copy.copy(self)
        bound.environment = environment
        bound.compiled = environment.compile()
        bound.preprocess(progress)
        return bound

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Everything standing between this technique and that environment.

        Empty means :meth:`bind` will work. Answered without binding anything,
        so a study can skip what a dataset cannot support before spending the
        preprocessing to find out — which only holds if this covers everything
        ``bind`` checks, so it does: cost models the technique does not accept,
        and (in subclasses) whatever the technique derives from the environment
        — a schedule, a timetable, coordinates — that the layers never supplied.

        Entries are short names either way, and each is owned by whoever
        answers for it: ``"timetable"`` for a cost model this algorithm cannot
        route over, ``"positions"`` from :class:`~routelab.Plane`, ``"schedule"``
        from :class:`~routelab.Schedule`.
        """
        return compiled.cost_models - self.accepts

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Work done once at bind time, before any query.

        Nothing to do for a plain search; this is where a landmark table or a
        set of shortcuts gets built. `progress` is passed straight through to
        whatever does that building, and is not kept: it describes one build,
        and a planner holding it afterwards would be holding a stopwatch that
        stopped.
        """

    @property
    def footprint(self) -> int:
        """Bytes of preprocessed data this planner is holding.

        Zero for a plain search, which is the honest answer rather than a
        missing one: preprocessing is a trade, and a table comparing techniques
        has to be able to print the cost side of it for every row.
        """
        self._bound()          # an unbound technique holds nothing, and says so
        return self._footprint()

    def _footprint(self) -> int:
        """The size of whatever :meth:`preprocess` built."""
        return 0

    @property
    def searches(self) -> "Tuple[str, int]":
        """What a query of this planner settles, and how many of them there
        are — the denominator a settled count is a share of.

        ``("nodes", n)`` for a search over the environment's graph. A technique
        that searches something else says so: the time-expanded model settles
        events, the timetable techniques stops.
        """
        return ("nodes", len(self._bound()))

    def _bound(self) -> CompiledEnvironment:
        """The compiled environment, or an error naming what is missing."""
        if self.compiled is None:
            raise ValueError(
                f"{self!r} is a technique, not a planner — bind it to an "
                f"environment first: {type(self).__name__}(...).bind(env)"
            )
        return self.compiled

    # --- the three questions ------------------------------------------------

    def route(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> Optional[Journey]:
        """The cheapest journey from ``origin`` to ``destination``.

        One implementation for every technique: options are checked once,
        origins are resolved once, and the technique supplies only the part
        that is its own — see :meth:`_route`.

        Args:
            origin: A label, several labels, or ``{label: initial_cost}`` — the
                last being how a multimodal query starts, with each entry point
                already costing an access walk. For a timetable technique the
                cost is seconds already spent when the query departs, so
                ``{"stop_b": 45}`` stands at stop_b forty-five seconds after
                ``departing``.
            destination: The label to route to.
            **options: What this technique takes — see :attr:`options`.
                ``max_cost``, ``max_depth`` bound a search; ``departing`` is
                when a clock-reading technique leaves; ``max_transfers`` caps
                a round-based one.

        Returns:
            A :class:`~routelab.Journey`, or ``None`` if the destination cannot
            be reached under the given bounds.
        """
        options = self._options(options)
        starts = self._origin_ids(origin)
        return self._route(starts, self.node_id(destination), destination, options)

    def _route(
        self,
        starts: "Dict[int, int]",
        target: int,
        destination: Hashable,
        options: "Dict[str, Any]",
    ) -> Optional[Journey]:
        """The technique's own half of :meth:`route`, ids in.

        The default is the search-based one: run :meth:`_search` toward the
        target and read the journey off the result. A technique whose answer is
        an itinerary rather than a cost table overrides this instead.
        """
        result = self._search(starts, targets=[target], **options)
        if result.cost(target) is None:
            return None
        return self.journey(result, destination)

    def search(self, origins: Origins, **options: Any) -> Result:
        """Run the search and return the raw, id-keyed result.

        The escape hatch: everything the kernel computed, without the journey
        packaging. Node ids here are dense, so use :meth:`node_id`/:meth:`label`
        to get between them and your labels. Options are the same as
        :meth:`route`'s, plus ``targets=[node, ...]`` to stop early.

        Every technique that keeps a table of costs answers this; one that
        answers only with a journey says so rather than handing back an empty
        table.
        """
        return self._search(self._origin_ids(origins), **self._options(options))

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        """The technique's own half of :meth:`search`, ids in — the one place
        it calls its kernel."""
        raise NotImplementedError(
            f"{type(self).__name__} answers with a journey rather than a cost "
            f"per node — use route(origin, destination, ...)."
        )

    def journey(self, result: Result, destination: Hashable) -> Optional[Journey]:
        """The journey a kept result holds for ``destination``, or ``None``.

        What :meth:`route` does with a search; public so a result asked for once
        can be read for several destinations, or drawn and then answered from,
        without running the search again.
        """
        if result.cost(self.node_id(destination)) is None:
            return None
        return Journey.from_result(self._bound(), result, destination)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """What the search looked at, in a form something can draw.

        Dijkstra and A* — and BFS — explore by growing a shortest-path tree, so
        that is what they report. An algorithm that explores differently returns
        a different :class:`~routelab.SearchSpace`; the promise is only that
        whatever it explored can be rendered.

        Options belong to the shape of the space, so they differ by algorithm and
        an algorithm that does not understand one says so — the same rule
        :meth:`SearchSpace.geojson` states, and the reason ``magnitude`` is not
        on this signature: it means something only to a tree.

        Args:
            result: A result from :meth:`search`, or from :meth:`route` if you
                kept one. The tree is rebuilt from it rather than recorded during
                the search, so asking costs nothing until you ask.
            magnitude: What each branch should carry from the subtree beyond it:
                ``"weight"`` for travel time, ``"nodes"`` for a count.
        """
        magnitude = options.pop("magnitude", "weight")
        self._no_other(options, "a shortest-path tree")
        return ShortestPathTree(self._bound(), result, magnitude)

    # --- the shared checks ---------------------------------------------------

    def _options(self, options: "Dict[str, Any]") -> "Dict[str, Any]":
        """Refuse what this technique does not take, and insist on what it does.

        One sentence for every planner: the option, the technique that owns it,
        and — for a departure time — which technique here could read this
        environment's clock, or that nothing can.
        """
        name = type(self).__name__
        for option in sorted(options):
            if option == "targets" or option in self.options:
                continue
            if option == "departing":
                readers = clock_readers(self._bound())
                advice = (
                    f"Use {readers} to read the clock, or drop departing= to search "
                    f"the always-open network knowingly."
                    if readers
                    else "Nothing in this environment is scheduled, so no technique "
                    "has a clock to read here; drop departing=."
                )
                raise ValueError(
                    f"{name} routes the network as though it were always open, so "
                    f"it has no departure time. {advice}"
                )
            described = OPTIONS.get(option, f"{option}")
            takers = names(owners(option))
            owner = f"{described} belongs to {takers}" if takers else f"no technique here takes {described}"
            raise ValueError(f"{name} takes no {option}; {owner}.")
        for option in sorted(self.required - options.keys()):
            if option == "departing":
                raise ValueError(
                    f"{name} needs a departure time: pass departing=time(8, 30), "
                    f"a datetime, or seconds on its clock. There is no default, "
                    f"for the reason Zero() has to be asked for out loud."
                )
            raise ValueError(f"{name} needs {option}=, and it was not given.")
        return dict(options)

    @staticmethod
    def _no_other(options: "Dict[str, Any]", what: str) -> None:
        """Reject leftover options rather than silently ignoring them."""
        if options:
            raise ValueError(
                f"{what} has no {', '.join(sorted(options))}; that option belongs "
                f"to a different kind of search."
            )

    def _single_target(self, options: "Dict[str, Any]", searches: str) -> int:
        """Pop the one target a goal-directed search needs, or explain.

        Shared because more than one technique here is goal-directed: A* aims a
        heuristic at somewhere, a hierarchy climbs toward somewhere, and neither
        has anything to estimate or climb toward without exactly one somewhere.
        """
        targets = options.pop("targets", None)
        if targets is None or len(targets) != 1:
            count = "no target" if not targets else f"{len(targets)} targets"
            raise ValueError(
                f"{searches} searches toward a single target, and got {count}. Use "
                f"route(origin, destination), or pass targets=[node]."
            )
        return targets[0]

    def node_id(self, label: Hashable) -> int:
        """The dense id this environment gave ``label``."""
        return self._bound().node_id(label)

    def label(self, node_id: int) -> Hashable:
        """The label behind a dense id."""
        return self._bound().label(node_id)

    def _origin_ids(self, origins: Origins) -> "Dict[int, int]":
        """Resolve labelled origins to ``{node_id: initial_cost}``.

        A label can be any hashable, including a tuple like ``("stop", 7)``, so
        "one label" and "several labels" cannot be told apart by iterability. The
        rule: a mapping is labels with costs, a list/set/iterator is several
        labels, and anything else — tuples and strings included — is one label.
        """
        if isinstance(origins, Mapping):
            items: Any = origins.items()
        elif isinstance(origins, Iterable) and not isinstance(
            origins, (str, bytes, tuple)
        ):
            items = [(label, 0) for label in origins]
        else:
            items = [(origins, 0)]
        return {self.node_id(label): int(cost) for label, cost in items}

    def __repr__(self) -> str:
        return self._describe()

    def _describe(self, *configuration: str) -> str:
        """A technique reads as its configuration; a planner adds its data."""
        inside = ", ".join(configuration)
        bound = "" if self.environment is None else f" bound to {self.environment!r}"
        return f"{type(self).__name__}({inside}){bound}"


class Dijkstra(Planner):
    """Cheapest-cost routing over fixed-cost edges."""

    options = frozenset({"max_cost"})

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        return dijkstra(self._bound().graph, starts, **options)


class BFS(Planner):
    """Fewest-hops routing, ignoring edge costs.

    Origins all start at depth 0: a hop count cannot express "you are already
    part of the way there", so an initial cost has nowhere to go.
    """

    options = frozenset({"max_depth"})

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        priced = {self.label(node) for node, cost in starts.items() if cost}
        if priced:
            raise ValueError(
                f"BFS counts hops, so origins cannot carry an initial cost: "
                f"{', '.join(repr(label) for label in sorted(priced, key=repr))}"
            )
        return bfs(self._bound().graph, list(starts), **options)


class AStar(Planner):
    """Cheapest-cost routing, guided toward the destination by a heuristic.

        AStar(Euclidean()).bind(env).route("a", "b")

    Returns exactly what :class:`Dijkstra` returns, by settling fewer nodes — how
    many fewer is the whole question, and ``len(result.order)`` is how you answer
    it.

    The heuristic is required. A* whose heuristic quietly fell back to zero is
    Dijkstra wearing its name, which is the one thing a benchmark must never be
    unable to detect — so :class:`~routelab.heuristics.Zero` has to be asked for
    out loud.
    """

    options = frozenset({"max_cost"})

    def __init__(self, heuristic: Heuristic):
        self.heuristic_spec = heuristic

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the heuristic needs, on top of what any planner needs."""
        return super().missing_from(compiled) | self.heuristic_spec.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Bind the heuristic to this environment — where a landmark table,
        and any preprocessing after it, gets built."""
        self.heuristic = self.heuristic_spec.bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.heuristic.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> SearchResult:
        """Run the guided search. Requires exactly one target.

        A* is goal-directed: the estimate is an estimate *to somewhere*. Without
        a target there is nothing to aim at, and with several there is no single
        thing the heuristic could be a bound on.
        """
        target = self._single_target(options, "A*")
        return astar(self._bound().graph, starts, target, self.heuristic, **options)

    def __repr__(self) -> str:
        return self._describe(repr(self.heuristic_spec))


class ContractionHierarchy(Planner):
    """Exact routing by rewriting the graph, then only ever climbing it.

        ContractionHierarchy().bind(env).route("a", "b")

    Geisberger, Sanders, Schultes and Delling. Preprocessing contracts nodes one
    at a time, least important first, inserting a **shortcut** wherever removing a
    node would otherwise have lengthened a shortest path. What comes out is the
    original graph plus shortcuts and a rank per node — and a query that searches
    upward from the source and upward from the target and meets above the trip,
    never looking sideways at the thousands of streets in between.

    The answers are exact. Not approximately exact: the tests hold every distance
    to Dijkstra's on every instance, because a routing technique that is usually
    right is not a routing technique.

    Unlike every other technique here, this one searches a graph the environment
    has never seen. Its answers are unpacked back into the environment's own
    edges before anyone sees them, so journeys, geometry and provenance work
    exactly as they do for Dijkstra — a technique may search whatever it likes,
    but it answers in the caller's terms.

    A hierarchy query takes no bounds: its search is over the contracted graph,
    where a cost bound would cut off paths that are still cheap in the original.

    Args:
        ordering: Which node to contract next; see :mod:`routelab.orderings`.
            A policy, never a correctness choice — every ordering gives the same
            distances, and a bad one just builds a bigger hierarchy.
    """

    def __init__(self, ordering: Optional[Ordering] = None):
        self.ordering = ordering if ordering is not None else EdgeDifference()

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the ordering needs, on top of what any planner needs."""
        return super().missing_from(compiled) | self.ordering.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Contract the graph. The expensive step, and the whole technique."""
        self.hierarchy = self.ordering.bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.hierarchy.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        """Run the bidirectional query. Requires exactly one target.

        Returns a :class:`~routelab._routelab.MeetingSearch` rather than a
        `SearchResult`: two searches met in the middle, and neither half alone
        is the answer. It reports costs and paths in the environment's own edges,
        which is all :class:`~routelab.Journey` ever asked of a result.
        """
        target = self._single_target(options, "A hierarchy")
        return self.hierarchy.query(list(starts.items()), target)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """The two halves of the search, and where they met."""
        self._no_other(options, "meeting trees")
        return MeetingTrees(self._bound(), result)

    def __repr__(self) -> str:
        return self._describe(repr(self.ordering))


class TimeDependentDijkstra(Planner):
    """Cheapest arrival when the network is not always open.

        TimeDependentDijkstra().bind(env).route("a", "b", departing=time(8, 30))

    Dreyfus, *An Appraisal of Some Shortest-Path Algorithms* (1969): Dijkstra's
    algorithm generalises to a time-dependent network unchanged, provided
    arrival is non-decreasing in departure. Here it is, because travel times are
    constant and only availability varies — a gate is shut, a lane runs the
    other way — so leaving later cannot arrive earlier.

    This is a *different technique*, not :class:`Dijkstra` with an extra
    argument, which is what keeps a schedule from being ignored by accident.
    Ask for `Dijkstra` and you get the always-open network, honestly and
    knowingly; ask for this one and you get the clock. A departure time is
    required, and there is no default for it: a time-dependent query without a
    time is not a query with a sensible fallback, it is a different question.

    Args:
        waiting: ``"unrestricted"`` waits at a shut edge and pays the wait as
            travel time — arriving five minutes early beats an hour's detour,
            and a ten-hour wait loses to one, without anyone deciding which.
            ``"forbidden"`` treats a shut edge as absent, which is the control
            that shows waiting is doing real work.
    """

    options = frozenset({"departing", "max_cost"})
    required = frozenset({"departing"})

    WAITING = ("unrestricted", "forbidden")

    def __init__(self, waiting: str = "unrestricted"):
        if waiting not in self.WAITING:
            raise ValueError(
                f"unknown waiting policy {waiting!r}; expected "
                f"{' or '.join(repr(name) for name in self.WAITING)}"
            )
        self.waiting = waiting

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """A schedule to read, on top of what any planner needs. Without one
        this is Dijkstra with extra steps, and this says so before anything
        is built."""
        return super().missing_from(compiled) | Schedule.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Gather the layers' schedules into the calendar every query reads.

        The environment does not keep one; this technique is what reads it,
        so this technique derives it — and refuses here, at bind, if there is
        nothing to derive.
        """
        self.calendar = Schedule().bind(self._bound(), progress)

    def _footprint(self) -> int:
        return self.calendar.footprint

    def _search(self, starts: "Dict[int, int]", **options: Any) -> Result:
        departing = options.pop("departing")
        compiled = self._bound()
        return _routelab.time_dependent_dijkstra(
            compiled.graph,
            self.calendar,
            list(starts.items()),
            weekly_seconds(departing),
            waiting=self.waiting,
            **options,
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.waiting) if self.waiting != "unrestricted" else "")


class TimetablePlanner(Planner):
    """Earliest arrival over a timetable — what the timetable techniques share.

    Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
    Information in Public Transportation Systems* (ACM JEA 12, Article 2.4,
    2007). The paper's subject is not an algorithm but a *modelling* decision:
    a timetable is a set of departures, and there are two ways to make one into
    something a shortest-path algorithm can read. :class:`TimeExpanded` spends
    nodes; :class:`TimeDependent` spends search. :class:`RAPTOR` came five
    years later and builds no graph at all. All three must agree on every
    query, which is the paper's thesis and this library's test.

    Each accepts a ``"scalar"`` layer alongside the timetable and reads its
    edges between stops as **footpaths** — walks a rider may make at any time,
    for the edge's weight, which is what lets a real feed's two sides of a
    street be one place (see :class:`~routelab.Footpaths`). Each **requires** a
    timetable — bind one to a plain road network and ``missing_from`` says so
    before anything is built — and a departure time, on the service day's clock.

    A query may start at several stops, each already reached at its own time
    (``{stop: seconds}``): that is the multimodal query, and every technique
    here takes it the same way.
    """

    accepts: "frozenset[str]" = frozenset({"scalar", "timetable"})
    options = frozenset({"departing"})
    required = frozenset({"departing"})

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Departures to read, on top of what any planner needs. Without them
        there is no timetable to route over."""
        return super().missing_from(compiled) | Departures.missing_from(compiled)

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        """Gather the layers' departures into the timetable every model reads,
        and their scalar edges between stops into the footpaths every model
        walks.

        Derived here rather than kept on the environment, for the reason
        :class:`TimeDependentDijkstra` gives — and refused here, at bind, so
        a technique that cannot route says so before it is asked to.
        """
        compiled = self._bound()
        self.timetable = Departures().bind(compiled, progress)
        self.footpaths = Walks().bind(compiled, progress)

    def _footprint(self) -> int:
        """The timetable and the walks, in kernel form: what every model holds
        before it builds anything of its own."""
        return self.timetable.footprint + self.footpaths.footprint

    @property
    def searches(self) -> "Tuple[str, int]":
        return ("stops", self.timetable.num_stops)

    def _sources(self, starts: "Dict[int, int]", options: "Dict[str, Any]") -> "Tuple[List[Tuple[int, int]], int]":
        """The query's sources as ``(stop, time)`` on the service-day clock, and
        the departure everything is elapsed from."""
        at = service_seconds(options["departing"])
        return [(stop, at + head_start) for stop, head_start in starts.items()], at

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        """Run the model. The one line the models do not share."""
        raise NotImplementedError

    def _route(
        self,
        starts: "Dict[int, int]",
        target: int,
        destination: Hashable,
        options: "Dict[str, Any]",
    ) -> Optional[Journey]:
        """The earliest arrival at ``destination``, as a journey.

        A timetable technique answers with an itinerary — which vehicles, when
        — rather than a cost per node, so this is where the shared
        :meth:`Planner.route` hands over.
        """
        sources, at = self._sources(starts, options)
        itinerary = self._earliest_arrival(sources, target, options)
        if itinerary is None:
            return None
        return Journey.from_itinerary(self._bound(), itinerary, destination, at)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """Not this: a model that answers with a journey keeps no search space.

        Said in so many words rather than left to fail on a result that was
        never a table. RAPTOR keeps its rounds and reports them.
        """
        raise NotImplementedError(
            f"{type(self).__name__} answers with a journey and keeps no search "
            f"space, so there is nothing to draw. RAPTOR() reports its rounds."
        )


class TimeDependent(TimetablePlanner):
    """A node per stop, and a search that reads the clock (Pyrga et al. §4).

        TimeDependent().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    The graph stays the size of the network. Relaxing an edge is not reading a
    weight but asking what leaves next along it, which is a binary search over
    that edge's departures — so the model is small and the search is bespoke.
    Only the timetable and the walks are built at bind, which is the other half
    of the trade.

    Changing vehicles is instantaneous here, which is the paper's *simple*
    model. Its *realistic* one charges a minimum change time and is not
    expressible with one label per stop: staying in your seat must not be
    charged, so the cost of boarding depends on which vehicle you arrived on.
    The paper's answer is more nodes (§4.2); RAPTOR's rounds are another.
    """

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        return self.timetable.earliest_arrival(sources, target, self.footpaths)


class TimeExpanded(TimetablePlanner):
    """A node per event, and then it is just a graph (Pyrga et al. §3).

        TimeExpanded().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    Every departure and every arrival becomes its own node; riding and waiting
    become edges. What comes out is an ordinary static graph, so
    :func:`~routelab.dijkstra` routes it unchanged — and so would A*, or
    landmarks, or a contraction hierarchy. That is the model's whole appeal.

    Its cost is size. A city's weekday is a few thousand stops and hundreds of
    thousands of events, so the graph is built once at bind time and
    :attr:`footprint` is worth reading before you build one.
    """

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._expanded = _routelab.TimeExpanded.build(self.timetable, self.footpaths)

    def _footprint(self) -> int:
        return super()._footprint() + self._expanded.footprint

    @property
    def num_events(self) -> int:
        """Nodes in the expanded graph — the number the model is judged on."""
        self._bound()
        return self._expanded.num_events

    @property
    def searches(self) -> "Tuple[str, int]":
        return ("events", self.num_events)

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        return self._expanded.earliest_arrival(sources, target)


class RAPTOR(TimetablePlanner):
    """Round-based public transit routing (Delling, Pajor & Werneck, 2012).

        RAPTOR().bind(env).route(origin, target, departing=time(8, 30))

    No graph. Round `k` scans, once each, every route touched in round `k-1`
    and rides the earliest trip that can be caught, so after `k` rounds every
    stop holds its earliest arrival with at most `k-1` changes. That is
    one-to-all by construction — a label per stop per round — which is why,
    alone among the timetable techniques, this one has a real :meth:`search`
    and something to draw. It is Pareto by construction too: arrival against
    changes, one incomparable journey per round that improved something, which
    is what :meth:`frontier` hands back.

    ``max_transfers`` is a query option, not a constructor argument, for the
    reason ``max_cost`` is: it bounds one question, not the technique.
    Changing vehicles is instantaneous, as it is for the two Pyrga models it is
    checked against; a minimum change time is a new kernel ``Transfer``
    constructor and would land in all three at once, so it is not a knob here.
    """

    options = frozenset({"departing", "max_transfers"})

    def preprocess(self, progress: "Optional[_routelab.Progress]" = None) -> None:
        super().preprocess(progress)
        self._raptor = _routelab.Raptor.build(self.timetable, self.footpaths)

    def _footprint(self) -> int:
        return super()._footprint() + self._raptor.footprint

    @property
    def num_routes(self) -> int:
        """Routes in the paper's sense — distinct stop sequences whose trips
        never overtake — which is more than a feed's own count of routes."""
        self._bound()
        return self._raptor.num_routes

    @property
    def num_trips(self) -> int:
        self._bound()
        return self._raptor.num_trips

    @staticmethod
    def _rounds(max_transfers: Optional[int]) -> Optional[int]:
        """`k` changes is `k + 1` trips is `k + 1` rounds."""
        if max_transfers is None:
            return None
        if max_transfers < 0:
            raise ValueError(
                f"max_transfers counts changes of vehicle, so it cannot be {max_transfers}"
            )
        return int(max_transfers) + 1

    # RAPTOR keeps a label per stop per round, so it has a cost table like any
    # graph search and `Planner.route` — search, then read the journey off the
    # result — is the whole implementation. The itinerary hook the two Pyrga
    # models need is not used here.
    _route = Planner._route

    def _search(self, starts: "Dict[int, int]", **options: Any) -> "_routelab.RaptorSearch":
        """Run the rounds — toward one target if given, else to every stop."""
        sources, at = self._sources(starts, options)
        target = None
        if options.get("targets"):
            target = self._single_target(options, "RAPTOR, pruning toward a target,")
        return self._raptor.search(
            sources, target, self._rounds(options.get("max_transfers")), at
        )

    def journey(self, result: Result, destination: Hashable) -> Optional[Journey]:
        """The earliest arrival at ``destination`` a kept search holds."""
        itinerary = result.itinerary(self.node_id(destination))  # type: ignore[attr-defined]
        if itinerary is None:
            return None
        return Journey.from_itinerary(self._bound(), itinerary, destination, result.departing)

    def journeys(self, result: Result, destination: Hashable) -> "List[Journey]":
        """Every journey a kept search holds for ``destination``: the earliest
        arrival for each number of changes, fewest changes first, none
        dominated by another.

        The counterpart to :meth:`journey`, and what :meth:`frontier` is in
        terms of — so a caller holding a search never pays for a second one.
        """
        compiled = self._bound()
        target = self.node_id(destination)
        return [
            Journey.from_itinerary(compiled, itinerary, destination, result.departing)
            for itinerary in result.itineraries(target)  # type: ignore[attr-defined]
        ]

    def frontier(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> "List[Journey]":
        """Every journey worth having, in one call: the Pareto front over
        arrival time and changes, where :meth:`route` returns only its last
        entry.
        """
        options = self._options(options)
        starts = self._origin_ids(origin)
        result = self._search(starts, targets=[self.node_id(destination)], **options)
        return self.journeys(result, destination)

    def explored(self, result: Result, **options: Any) -> SearchSpace:
        """Every stop the rounds reached, by the round that first got there."""
        self._no_other(options, "rounds")
        return Rounds(self._bound(), result)


def route(
    technique: Planner,
    environment: Environment,
    origin: Origins,
    destination: Hashable,
    **options: Any,
) -> Optional[Journey]:
    """One-shot routing: bind a technique, ask it one question, throw it away.

        route(AStar(Landmarks(16)), env, "a", "c")

    Convenient for a single query. When you are asking more than one, bind the
    technique yourself and keep the planner — that is what makes preprocessing
    worth doing, and this function throws it away every time.

    There is no registry of names here on purpose. A technique is a value: a
    dictionary of them is a line a caller writes, and no fixed set the library
    could ship would serve both a demo's dropdown and a parameter sweep.
    """
    return technique.bind(environment).route(origin, destination, **options)
