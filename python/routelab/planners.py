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
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Union

from . import _routelab
from .clock import service_seconds, weekly_seconds
from .environment import CompiledEnvironment, Environment
from .heuristics import Heuristic
from .journey import Journey
from .orderings import EdgeDifference, Ordering
from .search import Result, SearchResult, astar, bfs, dijkstra
from .searchspace import MeetingTrees, SearchSpace, ShortestPathTree

__all__ = [
    "AStar",
    "BFS",
    "ContractionHierarchy",
    "Dijkstra",
    "Planner",
    "TimeDependent",
    "TimeDependentDijkstra",
    "TimeExpanded",
    "TimetablePlanner",
    "route",
]

#: How a caller names where to start: one label, a list of labels, or a mapping
#: of labels to the cost of already being there. Tuples and strings are single
#: labels — a label may itself be a tuple, so iterability cannot decide this.
Origins = Union[Hashable, Iterable[Hashable], Mapping[Hashable, int]]


class Planner:
    """A routing technique: configured on construction, bound to data later."""

    #: Cost models this algorithm knows how to route over.
    accepts: "frozenset[str]" = frozenset({"scalar"})

    #: The environment this was bound to, or ``None`` while it is still just a
    #: configuration.
    environment: Optional[Environment] = None
    compiled: Optional[CompiledEnvironment] = None

    #: A counter this planner's preprocessing writes into, if it was given one
    #: and has anything honest to count. Set by :meth:`bind`.
    progress: "Optional[_routelab.Progress]" = None

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
        bound.progress = progress
        bound.preprocess()
        return bound

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Everything standing between this technique and that environment.

        Empty means :meth:`bind` will work. Answered without binding anything,
        so a study can skip what a dataset cannot support before spending the
        preprocessing to find out — which only holds if this covers everything
        ``bind`` checks, so it does: cost models the technique does not accept,
        and (in subclasses) capabilities the environment does not provide.

        Entries are short names either way: ``"timetable"`` for a cost model
        this algorithm cannot route over, ``"positions"`` for something the
        layers never supplied.
        """
        return compiled.cost_models - self.accepts

    def preprocess(self) -> None:
        """Work done once at bind time, before any query.

        Nothing to do for a plain search; this is where a landmark table or a
        set of shortcuts gets built.
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

    def _bound(self) -> CompiledEnvironment:
        """The compiled environment, or an error naming what is missing."""
        if self.compiled is None:
            raise ValueError(
                f"{self!r} is a technique, not a planner — bind it to an "
                f"environment first: {type(self).__name__}(...).bind(env)"
            )
        return self.compiled

    def search(self, origins: Origins, **options: Any) -> Result:
        """Run the search and return the raw, id-keyed result.

        The escape hatch: everything the kernel computed, without the journey
        packaging. Node ids here are dense, so use :meth:`node_id`/:meth:`label`
        to get between them and your labels.
        """
        raise NotImplementedError

    def route(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> Optional[Journey]:
        """The cheapest journey from ``origin`` to ``destination``.

        Args:
            origin: A label, several labels, or ``{label: initial_cost}`` — the
                last being how a multimodal query starts, with each entry point
                already costing an access walk.
            destination: The label to route to.
            **options: Passed through to the search (``max_cost``, ``max_depth``).

        Returns:
            A :class:`~routelab.Journey`, or ``None`` if the destination cannot
            be reached under the given bounds.
        """
        target = self.node_id(destination)
        result = self.search(origin, targets=[target], **options)
        if result.cost(target) is None:
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

    @staticmethod
    def _no_other(options: "Dict[str, Any]", what: str) -> None:
        """Reject leftover options rather than silently ignoring them."""
        if options:
            raise ValueError(
                f"{what} has no {', '.join(sorted(options))}; that option belongs "
                f"to a different kind of search."
            )

    def _reads_no_clock(self, options: "Dict[str, Any]") -> None:
        """Refuse a departure time, naming the technique that takes one.

        Every planner but one routes the network as though it were always open,
        which is a fair question to ask and a terrible one to answer by
        accident. So the refusal says what to ask instead, rather than letting
        the kernel complain about a keyword argument it has never heard of.
        """
        if "departing" in options:
            raise ValueError(
                f"{type(self).__name__} routes the network as though it were "
                f"always open, so it has no departure time. Use "
                f"TimeDependentDijkstra() to read the clock, or drop "
                f"departing= to search the always-open network knowingly."
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

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        self._reads_no_clock(options)
        return dijkstra(self._bound().graph, self._origin_ids(origins), **options)


class BFS(Planner):
    """Fewest-hops routing, ignoring edge costs.

    Origins all start at depth 0: a hop count cannot express "you are already
    part of the way there", so an initial cost has nowhere to go.
    """

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        self._reads_no_clock(options)
        starts = self._origin_ids(origins)
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

    def __init__(self, heuristic: Heuristic):
        self.heuristic_spec = heuristic

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Whatever the heuristic needs, on top of what any planner needs."""
        return super().missing_from(compiled) | self.heuristic_spec.missing_from(compiled)

    def preprocess(self) -> None:
        """Bind the heuristic to this environment — where a landmark table,
        and any preprocessing after it, gets built."""
        self.heuristic = self.heuristic_spec.bind(self._bound(), self.progress)

    def _footprint(self) -> int:
        return self.heuristic.footprint

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        """Run the guided search. Requires exactly one target.

        A* is goal-directed: the estimate is an estimate *to somewhere*. Without
        a target there is nothing to aim at, and with several there is no single
        thing the heuristic could be a bound on.
        """
        self._reads_no_clock(options)
        target = self._single_target(options, "A*")
        return astar(
            self._bound().graph, self._origin_ids(origins), target, self.heuristic, **options
        )

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

    def preprocess(self) -> None:
        """Contract the graph. The expensive step, and the whole technique."""
        self.hierarchy = self.ordering.bind(self._bound(), self.progress)

    def _footprint(self) -> int:
        return self.hierarchy.footprint

    def search(self, origins: Origins, **options: Any) -> Result:
        """Run the bidirectional query. Requires exactly one target.

        Returns a :class:`~routelab._routelab.MeetingSearch` rather than a
        `SearchResult`: two searches met in the middle, and neither half alone
        is the answer. It reports costs and paths in the environment's own edges,
        which is all :class:`~routelab.Journey` ever asked of a result.
        """
        self._reads_no_clock(options)
        target = self._single_target(options, "A hierarchy")
        if options:
            unsupported = ", ".join(sorted(options))
            raise ValueError(
                f"a hierarchy query takes no bounds; got {unsupported}. Its search "
                f"is over the contracted graph, where a cost bound would cut off "
                f"paths that are still cheap in the original."
            )
        return self.hierarchy.query(list(self._origin_ids(origins).items()), target)

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
    knowingly; ask for this one and you get the clock.

    Args:
        waiting: ``"unrestricted"`` waits at a shut edge and pays the wait as
            travel time — arriving five minutes early beats an hour's detour,
            and a ten-hour wait loses to one, without anyone deciding which.
            ``"forbidden"`` treats a shut edge as absent, which is the control
            that shows waiting is doing real work.
    """

    #: A schedule to read. Without one this is Dijkstra with extra steps, and
    #: `missing_from` says so before anything is built.
    requires: "frozenset[str]" = frozenset({"schedule"})

    WAITING = ("unrestricted", "forbidden")

    def __init__(self, waiting: str = "unrestricted"):
        if waiting not in self.WAITING:
            raise ValueError(
                f"unknown waiting policy {waiting!r}; expected "
                f"{' or '.join(repr(name) for name in self.WAITING)}"
            )
        self.waiting = waiting

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        return super().missing_from(compiled) | (self.requires - compiled.provides)

    def search(self, origins: Origins, **options: Any) -> Result:
        """Run the search from a departure time. There is no default for it.

        A time-dependent query without a time is not a query with a sensible
        fallback — it is a different question. Asking for one loudly is the same
        rule that makes :class:`~routelab.Zero` something you have to name.
        """
        departing = options.pop("departing", None)
        if departing is None:
            raise ValueError(
                "a time-dependent search needs a departure time: pass "
                "departing=datetime(...) or departing=time(8, 30). Use "
                "Dijkstra() to route the network as if it were always open."
            )
        compiled = self._bound()
        return _routelab.time_dependent_dijkstra(
            compiled.graph,
            compiled.calendar,
            list(self._origin_ids(origins).items()),
            weekly_seconds(departing),
            waiting=self.waiting,
            **options,
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.waiting) if self.waiting != "unrestricted" else "")


class TimetablePlanner(Planner):
    """Earliest arrival over a timetable — the shared half of two models.

    Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
    Information in Public Transportation Systems* (ACM JEA 12, Article 2.4,
    2007). The paper's subject is not an algorithm but a *modelling* decision:
    a timetable is a set of departures, and there are two ways to make one into
    something a shortest-path algorithm can read. :class:`TimeExpanded` spends
    nodes; :class:`TimeDependent` spends search. They must agree on every query,
    which is the paper's thesis and this library's test.

    Both accept a ``"scalar"`` layer alongside the timetable so an environment
    can carry stop geometry, and both **require** a timetable — bind one to a
    plain road network and ``missing_from`` says so before anything is built.
    """

    accepts: "frozenset[str]" = frozenset({"scalar", "timetable"})

    #: Departures to read. Without them there is no timetable to route over.
    requires: "frozenset[str]" = frozenset({"timetable"})

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        return super().missing_from(compiled) | (self.requires - compiled.provides)

    def _timetable(self) -> "_routelab.Timetable":
        timetable = self._bound().timetable
        if timetable is None:
            raise ValueError(
                f"{self!r} needs a timetable and this environment has none. "
                f"Register a layer that keeps departures — "
                f"routelab.sources.gtfs.GTFS(path, date)."
            )
        return timetable

    def _earliest_arrival(
        self, origin: Hashable, destination: Hashable, departing: int
    ) -> "Optional[_routelab.Itinerary]":
        """Run the model. The one line the two subclasses do not share."""
        raise NotImplementedError

    def search(self, origins: Origins, **options: Any) -> Result:
        """Not this. A timetable query answers with an itinerary.

        The other planners return a cost per node because that is what their
        kernels compute. These compute one journey and report what settling it
        cost, so there is no table to hand back — and returning an empty one
        would be worse than saying so.
        """
        raise NotImplementedError(
            f"{type(self).__name__} answers with an itinerary rather than a "
            f"cost per node — use route(origin, destination, departing=...)."
        )

    def route(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> Optional[Journey]:
        """The earliest arrival at ``destination``, leaving no earlier than
        ``departing``.

        Args:
            origin: One stop label. Several origins would each have their own
                departure time — an access walk of a different length to each —
                and that is the multimodal query, not this one.
            destination: The stop label to reach.
            departing: When you can first leave, on the service day's clock. No
                default, for the reason :class:`TimeDependentDijkstra` has none.
        """
        departing = options.pop("departing", None)
        if departing is None:
            raise ValueError(
                "a timetable search needs a departure time: pass "
                "departing=time(8, 30) or departing=30600, seconds since the "
                "service day's midnight."
            )
        self._no_other(options, "a timetable search")
        if isinstance(origin, Mapping) or (
            isinstance(origin, Iterable) and not isinstance(origin, (str, bytes, tuple))
        ):
            raise ValueError(
                f"{type(self).__name__} departs from one stop at one time; "
                f"several origins would each need their own departure time."
            )
        at = service_seconds(departing)
        itinerary = self._earliest_arrival(origin, destination, at)
        if itinerary is None:
            return None
        return Journey.from_itinerary(self._bound(), itinerary, origin, at)


class TimeDependent(TimetablePlanner):
    """A node per stop, and a search that reads the clock (Pyrga et al. §4).

        TimeDependent().bind(env).route("1_1234", "1_5678", departing=time(8, 30))

    The graph stays the size of the network. Relaxing an edge is not reading a
    weight but asking what leaves next along it, which is a binary search over
    that edge's departures — so the model is small and the search is bespoke.
    Nothing is built at bind time, which is the other half of the trade.

    Changing vehicles is instantaneous here, which is the paper's *simple*
    model. Its *realistic* one charges a minimum change time and is not
    expressible with one label per stop: staying in your seat must not be
    charged, so the cost of boarding depends on which vehicle you arrived on.
    The paper's answer is more nodes (§4.2), and that is its own increment.
    """

    def _earliest_arrival(
        self, origin: Hashable, destination: Hashable, departing: int
    ) -> "Optional[_routelab.Itinerary]":
        return self._timetable().earliest_arrival(
            self.node_id(origin), departing, self.node_id(destination)
        )


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

    def preprocess(self) -> None:
        self._expanded = _routelab.TimeExpanded.build(self._timetable())

    def _footprint(self) -> int:
        return self._expanded.footprint

    @property
    def num_events(self) -> int:
        """Nodes in the expanded graph — the number the model is judged on."""
        self._bound()
        return self._expanded.num_events

    def _earliest_arrival(
        self, origin: Hashable, destination: Hashable, departing: int
    ) -> "Optional[_routelab.Itinerary]":
        return self._expanded.earliest_arrival(
            self.node_id(origin), departing, self.node_id(destination)
        )


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
