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

from .environment import CompiledEnvironment, Environment
from .heuristics import Heuristic
from .journey import Journey
from .search import SearchResult, astar, bfs, dijkstra
from .searchspace import SearchSpace, ShortestPathTree

__all__ = ["AStar", "BFS", "Dijkstra", "Planner", "route"]

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

    def bind(self, environment: Environment) -> "Planner":
        """Attach this technique to an environment and do its preprocessing.

        Returns a *new* planner and leaves this one as it was, so a technique can
        be bound to several environments — which is the whole point of writing
        one down: the same configuration, measured across datasets.
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

    def _bound(self) -> CompiledEnvironment:
        """The compiled environment, or an error naming what is missing."""
        if self.compiled is None:
            raise ValueError(
                f"{self!r} is a technique, not a planner — bind it to an "
                f"environment first: {type(self).__name__}(...).bind(env)"
            )
        return self.compiled

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        """Run the one-to-all search and return the raw, id-keyed result.

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

    def explored(self, result: SearchResult, magnitude: str = "weight") -> SearchSpace:
        """What the search looked at, in a form something can draw.

        Dijkstra and A* — and BFS — explore by growing a shortest-path tree, so
        that is what they report. An algorithm that explores differently returns
        a different :class:`~routelab.SearchSpace`; the promise is only that
        whatever it explored can be rendered.

        Args:
            result: A result from :meth:`search`, or from :meth:`route` if you
                kept one. The tree is rebuilt from it rather than recorded during
                the search, so asking costs nothing until you ask.
            magnitude: What each branch should carry from the subtree beyond it:
                ``"weight"`` for travel time, ``"nodes"`` for a count.
        """
        return ShortestPathTree(self._bound(), result, magnitude)

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
        return dijkstra(self._bound().graph, self._origin_ids(origins), **options)


class BFS(Planner):
    """Fewest-hops routing, ignoring edge costs.

    Origins all start at depth 0: a hop count cannot express "you are already
    part of the way there", so an initial cost has nowhere to go.
    """

    def search(self, origins: Origins, **options: Any) -> SearchResult:
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
        self.heuristic = self.heuristic_spec.bind(self._bound())

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        """Run the guided search. Requires exactly one target.

        A* is goal-directed: the estimate is an estimate *to somewhere*. Without
        a target there is nothing to aim at, and with several there is no single
        thing the heuristic could be a bound on.
        """
        targets = options.pop("targets", None)
        if targets is None or len(targets) != 1:
            count = "no target" if not targets else f"{len(targets)} targets"
            raise ValueError(
                f"A* searches toward a single target, and got {count}. Use "
                f"route(origin, destination), or pass targets=[node]."
            )
        return astar(
            self._bound().graph, self._origin_ids(origins), targets[0], self.heuristic, **options
        )

    def __repr__(self) -> str:
        return self._describe(repr(self.heuristic_spec))


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
