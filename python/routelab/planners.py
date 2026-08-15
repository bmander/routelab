"""Planners: an algorithm bound to an environment.

    planner = Dijkstra(env)          # preprocess once
    planner.route("a", "c")          # query many times

The two-step shape is the point. Dijkstra has nothing to precompute, so
constructing one is nearly free — but every algorithm this project is heading
toward (ULTRA's transfer shortcuts, transfer patterns, contraction hierarchies)
earns its query speed with a preprocessing step whose cost is paid once and
amortized over queries. :meth:`Planner.preprocess` is where that work goes, so
the API does not have to change shape when it arrives.

A planner holds the environment as it was when the planner was built. Register
another layer and you have a different world, so build a new planner for it.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Type, Union

from .environment import CompiledEnvironment, Environment
from .heuristics import Heuristic
from .journey import Journey
from .search import SearchResult, astar, bfs, dijkstra

__all__ = ["AStar", "BFS", "Dijkstra", "PLANNERS", "Planner", "route"]

#: How a caller names where to start: one label, a list of labels, or a mapping
#: of labels to the cost of already being there. Tuples and strings are single
#: labels — a label may itself be a tuple, so iterability cannot decide this.
Origins = Union[Hashable, Iterable[Hashable], Mapping[Hashable, int]]


class Planner:
    """Base class: an algorithm bound to a compiled environment."""

    #: Cost models this algorithm knows how to route over.
    accepts: "frozenset[str]" = frozenset({"scalar"})

    def __init__(self, environment: Environment):
        unsupported = environment.cost_models - self.accepts
        if unsupported:
            raise TypeError(
                f"{type(self).__name__} cannot route over "
                f"{', '.join(sorted(unsupported))} layers; it accepts "
                f"{', '.join(sorted(self.accepts))}"
            )
        self.environment = environment
        self.compiled: CompiledEnvironment = environment.compile()
        self.preprocess()

    def preprocess(self) -> None:
        """Work done once, before any query. Nothing to do for a plain search."""

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
        return Journey.from_result(self.compiled, result, destination)

    def node_id(self, label: Hashable) -> int:
        """The dense id this environment gave ``label``."""
        return self.compiled.node_id(label)

    def label(self, node_id: int) -> Hashable:
        """The label behind a dense id."""
        return self.compiled.label(node_id)

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
        return f"{type(self).__name__}({self.environment!r})"


class Dijkstra(Planner):
    """Cheapest-cost routing over fixed-cost edges."""

    def search(self, origins: Origins, **options: Any) -> SearchResult:
        return dijkstra(self.compiled.graph, self._origin_ids(origins), **options)


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
        return bfs(self.compiled.graph, list(starts), **options)


class AStar(Planner):
    """Cheapest-cost routing, guided toward the destination by a heuristic.

        AStar(env, Euclidean()).route("a", "b")

    Returns exactly what :class:`Dijkstra` returns, by settling fewer nodes — how
    many fewer is the whole question, and ``len(result.order)`` is how you answer
    it.

    The heuristic is required. A* whose heuristic quietly fell back to zero is
    Dijkstra wearing its name, which is the one thing a benchmark must never be
    unable to detect — so :class:`~routelab.heuristics.Zero` has to be asked for
    out loud.
    """

    def __init__(self, environment: Environment, heuristic: Heuristic):
        self.heuristic_spec = heuristic
        super().__init__(environment)

    def preprocess(self) -> None:
        """Bind the heuristic to this environment, once, before any query."""
        self.heuristic = self.heuristic_spec.bind(self.compiled)

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
            self.compiled.graph, self._origin_ids(origins), targets[0], self.heuristic, **options
        )

    def __repr__(self) -> str:
        return f"AStar({self.environment!r}, {self.heuristic_spec!r})"


#: Planners by name, so a benchmark can loop over algorithms instead of naming
#: them. :class:`AStar` is deliberately absent: it needs a heuristic, and a
#: registry entry would have to invent one. Compare configured planners by
#: building them — ``[Dijkstra(env), AStar(env, Euclidean())]``.
PLANNERS: "Dict[str, Type[Planner]]" = {"dijkstra": Dijkstra, "bfs": BFS}


def route(
    planner: "Union[str, Type[Planner]]",
    environment: Environment,
    origin: Origins,
    destination: Hashable,
    **options: Any,
) -> Optional[Journey]:
    """One-shot routing: build a planner, ask it one question, throw it away.

        route(Dijkstra, env, "a", "c")

    Convenient for a single query and for comparing algorithms in a loop. When
    you are asking more than one question, build the planner yourself and keep
    it — that is what makes preprocessing worth doing.
    """
    if isinstance(planner, str):
        try:
            planner = PLANNERS[planner]
        except KeyError:
            known = ", ".join(sorted(PLANNERS))
            raise KeyError(f"unknown planner {planner!r}; known planners: {known}") from None
    return planner(environment).route(origin, destination, **options)
