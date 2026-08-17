"""The shape every technique has: configure, bind, then ask.

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

from .. import _routelab
from ..model.environment import CompiledEnvironment, Environment
from ..model.journey import Journey
from ..model.search import Result
from ..model.searchspace import SearchSpace, ShortestPathTree
from ..util.clock import service_seconds
from .departures import Departures, Walks
from .schedule import Schedule

__all__ = [
    "OPTIONS",
    "Front",
    "Origins",
    "Planner",
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
    """The techniques that take ``option`` on any of their verbs — who a
    refusal should point at."""
    return [cls for cls in techniques() if option in cls.options or option in cls.verbs]


def verb_for(option: str) -> Optional[str]:
    """The verb ``option`` belongs to, when it is not :meth:`Planner.route`'s
    or :meth:`Planner.search`'s and every technique that takes it agrees on
    which one — so a refusal can say *where* to pass it, not just to whom.
    ``None`` when they disagree, which is when the verb is not the useful half
    of the answer anyway."""
    where = {cls.verbs[option] for cls in techniques() if option in cls.verbs}
    return where.pop() if len(where) == 1 else None


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
    "until": "a departure window",
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

    #: Options this technique takes on a verb of its own rather than on
    #: :meth:`route` or :meth:`search`, as ``{option: verb}``. Declared for the
    #: same reason :attr:`options` is: a refusal names who takes the option and
    #: where to pass it by asking the shelf, so no sentence has to spell out a
    #: technique that may not be the only one tomorrow.
    verbs: "Dict[str, str]" = {}

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
            verb = verb_for(option)
            where = takers if verb is None else f"{verb}() on {takers}"
            owner = f"{described} belongs to {where}" if takers else f"no technique here takes {described}"
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


class TimetablePlanner(Planner):
    """Earliest arrival over a timetable — what the timetable techniques share.

    Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
    Information in Public Transportation Systems* (ACM JEA 12, Article 2.4,
    2007). The paper's subject is not an algorithm but a *modelling* decision:
    a timetable is a set of departures, and there are two ways to make one into
    something a shortest-path algorithm can read. :class:`TimeExpanded` spends
    nodes; :class:`TimeDependent` spends search. :class:`RAPTOR` came five
    years later and builds no graph at all; :class:`CSA` a year after that and
    keeps only the departures, sorted; :class:`TripBased` two years on again
    and labels trips, with the changes between them computed once. All five
    must agree on every query, which is the paper's thesis and this library's
    test.

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

    def _window(self, departing: Any, until: Any) -> "Tuple[int, int]":
        """A departure window as ``(opens, closes)`` on the service-day clock.

        Shared because more than one technique here answers over a range of
        departures rather than from one moment, and a window is refused the
        same way whoever asked: a missing end is a question that was not
        finished, and one that closes before it opens is not a window.
        """
        if departing is None or until is None:
            raise ValueError(
                f"{type(self).__name__}.profile needs a departure window: pass "
                f"departing=time(8, 30), until=time(10, 30) — times, datetimes, "
                f"or seconds on the service-day clock."
            )
        opens, closes = service_seconds(departing), service_seconds(until)
        if closes < opens:
            raise ValueError(
                f"a departure window cannot close before it opens: "
                f"departing={opens} is after until={closes}"
            )
        return opens, closes

    def _merge_departures(
        self,
        found: "List[Tuple[int, _routelab.Itinerary]]",
        destination: Hashable,
    ) -> "List[Journey]":
        """Several origins' profiles, merged on the query's clock.

        Shared because a profile from several origins is not the paper's
        question but this library's: each origin is read out on its own clock
        (its head start already subtracted) and what survives is every
        itinerary that arrives strictly earlier than anything leaving later.
        Latest departure first while merging, earliest first coming back.

        The journey is built from the survivors only, since building one asks
        the environment for an edge per leg and most origins lose most pairs.
        """
        compiled = self._bound()
        found.sort(key=lambda pair: (pair[0], -pair[1].arrives))
        kept: "List[Journey]" = []
        best: Optional[int] = None
        for departs, itinerary in reversed(found):
            if best is None or itinerary.arrives < best:
                kept.append(
                    Journey.from_itinerary(compiled, itinerary, destination, departs)
                )
                best = itinerary.arrives
        kept.reverse()
        return kept

    @staticmethod
    def _changes(max_transfers: Optional[int]) -> Optional[int]:
        """``max_transfers`` as a checked count of changes.

        Shared because the cap means the same thing to every technique that
        takes it — and because a technique that counts something else from it
        should still refuse a negative one in the same words. RAPTOR counts
        rounds, so it adds the one that a change of vehicle costs.
        """
        if max_transfers is None:
            return None
        if max_transfers < 0:
            raise ValueError(
                f"max_transfers counts changes of vehicle, so it cannot be {max_transfers}"
            )
        return int(max_transfers)

    def _earliest_arrival(
        self, sources: "List[Tuple[int, int]]", target: int, options: "Dict[str, Any]"
    ) -> "Optional[_routelab.Itinerary]":
        """Run the model. The one line the models do not share."""
        raise NotImplementedError

    def journey(self, result: Result, destination: Hashable) -> Optional[Journey]:
        """The earliest arrival at ``destination`` a kept search holds.

        A timetable technique that keeps a table — a label per stop, however
        it got there — reads an *itinerary* off it rather than an edge path,
        which is why this is the family's and not :class:`Planner`'s: the
        result knows which vehicles were ridden, and a leg is one of those.
        """
        itinerary = result.itinerary(self.node_id(destination))  # type: ignore[attr-defined]
        if itinerary is None:
            return None
        return Journey.from_itinerary(self._bound(), itinerary, destination, result.departing)

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
        never a table. Who to ask instead is read off the shelf — the kernels
        that override this — so a new one joins the sentence by existing,
        the way :func:`clock_readers` does it.
        """
        drawers = names(
            cls
            for cls in techniques()
            if issubclass(cls, TimetablePlanner)
            and cls.explored is not TimetablePlanner.explored
        )
        raise NotImplementedError(
            f"{type(self).__name__} answers with a journey and keeps no search "
            f"space, so there is nothing to draw. The techniques that keep a "
            f"table report one: ask {drawers}."
        )


class Front:
    """Answers with a Pareto front, not just its best entry.

    What a technique whose search counts changes as it goes can hand back: one
    journey per number of changes that arrives strictly earlier than any
    journey with fewer. :class:`RAPTOR` gets it from its rounds and
    :class:`TripBased` from its transfer counts, and both read it off a result
    the same way — so the two verbs are written here rather than twice.

    Not on :class:`TimetablePlanner`, because it is not the family's: a
    connection scan counts nothing but time, and :class:`CSA` has no front to
    read off a search that never distinguished one journey from another by
    changes. A mixin is what says *these* techniques, rather than widening the
    base until a technique inherits a verb it cannot honour.
    """

    def journeys(self, result: Result, destination: Hashable) -> "List[Journey]":
        """Every journey a kept search holds for ``destination``: the earliest
        arrival for each number of changes, fewest changes first, none
        dominated by another.

        The counterpart to :meth:`Planner.journey`, and what :meth:`frontier`
        is in terms of — so a caller holding a search never pays for a second
        one.
        """
        compiled = self._bound()  # type: ignore[attr-defined]
        target = self.node_id(destination)  # type: ignore[attr-defined]
        return [
            Journey.from_itinerary(compiled, itinerary, destination, result.departing)
            for itinerary in result.itineraries(target)  # type: ignore[attr-defined]
        ]

    def frontier(
        self, origin: Origins, destination: Hashable, **options: Any
    ) -> "List[Journey]":
        """Every journey worth having, in one call: the Pareto front over
        arrival time and changes, where :meth:`Planner.route` returns only its
        last entry.
        """
        options = self._options(options)  # type: ignore[attr-defined]
        starts = self._origin_ids(origin)  # type: ignore[attr-defined]
        result = self._search(  # type: ignore[attr-defined]
            starts, targets=[self.node_id(destination)], **options  # type: ignore[attr-defined]
        )
        return self.journeys(result, destination)


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
