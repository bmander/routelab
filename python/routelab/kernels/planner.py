"""The shape every technique has: configure, bind, then ask.

    technique = AStar(Landmarks(16))     # configuration, costing nothing
    planner = technique.bind(env)        # preprocessing, costing seconds
    planner.route("a", "c")              # queries, costing milliseconds

Three steps rather than two, and the middle one is why. Every algorithm this
project is heading toward — ULTRA's transfer shortcuts, transfer patterns,
contraction hierarchies — earns its query speed with preprocessing paid once and
amortized. Sixteen landmarks over a city is already a second and 33 MB. Spending
that inside a constructor hides it; spending it in :meth:`Technique.bind` gives
it a verb.

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

A :class:`Technique` and a :class:`Planner` are different types, and that is the
whole of the contract. A technique holds configuration and can be bound; only a
planner has ``route``. Each planner declares the query options it takes on its
own signature, keyword-only, so what a technique understands is read off the
signature rather than out of a registry — an option nobody takes is a
``TypeError`` from Python and an error from a type checker, in the caller's own
words, before anything runs.

A bound planner holds the environment as it was when it was bound. Register
another layer and you have a different world; bind the technique again.
"""

from __future__ import annotations

import copy
from typing import (
    Any,
    Generic,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .. import _routelab
from ..model.environment import CompiledEnvironment, Environment
from ..model.answer import Answer
from ..model.journey import Journey
from ..model.search import EdgeResult, FrontResult, StopResult
from ..model.searchspace import ShortestPathTree
from ..util.clock import Departure, service_seconds
from .departures import Departures, Walks

__all__ = [
    "Front",
    "GraphPlanner",
    "Origins",
    "Planner",
    "Tabled",
    "Technique",
    "TimetablePlanner",
    "TimetableTechnique",
    "TreePlanner",
    "route",
]


#: How a caller names where to start: one label, a list of labels, or a mapping
#: of labels to the cost of already being there. Tuples and strings are single
#: labels — a label may itself be a tuple, so iterability cannot decide this.
Origins = Union[Hashable, Iterable[Hashable], Mapping[Hashable, int]]

#: What a kept search is, for the techniques that keep one. Bound to
#: :class:`~routelab.model.search.StopResult` because reading a journey off a
#: search means asking it for an itinerary and when the query left.
_Result = TypeVar("_Result", bound=StopResult)

#: A technique, as itself: what :meth:`TimetableTechnique._with_walks` hands
#: back, so that a copy of a `RAPTOR` is still a `RAPTOR` and its `bind` still
#: promises a `RAPTORPlanner`.
_Technique = TypeVar("_Technique", bound="TimetableTechnique")


class Technique:
    """A routing algorithm, configured and not yet attached to anything.

    A value: write it down, name it, put it in a list, point it at more than
    one dataset. It answers two questions — whether a dataset can support it
    (:meth:`missing_from`) and what a planner over that dataset is
    (:meth:`bind`) — and nothing else, because a technique with no data has
    nothing to route.
    """

    #: Cost models this algorithm knows how to route over. A bind-time refusal
    #: rather than a query-time one: routing a timetable as though its edges
    #: had weights is a real path and a wrong answer, and the layers say which
    #: it is before anything is built.
    accepts: "frozenset[str]" = frozenset({"scalar"})

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

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "Planner":
        """Attach this technique to an environment and do its preprocessing.

        Returns a *new* planner and leaves this technique as it was, so one
        configuration can be bound to several environments — which is the whole
        point of writing it down: the same technique, measured across datasets.
        Each subclass narrows the return type to its own planner, so what a
        caller may ask next is known before anything runs.

        Args:
            progress: A counter to write into while preprocessing runs, for
                anything watching from another thread. Six seconds of
                contraction is long enough that "still working" and "hung" want
                telling apart. Techniques with no honest measure of their own
                progress leave it alone — see :mod:`routelab._routelab` and
                `Progress.fraction` returning ``None``.
        """
        raise NotImplementedError

    def _compile(self, environment: Environment) -> CompiledEnvironment:
        """The environment as this technique may read it, or a refusal.

        Where the cost-model check lives, because it is the one thing every
        technique checks and the one that has to happen before a kernel sees a
        weight it would misread.
        """
        unsupported = environment.cost_models - self.accepts
        if unsupported:
            raise TypeError(
                f"{type(self).__name__} cannot route over "
                f"{', '.join(sorted(unsupported))} layers; it accepts "
                f"{', '.join(sorted(self.accepts))}"
            )
        return environment.compile()

    def __repr__(self) -> str:
        return self._describe()

    def _describe(self, *configuration: str) -> str:
        """A technique reads as the call that would make it."""
        return f"{type(self).__name__}({', '.join(configuration)})"


class Planner:
    """A technique with an environment behind it: the thing that answers.

    Holds what binding produced and nothing about how to ask — the verbs live
    on the subclasses, each with the keyword-only options its own algorithm
    understands, because a base class wide enough for all of them would be a
    base class promising things half of them refuse.
    """

    def __init__(
        self,
        technique: Technique,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        #: The configuration this was bound from, unchanged.
        self.technique = technique
        #: The environment as it was when this was bound.
        self.environment = environment
        self.compiled = compiled

    @property
    def footprint(self) -> int:
        """Bytes of preprocessed data this planner is holding.

        Zero for a plain search, which is the honest answer rather than a
        missing one: preprocessing is a trade, and a table comparing techniques
        has to be able to print the cost side of it for every row.
        """
        return 0

    @property
    def searches(self) -> "Tuple[str, int]":
        """What a query of this planner settles, and how many of them there
        are — the denominator a settled count is a share of.

        ``("nodes", n)`` for a search over the environment's graph. A technique
        that searches something else says so: the time-expanded model settles
        events, the timetable techniques stops.
        """
        return ("nodes", len(self.compiled))

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
        return f"{self.technique!r} bound to {self.environment!r}"


class GraphPlanner(Planner):
    """A planner whose answer is a path through the environment's own edges.

    Every static search is one of these, and so is a hierarchy that searches a
    graph nobody else has seen and unpacks before answering. What they share is
    the reading: a cost table, and a leg per edge.
    """

    def journey(self, result: EdgeResult, destination: Hashable) -> Optional[Journey]:
        """The journey a kept result holds for ``destination``, or ``None``.

        What :meth:`route` does with a search; public so a result asked for once
        can be read for several destinations, or drawn and then answered from,
        without running the search again.
        """
        if result.cost(self.node_id(destination)) is None:
            return None
        return Journey.from_result(self.compiled, result, destination)

    def _answer(self, result: EdgeResult, destination: Hashable) -> Answer:
        """The routes and the table they were read off — one search, kept."""
        journey = self.journey(result, destination)
        return Answer(self, destination, [] if journey is None else [journey], result)


class TreePlanner(GraphPlanner):
    """A planner that explores by growing a shortest-path tree.

    Dijkstra and A* — and BFS, whose tree is by hops — settle outward from
    their sources, so what they looked at is a tree and ``magnitude`` is a
    question that can be asked of it. A planner that explores some other shape
    returns some other :class:`~routelab.SearchSpace` and takes no magnitude,
    which is why this is a class of its own rather than a method on
    :class:`Planner`.
    """

    def explored(self, result: EdgeResult, magnitude: str = "weight") -> ShortestPathTree:
        """What the search looked at, in a form something can draw.

        Args:
            result: A result from ``search``, or an answer's
                :attr:`~routelab.Answer.raw`. The tree is rebuilt from it rather
                than recorded during the search, so asking costs nothing until
                you ask — which is why an answer's
                :meth:`~routelab.Answer.searchspace` is a method.
            magnitude: What each branch should carry from the subtree beyond it:
                ``"weight"`` for travel time, ``"nodes"`` for a count.
        """
        return ShortestPathTree(self.compiled, result, magnitude)


class TimetableTechnique(Technique):
    """Earliest arrival over a timetable — what the timetable techniques share.

    Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
    Information in Public Transportation Systems* (ACM JEA 12, Article 2.4,
    2007). The paper's subject is not an algorithm but a *modelling* decision:
    a timetable is a set of departures, and there are two ways to make one into
    something a shortest-path algorithm can read. :class:`TimeExpanded` spends
    nodes; :class:`TimeDependent` spends search. :class:`RAPTOR` came five
    years later and builds no graph at all; :class:`CSA` a year after that and
    keeps only the departures, sorted; :class:`TripBased` two years on again
    and labels trips, with the changes between them computed once. Every
    technique here must agree on every query, which is the paper's thesis and
    this library's test.

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

    #: The walks this technique was handed instead of the ones it would derive.
    #: ``None`` for every technique a caller writes down; :class:`~routelab.ULTRA`
    #: is what sets it, and :meth:`_with_walks` is the only way it is set.
    _walks: Any = None

    def missing_from(self, compiled: CompiledEnvironment) -> "frozenset[str]":
        """Departures to read, on top of what any technique needs. Without them
        there is no timetable to route over."""
        return super().missing_from(compiled) | Departures.missing_from(compiled)

    def walks(self) -> Any:
        """Which walks this technique reads — a derivation, like the timetable.

        :class:`~routelab.Walks` for every model here: the scalar edges between
        stops, closed under composition, which is what a one-hop transfer set
        has to be. Declared rather than assumed so that a technique which walks
        something else can say so — :class:`~routelab.LabelConstrained` reads
        the arcs themselves and wants no closure at all.
        """
        return Walks() if self._walks is None else self._walks

    def _with_walks(self: _Technique, spec: Any) -> _Technique:
        """The same configuration, reading someone else's transfer set.

        :class:`~routelab.ULTRA`'s seam, and the whole of how a precomputed
        set of shortcuts reaches the technique underneath: a copy rather than a
        mutation, because the technique a caller wrote down is still theirs.
        """
        other = copy.copy(self)
        other._walks = spec
        return other


class TimetablePlanner(Planner):
    """A timetable technique with a feed behind it.

    Holds the two things every model here reads — the departures, and the walks
    between stops — and the clock arithmetic they all do: a query's sources are
    stops on the service day, each already some seconds along, and a journey is
    an itinerary rather than a path through edges.
    """

    def __init__(
        self,
        technique: TimetableTechnique,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        super().__init__(technique, environment, compiled, progress)
        #: Gathered here rather than kept on the environment, for the reason
        #: :class:`~routelab.TimeDependentDijkstra` gives about its calendar —
        #: and refused here, at bind, so a technique that cannot route says so
        #: before it is asked anything.
        self.timetable = Departures().bind(compiled, progress)
        self.footpaths = technique.walks().bind(compiled, progress)

    @property
    def footprint(self) -> int:
        """The timetable and the walks, in kernel form: what every model holds
        before it builds anything of its own.

        A model that goes on to build a kernel counts the walks *there* rather
        than here, because the kernel keeps its own copy of them — see
        :meth:`~routelab.kernels.RAPTORPlanner.footprint`. Counting both would
        report one structure twice.
        """
        return self.timetable.footprint + self.footpaths.footprint

    @property
    def searches(self) -> "Tuple[str, int]":
        return ("stops", self.timetable.num_stops)

    def _sources(
        self, starts: "Dict[int, int]", departing: Departure
    ) -> "Tuple[List[Tuple[int, int]], int]":
        """The query's sources as ``(stop, time)`` on the service-day clock, and
        the departure everything is elapsed from."""
        at = service_seconds(departing)
        return [(stop, at + head_start) for stop, head_start in starts.items()], at

    def _window(self, departing: Departure, until: Departure) -> "Tuple[int, int]":
        """A departure window as ``(opens, closes)`` on the service-day clock.

        Shared because more than one technique here answers over a range of
        departures rather than from one moment, and a window that closes before
        it opens is not a window whoever asked.
        """
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
        found.sort(key=lambda pair: (pair[0], -pair[1].arrives))
        kept: "List[Journey]" = []
        best: Optional[int] = None
        for departs, itinerary in reversed(found):
            if best is None or itinerary.arrives < best:
                kept.append(
                    Journey.from_itinerary(self.compiled, itinerary, destination, departs)
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

    def _answer_itinerary(
        self,
        itinerary: "Optional[_routelab.Itinerary]",
        destination: Hashable,
        at: int,
    ) -> Answer:
        """One itinerary, as an answer that holds no table.

        A model that answers a pair of stops rather than filling a cost table —
        Pyrga's two, PTL, the label-constrained pair — has nothing else to hand
        back, and says so with ``raw=None`` rather than an empty table.
        """
        journey = (
            None
            if itinerary is None
            else Journey.from_itinerary(self.compiled, itinerary, destination, at)
        )
        return Answer(self, destination, [] if journey is None else [journey], None)


class Tabled(TimetablePlanner, Generic[_Result]):
    """A timetable technique that keeps a label per stop, and can be read.

    Three of the nine do — :class:`~routelab.RAPTOR`,
    :class:`~routelab.CSA` and :class:`~routelab.TripBased`. The rest answer a
    pair of stops and keep nothing, which is why ``journey`` is here and not on
    :class:`TimetablePlanner`: a base that offered it would be promising a verb
    to five techniques that have no table to read it off, and
    :meth:`~TimetablePlanner._answer_itinerary` is what they have instead.

    The same line the kernels are split along —
    ``Reads`` for a technique with a search to read, ``EarliestArrival`` for
    one that answers and keeps nothing — so the two halves of the library
    disagree about nothing.

    Generic in what the search is, so that :class:`Front` can say its own
    result kept more without narrowing a parameter its base had widened.
    """

    def journey(self, result: _Result, destination: Hashable) -> Optional[Journey]:
        """The earliest arrival at ``destination`` a kept search holds.

        A timetable technique that keeps a table — a label per stop, however
        it got there — reads an *itinerary* off it rather than an edge path,
        which is why this is not :class:`GraphPlanner`'s: the result knows
        which vehicles were ridden, and a leg is one of those.
        """
        itinerary = result.itinerary(self.node_id(destination))
        if itinerary is None:
            return None
        return Journey.from_itinerary(
            self.compiled, itinerary, destination, result.departing
        )

    def _answer(self, result: _Result, destination: Hashable) -> Answer:
        """The routes a kept search holds, and the search."""
        journey = self.journey(result, destination)
        return Answer(self, destination, [] if journey is None else [journey], result)


class Front(Tabled[FrontResult]):
    """Answers with a front of more than one, where most techniques answer one.

    Every query here answers with a Pareto set — the journeys no other journey
    beats on every criterion — and a technique that tells them apart only by
    when they arrive has exactly one. What this says is that *these* two tell
    them apart by something else as well: one journey per number of changes
    that arrives strictly earlier than any journey with fewer.
    :class:`~routelab.RAPTOR` gets it from its rounds and
    :class:`~routelab.TripBased` from its transfer counts, and both read it off
    a result the same way — so it is written here rather than twice.

    Not on :class:`TimetablePlanner`, because it is not the family's: a
    connection scan counts nothing but time, and :class:`~routelab.CSA` has no
    front to read off a search that never distinguished one journey from
    another by changes. Saying *these* techniques is what keeps the base from
    promising a verb half of it cannot honour.
    """

    def journeys(self, result: FrontResult, destination: Hashable) -> "List[Journey]":
        """Every journey a kept search holds for ``destination``: the earliest
        arrival for each number of changes, fewest changes first, none
        dominated by another.

        The kernel's own order. An answer's :attr:`~routelab.Answer.routes`
        is this reversed, so that it leads with the earliest arrival — the
        journey every technique here would have given — whatever was asked.
        """
        target = self.node_id(destination)
        return [
            Journey.from_itinerary(
                self.compiled, itinerary, destination, result.departing
            )
            for itinerary in result.itineraries(target)
        ]

    def _answer(self, result: FrontResult, destination: Hashable) -> Answer:
        """The front, best first — which is the order an answer promises and
        the reverse of the order a search produces it in.

        :meth:`journeys` reads the kernel's own order, fewest changes first
        with arrivals decreasing. An answer leads with the journey every other
        technique here would have returned, so `routes[0]` means the same thing
        whatever was asked, and each entry after it trades arrival time for one
        fewer change.
        """
        routes = list(reversed(self.journeys(result, destination)))
        return Answer(self, destination, routes, result)


def route(
    technique: Technique,
    environment: Environment,
    origin: Origins,
    destination: Hashable,
    **options: Any,
) -> Answer:
    """One-shot routing: bind a technique, ask it one question, throw it away.

        route(AStar(Landmarks(16)), env, "a", "c")

    Convenient for a single query. When you are asking more than one, bind the
    technique yourself and keep the planner — that is what makes preprocessing
    worth doing, and this function throws it away every time.

    ``**options`` are the bound planner's own, and are checked by it: this
    knows which technique it was handed no earlier than the caller does, so it
    passes them straight through rather than pretending to.
    """
    planner: Any = technique.bind(environment)
    return planner.route(origin, destination, **options)
