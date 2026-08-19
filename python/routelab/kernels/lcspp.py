"""Barrett, Jacob & Marathe, *Formal-Language-Constrained Path Problems* (SIAM
Journal on Computing 30(3), 2000), in the multimodal form Dibbelt, Pajor &
Wagner give it in *User-Constrained Multi-Modal Route Planning* (ALENEX 2012)
§2.2."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, NamedTuple, Optional, Sequence, Tuple

from .. import _routelab
from ..model.answer import Answer
from ..model.environment import CompiledEnvironment, Environment
from ..util.clock import Departure
from .planner import Origins, TimetablePlanner, TimetableTechnique
from .ultra import Transfers

__all__ = [
    "Modes",
    "LabelConstrained",
    "LabelConstrainedPlanner",
    "UCCH",
    "UCCHPlanner",
]


class Language(NamedTuple):
    """A :class:`Modes` read against an environment: the automaton, the mode of
    every arc, and which symbols mean what."""

    automaton: Any
    labels: bytes
    riding: int
    symbol_of: "Dict[str, int]"

#: A mode the environment does not have, so no arc carries it. Distinct from
#: `_SCHEDULED`, which every timetable arc carries.
_NO_MODE = 254

#: The label an arc carries when no language may walk it. Timetable arcs get
#: this: they are relaxed by asking what leaves next along them, not by adding a
#: weight, so the scalar half of the search must pass them by. Outside any
#: alphabet, so the automaton refuses it without being told to.
_SCHEDULED = 255


def _mode_of(source: Any) -> str:
    """What kind of arc a layer emits, as a word of the alphabet.

    A timetable layer is ridden. Everything else is walked, unless it says
    otherwise — :class:`~routelab.Access` says ``"link"``, because joining two
    networks is what the paper's link arcs are and changing mode is only
    allowed on them.
    """
    if source.cost_model == "timetable":
        return "transit"
    return getattr(source, "mode", "foot")


class Modes:
    """Which sequences of transport modes a journey may use.

        Modes()                    # derived from the environment
        Modes(states={"foot": ["foot"], "aboard": ["transit"]}, link="link")

    The automaton of §2.2, and no more than that shape: a state stands for one
    or more modes, travelling within one of them is a self-loop, and distinct
    states are joined **only** by the link label — so a journey may change mode
    exactly where the networks were stitched together, and nowhere else. States
    are marked initial or final by whether their modes may begin or end a
    journey.

    Left to itself it reads the environment, which is enough for the two cases
    that come up. Where a link layer joins separate networks — streets and a
    feed, with :class:`~routelab.Access` between — it builds the paper's Figure
    1(a): a walking state and a riding state, joined both ways by the link.
    Where there is none, the timetable's stops *are* the walking network's
    nodes, so one state stands for both modes and a rider changes vehicles
    standing still, which is what every other timetable technique here assumes.
    """

    def __init__(
        self,
        states: "Optional[Dict[str, Sequence[str]]]" = None,
        link: Optional[str] = "link",
        start: "Optional[Sequence[str]]" = None,
        end: "Optional[Sequence[str]]" = None,
    ):
        self.states = None if states is None else {k: list(v) for k, v in states.items()}
        self.link = link
        self.start = None if start is None else list(start)
        self.end = None if end is None else list(end)

    def _derive(self, present: "Sequence[str]") -> "Tuple[Dict[str, List[str]], List[str], List[str]]":
        """The automaton to build when nobody said — see the class docstring."""
        walking = [mode for mode in present if mode not in ("transit", self.link)]
        if self.link in present and "transit" in present:
            states = {"foot": walking, "aboard": ["transit"]}
            # Every state begins and ends a journey. Marking only "foot" reads
            # as "a person starts and finishes on the pavement", which is true
            # of a doorway and false of a stop: a stop has no walking arc, only
            # link arcs, so a journey to one could never be in the walking state
            # when it got there and was refused outright.
            return states, ["foot", "aboard"], ["foot", "aboard"]
        # One network, so one state: nothing has to be crossed to board.
        both = {"anything": [mode for mode in present if mode != self.link]}
        return both, ["anything"], ["anything"]

    def compile(self, compiled: CompiledEnvironment) -> Language:
        """The automaton, the label of every arc, and which symbol is riding.

        Nothing is precomputed here in the sense the other techniques mean it:
        this reads the layers already in the environment and writes a byte per
        arc.
        """
        present: "List[str]" = []
        for _, _, source in compiled.spans:
            mode = _mode_of(source)
            if mode not in present:
                present.append(mode)

        states = self.states
        start, end = self.start, self.end
        if states is None:
            states, derived_start, derived_end = self._derive(present)
            start = start or derived_start
            end = end or derived_end
        if start is None or end is None:
            raise ValueError(
                "a language needs somewhere to begin and end: pass start= and "
                "end= alongside states=, naming states you defined"
            )

        # The alphabet: every mode the environment actually has, plus the link
        # whether or not it does, so that naming it is never an error.
        alphabet = list(present)
        if self.link is not None and self.link not in alphabet:
            alphabet.append(self.link)
        symbol_of = {mode: index for index, mode in enumerate(alphabet)}

        order = list(states)
        index_of = {name: index for index, name in enumerate(order)}
        for named in (start, end):
            for name in named:
                if name not in index_of:
                    raise ValueError(
                        f"no state named {name!r}; this language has "
                        f"{', '.join(repr(s) for s in order) or 'none'}"
                    )

        transitions: "List[Tuple[int, int, int]]" = []
        for name, modes in states.items():
            for mode in modes:
                if mode not in symbol_of:
                    raise ValueError(
                        f"state {name!r} travels by {mode!r}, which no layer here "
                        f"emits; this environment has "
                        f"{', '.join(repr(m) for m in present)}"
                    )
                transitions.append((index_of[name], symbol_of[mode], index_of[name]))
        # Distinct states are joined only by the link, both ways.
        if self.link is not None and self.link in symbol_of:
            for name in order:
                for other in order:
                    if name != other:
                        transitions.append(
                            (index_of[name], symbol_of[self.link], index_of[other])
                        )

        automaton = _routelab.Modes(
            len(order),
            len(alphabet),
            transitions,
            [index_of[name] for name in start],
            [index_of[name] for name in end],
        )

        # A label per arc, by position in the input edge list — a graph permutes
        # its edges into adjacency order, and the kernel reads through that.
        labels = bytearray([_SCHEDULED]) * compiled.graph.num_edges
        for first, last, source in compiled.spans:
            mode = _mode_of(source)
            if mode == "transit":
                continue  # relaxed by its schedule, not by a weight
            labels[first:last] = bytes([symbol_of[mode]]) * (last - first)
        return Language(
            automaton,
            bytes(labels),
            symbol_of.get("transit", _SCHEDULED),
            symbol_of,
        )

    def __repr__(self) -> str:
        if self.states is None:
            return "Modes()"
        return f"Modes({self.states!r}, link={self.link!r})"


class _NoWalks:
    """A closure this technique does not want.

    Every other timetable model reads its walks as one-hop transfers between
    stops, closed under composition. This one walks the arcs themselves, in the
    same search that rides — so the closure would be work done to be ignored.
    """

    name = "walks"

    @classmethod
    def missing_from(cls, compiled: CompiledEnvironment) -> "frozenset[str]":
        return frozenset()

    def bind(self, compiled: CompiledEnvironment, progress: Any = None) -> Any:
        return _routelab.Footpaths.none()

    def __repr__(self) -> str:
        return "NoWalks()"


class LabelConstrained(TimetableTechnique):
    """The best journey whose sequence of transport modes you allow.

        LabelConstrained().bind(env).route(doorstep, office, departing=time(8, 30))

    Merge a network per mode into one graph and the shortest path through it is
    often one nobody can take — it leaves a car mid-journey, or boards a train
    from a motorway. So label every arc with the mode it belongs to, say which
    sequences of labels are allowed, and search only those: the *label
    constrained shortest path problem*, which Barrett et al. proved tractable
    for regular languages, which is more than enough for "walk, ride, walk".

    **Nothing is precomputed.** That is what it is for. The shelf's other
    multimodal answer, :class:`~routelab.ULTRA`, spends minutes working out the
    walks worth taking so that a query costs milliseconds; this spends nothing
    and pays per query, searching the product of the merged graph and the
    automaton. Two corners of the same trade, and the paper this takes its model
    from contributes the technique that sits between them, which is not here.

    Its costs are worth knowing before choosing it. To write a constraint you
    have to know what modes the network has; and a language admits no journey
    that combines the modes differently, so this answers with one journey rather
    than a set of alternatives.

    Args:
        modes: The language, as a :class:`Modes`. Left out, it is read from the
            environment — see :class:`Modes`.
    """

    def __init__(self, modes: Optional[Modes] = None):
        self.modes = modes if modes is not None else Modes()

    def __repr__(self) -> str:
        return self._describe() if self.modes.states is None else self._describe(repr(self.modes))

    def walks(self) -> Any:
        """None: this reads the arcs, not a closure of them — see
        :meth:`TimetableTechnique.walks`."""
        return _NoWalks()

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "LabelConstrainedPlanner":
        return LabelConstrainedPlanner(
            self, environment, self._compile(environment), progress
        )


class LabelConstrainedPlanner(TimetablePlanner):
    """:class:`LabelConstrained` over one environment: a byte per arc and an
    automaton, and nothing else built."""

    def __init__(
        self,
        technique: LabelConstrained,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        """Read the layers, label the arcs, build the automaton. No search."""
        super().__init__(technique, environment, compiled, progress)
        #: The language read against this environment, kept because UCCH builds
        #: its hierarchy from the same labels and compiling twice is a byte
        #: written per arc twice — and two objects that must agree.
        self.language = technique.modes.compile(compiled)
        self.automaton = self.language.automaton
        if self.automaton.is_empty:
            raise ValueError(
                "this language admits no journey at all: it names no state to "
                "begin in, or none to end in"
            )
        self.network = _routelab.Multimodal(
            compiled.graph, self.language.labels, self.timetable, self.language.riding
        )

    @property
    def footprint(self) -> int:
        return super().footprint + self.network.footprint

    @property
    def searches(self) -> "Tuple[str, int]":
        """Product vertices — a stop per state, which is what the search is
        over and the number this model is judged on."""
        return ("states", len(self.compiled) * self.automaton.num_states)

    def route(
        self, origin: Origins, destination: Hashable, *, departing: Departure
    ) -> Answer:
        """The best journey to ``destination`` whose modes the language allows."""
        sources, at = self._sources(self._origin_ids(origin), departing)
        itinerary = self.network.earliest_arrival(
            self.automaton, sources, self.node_id(destination)
        )
        return self._answer_itinerary(itinerary, destination, at)


def _served(compiled: CompiledEnvironment) -> "List[int]":
    """The vertices a vehicle calls at.

    Never contracted, which is what the paper's practical variant means by
    leaving the time-dependent network alone. Most are link endpoints already —
    a stop is joined to the pavement outside it — but a stop the streets never
    reach is joined to nothing, and contracting it would take it out from under
    a trip that rides straight through.

    Which labels those are is :class:`~routelab.kernels.Transfers`'s question
    too, and asked there first, so it is asked there.
    """
    return sorted(compiled.node_id(label) for label in Transfers._stops(compiled))


class UCCH(LabelConstrained):
    """Label-constrained routing with the walking contracted first.

        UCCH().bind(env).route(doorstep, office, departing=time(8, 30))

    Dibbelt, Pajor & Wagner, *User-Constrained Multi-Modal Route Planning*
    (ALENEX 2012) §3 — the speedup for :class:`LabelConstrained`, and the middle
    of three corners. :class:`~routelab.ULTRA` precomputes for minutes and
    answers in milliseconds; that one precomputes nothing and searches the whole
    network; this contracts the streets in a few minutes and searches a core
    of about two per cent of them.

    Where nothing joins two networks — a feed and its footpaths, with no street
    layer — there is no core distinct from the network, so this is
    :class:`LabelConstrained` with a hierarchy built for nothing. That is the
    plain model rather than a refusal, and the answers are the same.

    What it does *not* do is bake the language in, which is the whole reason it
    exists rather than an ordinary hierarchy. Contract the merged network and a
    shortcut can span two modes, so it carries a modal transfer inside it that a
    query forbidding that transfer cannot use — and the path avoiding it may
    already have been discarded. UCCH contracts each mode's subnetwork alone and
    never contracts a vertex where the networks join, so no shortcut crosses a
    boundary and the automaton stays a query input.

    The honest measurement, on King County Metro and Seattle's pavements: about
    three and a half times faster than :class:`LabelConstrained`, for a few
    minutes of contraction. Most of what is left is the transit search inside
    the core, which this does not touch — the paper says as much — so that is
    close to its ceiling here.

    Args:
        modes: The language, as a :class:`Modes`. Left out, it is read from the
            environment.
        max_degree: Stop contracting once the core averages more than this many
            arcs a vertex. Lower leaves more standing and contracts faster.
    """

    def __init__(self, modes: Optional[Modes] = None, max_degree: float = 12.0):
        super().__init__(modes)
        self.max_degree = float(max_degree)

    def __repr__(self) -> str:
        inside = [] if self.modes.states is None else [repr(self.modes)]
        if self.max_degree != 12.0:
            inside.append(f"max_degree={self.max_degree:g}")
        return self._describe(*inside)

    def bind(
        self, environment: Environment, progress: "Optional[_routelab.Progress]" = None
    ) -> "UCCHPlanner":
        return UCCHPlanner(self, environment, self._compile(environment), progress)


class UCCHPlanner(LabelConstrainedPlanner):
    """:class:`UCCH` over one environment, its walking already contracted."""

    def __init__(
        self,
        technique: UCCH,
        environment: Environment,
        compiled: CompiledEnvironment,
        progress: "Optional[_routelab.Progress]" = None,
    ):
        """Contract the walking network around the vertices where it meets the
        rest. Minutes on a city, paid once."""
        super().__init__(technique, environment, compiled, progress)
        language = self.language
        # An environment with nothing to contract — no streets, or nothing
        # joining them to the feed — is the plain model rather than a refusal,
        # for the reason ULTRA's `Transfers` gives: there is simply no core
        # distinct from the network, and the search underneath runs as it would
        # have anyway.
        #
        # A mode this environment does not have is _NO_MODE, which is not
        # `_SCHEDULED`: that one is what every timetable arc carries, so reusing
        # it here would hand the whole timetable to the contractor as a walking
        # network.
        walking = language.symbol_of.get("foot", _NO_MODE)
        named = technique.modes.link
        link = _NO_MODE if named is None else language.symbol_of.get(named, _NO_MODE)
        self.hierarchy = _routelab.Ucch.build(
            compiled.graph,
            language.labels,
            walking,
            link,
            _served(compiled),
            technique.max_degree,
            progress=progress,
        )

    @property
    def footprint(self) -> int:
        return super().footprint + self.hierarchy.footprint

    @property
    def searches(self) -> "Tuple[str, int]":
        """Core vertices, times the automaton's states — what a query is over,
        and a fraction of what :class:`LabelConstrained` searches."""
        return ("states", self.hierarchy.num_core * self.automaton.num_states)

    @property
    def num_core(self) -> int:
        """Vertices the contraction left standing."""
        return self.hierarchy.num_core

    def route(
        self, origin: Origins, destination: Hashable, *, departing: Departure
    ) -> Answer:
        """The same answer :class:`LabelConstrainedPlanner` gives, over the core."""
        sources, at = self._sources(self._origin_ids(origin), departing)
        itinerary = self.hierarchy.earliest_arrival(
            self.network, self.automaton, sources, self.node_id(destination)
        )
        return self._answer_itinerary(itinerary, destination, at)
