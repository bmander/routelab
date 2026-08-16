"""What a technique derives from an environment, and how it says it cannot.

The environment is a merge — labels, one graph, provenance — and nothing else.
A calendar, a timetable, a coordinates table and a rate are each derived by the
technique that reads them, through four small specifications of one shape:
``missing_from(compiled)`` names what the layers withheld, ``bind(compiled)``
builds the thing or refuses with the fix in the sentence. These tests hold the
four to the same invariant a heuristic is held to: if nothing is missing,
binding works; if something is, it does not, and it says so.
"""

from __future__ import annotations

import pytest

import routelab as rl

from conftest import FIXTURES, TINY_DATE, TINY_GTFS

#: Every derivation, with an environment that can supply it.
SUPPLIED = [
    (rl.Schedule, lambda: rl.Environment(rl.OSM(FIXTURES / "conditional.osm", rl.Walking()))),
    (rl.Departures, lambda: rl.Environment(rl.GTFS(TINY_GTFS, TINY_DATE))),
    (
        rl.Plane,
        lambda: rl.Environment(
            rl.ScalarEdges(("a", "b", 1)), rl.Positions({"a": (0, 0), "b": (1, 0)})
        ),
    ),
    (rl.Pace, lambda: rl.Environment(rl.ScalarEdges(("a", "b", 1), cost_per_distance=1.0))),
    (
        rl.Walks,
        lambda: rl.Environment(rl.GTFS(TINY_GTFS, TINY_DATE), rl.ScalarEdges(("A", "B", 30))),
    ),
]

#: The words each answers with, and the sentence each refuses with.
REFUSALS = {
    rl.Schedule: ("schedule", "nothing here is scheduled"),
    rl.Departures: ("timetable", "needs a timetable"),
    rl.Plane: ("positions", "position for every node"),
    rl.Pace: ("cost_per_distance", "cost_per_distance"),
}


def bare() -> rl.Environment:
    """Edges and nothing else: no hours, no departures, no coordinates, no rate."""
    return rl.Environment(rl.ScalarEdges(("a", "b", 1)))


@pytest.mark.parametrize("derivation, supplied", SUPPLIED, ids=lambda x: getattr(x, "__name__", ""))
def test_a_derivation_binds_where_nothing_is_missing(derivation, supplied):
    compiled = supplied().compile()
    assert derivation.missing_from(compiled) == frozenset()
    assert derivation().bind(compiled) is not None


@pytest.mark.parametrize("derivation", list(REFUSALS), ids=lambda d: d.__name__)
def test_a_derivation_refuses_in_its_own_words_where_something_is(derivation):
    name, sentence = REFUSALS[derivation]
    compiled = bare().compile()
    assert derivation.missing_from(compiled) == {name}
    with pytest.raises(ValueError, match=sentence):
        derivation().bind(compiled)


def test_missing_words_are_owned_by_whoever_answers_for_them():
    # No central table: each derivation spells its own word, and a technique's
    # `missing_from` is the union of what its parts say.
    assert rl.Schedule.name == "schedule"
    assert rl.Departures.name == "timetable"
    assert rl.Plane.name == "positions"
    assert rl.Pace.name == "cost_per_distance"
    assert rl.Walks.name == "walks"
    compiled = bare().compile()
    assert rl.TimeDependentDijkstra().missing_from(compiled) == {"schedule"}
    assert rl.TimeDependent().missing_from(compiled) == {"timetable"}
    assert rl.AStar(rl.Euclidean()).missing_from(compiled) == {"positions", "cost_per_distance"}


def test_a_technique_derives_at_bind_and_keeps_what_it_derived():
    # The calendar belongs to the planner that reads it, not to the environment
    # it was read from — two planners over one environment each hold their own.
    env = rl.Environment(rl.OSM(FIXTURES / "conditional.osm", rl.Walking()))
    first = rl.TimeDependentDijkstra().bind(env)
    second = rl.TimeDependentDijkstra(waiting="forbidden").bind(env)
    assert len(first.calendar) == len(second.calendar) > 0
    assert first.calendar is not second.calendar


def test_a_derivation_that_never_refuses_binds_empty_on_a_bare_environment():
    # Walks are optional — no walks is the paper's plain model — so `bind` on
    # an environment with none answers with an empty table, not a refusal.
    compiled = bare().compile()
    assert rl.Walks.missing_from(compiled) == frozenset()
    assert len(rl.Walks().bind(compiled)) == 0


@pytest.mark.parametrize("derivation, supplied", SUPPLIED, ids=lambda x: getattr(x, "__name__", ""))
def test_every_bind_takes_a_progress_counter(derivation, supplied):
    # One shape for every derivation: bind(compiled, progress=None), whether or
    # not it has anything to write into the counter.
    compiled = supplied().compile()
    assert derivation().bind(compiled, progress=None) is not None
