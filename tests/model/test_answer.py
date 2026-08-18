"""One query, asked more than one thing.

`route` returns a journey and drops the search it was read off, which is right
when the journey is all you wanted. `ask` keeps it, so the space behind a route
and the front it belongs to are answered from the search already run — the
thing this file is actually about, since the answers themselves were already
covered wherever the technique is.
"""

from __future__ import annotations

from datetime import time

import pytest

import routelab as rl


@pytest.fixture
def streets() -> rl.Environment:
    return rl.Environment(
        rl.ScalarEdges(
            ("home", "a", 300), ("a", "b", 60), ("b", "work", 240), ("home", "work", 900)
        )
    )


def test_ask_answers_what_route_answers(streets):
    planner = rl.Dijkstra().bind(streets)
    assert planner.ask("home", "work").journey == planner.route("home", "work")


def test_the_answer_keeps_the_search_it_was_read_off(streets):
    planner = rl.Dijkstra().bind(streets)
    answer = planner.ask("home", "work")
    assert answer.result is not None
    assert answer.result.settled == 4
    # The space comes off the kept search rather than a second one, which is
    # the whole point: the same result object, not an equal one.
    assert repr(answer.explored()) == repr(planner.explored(answer.result))


def test_asking_searches_once(streets, monkeypatch):
    """The claim that makes `ask` worth having, held to by counting."""
    planner = rl.Dijkstra().bind(streets)
    searches = 0
    underlying = type(planner)._search

    def counted(self, starts, **options):
        nonlocal searches
        searches += 1
        return underlying(self, starts, **options)

    monkeypatch.setattr(type(planner), "_search", counted)

    answer = planner.ask("home", "work")
    answer.journey
    answer.explored()
    assert searches == 1, "the journey and the space came from one search"

    # What it replaces: the same two answers the long way round cost two.
    searches = 0
    planner.route("home", "work")
    planner.explored(planner.search("home", targets=[planner.node_id("work")]))
    assert searches == 2


def test_an_unreachable_destination_is_an_answer_with_no_journey(streets):
    planner = rl.Dijkstra().bind(streets)
    answer = planner.ask("home", "work", max_cost=100)
    assert answer.journey is None
    assert answer.result is not None, "it searched; it just did not arrive"


def test_a_front_is_read_off_the_same_search(feed):
    # RAPTOR keeps a front, and `frontier` is now that read rather than a
    # second search — the entry it ends on is what `route` returns.
    planner = rl.RAPTOR().bind(rl.Environment(feed))
    answer = planner.ask("A", "C", departing=time(8, 0))
    front = answer.frontier()
    assert [journey.transfers for journey in front] == [0, 1]
    assert front[-1] == answer.journey
    assert front == planner.frontier("A", "C", departing=time(8, 0))


def test_a_technique_with_no_front_refuses_by_name(feed):
    planner = rl.CSA().bind(rl.Environment(feed))
    answer = planner.ask("A", "C", departing=time(8, 0))
    with pytest.raises(NotImplementedError, match="RAPTOR\\(\\) or TripBased\\(\\)"):
        answer.frontier()


def test_a_technique_with_no_table_answers_and_says_so(feed):
    # The uniform part: every technique answers `ask`, and the ones that keep
    # no table hold the journey and refuse what needs one, in their own words.
    planner = rl.TimeDependent().bind(rl.Environment(feed))
    answer = planner.ask("A", "C", departing=time(8, 0))
    assert answer.journey is not None
    assert answer.result is None
    with pytest.raises(NotImplementedError, match="keeps no search space"):
        answer.explored()


def test_ultra_answers_the_same_way(feed):
    walks = rl.ScalarEdges(("A", "B", 60), ("B", "A", 60), ("B", "C", 60), ("C", "B", 60))
    planner = rl.ULTRA(rl.RAPTOR()).bind(rl.Environment(feed, walks))
    answer = planner.ask("A", "C", departing=time(8, 0))
    assert answer.journey == planner.route("A", "C", departing=time(8, 0))
    assert answer.result is None, "the search it brackets is the wrapped technique's"
