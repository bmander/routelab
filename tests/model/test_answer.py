"""One query, one answer, three things to ask it.

`route` runs a search that works out more than the journey it is usually asked
for, and hands back all of it: the routes, the space behind them, and the
kernel's own table. What this file is about is that they come from *one*
search, and that a technique which tells journeys apart only by arrival gives a
Pareto set of one rather than a lesser kind of answer.
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


def test_an_answer_is_routes_a_space_and_the_raw_table(streets):
    answer = rl.Dijkstra().bind(streets).route("home", "work")
    assert [journey.cost for journey in answer.routes] == [600]
    assert repr(answer.searchspace()) == (
        "ShortestPathTree(3 branches, magnitude='weight', peak=600)"
    )
    assert answer.raw.settled == 4


def test_one_route_is_a_pareto_set_of_one(streets):
    # Not a special case: a front is the set of journeys no other journey beats
    # on every criterion, and a search that knows only arrival has exactly one.
    answer = rl.Dijkstra().bind(streets).route("home", "work")
    assert len(answer.routes) == 1


def test_routing_searches_once(streets, monkeypatch):
    """The claim that makes the answer worth having, held to by counting."""
    planner = rl.Dijkstra().bind(streets)
    searches = 0
    underlying = type(planner).search

    def counted(self, *arguments, **options):
        nonlocal searches
        searches += 1
        return underlying(self, *arguments, **options)

    monkeypatch.setattr(type(planner), "search", counted)

    answer = planner.route("home", "work")
    answer.routes
    answer.searchspace()
    answer.searchspace(magnitude="nodes")
    assert searches == 1, "the routes and both spaces came from one search"


def test_the_space_takes_the_options_it_has_always_taken(streets):
    answer = rl.Dijkstra().bind(streets).route("home", "work")
    assert answer.searchspace(magnitude="nodes").magnitude == "nodes"
    with pytest.raises(ValueError, match="unknown magnitude"):
        answer.searchspace(magnitude="furlongs")


def test_nothing_reachable_is_an_empty_pareto_set(streets):
    answer = rl.Dijkstra().bind(streets).route("home", "work", max_cost=100)
    assert answer.routes == []
    assert answer.raw is not None, "it searched; it just did not arrive"


def test_a_round_based_technique_leads_with_the_earliest_arrival(feed):
    # The order an answer promises, and the reason it is not the kernel's:
    # `routes[0]` means the same thing whatever technique was asked.
    planner = rl.RAPTOR().bind(rl.Environment(feed))
    answer = planner.route("A", "C", departing=time(8, 0))
    assert [(j.transfers, j.cost) for j in answer.routes] == [(1, 1200), (0, 1800)]
    assert answer.routes[0].cost == min(journey.cost for journey in answer.routes)


def test_a_technique_with_no_table_answers_and_says_so(feed):
    # Every technique answers `route`; the ones that keep no table hold the
    # journey and refuse what needs one, in their own words.
    answer = rl.TimeDependent().bind(rl.Environment(feed)).route(
        "A", "C", departing=time(8, 0)
    )
    assert len(answer.routes) == 1
    assert answer.raw is None
    with pytest.raises(NotImplementedError, match="keeps no search space"):
        answer.searchspace()


def test_ultra_answers_the_same_way(feed):
    walks = rl.ScalarEdges(("A", "B", 60), ("B", "A", 60), ("B", "C", 60), ("C", "B", 60))
    planner = rl.ULTRA(rl.RAPTOR()).bind(rl.Environment(feed, walks))
    answer = planner.route("A", "C", departing=time(8, 0))
    assert [journey.cost for journey in answer.routes] == [120]
    assert answer.raw is None, "the search it brackets is the wrapped technique's"


def test_the_one_shot_helper_answers_the_same_shape(streets):
    assert rl.route(rl.Dijkstra(), streets, "home", "work").routes[0].cost == 600
