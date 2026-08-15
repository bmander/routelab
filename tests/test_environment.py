"""Environments: layers in, labelled graph out."""

from __future__ import annotations

import pytest

import routelab as rl


def test_scalar_edges_accepts_varargs_or_an_iterable():
    as_varargs = rl.ScalarEdges(("a", "b", 1), ("b", "c", 15))
    as_iterable = rl.ScalarEdges([("a", "b", 1), ("b", "c", 15)])
    assert list(as_varargs.edges()) == list(as_iterable.edges())
    assert len(as_varargs) == 2


def test_scalar_edges_can_go_both_ways():
    edges = list(rl.ScalarEdges(("a", "b", 5), bidirectional=True).edges())
    assert edges == [("a", "b", 5), ("b", "a", 5)]


def test_register_chains_and_reports_layers():
    env = rl.Environment()
    assert env.register(rl.ScalarEdges(("a", "b", 1))) is env
    assert len(env.sources) == 1
    assert repr(env) == "Environment(1 layer)"


def test_register_rejects_non_layers():
    with pytest.raises(TypeError, match="not an EdgeSource"):
        rl.Environment().register([("a", "b", 1)])


def test_labels_are_numbered_in_first_seen_order():
    env = rl.Environment(rl.ScalarEdges(("b", "a", 1), ("a", "c", 1)))
    compiled = env.compile()
    assert compiled.labels == ("b", "a", "c")
    assert compiled.node_id("b") == 0
    assert compiled.label(2) == "c"


def test_any_hashable_can_be_a_node():
    env = rl.Environment(rl.ScalarEdges((("stop", 7), ("bike", 3), 60)))
    assert rl.Dijkstra().bind(env).route(("stop", 7), ("bike", 3)).cost == 60


def test_compilation_is_cached_until_a_layer_is_registered():
    env = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
    first = env.compile()
    assert env.compile() is first
    env.register(rl.ScalarEdges(("b", "c", 1)))
    assert env.compile() is not first
    assert env.compile().graph.num_edges == 2


def test_unknown_labels_are_named_in_the_error():
    compiled = rl.Environment(rl.ScalarEdges(("a", "b", 1))).compile()
    with pytest.raises(KeyError, match="'nowhere' is not a node"):
        compiled.node_id("nowhere")


def test_every_edge_remembers_the_layer_it_came_from():
    # CSR reorders edges, so provenance has to survive the permutation.
    streets = rl.ScalarEdges(("a", "b", 300), ("b", "c", 300))
    transit = rl.ScalarEdges(("a", "c", 120))
    compiled = rl.Environment(streets, transit).compile()

    by_source = {}
    for edge_id in range(compiled.graph.num_edges):
        tail, head, _ = compiled.graph.edge(edge_id)
        by_source[(compiled.label(tail), compiled.label(head))] = compiled.source_of(
            edge_id
        )
    assert by_source[("a", "b")] is streets
    assert by_source[("b", "c")] is streets
    assert by_source[("a", "c")] is transit


class Timetable(rl.EdgeSource):
    """A two-stop shuttle, small enough to check the seam by hand.

    Edge weights are the shortest ride anyone makes, which is a lower bound and
    not a cost — the point of the ``"timetable"`` cost model.
    """

    cost_model = "timetable"

    #: `(trip, departs, arrives)` per edge, in the order `edges` yields them.
    DEPARTURES = [
        [(0, 8 * 3600, 8 * 3600 + 600), (1, 9 * 3600, 9 * 3600 + 500)],
        [(2, 10 * 3600, 10 * 3600 + 900)],
    ]

    def edges(self):
        return [("a", "b", 500), ("b", "c", 900)]

    def connections(self, index):
        return self.DEPARTURES[index]


class Untimetabled(Timetable):
    """A timetable layer with no departures — a service day nothing runs on."""

    def connections(self, index):
        return []


def test_cost_models_are_collected_from_the_layers():
    env = rl.Environment(rl.ScalarEdges(("a", "b", 1)), Timetable())
    assert env.cost_models == frozenset({"scalar", "timetable"})


def test_a_timetable_layer_compiles_and_brings_its_departures():
    compiled = rl.Environment(Timetable()).compile()
    assert "timetable" in compiled.provides
    assert compiled.timetable.num_stops == 3
    assert compiled.timetable.num_connections == 3
    # Two of the three run along the same pair of stops, so the timetable has
    # one fewer edge than it has connections.
    assert compiled.timetable.num_edges == 2


def test_departures_land_on_the_edge_they_were_filed_under():
    # The trap this keying exists to avoid: `Graph` permutes its edges into CSR
    # order, so a timetable keyed by edge id would put the 10:00 departure on
    # a → b. Leaving at 08:00 must therefore reach 'c' at 10:15 — catching the
    # 08:00 to 'b', waiting, then the only departure onward.
    compiled = rl.Environment(Timetable()).compile()
    itinerary = compiled.timetable.earliest_arrival(
        compiled.node_id("a"), 8 * 3600, compiled.node_id("c")
    )
    assert itinerary.arrives == 10 * 3600 + 900
    assert [ride[0] for ride in itinerary.rides()] == [0, 2]


def test_a_timetable_layer_with_no_departures_provides_no_timetable():
    # It compiles — the edges are real — but there is nothing to route over,
    # which is what `missing_from` is for rather than a failure at compile time.
    compiled = rl.Environment(Untimetabled()).compile()
    assert compiled.timetable is None
    assert "timetable" not in compiled.provides


def test_compiling_refuses_a_cost_model_nothing_can_carry():
    # The seam is still a seam: a model this class would have to lie about is
    # refused rather than flattened.
    class Continuous(rl.EdgeSource):
        cost_model = "flow"

        def edges(self):
            return [("a", "b", 1)]

    with pytest.raises(NotImplementedError, match="does not"):
        rl.Environment(Continuous()).compile()
