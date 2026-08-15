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
    """A stand-in for the layer that does not yet exist, to test the seam."""

    cost_model = "timetable"

    def edges(self):
        return [("a", "b", 1)]


def test_cost_models_are_collected_from_the_layers():
    env = rl.Environment(rl.ScalarEdges(("a", "b", 1)), Timetable())
    assert env.cost_models == frozenset({"scalar", "timetable"})


def test_compiling_refuses_a_layer_it_would_have_to_lie_about():
    # A time-dependent layer cannot flatten into fixed costs. Better to say so
    # than to compile its edges as if their weights told the whole story.
    env = rl.Environment(Timetable())
    with pytest.raises(NotImplementedError, match="does not"):
        env.compile()
