"""The graph type: a thin, friendly front for the Rust CSR kernel."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

from . import _routelab

__all__ = ["Graph"]


class Graph(_routelab.Graph):
    """An immutable directed graph with non-negative integer weights.

    Weights are ``u32`` and conventionally seconds — the resolution the transit
    routing literature works in. Parallel edges and self-loops are allowed and
    nothing is deduplicated.

    Edges are stored in CSR order (grouped by tail, input order within a tail),
    so edge ids are *not* positions in the list you passed. Use
    :meth:`input_index` to get back to your own per-edge data.

    Nodes here are dense integers. To route over labelled things — stops, street
    corners, ``('bikeshare', 42)`` — use an :class:`~routelab.Environment`, which
    keeps the label bookkeeping and builds one of these underneath.
    """

    __slots__ = ()

    @classmethod
    def from_edges(
        cls,
        edges: Iterable["tuple[int, int, int]"],
        *,
        num_nodes: Optional[int] = None,
        directed: bool = True,
    ) -> "Graph":
        """Build a graph from ``(tail, head, weight)`` triples.

        Args:
            edges: The edge list. Weights must be non-negative integers.
            num_nodes: Node count. Defaults to ``max(node id) + 1``, which means
                trailing isolated nodes disappear unless you say how many there are.
            directed: If false, each triple is inserted in both directions, so the
                graph has twice as many edges as triples.
        """
        edges = [(int(tail), int(head), int(weight)) for tail, head, weight in edges]
        if not directed:
            edges = [
                edge
                for tail, head, weight in edges
                for edge in ((tail, head, weight), (head, tail, weight))
            ]
        if num_nodes is None:
            num_nodes = 1 + max((max(tail, head) for tail, head, _ in edges), default=-1)
        return cls(num_nodes, edges)

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Mapping[int, Iterable["tuple[int, int]"]],
        *,
        num_nodes: Optional[int] = None,
    ) -> "Graph":
        """Build a graph from ``{tail: [(head, weight), ...]}``."""
        edges = [
            (tail, head, weight)
            for tail, out_edges in adjacency.items()
            for head, weight in out_edges
        ]
        return cls.from_edges(edges, num_nodes=num_nodes)
