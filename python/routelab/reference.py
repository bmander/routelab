"""Pure-Python reference implementations — the correctness oracle.

Every kernel in routelab has a twin here: the same algorithm written the obvious
way, slowly, with no tricks. The Rust kernel is fast; this one is *legible*, and
legibility is what makes it a check. The differential tests in ``tests/`` run both
over random graphs and demand the same answer, so the fast implementation has to
earn its speed rather than merely claim it.

The reference deliberately matches the kernel's tie-breaking (settle by
ascending ``(cost, node)``, relax out-edges in CSR order, first parent to reach a
node at its final cost wins), so the comparison covers the whole result — costs,
parents, and settle order — not just distances.

These functions take the same arguments as their :mod:`routelab` counterparts and
work on any object exposing ``num_nodes``, ``num_edges``, and ``neighbors(node)``.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, List, Optional, Sequence

from ._args import Nodes, Sources, normalize_nodes, normalize_sources

__all__ = ["ReferenceResult", "bfs", "dijkstra"]


class ReferenceResult:
    """The same shape as :class:`routelab.SearchResult`, in pure Python."""

    def __init__(self, num_nodes: int):
        self._costs: List[Optional[int]] = [None] * num_nodes
        self._parents: List[Optional[int]] = [None] * num_nodes
        self._parent_edges: List[Optional[int]] = [None] * num_nodes
        self.order: List[int] = []

    @property
    def costs(self) -> "List[Optional[int]]":
        """Cost of every node, ``None`` where a node was not reached."""
        return list(self._costs)

    def cost(self, node: int) -> Optional[int]:
        return self._costs[node]

    def parent(self, node: int) -> Optional[int]:
        return self._parents[node]

    def parent_edge(self, node: int) -> Optional[int]:
        return self._parent_edges[node]

    def reached(self) -> "List[int]":
        return list(self.order)

    def path(self, node: int) -> "Optional[List[int]]":
        if self._costs[node] is None:
            return None
        path = [node]
        while self._parents[path[-1]] is not None:
            parent = self._parents[path[-1]]
            assert parent is not None  # loop condition, restated for type checkers
            path.append(parent)
        path.reverse()
        return path

    def edge_path(self, node: int) -> "Optional[List[int]]":
        if self._costs[node] is None:
            return None
        edges: List[int] = []
        current = node
        while self._parent_edges[current] is not None:
            edge = self._parent_edges[current]
            parent = self._parents[current]
            assert edge is not None and parent is not None
            edges.append(edge)
            current = parent
        edges.reverse()
        return edges

    def __repr__(self) -> str:
        return f"ReferenceResult(num_nodes={len(self._costs)}, settled={len(self.order)})"


def dijkstra(
    graph: Any,
    sources: Sources,
    *,
    targets: Optional[Nodes] = None,
    max_cost: Optional[int] = None,
) -> ReferenceResult:
    """Textbook Dijkstra with a binary heap and lazy deletion."""
    seeds = normalize_sources(sources)
    _check_sources(graph, seeds)
    result = ReferenceResult(graph.num_nodes)
    outstanding = _outstanding_targets(normalize_nodes(targets))
    if outstanding is not None and not outstanding:
        return result

    queue: List[Any] = []
    for node, cost in seeds:
        if max_cost is not None and cost > max_cost:
            continue
        known = result._costs[node]
        if known is None or cost < known:
            result._costs[node] = cost
            heapq.heappush(queue, (cost, node))

    while queue:
        cost, node = heapq.heappop(queue)
        if cost > result._costs[node]:  # a stale heap entry
            continue
        result.order.append(node)
        if outstanding is not None:
            outstanding.discard(node)
            if not outstanding:
                break

        for head, weight, edge in graph.neighbors(node):
            next_cost = cost + weight
            if max_cost is not None and next_cost > max_cost:
                continue
            known = result._costs[head]
            if known is None or next_cost < known:
                result._costs[head] = next_cost
                result._parents[head] = node
                result._parent_edges[head] = edge
                heapq.heappush(queue, (next_cost, head))

    return result


def bfs(
    graph: Any,
    sources: Nodes,
    *,
    targets: Optional[Nodes] = None,
    max_depth: Optional[int] = None,
) -> ReferenceResult:
    """Textbook breadth-first search over a FIFO queue, ignoring edge weights."""
    seeds = normalize_nodes(sources) or []
    _check_sources(graph, [(node, 0) for node in seeds])
    result = ReferenceResult(graph.num_nodes)
    outstanding = _outstanding_targets(normalize_nodes(targets))
    if outstanding is not None and not outstanding:
        return result

    queue: "deque[int]" = deque()
    for node in seeds:
        if result._costs[node] is None:
            result._costs[node] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        depth = result._costs[node]
        assert depth is not None
        result.order.append(node)
        if outstanding is not None:
            outstanding.discard(node)
            if not outstanding:
                break
        if max_depth is not None and depth == max_depth:
            continue

        for head, _weight, edge in graph.neighbors(node):
            if result._costs[head] is None:
                result._costs[head] = depth + 1
                result._parents[head] = node
                result._parent_edges[head] = edge
                queue.append(head)

    return result


def _check_sources(graph: Any, seeds: Sequence["tuple[int, int]"]) -> None:
    for node, _cost in seeds:
        if not 0 <= node < graph.num_nodes:
            raise ValueError(
                f"source node {node} is out of range for a graph with "
                f"{graph.num_nodes} nodes"
            )


def _outstanding_targets(targets: "Optional[List[int]]") -> "Optional[set[int]]":
    """The set whose emptiness ends the search, or ``None`` to run to exhaustion."""
    if targets is None:
        return None
    # An out-of-range target can never be settled, so it keeps the set non-empty
    # and the search runs to exhaustion — same as the kernel.
    return set(targets)
