"""Argument coercion shared by the Rust-backed API and the pure-Python reference.

Both front doors accept the same spellings so that a test can hand the identical
call to each implementation and compare what comes back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

#: A node, a list of nodes, a list of ``(node, initial_cost)`` pairs, or a
#: ``{node: initial_cost}`` mapping.
Sources = Union[int, Iterable[int], Iterable["tuple[int, int]"], Mapping[int, int]]
Nodes = Union[int, Iterable[int]]


def normalize_sources(sources: Sources) -> "list[tuple[int, int]]":
    """Coerce the accepted source spellings into ``(node, initial_cost)`` pairs."""
    if isinstance(sources, int):
        return [(sources, 0)]
    if isinstance(sources, Mapping):
        # Annotated loosely on purpose: a mapping is also an iterable, so
        # narrowing the union above leaves the key and value types ambiguous,
        # and this function's whole job is to accept the loose spellings.
        pairs: Mapping[Any, Any] = sources
        return [(int(node), int(cost)) for node, cost in pairs.items()]

    normalized: "list[tuple[int, int]]" = []
    for item in sources:
        if isinstance(item, int):
            normalized.append((item, 0))
            continue
        try:
            node, cost = item
        except (TypeError, ValueError):
            raise TypeError(
                f"sources must be node ids or (node, initial_cost) pairs, got {item!r}"
            ) from None
        normalized.append((int(node), int(cost)))
    return normalized


def normalize_nodes(nodes: Optional[Nodes]) -> "Optional[list[int]]":
    if nodes is None:
        return None
    if isinstance(nodes, int):
        return [nodes]
    return [int(node) for node in nodes]
