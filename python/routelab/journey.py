"""What a query returns: a journey, in labels, with each leg's provenance."""

from __future__ import annotations

from typing import Hashable, List, NamedTuple, Tuple

from .environment import CompiledEnvironment, EdgeSource
from .search import SearchResult

__all__ = ["Journey", "Leg"]


class Leg(NamedTuple):
    """One edge of a journey, with the layer that supplied it."""

    tail: Hashable
    head: Hashable
    weight: int
    source: EdgeSource
    #: The graph edge this leg crossed. Keep it and you can ask the environment
    #: for anything else the edge knows — its shape, most usefully, since a leg
    #: between two labels may be a winding street on the ground.
    edge: int


class Journey(NamedTuple):
    """A path through an environment, with the cost the planner minimized.

    ``cost`` is what the planner was optimizing, which is not always the sum of
    the leg weights: :class:`~routelab.planners.BFS` counts hops, so its journeys
    report a hop count while their legs still carry real edge weights.
    """

    origin: Hashable
    destination: Hashable
    cost: int
    legs: "Tuple[Leg, ...]"

    @property
    def nodes(self) -> "List[Hashable]":
        """The labels along the journey, origin first."""
        return [self.origin] + [leg.head for leg in self.legs]

    @classmethod
    def from_result(
        cls,
        compiled: CompiledEnvironment,
        result: SearchResult,
        destination: Hashable,
    ) -> "Journey":
        """Rebuild a journey from a search result. Assumes the target was reached."""
        node_id = compiled.node_id(destination)
        legs = []
        for edge_id in result.edge_path(node_id):
            tail, head, weight = compiled.graph.edge(edge_id)
            legs.append(
                Leg(
                    tail=compiled.label(tail),
                    head=compiled.label(head),
                    weight=weight,
                    source=compiled.source_of(edge_id),
                    edge=edge_id,
                )
            )
        return cls(
            # Where the journey starts, which with several origins is whichever
            # of them won rather than the first one the caller named.
            origin=compiled.label(result.path(node_id)[0]),
            destination=destination,
            cost=result.cost(node_id),
            legs=tuple(legs),
        )

    def __repr__(self) -> str:
        arrow = " → ".join(repr(label) for label in self.nodes)
        return f"Journey({arrow}, cost={self.cost})"
