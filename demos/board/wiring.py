"""Reading a posted node graph: which node feeds which."""

from __future__ import annotations

import json
from typing import NamedTuple

from .catalogue import NODES

class Board(NamedTuple):
    """A graph as the page draws it: nodes by id, and the wires between them.

    The board *is* the query. There is no second representation of what the
    controls mean — evaluating this is how a route gets found, so a wire that is
    not plugged in is not a validation failure but a missing argument, and it
    says so in the same words the library would.
    """

    nodes: "dict[str, dict]"
    links: "list[dict]"

    @classmethod
    def parse(cls, text: str) -> "Board":
        raw = json.loads(text)
        nodes = {str(node["id"]): node for node in raw.get("nodes", [])}
        return cls(nodes, raw.get("links", []))

    def sources(self, node_id: str, port: str) -> "list[tuple[str, str]]":
        """`(id, output port)` feeding one input, in the order they were wired."""
        return [
            (link["from"], link.get("fromPort", ""))
            for link in self.links
            if link.get("to") == node_id and link.get("toPort", link.get("port")) == port
        ]

    def listeners(self, node_id: str, port: str) -> "list[str]":
        """Ids wired to one *output* — who is listening for it, if anyone.

        Which is a question worth being able to ask: nothing is listening for a
        search space means nothing has to build one.
        """
        return [
            link["to"]
            for link in self.links
            if link.get("from") == node_id and link.get("fromPort") == port
        ]

    def upstream(self, node_id: str, found: "list[str] | None" = None) -> "list[str]":
        """`node_id` and everything it takes as an argument, transitively.

        The one place the argument-port rule is written: a terminal node has no
        arguments — what comes out of the map does not depend on what is drawn
        on it — which is both what stops Map -> Query -> Map recurring for ever
        and what `signature` and `settled` are agreeing about when they agree.
        """
        found = [] if found is None else found
        if node_id in found:
            return found
        found.append(node_id)
        kind = self.nodes[node_id]["type"]
        if not NODES[kind].get("terminal"):
            for port in NODES[kind]["inputs"]:
                for source, _ in self.sources(node_id, port):
                    self.upstream(source, found)
        return found

    def only(self, kind: str) -> "str | None":
        """The one node of a kind, if there is exactly one. How the query is found."""
        found = [
            node_id
            for node_id, node in self.nodes.items()
            if NODES.get(node["type"], {}).get("kind") == kind
        ]
        return found[0] if len(found) == 1 else None


class Unwired(ValueError):
    """A node whose input has nothing plugged into it.

    Carries the node so the board can point at it. Everything else that can go
    wrong — a technique refusing a cost model, a heuristic with no positions —
    is the library's own error, raised in the library's own words, and this
    exists only because "nothing is connected" is a question the library never
    gets asked.
    """

    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id
