"""The HTTP layer: the page, the static files, and two JSON endpoints."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .router import Router
from .wiring import Board, Unwired

class Handler(BaseHTTPRequestHandler):
    router: Router
    center: "tuple[float, float]"

    #: Keep-alive. Every response here sets `Content-Length`, so this is safe,
    #: and a drag makes tens of requests a second — each one otherwise a fresh
    #: connection and a fresh thread.
    protocol_version = "HTTP/1.1"

    #: The page and the board's own code, read from disk beside this file. The
    #: editor is several hundred lines of JavaScript, which is a thing to keep
    #: in a `.js` file where it can be read and linted rather than in a Python
    #: string where it cannot.
    STATIC = Path(__file__).resolve().parent.parent / "static"
    TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        url = urlparse(self.path)
        if url.path == "/":
            self.respond(200, "text/html; charset=utf-8", self.page().encode("utf-8"))
        elif url.path == "/route":
            self.respond(200, "application/json", self.route(parse_qs(url.query)))
        elif url.path == "/progress":
            # Answered while another request is mid-build, which is the only
            # reason it can say anything. `ThreadingHTTPServer`, an atomic
            # counter, and no lock between them.
            self.respond(
                200,
                "application/json",
                json.dumps(self.router.working(), separators=(",", ":")).encode("utf-8"),
                cache=False,
            )
        elif url.path.startswith("/static/"):
            self.static(url.path[len("/static/") :])
        else:
            self.respond(404, "text/plain", b"not found")

    def static(self, name: str) -> None:
        """Serve one file from `static/`, and nothing outside it."""
        path = (self.STATIC / name).resolve()
        if not path.is_file() or self.STATIC not in path.parents:
            self.respond(404, "text/plain", b"not found")
            return
        kind = self.TYPES.get(path.suffix, "application/octet-stream")
        # Never cached. This is a demo you are meant to edit, and a stale
        # board.js served out of the browser's cache is an afternoon.
        self.respond(200, f"{kind}; charset=utf-8", path.read_bytes(), cache=False)

    def route(self, query: "dict[str, list[str]]") -> bytes:
        def point(name: str) -> "tuple[float, float]":
            lat, _, lon = query[name][0].partition(",")
            return float(lat), float(lon)

        try:
            payload = self.router.route(
                Board.parse(query["board"][0]),
                point("from"),
                point("to"),
                explore=query.get("explore", ["1"])[0] != "0",
            )
        except Unwired as error:
            # The board can point at the node that is missing an argument.
            payload = {"error": str(error), "node": error.node_id}
        except (KeyError, TypeError, ValueError) as error:
            # Everything else is the library refusing the graph in its own
            # words — a technique that cannot route a cost model, a heuristic
            # with no positions to work from — and those messages are better
            # than any this could write over them.
            payload = {"error": str(error)}
        # Compact separators: the payload is mostly numbers, and the default
        # ", " / ": " padding is about a tenth of ten megabytes.
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def page(self) -> str:
        return (self.STATIC / "index.html").read_text().replace(
            "__SETUP__",
            json.dumps({"center": list(self.center), **self.router.catalogue}),
        )

    def respond(
        self, status: int, content_type: str, body: bytes, cache: bool = True
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Quieter than the default, which logs every tile-less request."""
        if "/route" in str(args[0] if args else ""):
            print(f"  {str(args[0])[:110]}", flush=True)
