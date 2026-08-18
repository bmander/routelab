#!/usr/bin/env python3
"""Render the paper pages to a static site.

    python docs/build.py --out site

The pages are Markdown first and a website second: they are read on GitHub, and
their prompted blocks are run as tests by ``tests/test_docs.py``. This turns the
same files into HTML without asking them to change, which means the build has
exactly one interesting job — the links.

A page links three ways, and each needs different treatment:

* to a sibling page (``csa.md``) — becomes ``csa.html``, still relative;
* to a file outside ``docs/`` (``../../README.md``, a kernel's source) — becomes
  a URL into the repository on GitHub, because the site does not contain those
  files and a reader following the link wants to see the code;
* to the web — left alone.

Getting that wrong is the reason the obvious alternative, pointing GitHub's
built-in Jekyll at ``docs/``, does not work here.

The navigation is not configured anywhere. It is read out of ``index.md``: its
section headings become the groups and its table rows become the entries, in
the order they appear. A new page is therefore added to the site by adding it
to the shelf, which is where a reader would look for it anyway.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List, NamedTuple, Tuple

import markdown

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

#: Where a link that leaves ``docs/`` should point instead. The branch is
#: pinned to ``main`` rather than a commit: these are references to "the code",
#: which should follow the code.
REPO = "https://github.com/bmander/routelab/blob/main/"

#: The subtitle under the site title, and the page description.
TAGLINE = "Reference implementations of routing algorithms — Python veneer, Rust kernels."


class Entry(NamedTuple):
    """One page in the navigation."""

    title: str
    href: str


def nav_from_index(index: str) -> "List[Tuple[str, List[Entry]]]":
    """The section headings of ``index.md`` and the pages each links to.

    Rows are read in document order, so the shelf's own arrangement — road
    networks, then timetables, then multimodal — is the site's.
    """
    groups: "List[Tuple[str, List[Entry]]]" = []
    for line in index.splitlines():
        heading = re.match(r"^##\s+(.*)$", line)
        if heading:
            groups.append((heading.group(1).strip(), []))
            continue
        for title, href in re.findall(r"\[([^\]]+)\]\((papers/[^)]+\.md)\)", line):
            if groups:
                groups[-1][1].append(Entry(title, href[:-3] + ".html"))
    return [(name, entries) for name, entries in groups if entries]


def close_fences(text: str) -> str:
    """Drop the blank line a doctest block needs before its closing fence.

    A ``>>>`` example is terminated by a blank line, so every tested block on
    these pages ends with one. In Markdown that blank line is content, and it
    renders as a dead row inside the code box — visible slack that is really an
    artefact of the examples being executable. It is removed here rather than in
    the pages, because the pages are right as they stand.
    """
    lines = text.splitlines()
    out: "List[str]" = []
    inside = False
    for line in lines:
        fence = line.startswith("```")
        if fence and inside:
            while out and not out[-1].strip():
                out.pop()
        if fence:
            inside = not inside
        out.append(line)
    return "\n".join(out) + "\n"


def rewrite_link(url: str, page: Path) -> str:
    """One link, as the site needs it. See this module's docstring."""
    if url.startswith(("http://", "https://", "mailto:", "#")):
        return url
    path, hash_, fragment = url.partition("#")
    if not path:
        return url
    target = (page.parent / path).resolve()
    try:
        inside = target.relative_to(DOCS)
    except ValueError:
        # Outside docs/: the site has no copy, so point at the repository.
        return REPO + str(target.relative_to(ROOT)) + hash_ + fragment
    if inside.suffix == ".md":
        return path[:-3] + ".html" + hash_ + fragment
    return url


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{tagline}">
<link rel="stylesheet" href="{root}style.css">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<nav id="sidebar">
  <a class="brand" href="{root}index.html">routelab</a>
  <p class="tagline">{tagline}</p>
  {nav}
  <p class="source"><a href="https://github.com/bmander/routelab">Source on GitHub</a></p>
</nav>
<main id="content">
{body}
</main>
</body>
</html>
"""


def render_nav(groups, page_href: str, root: str) -> str:
    """The sidebar, with the page being rendered marked as current."""
    out = []
    for name, entries in groups:
        out.append(f"<h2>{name}</h2>\n<ul>")
        for entry in entries:
            here = ' class="here" aria-current="page"' if entry.href == page_href else ""
            out.append(f'<li><a href="{root}{entry.href}"{here}>{entry.title}</a></li>')
        out.append("</ul>")
    return "\n".join(out)


def title_of(text: str, fallback: str) -> str:
    """A page's first heading, which is what it calls itself."""
    heading = re.search(r"^#\s+(.*)$", text, re.M)
    return heading.group(1).strip() if heading else fallback


def build(out: Path) -> int:
    pages = sorted(DOCS.rglob("*.md"))
    groups = nav_from_index((DOCS / "index.md").read_text(encoding="utf-8"))
    renderer = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(DOCS / "style.css", out / "style.css")

    for page in pages:
        text = page.read_text(encoding="utf-8")
        relative = page.relative_to(DOCS)
        href = str(relative.with_suffix(".html"))
        root = "../" * (len(relative.parts) - 1)

        # Links are rewritten in the Markdown rather than the HTML: the source
        # is where a link's meaning is still unambiguous, and a regex over
        # rendered tags would have to know which attributes are URLs.
        text = re.sub(
            r"\]\(([^)\s]+)\)", lambda m: "](" + rewrite_link(m.group(1), page) + ")", text
        )
        text = close_fences(text)

        renderer.reset()
        destination = out / relative.with_suffix(".html")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            TEMPLATE.format(
                title=title_of(text, page.stem) + (" · routelab" if href != "index.html" else ""),
                tagline=TAGLINE,
                nav=render_nav(groups, href, root),
                body=renderer.convert(text),
                root=root,
            ),
            encoding="utf-8",
        )
    return len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the paper pages to a static site.")
    parser.add_argument("--out", default="site", type=Path, help="where to write the site")
    arguments = parser.parse_args()
    written = build(arguments.out.resolve())
    print(f"{written} pages → {arguments.out}")


if __name__ == "__main__":
    main()
