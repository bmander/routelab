"""The paper pages' own examples, run as tests.

Every prompted block under ``docs/`` is a doctest, for the reason the
docstrings are: a page describing an API that has since moved should fail the
suite rather than mislead a reader. Blocks written to be read rather than run —
the ones on a real city, with timings — carry no ``>>>`` and are not collected,
because no test suite should be downloading a 65 MB extract to check a number.

Run here rather than by pointing ``--doctest-glob`` at ``docs/`` because that
needs a ``conftest.py`` beside the pages to supply the fixture feed's path, and
a second top-level module named ``conftest`` shadows this suite's own — see the
note in ``pyproject.toml``. So the pages are collected by hand instead, which
costs one loop and keeps the arrangement the tests already depend on.

The only name a page may assume is ``TINY_GTFS``. Spelling that path out in the
page would tie the snippet to whichever directory pytest started in; naming it
says the one true thing about it, which is that a reader substitutes their own
feed.
"""

from __future__ import annotations

import doctest
from pathlib import Path
from typing import List

import pytest

from conftest import TINY_GTFS

DOCS = Path(__file__).resolve().parent.parent / "docs"

#: Every page, so a new one is covered by existing here rather than by being
#: added to a list someone has to remember.
PAGES: "List[Path]" = sorted(DOCS.rglob("*.md"))


def test_there_are_pages() -> None:
    """A glob that matches nothing passes every test it feeds."""
    assert PAGES, f"no documentation pages under {DOCS}"


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_page_examples(page: Path) -> None:
    """Run one page's prompted blocks against the fixtures it may assume."""
    failures, _ = doctest.testfile(
        str(page),
        module_relative=False,
        globs={"TINY_GTFS": TINY_GTFS},
        encoding="utf-8",
        # Reported rather than raised, so a page with three broken examples
        # says all three instead of stopping at the first.
        report=True,
        verbose=False,
    )
    assert failures == 0, f"{failures} failing example(s) in {page.name}"
