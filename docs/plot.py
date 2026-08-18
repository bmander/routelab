#!/usr/bin/env python3
"""Draw what each technique pays and what it buys, as SVG.

    python docs/plot.py

Reads ``docs/measurements.json`` and writes one scatter per class into
``docs/plots/``. Both axes are time, both logarithmic, because the whole point
is that the classes span four orders of magnitude: preprocessing runs from
nothing to a minute, and a query from a fifth of a millisecond to twenty.

The plots are committed rather than built, because these pages are read as
Markdown on GitHub as well as rendered into the site, and an image is the only
figure that works in both. Rerun this after rerunning a benchmark.

Design notes, since they are decisions rather than defaults:

* One hue per chart. Each point is a *named* technique, so identity is carried
  by the label beside it; colouring the points as well would spend the only
  free channel on information the labels already give.
* A technique with no preprocessing at all cannot go on a log axis, and moving
  it to "nearly zero" would be a quiet lie about a real qualitative difference.
  Those points get their own slot at the left, outside the axis, with a gap and
  a break mark saying so.
* Every point carries an SVG ``<title>``, which is a hover tooltip in every
  browser and needs no script. The numbers are also in a table beside the plot
  on the page, which is what makes the figure optional rather than load-bearing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, NamedTuple

DOCS = Path(__file__).resolve().parent
PLOTS = DOCS / "plots"

WIDTH, HEIGHT = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 74, 30, 46, 62
#: How much of the plot the "no preprocessing" slot takes, and the gap after it.
NONE_SLOT, NONE_GAP = 52, 26
#: Below this many seconds a bind is not preprocessing, it is nothing happening.
NEGLIGIBLE = 0.05

#: Slot 1 of the reference categorical palette, validated for both surfaces.
INK = {"light": "#2a78d6", "dark": "#3987e5"}
SURFACE = {"light": "#fdfdfc", "dark": "#16171a"}
PRIMARY = {"light": "#1a1a18", "dark": "#e6e5e0"}
SECONDARY = {"light": "#6a6a64", "dark": "#9a998f"}
RULE = {"light": "#e2e1dc", "dark": "#2c2e33"}


class Point(NamedTuple):
    label: str
    preprocess_s: float
    query_ms: float
    note: str


def decades(low: float, high: float) -> "List[float]":
    """Tick values bracketing the data, at 1-2-5 steps.

    Not whole decades: snapping to those left the timetable chart an entirely
    empty decade at the left, which reads as data that is not there. The empty
    space that *is* worth keeping — the corner where a query is fast and costs
    no preprocessing, which nothing occupies — survives either way.
    """
    steps = [step * 10.0**power for power in range(-4, 4) for step in (1, 2, 5)]
    first = max(i for i, step in enumerate(steps) if step <= low)
    last = min(i for i, step in enumerate(steps) if step >= high)
    return steps[first : last + 1]


def si(value: float, unit: str) -> str:
    """A tick label a person reads, not a float."""
    if value >= 1:
        text = f"{value:g}"
    else:
        text = f"{value:g}"
    return f"{text}{unit}"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def scatter(title: str, subtitle: str, points: "List[Point]") -> str:
    """One class of techniques, as a log-log scatter."""
    timed = [p for p in points if p.preprocess_s >= NEGLIGIBLE]
    instant = [p for p in points if p.preprocess_s < NEGLIGIBLE]

    plot_left = LEFT + (NONE_SLOT + NONE_GAP if instant else 0)
    plot_right = WIDTH - RIGHT
    plot_top, plot_bottom = TOP, HEIGHT - BOTTOM

    x_ticks = decades(min(p.preprocess_s for p in timed), max(p.preprocess_s for p in timed))
    y_ticks = decades(min(p.query_ms for p in points), max(p.query_ms for p in points))

    # The domain runs a little past the outermost tick, so a point that lands
    # near a round number — Dijkstra at 19.8 ms, against a 20 ms tick — keeps
    # air around it instead of being pressed into the frame with its label.
    PAD = 1.18

    def x_of(seconds: float) -> float:
        lo, hi = math.log10(x_ticks[0] / PAD), math.log10(x_ticks[-1] * PAD)
        return plot_left + (math.log10(seconds) - lo) / (hi - lo) * (plot_right - plot_left)

    def y_of(ms: float) -> float:
        lo, hi = math.log10(y_ticks[0] / PAD), math.log10(y_ticks[-1] * PAD)
        return plot_bottom - (math.log10(ms) - lo) / (hi - lo) * (plot_bottom - plot_top)

    out: "List[str]" = []
    add = out.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="{escape(title)}. {escape(subtitle)}">')
    add(f"<title>{escape(title)}</title>")
    add(STYLE)
    add(f'<rect width="{WIDTH}" height="{HEIGHT}" class="surface"/>')
    add(f'<text x="{LEFT - 46}" y="24" class="title">{escape(title)}</text>')
    add(f'<text x="{LEFT - 46}" y="40" class="subtitle">{escape(subtitle)}</text>')

    # Gridlines and ticks, hairline and recessive.
    for tick in y_ticks:
        y = y_of(tick)
        add(f'<line x1="{LEFT - 6}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" class="grid"/>')
        add(f'<text x="{LEFT - 12}" y="{y + 4:.1f}" class="tick end">{si(tick, " ms")}</text>')
    for tick in x_ticks:
        x = x_of(tick)
        add(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" class="grid"/>')
        add(f'<text x="{x:.1f}" y="{plot_bottom + 20}" class="tick mid">{si(tick, " s")}</text>')

    # The slot for techniques that precompute nothing, kept off the axis.
    if instant:
        centre = LEFT + NONE_SLOT / 2
        add(f'<text x="{centre:.1f}" y="{plot_bottom + 20}" class="tick mid">none</text>')
        broke = plot_left - NONE_GAP / 2
        add(f'<path d="M{broke - 4:.1f},{plot_bottom + 5} l8,-10 M{broke + 1:.1f},'
            f'{plot_bottom + 5} l8,-10" class="break"/>')

    add(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{HEIGHT - 16}" '
        f'class="axis mid">preprocessing, once per network →</text>')
    add(f'<text transform="translate(18,{(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" '
        f'class="axis mid">← median query</text>')

    for point in points:
        x = LEFT + NONE_SLOT / 2 if point.preprocess_s < NEGLIGIBLE else x_of(point.preprocess_s)
        y = y_of(point.query_ms)
        spent = "none" if point.preprocess_s < NEGLIGIBLE else f"{point.preprocess_s:.1f} s"
        add("<g class='mark'>")
        add(f"<title>{escape(point.label)}: {spent} preprocessing, "
            f"{point.query_ms:.3f} ms per query{escape(point.note)}</title>")
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="dot"/>')
        # Labels sit right of the dot, and flip left where that would overflow.
        flip = x > plot_right - 140
        anchor = "end" if flip else "start"
        offset = -11 if flip else 11
        add(f'<text x="{x + offset:.1f}" y="{y + 4:.1f}" class="label {anchor}">'
            f"{escape(point.label)}</text>")
        add("</g>")

    add("</svg>")
    return "\n".join(out) + "\n"


STYLE = f"""<style>
  .surface {{ fill: {SURFACE['light']}; }}
  .title {{ font: 600 15px system-ui, sans-serif; fill: {PRIMARY['light']}; }}
  .subtitle {{ font: 12px system-ui, sans-serif; fill: {SECONDARY['light']}; }}
  .tick {{ font: 11px system-ui, sans-serif; fill: {SECONDARY['light']}; }}
  .axis {{ font: 11px system-ui, sans-serif; fill: {SECONDARY['light']}; }}
  .label {{ font: 12px system-ui, sans-serif; fill: {PRIMARY['light']}; }}
  .grid {{ stroke: {RULE['light']}; stroke-width: 1; }}
  .break {{ stroke: {RULE['light']}; stroke-width: 1.5; fill: none; }}
  .dot {{ fill: {INK['light']}; stroke: {SURFACE['light']}; stroke-width: 2; }}
  .end {{ text-anchor: end; }}
  .mid {{ text-anchor: middle; }}
  .start {{ text-anchor: start; }}
  @media (prefers-color-scheme: dark) {{
    .surface {{ fill: {SURFACE['dark']}; }}
    .title {{ fill: {PRIMARY['dark']}; }}
    .subtitle, .tick, .axis {{ fill: {SECONDARY['dark']}; }}
    .label {{ fill: {PRIMARY['dark']}; }}
    .grid, .break {{ stroke: {RULE['dark']}; }}
    .dot {{ fill: {INK['dark']}; stroke: {SURFACE['dark']}; }}
  }}
</style>"""


def main() -> None:
    data = json.loads((DOCS / "measurements.json").read_text(encoding="utf-8"))
    PLOTS.mkdir(exist_ok=True)
    for name, group in data["classes"].items():
        points = [
            Point(row["technique"], row["preprocess_s"], row["query_ms"], row.get("note", ""))
            for row in group["techniques"]
        ]
        svg = scatter(group["title"], group["instance"], points)
        (PLOTS / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"{name}.svg — {len(points)} techniques")


if __name__ == "__main__":
    main()
