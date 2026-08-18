# Time-dependent Dijkstra

> Dreyfus, S. E. *An appraisal of some shortest-path algorithms.* Operations
> Research **17**(3), 395–412 (1969).

`routelab.TimeDependentDijkstra` · [source](../../python/routelab/kernels/timedep.py) ·
checked against a brute-force search over every departure

A gate shut at night. A lane that reverses in the morning. A trail closed
overnight. The network is the same shape all day, but not all of it is
available all the time — so the cost of an edge depends on when you get there.
Dreyfus's appraisal is where the result lives that says Dijkstra's algorithm
handles this **unchanged**, provided one condition holds.

## The algorithm

Dijkstra's, with the edge weight replaced by a function of arrival time:

```
time[v] ← ∞                       for every node v
time[o] ← departure               for every origin o
queue  ← the origins

while queue:
    u ← pop earliest              # settle u: time[u] is now final
    for each edge u → v:
        t ← arrival(u → v, time[u])       # ∞ if shut and waiting is forbidden
        if t < time[v]:
            time[v] ← t
            parent[v] ← the edge u → v
            push v at t
```

`arrival(e, t)` is: if `e` is open at `t`, `t + weight(e)`; if it is shut, wait
until it opens and then cross — or give up, depending on the policy.

The condition is the **FIFO property**: arrival must be non-decreasing in
departure, so leaving later can never arrive earlier. It holds here because
travel times are constant and only availability varies. When it holds, the
settle-and-never-revisit argument survives intact and the algorithm is exactly
Dijkstra's; when it does not — an overtaking express, a departure board — it
fails, and a different technique is needed.

This is a *different technique* rather than `Dijkstra` with an extra argument,
which is what stops a schedule from being ignored by accident. Ask for
`Dijkstra` and you get the always-open network, knowingly. Ask for this one and
you get the clock.

## Hello world

`CONDITIONAL_OSM` below is a fixture this repository ships: a gate in the shape
of the Ballard Locks, shut outside 07:00–21:00, with a much longer way round it.
Which route a search takes says plainly whether it read the schedule.

```
    1 ---- 2 ==gate== 3          the short way, open 07:00–21:00
    |                 |
    +------ 4 --------+          the long way, always open
```

```python
>>> import routelab as rl
>>> from datetime import time

>>> env = rl.Environment(rl.OSM(CONDITIONAL_OSM, rl.Walking()))
>>> planner = rl.TimeDependentDijkstra().bind(env)

>>> planner.route(1, 3, departing=time(12, 0))
Journey(1 → 2 → 3, cost=160)
>>> planner.route(1, 3, departing=time(3, 0))
Journey(1 → 3, cost=654)

```

At noon the gate is open and the short way costs 160 seconds. At three in the
morning it is shut, and the same query walks the long way for 654. Nodes keep
their OSM ids, so those are the ids from the file.

The clock here is a **week** — seconds since Monday 00:00 — because every
restriction OpenStreetMap can express repeats weekly. A schedule naming a date,
a month or a public holiday is refused at parse time rather than approximated.
That is a real limit and worth knowing rather than discovering: "next Tuesday
at nine" and "this Tuesday at nine" are the same question here.

## Waiting is a policy

At three in the morning, waiting four hours for the gate loses to an eleven
minute detour. Just before it opens, it does not:

```python
>>> planner.route(1, 3, departing=time(6, 55))
Journey(1 → 2 → 3, cost=380)

```

Five minutes to seven, the search waits two hundred and twenty seconds at the
gate and still beats the way round. Cost counts the wait as travel time, and
the journey keeps the split:

```python
>>> journey = planner.route(1, 3, departing=time(6, 55))
>>> journey.waiting, journey.moving
(220, 160)

```

Nobody had to decide that five minutes of waiting is acceptable and four hours
is not — the arithmetic decides, which is the point. `waiting="forbidden"` is
the control that shows it doing real work: a shut edge is simply absent.

```python
>>> control = rl.TimeDependentDijkstra(waiting="forbidden").bind(env)
>>> control.route(1, 3, departing=time(6, 55))
Journey(1 → 3, cost=654)

```

## What it refuses

A departure time is required, with no default — a time-dependent query without
a time is not a query with a sensible fallback, it is a different question:

```python
>>> planner.route(1, 3)                        # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: TimeDependentDijkstra needs a departure time: pass departing=time(8, 30), ...

```

And an environment where nothing is scheduled is refused at bind, because this
technique on such a network is Dijkstra with extra steps:

```python
>>> plain = rl.Environment(rl.ScalarEdges(("a", "b", 1)))
>>> sorted(rl.TimeDependentDijkstra().missing_from(plain.compile()))
['schedule']

```

## What OpenStreetMap can say

The fixture carries the three tag forms that matter, and the parser reads all
three:

- `access:conditional = yes @ (07:00-21:00)` beside a plain `foot=yes` — shut
  by default, open during the window. The `foot=yes` must not be mistaken for a
  default, or the gate disappears.
- `access:conditional = no @ (23:00-05:00)` — open by default, shut overnight.
- `oneway:conditional = -1 @ (Mo-Fr 05:00-11:00); yes @ (Mo-Fr 11:15-23:00)` — a
  reversible lane, running against its drawn direction on weekday mornings.

```python
env = rl.Environment(rl.OSM("seattle.osm.pbf", rl.Driving()))
rl.TimeDependentDijkstra().bind(env)      # gathers every layer's windows
```

## See also

- [Dijkstra](dijkstra.md) — the same algorithm on a network that is always open.
- [Pyrga et al.](pyrga.md) — the other thing "the clock matters" can mean: not
  an edge that is sometimes shut, but a vehicle that leaves at 08:15.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
