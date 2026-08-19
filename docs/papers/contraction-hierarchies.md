# Contraction hierarchies

> Geisberger, R., Sanders, P., Schultes, D. & Delling, D. *Contraction
> hierarchies: faster and simpler hierarchical routing in road networks.* WEA
> 2008, 319–333.

`routelab.ContractionHierarchy` · [source](../../python/routelab/kernels/contraction.py) ·
every distance held to [Dijkstra](dijkstra.md)'s, on every instance

[A*](astar.md) and [landmarks](landmarks.md) narrow the search. This stops
doing one. Preprocessing rewrites the network into a hierarchy, and a query
climbs it from both ends and meets above the trip, never looking sideways at
the thousands of streets in between.

## The algorithm

Preprocessing removes nodes one at a time, least important first, repairing the
damage as it goes:

```
rank ← 0
while some node is left:
    v ← the least important node left            # see "ordering" below
    for each pair (u, v, w) of its remaining neighbours:
        if the only shortest u→w path went through v:
            add a shortcut u→w of cost dist(u,v) + dist(v,w)
    rank[v] ← rank;  rank ← rank + 1
    remove v
```

"The only shortest path" is settled by a **witness search**: a local Dijkstra
from `u` bounded by `dist(u,v) + dist(v,w)`. If it finds a route to `w` that
avoids `v`, no shortcut is needed.

The result is the original graph plus its shortcuts, and a rank per node. A
query then only ever goes *up*:

```
forward  ← dijkstra from the origin,      following only edges to a higher rank
backward ← dijkstra from the destination, following only edges to a higher rank
                                          on the reversed graph
best ← min over nodes v settled by both of forward[v] + backward[v]
```

The paper's claim, and the reason this is exact: for every shortest path there
is an *up-down* path of the same cost in the hierarchy — up from the origin, up
from the destination, meeting at the highest-ranked node on the way. So
searching only upward loses nothing, and each half touches a few hundred nodes
instead of a sixth of the city.

Shortcuts stand for whole runs of original edges, so a shortcut is unpacked
back into them before anyone sees the answer.

## Hello world

```python
>>> import routelab as rl

>>> corridor = [(f"m{i}", f"m{i + 1}", 100) for i in range(5)]
>>> spur = [("m0", "s1", 100)] + [(f"s{i}", f"s{i + 1}", 100) for i in range(1, 4)]
>>> env = rl.Environment(rl.ScalarEdges(*corridor, *spur, bidirectional=True))

>>> planner = rl.ContractionHierarchy().bind(env)     # the expensive step
>>> planner.route("m0", "m5").routes[0]
Journey('m0' → 'm1' → 'm2' → 'm3' → 'm4' → 'm5', cost=500)

```

The answer is in the caller's own nodes, not the hierarchy's. This is the first
technique here that searches a graph the environment has never seen, and the
constraint that keeps it honest is worth stating out loud: **a technique may
search whatever graph it likes, but it answers in the caller's terms.** Every
shortcut is unpacked before anything leaves the planner, so journeys, legs,
geometry and provenance work exactly as they do under Dijkstra.

## The climb, and what it leaps over

The search is not one tree, so `explored` reports something else — the two
halves, and where they met:

```python
>>> result = planner.search("m0", target=planner.node_id("m5"))
>>> space = planner.explored(result)
>>> space
MeetingTrees(4 branches, longest span=2)
>>> for leap in space.branches():
...     print(leap.direction, leap.tail, "→", leap.head, "spans", len(leap.edges))
forward m0 → m1 spans 1
forward m0 → s1 spans 1
forward m1 → m3 spans 2
backward m5 → m3 spans 2

```

Four branches for a ten-node network, and two of them are shortcuts standing
for two original edges each. `m1 → m3` is the query crossing the corridor in
one step, which is the whole trick in miniature: on a cross-town Seattle query
the longest branch stands for a couple of hundred edges.

Branches are drawn with their unpacked geometry, so every line is real road
rather than a straight cut through the buildings a shortcut jumped over. The
hierarchy shows up in two properties instead: `span`, how many original edges a
branch stands for, and `level`, the rank of its higher end.

## Ordering is a policy

Which node to contract next decides everything about the hierarchy that comes
out, and nothing about what it answers.

```python
>>> rl.ContractionHierarchy(rl.EdgeDifference())        # the paper's recipe, the default
ContractionHierarchy(EdgeDifference(deleted_neighbours=True))
>>> rl.ContractionHierarchy(rl.RandomOrder(seed=0))     # the control
ContractionHierarchy(RandomOrder(seed=0))

```

Both answer identical distances; the random one just builds a far bigger
hierarchy to do it. Contract a city's ring road early and every route across
town needs a shortcut to stand in for it; contract the cul-de-sacs first and
most of them cost nothing at all.

The same holds for the witness-search limits. A witness search that gives up
early adds a shortcut that was not needed — costing space and query time, never
accuracy — which is why `max_settled` and `max_hops` are tuning knobs rather
than correctness ones.

## What it refuses

A hierarchy query takes no bounds. Its search runs over the contracted graph,
where a cost bound would cut off paths that are still cheap in the original:

```python
>>> planner.route("m0", "m5", max_cost=100).routes[0]    # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
TypeError: ...route() got an unexpected keyword argument 'max_cost'

```

And, like [A*](astar.md), it needs exactly one destination — a bidirectional
search has to know what the other half is searching from — so `search` takes
`target=` and nothing else.

## On a real map

Seattle, driving, 25 random trips (`benchmarks/bench_contraction.py`):

| | preprocessing | memory | settled | query |
|---|---|---:|---:|---:|
| [Dijkstra](dijkstra.md) | — | — | 129,992 | 16.920 ms |
| [A* (euclidean)](astar.md) | — | 4 MB | 65,523 | 12.751 ms |
| [A* (16 landmarks)](landmarks.md) | 1.3 s | 33 MB | 10,509 | 2.769 ms |
| Contraction hierarchy | 6.0 s | 19 MB | **251** | **0.152 ms** |

Identical costs on every trip — the benchmark fails if any row disagrees. Not
necessarily the identical *path*: where two routes tie, a search climbing from
both ends cannot honour the tie-break a one-directional search does, so it may
return a different equally-cheap way. The claim is exact distances, and that is
what the tests hold it to.

Preprocessing adds 504,022 shortcuts to Seattle's 590,671 edges — an 85% larger
graph, in exchange for touching 0.1% of it per query. Watch the progress bar on
a big network and it teaches something the timings hide: contracting Seattle's
walking graph settles 99.5% of its 554,393 nodes in the first minute and spends
two more on the rest. The last nodes left are the most connected, and they are
where the witness searches do their real work.

## See also

- [Landmarks](landmarks.md) — the other way to spend preprocessing, and a
  smaller bill.
- [UCCH](ucch.md) — this idea on a multimodal network, contracting each mode
  separately so no shortcut can cross between them.
- [Dijkstra](dijkstra.md) — the control every distance here is held to.
- [What preprocessing buys](../tradeoffs.md) — this technique's class, measured side by side.
- [The shelf](../index.md) — every paper implemented here, and how to install it.
