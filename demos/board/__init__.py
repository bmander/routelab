"""The node board's server side.

The page draws a graph of nodes and wires; this package turns one of those
graphs into a routing question and answers it. Split by job: what nodes exist
(:mod:`catalogue`), how a posted graph is read (:mod:`wiring`), what gets built
and asked (:mod:`router`), and how it reaches the browser (:mod:`handler`).
"""
