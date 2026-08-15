"""Layers built from real data.

An environment is only as interesting as what you can register into it. These
turn the files the world actually publishes — OSM extracts today, timetables
later — into the labelled, weighted layers the kernels understand.
"""

from __future__ import annotations

from .osm import OSM, Cycling, Driving, Profile, Walking

__all__ = ["OSM", "Cycling", "Driving", "Profile", "Walking"]
