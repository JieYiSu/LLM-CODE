"""Reproducible underwater current and obstacle layouts.

The geometry is deliberately low-dimensional and deterministic so that all
comparison algorithms see exactly the same digital-twin task for a seed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


WORKSPACE_LOW = np.asarray([-2.0, -2.0, -4.0], dtype=float)
WORKSPACE_HIGH = np.asarray([2.0, 2.0, -0.5], dtype=float)


SCENARIOS: dict[str, dict[str, Any]] = {
    "uniform_cross_current": {
        "title": "Uniform cross-current corridor",
        "current": "uniform_cross_current",
        "start": (-1.65, -1.20, -2.20),
        "goal": (1.55, 1.15, -2.20),
        "obstacles": [
            {"type": "box", "center": (0.0, 0.0, -2.20), "half_size": (0.34, 0.58, 0.75)},
        ],
        "reference_feasible": True,
    },
    "shear_vortex": {
        "title": "Shear-vortex field with rock pillar",
        "current": "shear_vortex",
        "start": (-1.65, -1.35, -2.90),
        "goal": (1.55, 1.25, -1.35),
        "obstacles": [
            {"type": "cylinder", "center": (0.05, 0.05, -2.15), "radius": 0.42, "half_height": 1.15},
        ],
        "reference_feasible": True,
    },
    "time_varying_multi_obstacle": {
        "title": "Time-varying current and multi-obstacle passage",
        "current": "time_varying_multi_obstacle",
        "start": (-1.70, -1.35, -2.75),
        "goal": (1.60, 1.30, -1.10),
        "obstacles": [
            {"type": "box", "center": (-0.55, -0.25, -2.25), "half_size": (0.24, 0.65, 0.72)},
            {"type": "box", "center": (0.55, 0.35, -1.70), "half_size": (0.27, 0.58, 0.70)},
            {"type": "cylinder", "center": (0.0, 0.95, -2.85), "radius": 0.28, "half_height": 0.58},
        ],
        "reference_feasible": True,
    },
}


def get_scenario(name: str) -> dict[str, Any]:
    try:
        item = deepcopy(SCENARIOS[str(name)])
    except KeyError as error:
        raise ValueError(f"unknown underwater scenario: {name}") from error
    item["start"] = np.asarray(item["start"], dtype=float)
    item["goal"] = np.asarray(item["goal"], dtype=float)
    return item


def scenario_bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
    get_scenario(name)
    return WORKSPACE_LOW.copy(), WORKSPACE_HIGH.copy()


def point_in_obstacle(point: np.ndarray, obstacle: dict[str, Any], margin: float = 0.0) -> bool:
    """Return whether a 3-D point lies inside an obstacle plus ``margin``."""

    point = np.asarray(point, dtype=float)
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle["type"] == "box":
        half = np.asarray(obstacle["half_size"], dtype=float) + float(margin)
        return bool(np.all(np.abs(point - center) <= half))
    if obstacle["type"] == "cylinder":
        radius = float(obstacle["radius"]) + float(margin)
        radial = np.linalg.norm(point[:2] - center[:2])
        vertical = abs(float(point[2] - center[2]))
        return bool(radial <= radius and vertical <= float(obstacle["half_height"]) + float(margin))
    raise ValueError(f"unsupported obstacle type: {obstacle['type']}")


def minimum_clearance(point: np.ndarray, obstacle: dict[str, Any]) -> float:
    """Approximate signed clearance from an obstacle surface."""

    point = np.asarray(point, dtype=float)
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle["type"] == "box":
        half = np.asarray(obstacle["half_size"], dtype=float)
        distance_to_surface = np.abs(point - center) - half
        outside = np.maximum(distance_to_surface, 0.0)
        outside_distance = float(np.linalg.norm(outside))
        inside_depth = float(np.max(distance_to_surface))
        return outside_distance if inside_depth >= 0.0 else inside_depth
    if obstacle["type"] == "cylinder":
        radial = np.linalg.norm(point[:2] - center[:2])
        radial_clearance = radial - float(obstacle["radius"])
        vertical_clearance = abs(float(point[2] - center[2])) - float(obstacle["half_height"])
        outside_distance = float(
            np.hypot(max(radial_clearance, 0.0), max(vertical_clearance, 0.0))
        )
        inside_depth = min(max(radial_clearance, vertical_clearance), 0.0)
        return float(outside_distance + inside_depth)
    raise ValueError(f"unsupported obstacle type: {obstacle['type']}")


__all__ = [
    "SCENARIOS",
    "WORKSPACE_LOW",
    "WORKSPACE_HIGH",
    "get_scenario",
    "scenario_bounds",
    "point_in_obstacle",
    "minimum_clearance",
]
