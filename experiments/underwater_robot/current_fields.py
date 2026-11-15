"""Analytic current fields used by the underwater digital twin."""

from __future__ import annotations

import numpy as np


class CurrentField:
    def __init__(self, scenario: str, *, max_speed: float = 0.85) -> None:
        self.scenario = str(scenario)
        self.max_speed = float(max_speed)

    def velocity(self, position: np.ndarray, time: float | np.ndarray) -> np.ndarray:
        points = np.asarray(position, dtype=float)
        original_shape = points.shape
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[-1] != 3:
            raise ValueError("position must have a final dimension of three")
        t = np.asarray(time, dtype=float)
        if t.ndim == 0:
            t = np.full(points.shape[0], float(t))
        else:
            t = np.broadcast_to(t, (points.shape[0],))
        x, y, z = points.T
        if self.scenario == "uniform_cross_current":
            flow = np.column_stack((0.18 + 0.02 * np.sin(t), 0.48 + 0.04 * np.cos(t), np.zeros_like(x)))
        elif self.scenario == "shear_vortex":
            radius = np.maximum(0.25, np.sqrt(x * x + y * y))
            flow = np.column_stack(
                (
                    0.15 + 0.18 * np.tanh(z + 2.0) - 0.30 * y / radius,
                    0.08 + 0.20 * np.tanh(x) + 0.30 * x / radius,
                    0.06 * np.sin(0.7 * x + 0.4 * t),
                )
            )
        elif self.scenario == "time_varying_multi_obstacle":
            phase = 0.35 * t + 0.45 * x - 0.25 * y
            flow = np.column_stack(
                (
                    0.22 + 0.20 * np.sin(phase),
                    0.18 + 0.26 * np.cos(phase + 0.7),
                    0.10 * np.sin(0.5 * t + z),
                )
            )
        else:
            raise ValueError(f"unknown current-field scenario: {self.scenario}")
        speed = np.linalg.norm(flow, axis=1)
        scale = np.minimum(1.0, self.max_speed / np.maximum(speed, 1e-9))
        flow = flow * scale[:, None]
        return flow[0] if len(original_shape) == 1 else flow


__all__ = ["CurrentField"]
