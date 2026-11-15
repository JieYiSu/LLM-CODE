"""Single-objective path cost used by every underwater comparison method."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .underwater_twin import UnderwaterTwin
from .underwater_scenarios import WORKSPACE_HIGH, WORKSPACE_LOW


class UnderwaterPathObjective:
    OBJECTIVE_WEIGHTS = {
        "length": 1.0,
        "time": 0.20,
        "energy": 0.25,
        "drift": 0.60,
        "smoothness": 0.35,
        "dynamic": 1.35,
        "terminal": 8.0,
        "clearance": 2.0,
        "collision": 250.0,
    }

    def __init__(self, scenario: str, *, intermediate_points: int = 8) -> None:
        self.scenario = str(scenario)
        self.twin = UnderwaterTwin(self.scenario, intermediate_points=intermediate_points)
        self.intermediate_points = int(intermediate_points)
        self.dimension = int(self.twin.dimension)
        self.bounds = self.twin.bounds

    def decode(self, vector: np.ndarray) -> np.ndarray:
        return self.twin.replay(np.asarray(vector, dtype=float))["path"]

    def _dynamic_proxy(self, path: np.ndarray, times: np.ndarray) -> dict[str, float]:
        """Estimate trackability without spending an additional MuJoCo run.

        The planner is evaluated with the reduced-order twin, so the proxy
        uses the same current field and the same hull parameters to penalize
        excessive acceleration, jerk, speed overshoot, and approximate force
        demand.  It is a ranking term, not a replacement for the final
        MuJoCo replay.
        """

        path = np.asarray(path, dtype=float)
        times = np.asarray(times, dtype=float)
        if len(path) < 3:
            return {
                "acceleration": 0.0,
                "jerk": 0.0,
                "speed_excess": 0.0,
                "force": 0.0,
                "dynamic_proxy": 0.0,
            }
        dt = np.maximum(np.diff(times), 1e-4)
        velocity = np.diff(path, axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        midpoint_dt = np.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-4)
        acceleration = np.diff(velocity, axis=0) / midpoint_dt[:, None]
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        if len(acceleration) > 1:
            jerk_dt = np.maximum(0.5 * (midpoint_dt[1:] + midpoint_dt[:-1]), 1e-4)
            jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
            jerk_norm = np.linalg.norm(jerk, axis=1)
        else:
            jerk_norm = np.zeros(1, dtype=float)
        current = self.twin.current.velocity(path[:-1], times[:-1])
        relative = velocity - current
        relative_speed = np.linalg.norm(relative, axis=1)
        drag_scale = (
            0.5
            * float(self.twin.parameters.water_density)
            * float(self.twin.parameters.drag_coefficient)
            * float(self.twin.parameters.reference_area)
        )
        drag = -drag_scale * relative_speed[:, None] * relative
        accel_for_force = np.vstack((acceleration, acceleration[-1:]))
        force = float(12.0) * accel_for_force + drag
        force_norm = np.linalg.norm(force, axis=1)
        acceleration_term = float(np.mean(acceleration_norm) / 2.5)
        jerk_term = float(np.mean(jerk_norm) / 8.0)
        speed_excess_term = float(np.mean(np.maximum(speed - 1.5, 0.0)) / 1.5)
        force_term = float(np.mean(force_norm) / 180.0)
        dynamic_proxy = (
            0.40 * acceleration_term
            + 0.20 * jerk_term
            + 0.15 * speed_excess_term
            + 0.25 * force_term
        )
        return {
            "acceleration": acceleration_term,
            "jerk": jerk_term,
            "speed_excess": speed_excess_term,
            "force": force_term,
            "dynamic_proxy": float(dynamic_proxy),
        }

    def evaluate_one(self, vector: np.ndarray) -> tuple[float, dict[str, Any], np.ndarray]:
        replay = self.twin.replay(vector)
        reference_distance = max(
            float(np.linalg.norm(self.twin.scenario["goal"] - self.twin.scenario["start"])),
            1e-6,
        )
        workspace_scale = max(float(np.linalg.norm(WORKSPACE_HIGH - WORKSPACE_LOW)), 1e-6)
        length_scale = reference_distance
        time_scale = reference_distance / 0.8
        energy_scale = max(reference_distance, 1.0)
        smoothness_scale = max(float(replay["path_length"]), reference_distance)
        normalized = {
            "length": float(replay["path_length"]) / length_scale,
            "time": float(replay["travel_time"]) / max(time_scale, 1e-6),
            "energy": float(replay["energy_cost"]) / energy_scale,
            "drift": float(replay["drift_error"]) / workspace_scale,
            "smoothness": float(replay["smoothness_cost"]) / smoothness_scale,
            "terminal": float(replay["terminal_error"]) / workspace_scale,
            "clearance": max(
                0.0,
                (self.twin.parameters.hull_radius - float(replay["minimum_clearance"]))
                / max(self.twin.parameters.hull_radius, 1e-6),
            ) ** 2,
            "collision": float(replay["collision_count"]),
        }
        dynamic_terms = self._dynamic_proxy(replay["path"], replay["times"])
        collision_penalty = self.OBJECTIVE_WEIGHTS["collision"] * normalized["collision"]
        terminal_penalty = self.OBJECTIVE_WEIGHTS["terminal"] * normalized["terminal"]
        clearance_penalty = self.OBJECTIVE_WEIGHTS["clearance"] * normalized["clearance"]
        dynamic_penalty = self.OBJECTIVE_WEIGHTS["dynamic"] * dynamic_terms["dynamic_proxy"]
        value = (
            self.OBJECTIVE_WEIGHTS["length"] * normalized["length"]
            + self.OBJECTIVE_WEIGHTS["time"] * normalized["time"]
            + self.OBJECTIVE_WEIGHTS["energy"] * normalized["energy"]
            + self.OBJECTIVE_WEIGHTS["drift"] * normalized["drift"]
            + self.OBJECTIVE_WEIGHTS["smoothness"] * normalized["smoothness"]
            + dynamic_penalty
            + collision_penalty
            + terminal_penalty
            + clearance_penalty
        )
        metrics = dict(replay)
        metrics.update(
            {
                "objective": float(value),
                "collision_penalty": collision_penalty,
                "terminal_penalty": terminal_penalty,
                "clearance_penalty": clearance_penalty,
                "dynamic_penalty": dynamic_penalty,
                "dynamic_proxy": dynamic_terms["dynamic_proxy"],
                "objective_weights": dict(self.OBJECTIVE_WEIGHTS),
                "normalized_terms": {**normalized, **dynamic_terms},
                "normalization": {
                    "reference_distance": reference_distance,
                    "workspace_scale": workspace_scale,
                    "time_scale": time_scale,
                    "energy_scale": energy_scale,
                    "smoothness_scale": smoothness_scale,
                },
            }
        )
        return float(value), metrics, np.asarray(replay["path"], dtype=float)

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        array = points.detach().cpu().numpy()
        values = np.asarray([self.evaluate_one(row)[0] for row in array], dtype=np.float32)
        return torch.as_tensor(values, dtype=torch.float32, device=points.device)


__all__ = ["UnderwaterPathObjective"]
