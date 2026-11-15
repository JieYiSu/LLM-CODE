"""Reduced-order hydrodynamic twin with optional MuJoCo collision replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .current_fields import CurrentField
from .underwater_scenarios import (
    WORKSPACE_HIGH,
    WORKSPACE_LOW,
    get_scenario,
    minimum_clearance,
    point_in_obstacle,
)


@dataclass(frozen=True)
class TwinParameters:
    water_density: float = 1025.0
    drag_coefficient: float = 0.9
    reference_area: float = 0.12
    hull_radius: float = 0.16
    drift_gain: float = 0.22
    energy_gain: float = 0.18
    smoothness_gain: float = 0.42
    samples_per_segment: int = 10


class UnderwaterTwin:
    """Deterministic current-aware path replay.

    The optimization-facing replay is a fast reduced-order model.  The same
    trajectory can be sent through ``mujoco_replay`` for geometry/contact
    validation after an algorithm finishes.
    """

    def __init__(
        self,
        scenario: str,
        *,
        intermediate_points: int = 8,
        parameters: TwinParameters | None = None,
    ) -> None:
        self.scenario_name = str(scenario)
        self.scenario = get_scenario(self.scenario_name)
        self.intermediate_points = int(intermediate_points)
        self.parameters = parameters or TwinParameters()
        self.current = CurrentField(self.scenario["current"])

    @property
    def dimension(self) -> int:
        return 4 * self.intermediate_points

    @property
    def bounds(self) -> np.ndarray:
        lower = np.tile(np.asarray([-2.0, -2.0, -4.0, 0.20], dtype=float), self.intermediate_points)
        upper = np.tile(np.asarray([2.0, 2.0, -0.5, 1.50], dtype=float), self.intermediate_points)
        return np.column_stack((lower, upper))

    def decode(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(vector, dtype=float).reshape(self.intermediate_points, 4)
        waypoints = np.vstack((self.scenario["start"], values[:, :3], self.scenario["goal"]))
        speeds = np.clip(values[:, 3], 0.20, 1.50)
        return waypoints, speeds

    def _segment_samples(self, waypoints: np.ndarray, speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = [waypoints[0].copy()]
        times: list[float] = [0.0]
        elapsed = 0.0
        for index in range(len(waypoints) - 1):
            start = waypoints[index]
            end = waypoints[index + 1]
            previous = waypoints[max(0, index - 1)]
            following = waypoints[min(len(waypoints) - 1, index + 2)]
            delta = end - start
            distance = float(np.linalg.norm(delta))
            speed = float(speeds[min(index, len(speeds) - 1)])
            dt = distance / max(speed, 0.20)
            count = max(2, int(self.parameters.samples_per_segment))
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
                f2 = fraction * fraction
                f3 = f2 * fraction
                nominal = 0.5 * (
                    2.0 * start
                    + (-previous + end) * fraction
                    + (2.0 * previous - 5.0 * start + 4.0 * end - following) * f2
                    + (-previous + 3.0 * start - 3.0 * end + following) * f3
                )
                local_time = elapsed + fraction * dt
                current = self.current.velocity(nominal, local_time)
                # Keep the prescribed start and goal as hard geometric
                # endpoints.  Applying a full drift offset at ``fraction=1``
                # moved the final goal itself and made every optimizer spend
                # evaluations compensating an artefact of the sampler.  The
                # current still perturbs the interior trajectory, with a
                # smooth zero-drift boundary at both ends of each segment.
                drift_fraction = 4.0 * fraction * (1.0 - fraction)
                drift = self.parameters.drift_gain * current * drift_fraction * dt
                sample = nominal + drift
                if fraction >= 1.0 - 1e-12:
                    sample = end.copy()
                points.append(sample)
                times.append(local_time)
            elapsed += dt
        return np.asarray(points, dtype=float), np.asarray(times, dtype=float)

    def replay(self, vector: np.ndarray, *, engine: str = "reduced") -> dict[str, Any]:
        waypoints, speeds = self.decode(vector)
        path, times = self._segment_samples(waypoints, speeds)
        collision_count = 0
        minimum_clearance_value = float("inf")
        for point in path:
            if np.any(point < WORKSPACE_LOW) or np.any(point > WORKSPACE_HIGH):
                collision_count += 1
            for obstacle in self.scenario["obstacles"]:
                surface_clearance = minimum_clearance(point, obstacle) - self.parameters.hull_radius
                minimum_clearance_value = min(minimum_clearance_value, surface_clearance)
                if point_in_obstacle(point, obstacle, margin=self.parameters.hull_radius):
                    collision_count += 1

        deltas = np.diff(path, axis=0)
        path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(deltas) else 0.0
        travel_time = float(times[-1]) if len(times) else 0.0
        current_samples = self.current.velocity(path, times)
        relative_speed = np.linalg.norm(current_samples, axis=1)
        propulsion_speed = np.linalg.norm(
            np.vstack((deltas, np.zeros((1, 3)))), axis=1
        ) / np.maximum(np.gradient(times), 1e-6)
        energy_signal = (relative_speed + propulsion_speed[: len(path)]) ** 2
        energy_cost = float(self.parameters.energy_gain * np.trapezoid(energy_signal, times))
        if len(path) > 1:
            drift_error = float(np.linalg.norm(path[-1] - self.scenario["goal"]))
            smoothness_cost = float(np.linalg.norm(np.diff(path, n=2, axis=0), axis=1).sum())
        else:
            drift_error = float(np.linalg.norm(path[-1] - self.scenario["goal"]))
            smoothness_cost = 0.0
        terminal_error = drift_error
        feasible = bool(collision_count == 0 and terminal_error <= 0.45)
        result = {
            "path": path,
            "times": times,
            "waypoints": waypoints,
            "path_length": path_length,
            "travel_time": travel_time,
            "energy_cost": energy_cost,
            "drift_error": drift_error,
            "smoothness_cost": smoothness_cost,
            "collision_count": int(collision_count),
            "minimum_clearance": float(minimum_clearance_value),
            "terminal_error": terminal_error,
            "feasible": feasible,
            "engine": str(engine),
        }
        if engine in {"mujoco", "mujoco_dynamic"}:
            if engine == "mujoco_dynamic":
                replay = self._mujoco_dynamic_replay(path, times)
                replay_mode = "dynamic_force_tracking"
            else:
                replay = self._mujoco_replay(path, times)
                replay_mode = "kinematic_contact_replay"
            result.update(
                {
                    "mujoco_path": replay["path"],
                    "mujoco_collision_count": replay["collision_count"],
                    "mujoco_collision_fraction": replay["collision_fraction"],
                    "mujoco_first_contact_time": replay["first_contact_time"],
                    "mujoco_terminal_error": replay["terminal_error"],
                    "mujoco_energy_cost": replay["energy_cost"],
                    "mujoco_replay_mode": replay_mode,
                }
            )
        elif engine != "reduced":
            raise ValueError("engine must be 'reduced', 'mujoco' or 'mujoco_dynamic'")
        return result

    def _mujoco_dynamic_replay(self, path: np.ndarray, times: np.ndarray) -> dict[str, Any]:
        """Track the planned path with a stable force-controlled free joint.

        This is a reduced six-DoF rigid-body replay: MuJoCo integrates the
        free-joint translational state, while relative-flow drag is injected
        as a Cartesian wrench.  It is intentionally separated from the
        kinematic contact replay so the experiment can report both geometry
        validity and dynamic tracking fidelity.
        """

        try:
            import mujoco
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("MuJoCo is required for engine='mujoco_dynamic'") from error
        model = mujoco.MjModel.from_xml_string(self._mjcf())
        data = mujoco.MjData(model)
        obstacle_ids = {model.geom(name).id for name in self._obstacle_names()}
        body_id = model.body("auv").id
        start = np.asarray(path[0], dtype=float)
        data.qpos[:3] = start
        data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        actual: list[np.ndarray] = [np.asarray(data.xpos[body_id], dtype=float).copy()]
        collisions = 0
        total_steps = 0
        first_contact_time: float | None = None
        energy = 0.0
        previous_time = 0.0
        mass = 12.0
        # MuJoCo's free-joint velocity layout is
        # ``[linear_velocity(3), angular_velocity(3)]``.  The translational
        # controller therefore uses qvel[:3].  The force budget is deliberately
        # larger than the peak relative-flow drag so the replay can test path
        # tracking rather than immediately saturating and drifting away.
        response_time = 0.45
        max_force = 180.0
        for target, target_time in zip(np.asarray(path[1:], dtype=float), np.asarray(times[1:], dtype=float)):
            target_time = float(target_time)
            interval = max(target_time - previous_time, float(model.opt.timestep))
            substeps = max(1, int(np.ceil(interval / float(model.opt.timestep))))
            dt = interval / substeps
            for _ in range(substeps):
                position = np.asarray(data.xpos[body_id], dtype=float)
                velocity = np.asarray(data.qvel[:3], dtype=float)
                flow = np.asarray(self.current.velocity(position, float(data.time)), dtype=float)
                relative = velocity - flow
                drag = -0.5 * self.parameters.water_density * self.parameters.drag_coefficient * self.parameters.reference_area * np.linalg.norm(relative) * relative
                desired_velocity = np.clip((target - position) / max(response_time, dt), -1.5, 1.5)
                # Treat drag as a physical external force and compensate it in
                # the commanded actuator force.  The previous implementation
                # computed the compensation term but applied only ``control``;
                # MuJoCo never received the drag force, so its sign became a
                # positive feedback loop.  Applying control + drag preserves
                # the stated relative-flow model and remains bounded through
                # the actuator clamp.
                control = mass * (desired_velocity - velocity) / max(response_time, dt) - drag
                norm = float(np.linalg.norm(control))
                if norm > max_force:
                    control = control * (max_force / norm)
                total_force = control + drag
                data.xfrc_applied[body_id, :3] = total_force
                data.xfrc_applied[body_id, 3:] = 0.0
                mujoco.mj_step(model, data)
                energy += float(np.dot(control, control)) * float(model.opt.timestep)
                actual_point = np.asarray(data.xpos[body_id], dtype=float).copy()
                actual.append(actual_point)
                total_steps += 1
                in_contact = any(
                    obstacle_ids.intersection({int(c.geom1), int(c.geom2)})
                    for c in data.contact[: data.ncon]
                )
                if in_contact:
                    collisions += 1
                    if first_contact_time is None:
                        first_contact_time = float(data.time)
            previous_time = target_time
        actual_path = np.asarray(actual, dtype=float)
        return {
            "path": actual_path,
            "collision_count": int(collisions),
            "collision_fraction": float(collisions / max(total_steps, 1)),
            "first_contact_time": float(first_contact_time) if first_contact_time is not None else float("nan"),
            "terminal_error": float(np.linalg.norm(actual_path[-1] - self.scenario["goal"])),
            "energy_cost": float(energy),
        }

    def _mujoco_replay(self, path: np.ndarray, times: np.ndarray) -> dict[str, Any]:
        try:
            import mujoco
        except ImportError as error:  # pragma: no cover - exercised only without MuJoCo installed
            raise RuntimeError("MuJoCo is required for engine='mujoco'") from error
        model = mujoco.MjModel.from_xml_string(self._mjcf())
        data = mujoco.MjData(model)
        obstacle_ids = {model.geom(name).id for name in self._obstacle_names()}
        collisions = 0
        total_samples = 0
        first_contact_time: float | None = None
        body_id = model.body("auv").id
        actual: list[np.ndarray] = []
        energy = 0.0
        previous = None
        for point, time_value in zip(np.asarray(path, dtype=float), np.asarray(times, dtype=float)):
            data.qpos[:3] = point
            data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            actual_point = np.asarray(data.xpos[body_id], dtype=float).copy()
            actual.append(actual_point)
            if previous is not None:
                dt = max(float(time_value - previous[1]), 1e-6)
                velocity = (actual_point - previous[0]) / dt
                flow = np.asarray(self.current.velocity(actual_point, float(time_value)), dtype=float)
                relative = velocity - flow
                drag = -0.5 * self.parameters.water_density * self.parameters.drag_coefficient * self.parameters.reference_area * np.linalg.norm(relative) * relative
                energy += float(np.dot(drag, drag)) * dt
            previous = (actual_point, float(time_value))
            total_samples += 1
            in_contact = any(
                obstacle_ids.intersection({int(c.geom1), int(c.geom2)})
                for c in data.contact[: data.ncon]
            )
            if in_contact:
                collisions += 1
                if first_contact_time is None:
                    first_contact_time = float(time_value)
        actual_path = np.asarray(actual, dtype=float)
        return {
            "path": actual_path,
            "collision_count": int(collisions),
            "collision_fraction": float(collisions / max(total_samples, 1)),
            "first_contact_time": float(first_contact_time) if first_contact_time is not None else float("nan"),
            "terminal_error": float(np.linalg.norm(actual_path[-1] - self.scenario["goal"])),
            "energy_cost": float(energy),
        }

    def _obstacle_names(self) -> list[str]:
        return [f"obstacle_{index}" for index in range(len(self.scenario["obstacles"]))]

    def _mjcf(self) -> str:
        geoms: list[str] = []
        for index, obstacle in enumerate(self.scenario["obstacles"]):
            center = " ".join(f"{float(value):.6f}" for value in obstacle["center"])
            if obstacle["type"] == "box":
                size = " ".join(f"{float(value):.6f}" for value in obstacle["half_size"])
                geoms.append(
                    f'<geom name="obstacle_{index}" type="box" pos="{center}" size="{size}" rgba="0.78 0.18 0.10 0.82"/>'
                )
            else:
                geoms.append(
                    f'<geom name="obstacle_{index}" type="cylinder" pos="{center}" '
                    f'size="{float(obstacle["radius"]):.6f} {float(obstacle["half_height"]):.6f}" rgba="0.78 0.18 0.10 0.82"/>'
                )
        return f"""
<mujoco model="underwater_auv_twin">
  <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
  <default>
    <joint damping="1.5" armature="0.02"/>
    <geom contype="1" conaffinity="1" friction="0.6 0.05 0.02"/>
  </default>
  <worldbody>
    <body name="auv" pos="0 0 -2">
      <freejoint name="auv_freejoint"/>
      <geom name="auv_hull" type="sphere" size="0.16" mass="12" rgba="0.10 0.38 0.72 1"/>
    </body>
    {''.join(geoms)}
  </worldbody>
  <actuator/>
</mujoco>
"""


__all__ = ["TwinParameters", "UnderwaterTwin"]
