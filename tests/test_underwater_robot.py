import sys
import shutil
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evox" / "src"))
sys.path.insert(0, str(ROOT))


class TestUnderwaterRobot(unittest.TestCase):
    def test_config_uses_expensive_comparison_protocol(self):
        from experiments.underwater_robot.experiment_config import CONFIG

        self.assertEqual(CONFIG.total_evals, 1000)
        self.assertEqual(CONFIG.initial_evals, 100)
        self.assertEqual(CONFIG.population_size, 30)
        self.assertEqual(CONFIG.repeats, 20)
        self.assertEqual(CONFIG.island_count, 3)
        self.assertEqual(CONFIG.llm_interval, 3)
        self.assertEqual(CONFIG.candidate_validation_evals, 60)
        self.assertEqual(CONFIG.intermediate_points, 8)
        self.assertEqual(CONFIG.trajectory_dimension, 32)
        self.assertEqual(CONFIG.candidate_validation_evals, 60)

    def test_protocol_manifest_contains_reproducibility_fields(self):
        from experiments.underwater_robot.run_underwater_comparison import protocol_manifest
        from experiments.underwater_robot.experiment_config import CONFIG

        manifest = protocol_manifest(CONFIG, [])
        self.assertEqual(manifest["protocol"]["total_evals"], 1000)
        self.assertEqual(manifest["protocol"]["initial_evals"], 100)
        self.assertEqual(manifest["protocol"]["island_count"], 3)
        self.assertEqual(manifest["protocol"]["repeats"], 20)
        self.assertIn("source_hashes", manifest)
        self.assertIn("validation", manifest)

    def test_protocol_manifest_reads_nested_replay_metrics(self):
        from experiments.underwater_robot.run_underwater_comparison import protocol_manifest
        from experiments.underwater_robot.experiment_config import CONFIG

        rows = [{
            "scenario": CONFIG.scenarios[0],
            "algorithm": "LLMOperatorDE",
            "seed": 1,
            "real_evals": CONFIG.total_evals,
            "status": "ok",
            "metrics": {"feasible": True},
            "mujoco_replay": {"mujoco_replay_mode": "dynamic_force_tracking"},
        }]
        validation = protocol_manifest(CONFIG, rows)["validation"]
        self.assertEqual(validation["reduced_feasible_rows"], 1)
        self.assertEqual(validation["mujoco_replay_mode_counts"], {"dynamic_force_tracking": 1})

    def test_statistics_read_structured_metrics(self):
        from experiments.underwater_robot.run_underwater_comparison import _row_metric

        row = {
            "metrics": {"feasible": True},
            "mujoco_replay": {"mujoco_collision_count": 2},
        }
        self.assertTrue(_row_metric(row, "feasible"))
        self.assertEqual(_row_metric(row, "mujoco_collision_count"), 2)

    def test_statistics_exclude_partial_api_runs_and_keep_audit(self):
        from experiments.underwater_robot.run_underwater_comparison import write_underwater_statistics

        output = ROOT / ".underwater_statistics_test"
        shutil.rmtree(output, ignore_errors=True)
        self.addCleanup(lambda: shutil.rmtree(output, ignore_errors=True))
        rows = [
            {
                "scenario": "uniform_cross_current",
                "algorithm": "LLMOperatorDE",
                "seed": 1,
                "best_f": 1.0,
                "status": "ok",
                "api_calls": 1,
                "api_successes": 1,
                "api_failures": 0,
            },
            {
                "scenario": "uniform_cross_current",
                "algorithm": "ESA",
                "seed": 1,
                "best_f": 2.0,
                "status": "ok",
            },
            {
                "scenario": "uniform_cross_current",
                "algorithm": "LLMOperatorDE",
                "seed": 2,
                "best_f": 99.0,
                "status": "llm_api_partial_failure",
                "api_calls": 1,
                "api_successes": 0,
                "api_failures": 1,
            },
            {
                "scenario": "uniform_cross_current",
                "algorithm": "ESA",
                "seed": 2,
                "best_f": 3.0,
                "status": "ok",
            },
        ]
        paths = write_underwater_statistics(rows, output)
        statistical = paths["csv"].read_text(encoding="utf-8-sig")
        invalid = paths["invalid_csv"].read_text(encoding="utf-8-sig")
        api = paths["api_csv"].read_text(encoding="utf-8-sig")
        self.assertIn(",ESA,1,2.0,0.0,2.0,1.0,1.0,", statistical)
        self.assertIn("paired_sign_effect", statistical)
        self.assertNotIn("cliffs_delta", statistical)
        self.assertNotIn(",2,", statistical)
        self.assertIn("llm_api_partial_failure", invalid)
        self.assertIn("valid_runs,invalid_runs", api)
        self.assertIn("uniform_cross_current,2,1,1,", api)

    def test_current_fields_are_bounded_and_time_dependent(self):
        from experiments.underwater_robot.current_fields import CurrentField

        field = CurrentField("time_varying_multi_obstacle")
        position = np.array([[0.0, 0.0, -1.0], [1.0, 0.5, -2.0]])
        first = field.velocity(position, 0.0)
        later = field.velocity(position, 1.0)
        self.assertEqual(first.shape, position.shape)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(later).all())
        self.assertFalse(np.allclose(first, later))
        self.assertLessEqual(float(np.max(np.linalg.norm(first, axis=1))), field.max_speed)

    def test_scenarios_have_feasible_reference_geometry(self):
        from experiments.underwater_robot.underwater_scenarios import (
            SCENARIOS,
            minimum_clearance,
            point_in_obstacle,
            scenario_bounds,
        )

        self.assertEqual(set(SCENARIOS), {
            "uniform_cross_current",
            "shear_vortex",
            "time_varying_multi_obstacle",
        })
        for name, scenario in SCENARIOS.items():
            self.assertEqual(len(scenario["start"]), 3)
            self.assertEqual(len(scenario["goal"]), 3)
            low, high = scenario_bounds(name)
            self.assertTrue(np.all(np.asarray(low) < np.asarray(high)))
            self.assertTrue(bool(scenario["reference_feasible"]))
        cylinder = SCENARIOS["shear_vortex"]["obstacles"][0]
        lateral_outside = np.asarray([1.0, 0.0, cylinder["center"][2]], dtype=float)
        self.assertGreaterEqual(minimum_clearance(lateral_outside, cylinder), 0.0)
        self.assertFalse(point_in_obstacle(lateral_outside, cylinder))

    def test_objective_returns_decomposed_finite_metrics(self):
        from experiments.underwater_robot.underwater_objective import UnderwaterPathObjective

        objective = UnderwaterPathObjective("uniform_cross_current", intermediate_points=8)
        vector = np.zeros(objective.dimension, dtype=float)
        value, metrics, trajectory = objective.evaluate_one(vector)
        self.assertTrue(np.isfinite(value))
        self.assertIn("path_length", metrics)
        self.assertIn("energy_cost", metrics)
        self.assertIn("collision_count", metrics)
        self.assertIn("dynamic_proxy", metrics)
        self.assertTrue(np.isfinite(float(metrics["dynamic_proxy"])))
        self.assertIn("acceleration", metrics["normalized_terms"])
        self.assertEqual(trajectory.shape[1], 3)

    def test_planning_collision_uses_the_same_hull_radius_as_mujoco(self):
        from experiments.underwater_robot.underwater_objective import UnderwaterPathObjective

        objective = UnderwaterPathObjective("uniform_cross_current", intermediate_points=8)
        obstacle = objective.twin.scenario["obstacles"][0]
        values = np.tile(np.asarray([obstacle["center"][0], obstacle["center"][1], obstacle["center"][2], 1.0]), 8)
        _, metrics, _ = objective.evaluate_one(values)
        self.assertGreater(int(metrics["collision_count"]), 0)
        self.assertLess(float(metrics["minimum_clearance"]), 0.0)

    def test_twin_replay_returns_terminal_and_energy_metrics(self):
        from experiments.underwater_robot.underwater_twin import UnderwaterTwin

        twin = UnderwaterTwin("uniform_cross_current")
        result = twin.replay(np.linspace(0.0, 1.0, 32, dtype=float))
        for key in ("path", "energy_cost", "drift_error", "collision_count", "terminal_error"):
            self.assertIn(key, result)
        self.assertEqual(result["path"].shape[1], 3)
        self.assertTrue(np.isfinite(float(result["energy_cost"])))

    def test_replay_preserves_hard_start_and_goal_endpoints(self):
        from experiments.underwater_robot.underwater_twin import UnderwaterTwin

        twin = UnderwaterTwin("time_varying_multi_obstacle", intermediate_points=8)
        lower, upper = twin.bounds[:, 0], twin.bounds[:, 1]
        vector = lower + (upper - lower) * np.linspace(0.0, 1.0, twin.dimension)
        replay = twin.replay(vector)
        np.testing.assert_allclose(replay["path"][0], twin.scenario["start"], atol=1e-10)
        np.testing.assert_allclose(replay["path"][-1], twin.scenario["goal"], atol=1e-10)
        self.assertAlmostEqual(float(replay["terminal_error"]), 0.0, places=10)

    def test_shared_initial_design_contains_path_aware_warm_starts(self):
        from experiments.underwater_robot.run_underwater_comparison import _common_initial_design
        from experiments.underwater_robot.underwater_objective import UnderwaterPathObjective

        objective = UnderwaterPathObjective("uniform_cross_current", intermediate_points=8)
        first = _common_initial_design(30, objective.bounds, 7, objective)
        second = _common_initial_design(30, objective.bounds, 7, objective)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, (30, objective.dimension))
        _, metrics, path = objective.evaluate_one(first[0])
        self.assertTrue(np.isfinite(float(metrics["objective"])))
        np.testing.assert_allclose(path[-1], objective.twin.scenario["goal"], atol=1e-10)
        for scenario in ("uniform_cross_current", "shear_vortex", "time_varying_multi_obstacle"):
            task = UnderwaterPathObjective(scenario, intermediate_points=8)
            design = _common_initial_design(30, task.bounds, 7, task)
            feasible_count = sum(bool(task.evaluate_one(row)[1]["feasible"]) for row in design[:8])
            self.assertGreaterEqual(feasible_count, 1)

    def test_mujoco_replay_tracks_a_straight_reference_path(self):
        from experiments.underwater_robot.underwater_objective import UnderwaterPathObjective

        objective = UnderwaterPathObjective("uniform_cross_current", intermediate_points=8)
        scenario = objective.twin.scenario
        knots = np.linspace(scenario["start"], scenario["goal"], 10)[1:-1]
        vector = np.column_stack((knots, np.full(len(knots), 1.0))).reshape(-1)
        replay = objective.twin.replay(vector, engine="mujoco")
        self.assertLess(float(replay["mujoco_terminal_error"]), 2.0)

    def test_dynamic_mujoco_replay_is_finite_and_reports_force_energy(self):
        from experiments.underwater_robot.underwater_objective import UnderwaterPathObjective

        objective = UnderwaterPathObjective("uniform_cross_current", intermediate_points=8)
        values = np.tile(np.asarray([0.0, 0.0, -2.0, 0.8]), 8)
        replay = objective.twin.replay(values, engine="mujoco_dynamic")
        self.assertTrue(np.isfinite(float(replay["mujoco_terminal_error"])))
        # A force-controlled replay must remain bounded; an earlier sign bug
        # made this same test diverge to hundreds of metres.
        self.assertLess(float(replay["mujoco_terminal_error"]), 5.0)
        self.assertGreaterEqual(float(replay["mujoco_collision_fraction"]), 0.0)
        self.assertLessEqual(float(replay["mujoco_collision_fraction"]), 1.0)
        self.assertTrue(np.isfinite(float(replay["mujoco_energy_cost"])))
        self.assertEqual(replay["mujoco_replay_mode"], "dynamic_force_tracking")

    def test_llm_parser_prefers_complete_operator_over_nested_expression_fragment(self):
        from llm_operator_de.core import LLMOperatorGenerator

        content = (
            'Draft AST: {"op":"ref","name":"x"}\n'
            '{"operator_id":"current_aware_v1",'
            '"operator_ast":{"op":"vector_expr","expr":{"op":"add",'
            '"left":{"op":"ref","name":"x"},'
            '"right":{"op":"scale","factor":0.2,"child":{"op":"ref","name":"r1"}}}}}'
        )
        spec = LLMOperatorGenerator.parse_response(content)
        self.assertEqual(spec.operator_id, "current_aware_v1")

    def test_llm_parser_accepts_constant_state_scale_factor(self):
        from llm_operator_de.core import LLMOperatorGenerator

        content = (
            '{"operator_id":"current_aware_v2",'
            '"operator_ast":{"op":"vector_expr","expr":{"op":"state_scale",'
            '"factor":0.25,"child":{"op":"ref","name":"history"}}}}'
        )
        spec = LLMOperatorGenerator.parse_response(content)
        self.assertEqual(spec.operator_id, "current_aware_v2")


if __name__ == "__main__":
    unittest.main()
