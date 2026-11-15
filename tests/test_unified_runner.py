import unittest
from pathlib import Path


class TestUnifiedRunner(unittest.TestCase):
    def test_phase_order_and_independent_outputs(self):
        from run_all_experiments import UnifiedExperimentConfig, _phase_plan

        root = Path(__file__).resolve().parents[1] / ".unified_runner_test_output"
        config = UnifiedExperimentConfig(output_root=root)
        plan = _phase_plan(config)
        self.assertEqual([item[0] for item in plan], ["comparison", "ablation", "engineering"])
        outputs = [item[1] for item in plan]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertTrue(all(str(root) in str(path) for path in outputs))
        self.assertEqual(config.comparison_total_evals, 300)
        self.assertEqual(config.comparison_initial_evals, 100)

    def test_dry_run_does_not_execute_a_phase(self):
        from run_all_experiments import UnifiedExperimentConfig, run_pipeline

        root = Path(__file__).resolve().parents[1] / ".unified_runner_dry_run_output"
        manifest = run_pipeline(UnifiedExperimentConfig(output_root=root, dry_run=True))
        self.assertEqual(manifest["phase_order"], ["comparison", "ablation", "engineering"])
        self.assertEqual(manifest["phases"], {})
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
