import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evox" / "src"))
sys.path.insert(0, str(ROOT))


class TestAlgorithmPolicy(unittest.TestCase):
    def test_policy_accepts_bounded_algorithm_modes(self):
        from llm_operator_de.policy import AlgorithmPolicy

        policy = AlgorithmPolicy.from_dict(
            {
                "selection": {"mode": "epsilon", "epsilon": 0.05},
                "survivor": {"mode": "elitist", "elite_fraction": 0.3},
                "constraint": {"mode": "feasibility_first", "epsilon": 0.01},
                "parent_sampling": {"mode": "pbest_biased", "pbest_fraction": 0.2},
                "island": {
                    "migration_mode": "adaptive",
                    "migration_interval": 7,
                    "role_mode": "preserve",
                },
            }
        )

        self.assertEqual(policy.selection_mode, "epsilon")
        self.assertEqual(policy.constraint_mode, "feasibility_first")
        self.assertEqual(policy.parent_sampling_mode, "pbest_biased")
        self.assertEqual(policy.migration_mode, "adaptive")
        self.assertEqual(policy.migration_interval, 7)

    def test_policy_rejects_unbounded_or_unknown_modes(self):
        from llm_operator_de.policy import AlgorithmPolicy

        with self.assertRaises(ValueError):
            AlgorithmPolicy.from_dict({"selection": {"mode": "arbitrary_python"}})
        with self.assertRaises(ValueError):
            AlgorithmPolicy.from_dict({"parent_sampling": {"pbest_fraction": 1.5}})

    def test_operator_spec_roundtrip_keeps_algorithm_policy(self):
        from llm_operator_de import DEOperatorSpec

        spec = DEOperatorSpec.from_dict(
            {
                "operator_id": "policy-op",
                "mutation": "rand1",
                "algorithm_policy": {
                    "selection": {"mode": "epsilon", "epsilon": 0.02}
                },
            }
        )
        self.assertEqual(spec.algorithm_policy["selection"]["mode"], "epsilon")
        self.assertEqual(
            DEOperatorSpec.from_dict(spec.to_dict()).algorithm_policy["selection"]["mode"],
            "epsilon",
        )


if __name__ == "__main__":
    unittest.main()
