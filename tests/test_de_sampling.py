import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evox" / "src"))
sys.path.insert(0, str(ROOT))


class TestDESampling(unittest.TestCase):
    def test_shared_parent_pool_draws_from_other_islands(self):
        import torch
        from llm_operator_de import DEOperatorLibrary

        populations = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0]],
                [[100.0], [101.0], [102.0], [103.0]],
            ]
        )
        fitnesses = populations.squeeze(-1)
        parents = DEOperatorLibrary.sample_shared_parent_indices(
            populations, fitnesses, cross_island_rate=1.0
        )

        self.assertEqual(tuple(parents.shape), (2, 4, 4, 2))
        # With cross-island mating forced, every target in island 0 must have
        # at least one parent from island 1, and vice versa.
        values = populations[parents[..., 0], parents[..., 1], 0]
        self.assertTrue(torch.all((values[0] >= 100.0).any(dim=1)))
        self.assertTrue(torch.all((values[1] < 100.0).any(dim=1)))

    def test_single_island_batch_falls_back_to_local_mating(self):
        import torch
        from llm_operator_de import DEOperatorLibrary, DEOperatorSpec

        population = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
        fitness = population.square().sum(dim=2)
        bounds_low = -torch.ones(1, 3)
        bounds_high = torch.ones(1, 3)
        spec = DEOperatorSpec("rand1")
        trials = DEOperatorLibrary().apply_islands(
            [spec], population, fitness, bounds_low, bounds_high
        )
        self.assertEqual(tuple(trials.shape), (1, 8, 3))
        self.assertTrue(torch.isfinite(trials).all())

    def test_success_history_adapts_f_and_cr_after_improvement(self):
        import torch
        from llm_operator_de import LLMOperatorDE

        island = LLMOperatorDE(
            6, -torch.ones(3), torch.ones(3), max_gen=20, device=torch.device("cpu")
        )
        old_f = island.mu_f
        old_cr = island.mu_cr
        island.record_success_history(
            torch.tensor([0.9, 0.8]), torch.tensor([0.95, 0.85]), torch.tensor([1.0, 0.5])
        )
        self.assertGreater(island.mu_f, old_f)
        self.assertGreater(island.mu_cr, old_cr)

    def test_gate_fallback_is_attributed_to_the_fallback_operator(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE

        island = LLMOperatorDE(
            8, -torch.ones(3), torch.ones(3), max_gen=20, device=torch.device("cpu")
        )
        island.library.add(
            DEOperatorSpec.from_dict(
                {
                    "operator_id": "gated-candidate",
                    "mutation": "rand1",
                    "gating_policy": {"condition": "stagnation_ge", "threshold": 8},
                    "fallback_operator": "rand1",
                }
            )
        )
        island.active_operator_id = "gated-candidate"
        island.stagnation = 0
        _, actual_operator_id, _ = island.prepare_operator()
        self.assertEqual(actual_operator_id, "rand1")

    def test_parent_sampling_uses_full_population_and_excludes_current(self):
        import torch
        from llm_operator_de import DEOperatorLibrary

        torch.manual_seed(11)
        population_size = 16
        parents = DEOperatorLibrary._sample_parent_indices(
            population_size, torch.device("cpu")
        )

        self.assertEqual(tuple(parents.shape), (population_size, 4))
        self.assertGreaterEqual(int(parents.min()), 0)
        self.assertLess(int(parents.max()), population_size)
        self.assertGreaterEqual(int(parents.max()), 4)

        current = torch.arange(population_size)[:, None]
        self.assertTrue(torch.all(parents != current))
        self.assertTrue(
            all(len(set(row.tolist())) == 4 for row in parents)
        )

    def test_pbest_reference_is_sampled_from_current_top_fraction(self):
        import torch
        from llm_operator_de import DEOperatorLibrary

        population = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        fitness = torch.arange(10, dtype=torch.float32)
        references = DEOperatorLibrary._sample_pbest_references(
            population.unsqueeze(0), fitness.unsqueeze(0), p_fraction=0.2
        )

        self.assertEqual(tuple(references.shape), (1, 10, 2))
        # The p-best pool contains the two current best vectors, not stale
        # archive points that are unrelated to the current population.
        allowed = {tuple(row.tolist()) for row in population[:2]}
        self.assertTrue(all(tuple(row.tolist()) in allowed for row in references[0]))


if __name__ == "__main__":
    unittest.main()
