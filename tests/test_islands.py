import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evox" / "src"))
sys.path.insert(0, str(ROOT))


class TestIslandCoordinator(unittest.TestCase):
    def test_valid_ensemble_is_trialed_when_innovation_request_is_not_met(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.zeros(4, 2)
            island.fit = torch.ones(4)
            island.best = island.pop[0].clone()
            island.best_fit = island.fit[0].clone()
            island.stagnation = 10
            island._update_archives()
        candidate = DEOperatorSpec.from_dict(
            {
                "operator_id": "valid-ensemble",
                "operator_ast": {
                    "op": "weighted_sum",
                    "terms": [
                        {"op": "primitive", "name": "rand1", "weight": 0.5},
                        {"op": "primitive", "name": "best1", "weight": 0.5},
                    ],
                },
            }
        )

        class Generator:
            def generate(self, state, context):
                self.state = state
                return candidate

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=Generator(),
            warmup_generations=0,
            llm_interval=1,
        )
        coordinator._maybe_synthesize()
        self.assertEqual(coordinator.candidate_operator_id, "valid-ensemble")
        self.assertEqual(coordinator.innovation_misses, 1)

    def test_innovation_candidate_is_trialed_on_balanced_island_before_broadcast(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.zeros(4, 2)
            island.fit = torch.ones(4)
            island.best = island.pop[0].clone()
            island.best_fit = island.fit[0].clone()
            island.stagnation = 10
            island._update_archives()

        candidate = DEOperatorSpec.from_dict(
            {
                "operator_id": "innovation_candidate",
                "operator_ast": {
                    "op": "vector_expr",
                    "expr": {
                        "op": "add",
                        "left": {"op": "ref", "name": "x"},
                        "right": {
                            "op": "scale",
                            "factor": 0.2,
                            "child": {
                                "op": "sub",
                                "left": {"op": "ref", "name": "pbest"},
                                "right": {"op": "ref", "name": "x"},
                            },
                        },
                    },
                },
            }
        )

        class Generator:
            def generate(self, state, context):
                self.state = state
                return candidate

        generator = Generator()
        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=generator,
            warmup_generations=0,
            llm_interval=10,
        )
        coordinator._maybe_synthesize()

        self.assertTrue(generator.state["innovation_required"])
        self.assertEqual(islands[0].active_operator_id, "current_to_rand1")
        self.assertEqual(islands[1].active_operator_id, "innovation_candidate")
        self.assertEqual(islands[2].active_operator_id, "best1")

    def test_candidate_promotes_after_a_real_selection_win(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.zeros(4, 2)
            island.fit = torch.ones(4)
            island.best = island.pop[0].clone()
            island.best_fit = island.fit[0].clone()
            island.stagnation = 6
            island._update_archives()

        candidate = DEOperatorSpec.from_dict(
            {
                "operator_id": "promote-me",
                "operator_ast": {
                    "op": "vector_expr",
                    "expr": {"op": "ref", "name": "x"},
                },
            }
        )

        class Generator:
            def generate(self, state, context):
                return candidate

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=Generator(),
            warmup_generations=0,
        )
        coordinator._maybe_synthesize()
        self.assertEqual(coordinator.innovation_attempts, 1)
        # A target island of size four has supplied the requested four real
        # candidate trials after one completed generation.  Promotion must
        # therefore not wait for the old hard-coded three generations.
        islands[1].gen = 1
        islands[1].operator_success["promote-me"] = 1
        islands[1].improvement_history.append(0.5)
        coordinator._maybe_promote_candidate()

        # The candidate is accepted on the balanced island and added to every
        # library, while exploration and exploitation keep their role-specific
        # operators instead of being overwritten by one noisy trial.
        self.assertEqual(islands[0].active_operator_id, "current_to_rand1")
        self.assertEqual(islands[1].active_operator_id, "promote-me")
        self.assertEqual(islands[2].active_operator_id, "best1")
        self.assertTrue(all("promote-me" in {spec.operator_id for spec in island.library.all()} for island in islands))
        self.assertEqual(coordinator.innovation_successes, 1)

    def test_candidate_validation_evals_rounds_to_a_complete_island_generation(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
        ]
        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            candidate_validation_evals=5,
        )

        # Five requested trials require two complete four-vector target
        # batches, hence eight actual candidate-island evaluations.
        self.assertEqual(coordinator._candidate_requirement(islands[1]), (2, 8))

    def test_promoted_candidate_installs_algorithm_policy_on_all_islands(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=10, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.zeros(4, 2)
            island.fit = torch.ones(4)
            island.best = island.pop[0].clone()
            island.best_fit = island.fit[0].clone()
            island.stagnation = 10
            island._update_archives()

        candidate = DEOperatorSpec.from_dict(
            {
                "operator_id": "policy-candidate",
                "operator_ast": {
                    "op": "vector_expr",
                    "expr": {"op": "ref", "name": "x"},
                },
                "algorithm_policy": {
                    "selection": {"mode": "epsilon", "epsilon": 0.02},
                    "parent_sampling": {"mode": "pbest_biased", "pbest_fraction": 0.2},
                    "island": {"migration_mode": "adaptive", "migration_interval": 7},
                },
            }
        )

        class Generator:
            def generate(self, state, context):
                return candidate

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=Generator(),
            warmup_generations=0,
        )
        coordinator._maybe_synthesize()
        islands[1].gen = 1
        islands[1].operator_success["policy-candidate"] = 1
        coordinator._maybe_promote_candidate()

        self.assertTrue(all(island.algorithm_policy["selection"]["mode"] == "epsilon" for island in islands))
        self.assertTrue(all(island.algorithm_policy["parent_sampling"]["mode"] == "pbest_biased" for island in islands))
        self.assertEqual(coordinator.migration_interval, 7)

    def test_parallel_island_batch_executes_generated_vector_expression(self):
        import torch
        from llm_operator_de import DEOperatorLibrary, DEOperatorSpec

        spec = DEOperatorSpec.from_dict(
            {
                "operator_id": "generated-island-expr",
                "operator_ast": {
                    "op": "vector_expr",
                    "expr": {
                        "op": "add",
                        "left": {"op": "ref", "name": "x"},
                        "right": {
                            "op": "scale",
                            "factor": 0.2,
                            "child": {
                                "op": "sub",
                                "left": {"op": "ref", "name": "best"},
                                "right": {"op": "ref", "name": "x"},
                            },
                        },
                    },
                },
            }
        )
        populations = torch.randn(3, 4, 2)
        fitnesses = populations.square().sum(dim=2)
        bounds = torch.ones(3, 2)
        trials = DEOperatorLibrary().apply_islands(
            [spec, spec, spec],
            populations,
            fitnesses,
            -bounds,
            bounds,
            states=[{}, {}, {}],
        )
        self.assertEqual(tuple(trials.shape), (3, 4, 2))
        self.assertTrue(torch.isfinite(trials).all())

    def test_llm_synthesis_waits_for_warmup_and_periodic_state(self):
        import torch
        from llm_operator_de import DEOperatorSpec, LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.randn(4, 2)
            island.fit = island.pop.square().sum(dim=1)
            index = torch.argmin(island.fit)
            island.best = island.pop[index].clone()
            island.best_fit = island.fit[index].clone()
            island._update_archives()

        class CountingGenerator:
            def __init__(self):
                self.calls = 0
                self.states = []

            def generate(self, state, context):
                self.calls += 1
                self.states.append(dict(state))
                return DEOperatorSpec("warmup_operator", mutation="rand1")

        generator = CountingGenerator()
        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=generator,
            llm_interval=10,
            warmup_generations=5,
        )

        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 0)
        coordinator.generation = 5
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 1)
        self.assertTrue(generator.states[-1]["innovation_required"])
        for island in islands:
            island.stagnation = 15
            island.operator_failures = 3
            island.improvement_history.extend([0.0] * 5)
        coordinator.generation = 15
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 2)

    def test_llm_synthesis_respects_real_evaluation_cooldown(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.randn(4, 2)
            island.fit = island.pop.square().sum(dim=1)
            best_index = torch.argmin(island.fit)
            island.best = island.pop[best_index].clone()
            island.best_fit = island.fit[best_index].clone()
            island.stagnation = 10
            island.operator_failures = 3
            island._update_archives()

        class CountingGenerator:
            def __init__(self):
                self.calls = 0

            def generate(self, state, context):
                self.calls += 1
                return None

        generator = CountingGenerator()
        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=generator,
            warmup_generations=0,
            llm_interval=1,
            llm_eval_interval=8,
            steady_state=True,
        )

        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 1)
        coordinator.generation = 10
        islands[0].trial_evals = 4
        islands[1].trial_evals = 0
        islands[2].trial_evals = 0
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 1)
        islands[0].trial_evals = 8
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 2)

    def test_llm_trigger_score_waits_for_a_real_search_failure(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=30, device=torch.device("cpu")),
        ]
        for island in islands:
            island.pop = torch.randn(4, 2)
            island.fit = island.pop.square().sum(dim=1)
            best_index = torch.argmin(island.fit)
            island.best = island.pop[best_index].clone()
            island.best_fit = island.fit[best_index].clone()
            island._update_archives()

        class CountingGenerator:
            def __init__(self):
                self.calls = 0

            def generate(self, state, context):
                self.calls += 1
                self.state = state
                return None

        generator = CountingGenerator()
        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            operator_generator=generator,
            warmup_generations=0,
            llm_interval=1,
        )
        coordinator.generation = 10
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 0)

        for island in islands:
            island.pop.zero_()
            island.fit.fill_(1.0)
            island.stagnation = 15
            island.operator_failures = 4
            island.improvement_history.extend([0.0] * 5)
            island._update_archives()
        coordinator.generation = 20
        coordinator._maybe_synthesize()
        self.assertEqual(generator.calls, 1)
        self.assertGreaterEqual(generator.state["trigger_score"], 0.7)

    def test_trigger_diagnostics_use_priority_rules_without_weights(self):
        from llm_operator_de.islands import IslandCoordinator

        score, reason, _, _ = IslandCoordinator._trigger_diagnostics(
            {
                "stagnation": 10,
                "diversity": 0.9,
                "improvement_rate": 0.0,
                "operator_failures": 0,
            },
            threshold=10,
            progress=0.2,
        )
        self.assertEqual(reason, "stagnation")
        self.assertEqual(score, 0.9)

        score, reason, _, _ = IslandCoordinator._trigger_diagnostics(
            {
                "stagnation": 10,
                "diversity": 0.1,
                "improvement_rate": 0.0,
                "operator_failures": 0,
            },
            threshold=10,
            progress=0.2,
        )
        self.assertEqual(reason, "diversity_collapse")
        self.assertEqual(score, 1.0)

    def test_parallel_step_batches_island_trials_into_one_evaluation(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
        ]
        for index, island in enumerate(islands):
            island.pop = torch.tensor(
                [[-0.8 + index * 0.1, 0.5], [0.2, -0.4], [0.6, 0.3], [-0.1, 0.1]],
                dtype=torch.float32,
            )
            island.fit = island.pop.square().sum(dim=1)
            best_index = torch.argmin(island.fit)
            island.best = island.pop[best_index].clone()
            island.best_fit = island.fit[best_index].clone()
            island._update_archives()

        calls = []

        def evaluate(batch):
            calls.append(tuple(batch.shape))
            return batch.square().sum(dim=1)

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            migration_interval=10,
        )
        coordinator.step(evaluate=evaluate, parallel=True)

        self.assertEqual(calls, [(12, 2)])
        self.assertEqual(coordinator.generation, 1)
        self.assertTrue(all(island.gen == 1 for island in islands))

    def test_steady_state_step_uses_one_real_evaluation(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
        ]
        for index, island in enumerate(islands):
            island.pop = torch.tensor(
                [[-0.8 + index * 0.1, 0.5], [0.2, -0.4], [0.6, 0.3], [-0.1, 0.1]],
                dtype=torch.float32,
            )
            island.fit = island.pop.square().sum(dim=1)
            best_index = torch.argmin(island.fit)
            island.best = island.pop[best_index].clone()
            island.best_fit = island.fit[best_index].clone()
            island._update_archives()

        calls = []

        def evaluate(batch):
            calls.append(tuple(batch.shape))
            return batch.square().sum(dim=1)

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            migration_interval=100,
            steady_state=True,
        )
        coordinator.step(evaluate=evaluate, parallel=False)
        coordinator.step(evaluate=evaluate, parallel=False)

        self.assertEqual(calls, [(1, 2), (1, 2)])
        self.assertEqual(coordinator.generation, 2)
        self.assertEqual([island.gen for island in islands], [1, 1, 0])
        self.assertEqual([island.trial_evals for island in islands], [1, 1, 0])

    def test_dynamic_threshold_and_elite_migration(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=20, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=20, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=20, device=torch.device("cpu")),
        ]
        for index, island in enumerate(islands):
            island.pop = torch.full((4, 2), float(index + 1))
            island.fit = torch.full((4,), float(index + 1))
            island.best = island.pop[0].clone()
            island.best_fit = island.fit[0].clone()
            island._update_archives()

        coordinator = IslandCoordinator(
            islands,
            operator_archive=OperatorArchive(),
            migration_interval=1,
            llm_interval=3,
        )
        threshold_early = coordinator.dynamic_stagnation_threshold(0.0, 1.0)
        threshold_late = coordinator.dynamic_stagnation_threshold(0.9, 0.05)
        self.assertGreaterEqual(threshold_early, threshold_late)

        coordinator.migrate()
        global_best = min(float(island.best_fit) for island in islands)
        self.assertTrue(all(float(island.best_fit) <= 2.0 for island in islands))
        self.assertEqual(global_best, 1.0)

    def test_island_state_roundtrip(self):
        import torch
        from llm_operator_de import LLMOperatorDE, OperatorArchive
        from llm_operator_de.islands import IslandCoordinator

        islands = [
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
        ]
        for island in islands:
            island.fit = torch.ones(4)
            island.best_fit = torch.tensor(1.0)
        coordinator = IslandCoordinator(islands, operator_archive=OperatorArchive())
        coordinator.generation = 3
        coordinator.llm_attempts = 2
        state = coordinator.checkpoint_state_dict()

        restored = IslandCoordinator(
            [
                LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
                LLMOperatorDE(4, -torch.ones(2), torch.ones(2), max_gen=5, device=torch.device("cpu")),
            ],
            operator_archive=OperatorArchive(),
        )
        restored.restore_checkpoint_state(state)
        self.assertEqual(restored.generation, 3)
        self.assertEqual(restored.llm_attempts, 2)
        self.assertEqual(len(restored.islands), 2)


if __name__ == "__main__":
    unittest.main()
