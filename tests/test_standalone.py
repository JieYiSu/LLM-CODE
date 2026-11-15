import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVOX_SOURCE = ROOT / "evox" / "src"
sys.path.insert(0, str(EVOX_SOURCE))
sys.path.insert(0, str(ROOT))


class TestStandaloneLLMOperatorDE(unittest.TestCase):
    def test_standalone_package_and_ast_compile(self):
        from llm_operator_de import OperatorASTCompiler

        spec = OperatorASTCompiler.compile(
            {
                "op": "weighted_sum",
                "terms": [
                    {"op": "primitive", "name": "rand1", "weight": 0.5},
                    {"op": "primitive", "name": "best1", "weight": 0.5},
                ],
            },
            operator_id="standalone_ast",
        )
        self.assertEqual(spec.mutation, "weighted_ensemble")

    def test_standalone_algorithm_runs_without_api(self):
        import torch
        from evox.core import Problem
        from evox.workflows import StdWorkflow
        from llm_operator_de import LLMOperatorDE

        class Sphere(Problem):
            def evaluate(self, population):
                return population.square().sum(dim=1)

        algorithm = LLMOperatorDE(
            pop_size=8,
            lb=-torch.ones(3),
            ub=torch.ones(3),
        )
        workflow = StdWorkflow(algorithm, Sphere())
        workflow.init_step()
        workflow.step()
        self.assertTrue(torch.isfinite(workflow.algorithm.fit).all())


if __name__ == "__main__":
    unittest.main()
