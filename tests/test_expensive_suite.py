import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evox" / "src"))
sys.path.insert(0, str(ROOT))


class TestExpensiveSuite(unittest.TestCase):
    def test_algorithms_have_separate_entry_modules(self):
        from experiments.expensive_algorithms.bgp_saea import RecentPaperController
        from experiments.expensive_algorithms.esa import ESAPort
        from experiments.expensive_algorithms.sade_ss import SADE_SSController
        from experiments.expensive_algorithms.sgdlco import SGDLCOController
        from experiments.expensive_algorithms.tsddeo import TSDDEOPort

        self.assertEqual(TSDDEOPort.__module__, "experiments.expensive_algorithms.tsddeo")
        self.assertEqual(ESAPort.__module__, "experiments.expensive_algorithms.esa")
        self.assertEqual(SADE_SSController.__module__, "experiments.expensive_algorithms.paper_surrogates")
        self.assertEqual(SGDLCOController.__module__, "experiments.expensive_algorithms.paper_surrogates")
        self.assertEqual(RecentPaperController.__module__, "experiments.expensive_algorithms.paper_surrogates")

    def test_problem_set_removes_duplicate_and_has_expected_bounds(self):
        from experiments.expensive_suite import PROBLEMS, get_suite_benchmark

        self.assertEqual(
            PROBLEMS,
            (
                "Ackley",
                "Griewank",
                "Rastrigin",
                "Rosenbrock",
                "Schwefel",
                "Sphere",
                "Ellipsoid",
            ),
        )
        benchmark = get_suite_benchmark("Ackley", 3)
        self.assertEqual(benchmark.dimension, 3)
        self.assertEqual(benchmark.bounds.shape, (3, 2))
        self.assertEqual(benchmark.optimal_point, "[0, ..., 0]")
        self.assertEqual(benchmark.optimal_value, 0.0)

    def test_python_ports_use_exact_real_evaluation_budget(self):
        from experiments.expensive_suite import SuiteConfig, run_ported_algorithm

        config = SuiteConfig(
            total_evals=40,
            initial_evals=20,
            population_size=8,
            convergence_points=5,
        )
        for method in ("TSDDEO", "ESA"):
            result = run_ported_algorithm(method, "Sphere", 3, 1, config)
            self.assertEqual(result["real_evals"], 40)
            self.assertEqual(result["best_f_history"][-1]["evaluations"], 40)
            self.assertLessEqual(result["best_f"], result["initial_best_f"])

    def test_paper_baselines_are_local_and_do_not_need_external_path(self):
        from experiments.expensive_algorithms.paper_surrogates import (
            run_paper_algorithm,
        )
        from experiments.expensive_suite import SuiteConfig

        config = SuiteConfig(
            total_evals=20,
            initial_evals=10,
            population_size=5,
            convergence_points=4,
            device="cpu",
        )
        result = run_paper_algorithm("SADE-SS", "Sphere", 3, 1, config)
        self.assertEqual(result["algorithm"], "SADE-SS")
        self.assertEqual(result["real_evals"], 20)
        self.assertEqual(result["best_f_history"][-1]["evaluations"], 20)
        self.assertEqual(result["audit"]["implementation"], "local_expensive_algorithms")

    def test_main_llm_runner_uses_local_package_without_legacy_comparison_module(self):
        from experiments.expensive_suite import SuiteConfig, run_llm_algorithm

        config = SuiteConfig(
            total_evals=60,
            initial_evals=30,
            population_size=12,
            convergence_points=4,
            device="cpu",
        )
        result = run_llm_algorithm("Sphere", 3, 1, config, use_llm=False)
        self.assertEqual(result["algorithm"], "LLMOperatorDE")
        self.assertEqual(result["real_evals"], 60)
        self.assertEqual(result["audit"]["legacy_comparison_runner"], False)
        self.assertEqual(result["audit"]["island_sizes"], [4, 4, 4])
        self.assertEqual(result["audit"]["active_population_size"], 12)
        self.assertEqual(result["audit"]["update_batch_evals"], 12)

    def test_suite_completes_all_repeats_for_method_before_next_method(self):
        import shutil

        from experiments.expensive_suite import SuiteConfig, run_suite

        calls = []

        def fake_ported(method, problem, dimension, seed, config):
            calls.append((method, problem, int(dimension), int(seed)))
            return {
                "algorithm": method,
                "problem": problem,
                "dimension": int(dimension),
                "seed": int(seed),
                "best_f": float(seed),
                "initial_best_f": float(seed + 1),
                "initial_evals": 4,
                "real_evals": 8,
                "best_f_history": [
                    {"evaluations": 4, "best_f": float(seed + 1)},
                    {"evaluations": 8, "best_f": float(seed)},
                ],
                "audit": {},
            }

        directory = ROOT / "experiments" / ".suite_order_test"
        shutil.rmtree(directory, ignore_errors=True)
        config = SuiteConfig(
            total_evals=8,
            initial_evals=4,
            population_size=4,
            repeats=2,
            dimensions=(3,),
            problems=("Sphere",),
            algorithms=("TSDDEO", "ESA"),
            output_dir=directory,
            device="cpu",
        )
        try:
            with patch("experiments.expensive_suite.run_ported_algorithm", side_effect=fake_ported):
                run_suite(config)
            self.assertEqual(
                calls,
                [
                    ("TSDDEO", "Sphere", 3, 1),
                    ("TSDDEO", "Sphere", 3, 2),
                    ("ESA", "Sphere", 3, 1),
                    ("ESA", "Sphere", 3, 2),
                ],
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_suite_progress_resumes_after_interrupt_at_completed_run_boundary(self):
        import json
        import shutil

        from experiments.expensive_suite import SuiteConfig, run_suite

        calls = []
        interrupt = {"enabled": True}

        def fake_ported(method, problem, dimension, seed, config):
            calls.append((method, problem, int(dimension), int(seed)))
            if interrupt["enabled"] and len(calls) == 2:
                raise KeyboardInterrupt
            return {
                "algorithm": method,
                "problem": problem,
                "dimension": int(dimension),
                "seed": int(seed),
                "best_f": float(seed),
                "initial_best_f": float(seed + 1),
                "initial_evals": 4,
                "real_evals": 8,
                "best_f_history": [
                    {"evaluations": 4, "best_f": float(seed + 1)},
                    {"evaluations": 8, "best_f": float(seed)},
                ],
                "audit": {},
            }

        directory = ROOT / "experiments" / ".suite_resume_test"
        shutil.rmtree(directory, ignore_errors=True)
        config = SuiteConfig(
            total_evals=8,
            initial_evals=4,
            population_size=4,
            repeats=2,
            dimensions=(3,),
            problems=("Sphere",),
            algorithms=("TSDDEO", "ESA"),
            output_dir=directory,
            device="cpu",
        )
        try:
            with patch("experiments.expensive_suite.run_ported_algorithm", side_effect=fake_ported):
                with self.assertRaises(KeyboardInterrupt):
                    run_suite(config)
            progress = directory / "suite_progress.json"
            self.assertTrue(progress.exists())
            payload = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "interrupted")
            self.assertEqual(payload["completed_runs"], 1)

            calls_before_resume = len(calls)
            interrupt["enabled"] = False
            with patch("experiments.expensive_suite.run_ported_algorithm", side_effect=fake_ported):
                rows = run_suite(config)
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                calls[:calls_before_resume],
                [
                    ("TSDDEO", "Sphere", 3, 1),
                    ("TSDDEO", "Sphere", 3, 2),
                ],
            )
            self.assertEqual(
                calls[calls_before_resume:],
                [
                    ("TSDDEO", "Sphere", 3, 2),
                    ("ESA", "Sphere", 3, 1),
                    ("ESA", "Sphere", 3, 2),
                ],
            )
            self.assertEqual(json.loads(progress.read_text(encoding="utf-8"))["status"], "completed")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_output_writer_creates_error_and_convergence_workbooks(self):
        import shutil
        import openpyxl
        from experiments.expensive_suite import write_suite_workbooks

        rows = [
            {
                "algorithm": "TSDDEO",
                "problem": "Sphere",
                "dimension": 3,
                "seed": 1,
                "best_f": 1.0,
                "real_evals": 40,
                "best_f_history": [
                    {"evaluations": 20, "best_f": 2.0},
                    {"evaluations": 40, "best_f": 1.0},
                ],
            }
        ]
        directory = ROOT / "experiments" / ".suite_test_output"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            paths = write_suite_workbooks(rows, directory, prefix="smoke")
            error_book = openpyxl.load_workbook(paths["error_stats"], data_only=True)
            convergence_book = openpyxl.load_workbook(
                paths["convergence"], data_only=True
            )
            self.assertIn("Error_Mean_Std", error_book.sheetnames)
            self.assertIn("Sphere_D3", convergence_book.sheetnames)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_output_writer_orders_llm_last_bolds_best_and_adds_summary(self):
        import shutil
        import openpyxl
        from experiments.expensive_suite import write_suite_workbooks

        algorithms = ("BGP-SAEA", "ESA", "SADE-SS", "SGDLCO", "TSDDEO", "LLMOperatorDE")
        rows = []
        means = {
            "BGP-SAEA": 6.0,
            "ESA": 5.0,
            "SADE-SS": 4.0,
            "SGDLCO": 3.0,
            "TSDDEO": 2.0,
            "LLMOperatorDE": 1.0,
        }
        for algorithm in algorithms:
            for seed in (1, 2):
                value = means[algorithm] + (0.1 if seed == 2 else -0.1)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "problem": "Sphere",
                        "dimension": 3,
                        "seed": seed,
                        "best_f": value,
                        "initial_evals": 20,
                        "real_evals": 40,
                        "best_f_history": [
                            {"evaluations": 20, "best_f": value + 1.0},
                            {"evaluations": 40, "best_f": value},
                        ],
                    }
                )
        directory = ROOT / "experiments" / ".suite_format_test"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            paths = write_suite_workbooks(rows, directory, prefix="format")
            workbook = openpyxl.load_workbook(paths["error_stats"], data_only=True)
            sheet = workbook["Error_Mean_Std"]
            self.assertEqual(
                [sheet.cell(1, index).value for index in range(1, 9)],
                ["Problem", "Dimension", "BGP-SAEA", "ESA", "SADE-SS", "SGDLCO", "TSDDEO", "LLMOperatorDE"],
            )
            self.assertTrue(sheet.cell(2, 8).font.bold)
            self.assertIn("Summary_BestCount", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Summary_MeanRank", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Summary_+_vs_LLMOperatorDE", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Summary_-_vs_LLMOperatorDE", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Summary_=_vs_LLMOperatorDE", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Summary_+/-/=_vs_LLMOperatorDE", [sheet.cell(row, 2).value for row in range(1, sheet.max_row + 1)])
            self.assertIn("Statistics", workbook.sheetnames)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_ablation_contribution_report_normalizes_positive_log_losses(self):
        from experiments.expensive_suite import compute_ablation_contributions

        rows = []
        values = {
            "LLMOperatorDE-Full": 1.0,
            "LLMOperatorDE-NoAdaptiveFCR": 2.0,
            "LLMOperatorDE-NoLLM": 4.0,
            "LLMOperatorDE-NoSharedMating": 1.0,
        }
        for algorithm, value in values.items():
            for seed in (1, 2):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "problem": "Sphere",
                        "dimension": 3,
                        "seed": seed,
                        "best_f": value,
                    }
                )
        report = compute_ablation_contributions(rows)
        by_component = {row["component"]: row for row in report}
        self.assertAlmostEqual(by_component["NoAdaptiveFCR"]["contribution_share_pct"], 33.3333, places=3)
        self.assertAlmostEqual(by_component["NoLLM"]["contribution_share_pct"], 66.6667, places=3)
        self.assertEqual(by_component["NoSharedMating"]["contribution_share_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
