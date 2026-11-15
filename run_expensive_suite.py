"""One-click entry point for the expensive single-objective suite."""

from __future__ import annotations

from experiments.expensive_suite import SuiteConfig, run_ablation_suite, run_suite


# Edit these values for a pilot run before launching the full paper sweep.
CONFIG = SuiteConfig(
    total_evals=300,
    initial_evals=100,
    population_size=30,
    repeats=20,
    dimensions=(30, 50, 100, 200, 500, 1000),
    problems=(
        "Ackley",
        "Griewank",
        "Rastrigin",
        "Rosenbrock",
        "Schwefel",
        "Sphere",
        "Ellipsoid",
    ),
    device="cuda",
    resume=True,
    checkpoint_interval=1,
)

# The main comparison and ablation are separate output folders. Set this to
# True only after the main comparison has completed and been inspected.
RUN_ABLATION = True


def main() -> None:
    try:
        run_suite(CONFIG)
        if RUN_ABLATION:
            run_ablation_suite(CONFIG)
    except KeyboardInterrupt:
        # The suite has already persisted its latest completed-run progress.
        # Keep Ctrl+C quiet; rerunning this same entry point resumes it.
        return


if __name__ == "__main__":
    main()
