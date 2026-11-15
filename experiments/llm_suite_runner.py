"""Runner for the bundled ``llm_operator_de`` package.

This module owns the EvoX/island orchestration needed by the expensive
single-objective suite and returns the suite's common result schema. Baseline
implementations are loaded through the experiment adapters rather than through
private external project paths.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVOX_SOURCE = _PROJECT_ROOT / "evox" / "src"
if str(_EVOX_SOURCE) not in sys.path:
    sys.path.insert(0, str(_EVOX_SOURCE))

from evox.problems.numerical import (  # noqa: F401  (import path is validated at runtime)
    Ackley,
    Ellipsoid,
    Griewank,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
)

from llm_operator_de import IslandCoordinator, LLMOperatorDE, LLMOperatorGenerator, OperatorArchive


def _device_for(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _island_sizes(total_population: int, island_count: int) -> list[int]:
    base, remainder = divmod(int(total_population), int(island_count))
    sizes = [base + (index < remainder) for index in range(int(island_count))]
    if min(sizes) < 4:
        raise ValueError("population_size must provide at least four individuals per island")
    return [int(size) for size in sizes]


def _initialize_island(
    island: LLMOperatorDE,
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    *,
    fitness: torch.Tensor | None = None,
    evaluation_count: int | None = None,
) -> None:
    if fitness is None:
        fitness = evaluate(island.pop)
        evaluation_count = island.pop_size if evaluation_count is None else evaluation_count
    elif evaluation_count is None:
        evaluation_count = island.pop_size
    island.fit = fitness
    island.cons = None
    scores = island._score(island.fit, island.cons)
    best_index = torch.argmin(scores)
    island.best = island.pop[best_index]
    island.best_fit = scores[best_index]
    island._update_archives()
    island.trial_evals += int(evaluation_count)


def run_local_llm_operator_de(
    problem: str,
    dimension: int,
    seed: int,
    config: Any,
    *,
    use_llm: bool = True,
    force_warmup_llm: bool | None = None,
    adaptive_parameters: bool = True,
    cross_island_rate: float = 0.35,
    global_guidance: float = 0.25,
    island_count: int = 3,
    migration_interval: int = 5,
    candidate_validation_evals: int = 20,
    llm_eval_interval: int = 200,
    rag_top_k: int = 2,
    knowledge_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the current three-island LLMOperatorDE under an exact FE budget."""

    from .expensive_suite import _lhs, get_suite_benchmark

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = _device_for(config.device)
    benchmark = get_suite_benchmark(problem, int(dimension))
    benchmark.initial_design = _lhs(
        int(config.initial_evals),
        benchmark.bounds,
        np.random.default_rng(int(seed)),
    )
    benchmark.problem_object = benchmark.problem_object.to(device=device)
    lower = torch.as_tensor(benchmark.bounds[:, 0], dtype=torch.float32, device=device)
    upper = torch.as_tensor(benchmark.bounds[:, 1], dtype=torch.float32, device=device)

    def evaluate(points: torch.Tensor) -> torch.Tensor:
        values = benchmark.problem_object.evaluate(points.to(device=device))
        return values.reshape(-1).to(device=device, dtype=torch.float32)

    generator = None
    if use_llm:
        generator = LLMOperatorGenerator.from_config(config.llm_config_path)
    archive_storage = (
        Path(knowledge_path)
        if knowledge_path is not None
        else Path(config.output_dir) / "operator_knowledge.json"
    )
    archive = OperatorArchive(storage_path=archive_storage)
    active_population_size = int(getattr(config, "population_size", 30))
    island_sizes = _island_sizes(active_population_size, island_count)
    remaining = max(0, int(config.total_evals) - int(config.initial_evals))
    full_generations, remainder = divmod(remaining, active_population_size)
    max_generations = max(1, int(full_generations))
    islands = [
        LLMOperatorDE(
            pop_size=size,
            lb=lower,
            ub=upper,
            max_gen=max_generations,
            llm_interval=10,
            stagnation_window=10,
            max_llm_calls=None,
            # The coordinator performs one aggregated synthesis and broadcasts
            # it; islands must not issue duplicate requests independently.
            operator_generator=None,
            operator_archive=archive,
            knowledge_path=None,
            task_id=f"{problem}-D{dimension}-seed{seed}-island{index}",
            adaptive_parameters=bool(adaptive_parameters),
            cross_island_rate=float(cross_island_rate),
            global_guidance=float(global_guidance),
            device=device,
        )
        for index, size in enumerate(island_sizes)
    ]
    coordinator = IslandCoordinator(
        islands,
        operator_archive=archive,
        operator_generator=generator,
        migration_interval=int(migration_interval),
        llm_interval=10,
        warmup_generations=1 if max_generations > 1 else 0,
        force_warmup_llm=(
            bool(config.force_warmup_llm)
            if force_warmup_llm is None
            else bool(force_warmup_llm)
        ),
        max_llm_calls=None,
        candidate_validation_evals=int(candidate_validation_evals),
        steady_state=False,
        llm_eval_interval=int(llm_eval_interval),
        rag_top_k=int(rag_top_k),
    )
    initial_points = torch.as_tensor(
        benchmark.initial_design,
        dtype=torch.float32,
        device=device,
    )
    initial_values = evaluate(initial_points)
    initial_order = torch.argsort(initial_values.detach())
    active_count = int(sum(island_sizes))
    active_indices = list(range(min(active_count, int(config.initial_evals))))
    best_initial_index = int(initial_order[0].detach().cpu())
    if active_indices and best_initial_index not in active_indices:
        active_indices[-1] = best_initial_index
    initial_archive = initial_points.detach().clone()
    initial_eval_shares = _island_sizes(int(config.initial_evals), int(island_count))
    initial_offset = 0
    for index, island in enumerate(islands):
        indices = active_indices[initial_offset : initial_offset + island.pop_size]
        island.pop = initial_points[indices].clone()
        island.history_archive = initial_archive.clone()
        _initialize_island(
            island,
            evaluate,
            fitness=initial_values[indices].clone(),
            evaluation_count=initial_eval_shares[index],
        )
        initial_offset += island.pop_size

    def current_best() -> float:
        return min(float(island.best_fit.detach().cpu()) for island in islands)

    history: list[dict[str, Any]] = [
        {"evaluations": int(config.initial_evals), "best_f": current_best()}
    ]
    real_evals = int(config.initial_evals)
    if progress_callback is not None:
        progress_callback({"real_evals": real_evals, "best_f": history[-1]["best_f"], "generation": 0})

    for generation in range(int(full_generations)):
        coordinator.step(evaluate=evaluate, parallel=True)
        real_evals += active_population_size
        best_value = current_best()
        history.append({"evaluations": real_evals, "best_f": best_value})
        if progress_callback is not None:
            progress_callback(
                {
                    "real_evals": real_evals,
                    "best_f": best_value,
                    "generation": generation + 1,
                    "api_calls": int(coordinator.llm_attempts),
                    "api_successes": int(generator.successful_generations) if generator else 0,
                }
            )

    # A non-divisible remainder is still counted against the true budget.  It
    # is used as a final random fill because a partial island generation cannot
    # be passed through the tensorized coordinator safely.
    if remainder:
        points = lower + torch.rand((int(remainder), int(dimension)), device=device) * (upper - lower)
        values = evaluate(points)
        real_evals += int(remainder)
        best_value = min(current_best(), float(values.min().detach().cpu()))
        history.append({"evaluations": real_evals, "best_f": best_value})

    best_value = min(float(item["best_f"]) for item in history)
    audit = {
        "implementation": "direct_local_llm_operator_de",
        "legacy_comparison_runner": False,
        "device": str(device),
        "island_count": int(island_count),
        "island_sizes": island_sizes,
        "active_population_size": active_population_size,
        "initial_design_evals": int(config.initial_evals),
        "update_batch_evals": active_population_size,
        "update_protocol": "three-island-generational-batch",
        "true_evaluations": int(real_evals),
        "use_llm": bool(use_llm),
        "adaptive_parameters": bool(adaptive_parameters),
        "cross_island_rate": float(cross_island_rate),
        "global_guidance": float(global_guidance),
        "remainder_fill_evals": int(remainder),
    }
    return {
        "algorithm": "LLMOperatorDE",
        "problem": problem,
        "dimension": int(dimension),
        "seed": int(seed),
        "best_f": float(best_value),
        "initial_best_f": float(history[0]["best_f"]),
        "initial_evals": int(config.initial_evals),
        "real_evals": int(real_evals),
        "best_f_history": history,
        "api_calls": int(coordinator.llm_attempts),
        "api_successes": int(generator.successful_generations) if generator else 0,
        "api_failures": int(generator.failed_generations) if generator else 0,
        "api_prompt_tokens": int(generator.prompt_tokens) if generator else 0,
        "api_completion_tokens": int(generator.completion_tokens) if generator else 0,
        "api_total_tokens": int(generator.total_tokens) if generator else 0,
        "api_last_error": generator.last_error if generator else "",
        "api_last_response_preview": generator.last_response_preview if generator else "",
        "api_last_finish_reason": generator.last_finish_reason if generator else "",
        "status": (
            "ok"
            if not generator or int(generator.failed_generations) == 0
            else "llm_api_partial_failure"
        ),
        "error": generator.last_error if generator and generator.failed_generations else "",
        "innovation_attempts": int(coordinator.innovation_attempts),
        "innovation_successes": int(coordinator.innovation_successes),
        "innovation_misses": int(coordinator.innovation_misses),
        "audit": audit,
    }


__all__ = ["run_local_llm_operator_de"]
