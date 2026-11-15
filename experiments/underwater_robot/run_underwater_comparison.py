"""Run the common expensive comparison protocol on underwater scenarios."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
EVOX_SOURCE = PROJECT / "evox" / "src"
if str(EVOX_SOURCE) not in sys.path:
    sys.path.insert(0, str(EVOX_SOURCE))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from llm_operator_de import IslandCoordinator, LLMOperatorDE, LLMOperatorGenerator, OperatorArchive

from experiments.expensive_algorithms.paper_surrogates import (
    RecentPaperController,
    SADE_SSController,
    SGDLCOController,
    _LocalPaperConfig,
    _history_rows,
)
from experiments.expensive_algorithms.esa import ESAPort
from experiments.expensive_algorithms.tsddeo import TSDDEOPort
from experiments.expensive_suite import write_suite_workbooks
from .experiment_config import CONFIG, UnderwaterExperimentConfig
from .plot_3d_paths import plot_3d_paths
from .underwater_objective import UnderwaterPathObjective
from .underwater_scenarios import minimum_clearance


class UnderwaterBenchmark:
    """Adapter expected by the existing comparison algorithm ports."""

    def __init__(self, objective: UnderwaterPathObjective):
        self.problem_id = f"UnderwaterAUV-{objective.scenario}"
        self.dimension = int(objective.dimension)
        self.problem = objective
        self.problem_object = objective
        self.bounds = np.asarray(objective.bounds, dtype=float)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        values = [self.problem.evaluate_one(row)[0] for row in array]
        return np.asarray(values, dtype=float).reshape(-1)


def _common_initial_design(
    count: int,
    bounds: np.ndarray,
    seed: int,
    objective: UnderwaterPathObjective | None = None,
) -> np.ndarray:
    """Create a shared design with reproducible path-aware warm starts.

    The Latin-hypercube part keeps the comparison broad, while a small set of
    deterministic depth and lateral-detour trajectories prevents every
    island from starting with only colliding random paths.  The exact same
    design is passed to every algorithm for a given scenario and seed.
    """

    rng = np.random.default_rng(int(seed))
    count = int(count)
    unit = np.empty((count, bounds.shape[0]), dtype=float)
    for column in range(bounds.shape[0]):
        unit[:, column] = (rng.permutation(count) + rng.random(count)) / count
    design = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    if objective is None or count <= 0:
        return design

    twin = objective.twin
    points = np.linspace(twin.scenario["start"], twin.scenario["goal"], twin.intermediate_points + 2)[1:-1]
    fraction = np.linspace(0.0, 1.0, twin.intermediate_points + 2)[1:-1]
    directions = np.asarray(
        [
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
        dtype=float,
    )
    amplitudes = (1.25, 1.25, 1.50, 1.50, 1.25, 1.25)
    speeds = (0.70, 0.70, 0.85, 0.85, 1.00, 1.00, 1.20)
    templates: list[np.ndarray] = []
    # Constant high and low depth passages are particularly useful in the
    # multi-obstacle scene, where lateral detours can still intersect the
    # second obstacle.  Their endpoints remain those prescribed by the task.
    for depth in (-0.70, -3.40):
        depth_points = points.copy()
        depth_points[:, 2] = depth
        templates.append(np.column_stack((depth_points, np.full(len(depth_points), 0.85))).reshape(-1))
    for index, (direction, amplitude) in enumerate(zip(directions, amplitudes)):
        offset = amplitude * np.sin(np.pi * fraction)[:, None] * direction[None, :]
        path_points = points + offset
        speed = speeds[index % len(speeds)]
        templates.append(np.column_stack((path_points, np.full(len(path_points), speed))).reshape(-1))
    template_array = np.asarray(templates, dtype=float)
    template_array = np.maximum(np.minimum(template_array, bounds[:, 1]), bounds[:, 0])
    take = min(int(count), int(template_array.shape[0]))
    design[:take] = template_array[:take]
    return design


def _design_hash(design: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(design, dtype=np.float64).tobytes()).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """Return a short content hash for a source/configuration file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def protocol_manifest(config: UnderwaterExperimentConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an auditable, self-contained description of one suite run.

    The manifest is deliberately separate from the result table.  It records
    the protocol that a reviewer needs to reproduce the table and a few
    machine-checkable completeness checks, while retaining raw rows even when
    a run is invalid.  No performance claim is inferred from this metadata.
    """

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("experiment_config.py"),
        Path(__file__).with_name("underwater_objective.py"),
        Path(__file__).with_name("underwater_scenarios.py"),
        Path(__file__).with_name("underwater_twin.py"),
        PROJECT / "llm_operator_de" / "core.py",
        PROJECT / "llm_operator_de" / "islands.py",
    ]
    source_hashes = {
        str(path.relative_to(PROJECT)).replace("\\", "/"): _file_sha256(path)
        for path in source_paths
        if path.exists()
    }
    keys = [
        (str(row.get("scenario")), str(row.get("algorithm")), int(row.get("seed", -1)))
        for row in rows
    ]
    expected = len(config.scenarios) * len(config.algorithms) * len(config.seeds)
    scenario_seed_hashes: dict[str, set[str]] = {}
    for row in rows:
        key = f"{row.get('scenario')}/seed_{int(row.get('seed', -1)):02d}"
        scenario_seed_hashes.setdefault(key, set()).add(str(row.get("initial_design_hash", "")))
    normalized_hashes = {
        key: sorted(value) for key, value in sorted(scenario_seed_hashes.items())
    }
    statuses = {}
    for row in rows:
        status = str(row.get("status", ""))
        statuses[status] = statuses.get(status, 0) + 1
    real_eval_values = sorted({int(row.get("real_evals", -1)) for row in rows})
    dynamic_modes = {}
    for row in rows:
        mode = str(_row_metric(row, "mujoco_replay_mode", ""))
        dynamic_modes[mode] = dynamic_modes.get(mode, 0) + 1
    return {
        "manifest_version": "underwater_protocol_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "total_evals": int(config.total_evals),
            "initial_evals": int(config.initial_evals),
            "population_size": int(config.population_size),
            "active_island_sizes": _island_sizes(config.population_size, config.island_count),
            "initial_design_evals": int(config.initial_evals),
            "update_batch_evals": int(config.population_size),
            "update_protocol": "three-island-generational-batch; one trial per active individual",
            "baseline_update_batch_evals": 1,
            "baseline_update_protocol": "single-candidate-surrogate-infill",
            "dimension": int(config.trajectory_dimension),
            "robot_dof": int(config.robot_dof),
            "intermediate_points": int(config.intermediate_points),
            "island_count": int(config.island_count),
            "migration_interval": int(config.migration_interval),
            "llm_interval": int(config.llm_interval),
            "llm_eval_interval": int(config.llm_eval_interval),
            "candidate_validation_evals": int(config.candidate_validation_evals),
            "rag_top_k": int(config.rag_top_k),
            "repeats": int(config.repeats),
            "seed_start": int(config.seed_start),
            "seeds": [int(seed) for seed in config.seeds],
            "algorithms": list(config.algorithms),
            "scenarios": list(config.scenarios),
            "objective_version": "underwater_v9_trackability_proxy_dynamic_replay_warmstart",
            "dynamic_proxy": {
                "terms": ["acceleration", "jerk", "speed_excess", "force"],
                "role": "low-cost planner ranking term; MuJoCo replay remains final validation",
            },
            "initial_design": "shared Latin-hypercube design plus deterministic path-aware warm starts per scenario and seed",
            "device_requested": str(config.device),
            "llm_enabled": bool(config.use_llm),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "torch": _package_version("torch"),
            "mujoco": _package_version("mujoco"),
        },
        "source_hashes": source_hashes,
        "validation": {
            "observed_rows": len(rows),
            "expected_rows": expected,
            "unique_keys": len(set(keys)),
            "duplicate_keys": len(keys) - len(set(keys)),
            "status_counts": statuses,
            "real_eval_values": real_eval_values,
            "all_rows_at_requested_budget": bool(
                rows and real_eval_values == [int(config.total_evals)]
            ),
            "initial_design_hashes_by_scenario_seed": normalized_hashes,
            "shared_initial_design_within_scenario_seed": bool(
                all(len(values) <= 1 for values in normalized_hashes.values())
            ),
            "dynamic_feasible_rows": sum(
                str(row.get("dynamic_feasible", "false")).lower() == "true"
                for row in rows
            ),
            "reduced_feasible_rows": sum(
                _as_bool(_row_metric(row, "feasible", False)) for row in rows
            ),
            "mujoco_replay_mode_counts": dynamic_modes,
            "objective_versions": sorted({str(row.get("objective_version", "")) for row in rows}),
        },
    }


def write_protocol_manifest(
    config: UnderwaterExperimentConfig,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Persist :func:`protocol_manifest` beside the suite tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol_manifest.json"
    path.write_text(
        json.dumps(protocol_manifest(config, rows), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


class ComparisonConfig:
    def __init__(self, config: UnderwaterExperimentConfig):
        self.total_evals = int(config.total_evals)
        self.initial_evals = int(config.initial_evals)
        self.population_size = int(config.population_size)

    def max_evaluations_for(self, dimension: int) -> int:
        del dimension
        return self.total_evals


def _device_for(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _island_sizes(total_population: int, island_count: int) -> list[int]:
    base, remainder = divmod(int(total_population), int(island_count))
    sizes = [base + int(index < remainder) for index in range(int(island_count))]
    if min(sizes) < 4:
        raise ValueError("population_size must provide at least four candidates per island")
    return sizes


def _initialize_island(
    island: LLMOperatorDE,
    evaluate: Any,
    *,
    fitness: torch.Tensor | None = None,
    evaluation_count: int | None = None,
) -> None:
    """Initialize an island, reusing fitness values from the common design."""

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


def _run_llm(config: UnderwaterExperimentConfig) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    objective = UnderwaterPathObjective(config.scenario, intermediate_points=config.intermediate_points)
    benchmark = UnderwaterBenchmark(objective)
    benchmark.initial_design = _common_initial_design(
        config.initial_evals, benchmark.bounds, config.seed, objective
    )
    device = _device_for(config.device)
    lower = torch.as_tensor(benchmark.bounds[:, 0], dtype=torch.float32, device=device)
    upper = torch.as_tensor(benchmark.bounds[:, 1], dtype=torch.float32, device=device)
    generator = LLMOperatorGenerator.from_config(config.config_path) if config.use_llm else None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = OperatorArchive(storage_path=output_dir / "operator_knowledge.json")
    # The active DE population is the configured total population (30), not
    # the initialization budget.  Three islands therefore perform one
    # tensorized update with 10 candidates each, consuming 30 real evaluations.
    sizes = _island_sizes(config.population_size, config.island_count)
    remaining = max(0, int(config.total_evals) - int(config.initial_evals))
    full_generations, remainder = divmod(remaining, int(config.population_size))
    max_generations = max(1, int(full_generations))
    midpoint = 0.5 * (objective.twin.scenario["start"] + objective.twin.scenario["goal"])
    local_flow = objective.twin.current.velocity(midpoint, 0.0)
    local_clearance = min(
        float(minimum_clearance(midpoint, obstacle))
        for obstacle in objective.twin.scenario["obstacles"]
    )
    environment_context = {
        "environment_domain": "underwater_auv_path_planning",
        "current_field": objective.twin.scenario["current"],
        "local_current_speed": float(np.linalg.norm(local_flow)),
        "local_current_direction": np.asarray(local_flow, dtype=float).round(6).tolist(),
        "local_obstacle_clearance": local_clearance,
        "hull_radius": float(objective.twin.parameters.hull_radius),
        "trajectory_representation": "endpoint_anchored_catmull_rom_waypoints_xyz_speed",
        "trackability_proxy": "acceleration_jerk_speed_force",
        "active_population_size": int(config.population_size),
        "island_population_sizes": list(sizes),
    }

    def evaluate(points: torch.Tensor) -> torch.Tensor:
        return objective.evaluate(points)

    islands = [
        LLMOperatorDE(
            pop_size=size,
            lb=lower,
            ub=upper,
            max_gen=max_generations,
            llm_interval=int(config.llm_interval),
            stagnation_window=int(config.llm_interval),
            max_llm_calls=None,
            # LLM synthesis is coordinated once from the aggregated island
            # state.  Individual islands must not call the shared generator,
            # otherwise one trigger becomes three independent API requests.
            operator_generator=None,
            operator_archive=archive,
            knowledge_path=None,
            task_id=f"underwater-{config.scenario}-seed{config.seed}-island{index}",
            adaptive_parameters=True,
            cross_island_rate=0.35,
            global_guidance=0.25,
            device=device,
        )
        for index, size in enumerate(sizes)
    ]
    coordinator = IslandCoordinator(
        islands,
        operator_archive=archive,
        operator_generator=generator,
        migration_interval=int(config.migration_interval),
        llm_interval=int(config.llm_interval),
        warmup_generations=1 if max_generations > 1 else 0,
        force_warmup_llm=bool(config.force_warmup_llm and config.use_llm),
        max_llm_calls=None,
        candidate_validation_evals=int(config.candidate_validation_evals),
        steady_state=False,
        llm_eval_interval=int(config.llm_eval_interval),
        rag_top_k=int(config.rag_top_k),
        state_context=environment_context,
    )
    # Evaluate the 100-point common initialization exactly once.  The active
    # populations reuse a deterministic subset containing the best initial
    # point, while the full design is retained in each island history archive
    # for parent sampling.  This keeps the 100-FE initialization protocol and
    # the 30-FE generational update protocol simultaneously explicit.
    initial_points = torch.as_tensor(benchmark.initial_design, dtype=torch.float32, device=device)
    initial_values = evaluate(initial_points)
    initial_order = torch.argsort(initial_values.detach())
    active_count = int(sum(sizes))
    active_indices = list(range(min(active_count, int(config.initial_evals))))
    best_initial_index = int(initial_order[0].detach().cpu())
    if active_indices and best_initial_index not in active_indices:
        active_indices[-1] = best_initial_index
    initial_archive = initial_points.detach().clone()
    initial_eval_shares = _island_sizes(config.initial_evals, config.island_count)
    initial_offset = 0
    for island_index, island in enumerate(islands):
        indices = active_indices[initial_offset : initial_offset + island.pop_size]
        island.pop = initial_points[indices].clone()
        island.history_archive = initial_archive.clone()
        initial_offset += island.pop_size
        _initialize_island(
            island,
            evaluate,
            fitness=initial_values[indices].clone(),
            evaluation_count=initial_eval_shares[island_index],
        )

    def current_best() -> float:
        return min(float(island.best_fit.detach().cpu()) for island in islands)

    history: list[dict[str, Any]] = [
        {"evaluations": int(config.initial_evals), "best_f": current_best()}
    ]
    print(
        f"Underwater {config.scenario:30s} LLMOperatorDE seed={config.seed:02d} "
        f"FE={config.initial_evals}/{config.total_evals} best={history[-1]['best_f']:.6e}",
        flush=True,
    )
    real_evals = int(config.initial_evals)
    for generation in range(int(full_generations)):
        coordinator.step(evaluate=evaluate, parallel=True)
        real_evals += int(config.population_size)
        history.append({"evaluations": real_evals, "best_f": current_best()})
        print(
            f"Underwater {config.scenario:30s} LLMOperatorDE seed={config.seed:02d} "
            f"gen={generation + 1}/{full_generations} FE={real_evals}/{config.total_evals} "
            f"best={history[-1]['best_f']:.6e} api={coordinator.llm_attempts}",
            flush=True,
        )
    if remainder:
        points = lower + torch.rand((int(remainder), objective.dimension), device=device) * (upper - lower)
        values = evaluate(points)
        real_evals += int(remainder)
        history.append({"evaluations": real_evals, "best_f": min(current_best(), float(values.min().detach().cpu()))})

    best_island = min(islands, key=lambda island: float(island.best_fit.detach().cpu()))
    best_vector = best_island.best.detach().cpu().numpy()
    best_value, metrics, path = objective.evaluate_one(best_vector)
    replay = objective.twin.replay(best_vector, engine="mujoco_dynamic")
    builtin_ids = {"best1", "rand1", "current_to_best1", "current_to_rand1"}
    generated_ids = [
        str(record.get("operator", {}).get("operator_id"))
        for record in archive.records
        if str(record.get("operator", {}).get("operator_id")) not in builtin_ids
    ]
    actual_api_calls = (
        int(generator.successful_generations) + int(generator.failed_generations)
        if generator
        else 0
    )
    api_failures = int(generator.failed_generations) if generator else 0
    api_successes = int(generator.successful_generations) if generator else 0
    status = "ok" if api_failures == 0 else "llm_api_partial_failure"
    dynamic_feasible = bool(
        int(replay.get("mujoco_collision_count", 0)) == 0
        and float(replay.get("mujoco_terminal_error", float("inf"))) <= 0.75
    )
    return {
        "algorithm": "LLMOperatorDE",
        "problem": benchmark.problem_id,
        "scenario": config.scenario,
        "objective_version": "underwater_v9_trackability_proxy_dynamic_replay_warmstart",
        "initial_design_hash": _design_hash(benchmark.initial_design),
        "dimension": objective.dimension,
        "robot_dof": config.robot_dof,
        "intermediate_points": config.intermediate_points,
        "seed": int(config.seed),
        "pop_size": int(config.population_size),
        "population_size": int(config.population_size),
        "island_count": config.island_count,
        "island_sizes": sizes,
        "active_population_size": int(sum(sizes)),
        "initial_design_evals": int(config.initial_evals),
        "update_batch_evals": int(sum(sizes)),
        "update_protocol": "three-island-generational-batch",
        "requested_evals": config.total_evals,
        "initial_evals": config.initial_evals,
        "real_evals": real_evals,
        "generations_run": len(history) - 1,
        "stop_reason": "evaluation_budget",
        "initial_best_f": history[0]["best_f"],
        "best_f": float(best_value),
        "wall_time_s": float(time.perf_counter() - started),
        "best_f_history": history,
        "path": path,
        "metrics": metrics,
        "mujoco_replay": {key: value for key, value in replay.items() if key != "path"},
        "dynamic_feasible": dynamic_feasible,
        # Each island owns the shared generator and may request a proposal at
        # its own interval.  Count actual HTTP attempts from the generator;
        # coordinator.llm_attempts only counts coordinator-level triggers.
        "api_calls": actual_api_calls,
        "coordinator_llm_triggers": int(coordinator.llm_attempts),
        "api_successes": api_successes,
        "api_failures": api_failures,
        "api_prompt_tokens": int(generator.prompt_tokens) if generator else 0,
        "api_completion_tokens": int(generator.completion_tokens) if generator else 0,
        "api_total_tokens": int(generator.total_tokens) if generator else 0,
        "api_last_error": generator.last_error if generator else "",
        "api_last_response_preview": generator.last_response_preview if generator else "",
        "api_last_finish_reason": generator.last_finish_reason if generator else "",
        "generated_operator_ids": generated_ids,
        "innovation_attempts": int(coordinator.innovation_attempts),
        "innovation_successes": int(coordinator.innovation_successes),
        "innovation_misses": int(coordinator.innovation_misses),
        "audit": {
            "implementation": "underwater_reduced_order_plus_mujoco_dynamic_replay",
            "true_evaluations": real_evals,
            "device": str(device),
            "objective_device": "cpu_numpy_reduced_order",
            "gpu_acceleration_scope": "DE variation and island batching",
            "legacy_comparison_runner": False,
            "initial_design": "shared_lhs_plus_path_aware_warm_starts_per_seed",
        },
        "status": status,
        "error": generator.last_error if generator and api_failures else "",
    }, best_vector


def _run_baseline(method: str, config: UnderwaterExperimentConfig) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    objective = UnderwaterPathObjective(config.scenario, intermediate_points=config.intermediate_points)
    benchmark = UnderwaterBenchmark(objective)
    benchmark.initial_design = _common_initial_design(
        config.initial_evals, benchmark.bounds, config.seed, objective
    )
    suite_config = ComparisonConfig(config)
    if method == "TSDDEO":
        controller = TSDDEOPort(benchmark, suite_config, int(config.seed))
        result = controller.run()
        best_index = int(np.argmin(controller.archive_y))
        best_vector = np.asarray(controller.archive_x[best_index], dtype=float)
        history = result.get("best_f_history", [])
        audit = result.get("audit", {})
    elif method == "ESA":
        controller = ESAPort(benchmark, suite_config, int(config.seed))
        result = controller.run()
        best_index = int(np.argmin(controller.archive_y))
        best_vector = np.asarray(controller.archive_x[best_index], dtype=float)
        history = result.get("best_f_history", [])
        audit = result.get("audit", {})
    else:
        local_config = _LocalPaperConfig(config)
        if method == "SADE-SS":
            controller = SADE_SSController(benchmark, local_config, int(config.seed))
            controller.tau = min(local_config.initial_evals, local_config.total_evals)
            controller.gs = controller.tau
            best_vector, _, evaluations, raw_history, audit = controller.run()
        elif method == "SGDLCO":
            controller = SGDLCOController(benchmark, local_config, int(config.seed))
            controller.initial_size = min(max(3, local_config.initial_evals), local_config.total_evals)
            best_vector, _, evaluations, raw_history, audit = controller.run()
        elif method == "BGP-SAEA":
            controller = RecentPaperController(benchmark, local_config, "bgp_saea", int(config.seed))
            rng = np.random.default_rng(int(config.seed))
            while controller.evaluations < local_config.total_evals:
                remaining = local_config.total_evals - controller.evaluations
                count = min(
                    local_config.initial_evals if controller.evaluations == 0 else local_config.population_size,
                    remaining,
                )
                points = (
                    benchmark.initial_design[:count].copy()
                    if controller.evaluations == 0
                    else rng.uniform(benchmark.bounds[:, 0], benchmark.bounds[:, 1], size=(count, benchmark.dimension))
                )
                controller.evaluate(torch.as_tensor(points, dtype=torch.float64))
            best_index = int(np.argmin(controller.archive_y))
            best_vector = np.asarray(controller.archive_x[best_index], dtype=float)
            evaluations = int(controller.evaluations)
            raw_history = list(controller.history)
            audit = controller.audit()
        else:
            raise ValueError(f"unsupported underwater comparison method: {method}")
        history = _history_rows(raw_history, int(evaluations), float(np.min(controller.archive_y)))
    value, metrics, path = objective.evaluate_one(best_vector)
    replay = objective.twin.replay(best_vector, engine="mujoco_dynamic")
    evaluations = int(result["real_evals"]) if method in {"TSDDEO", "ESA"} else int(evaluations)
    if not history:
        history = [{"evaluations": config.initial_evals, "best_f": value}, {"evaluations": evaluations, "best_f": value}]
    return {
        "algorithm": method,
        "problem": benchmark.problem_id,
        "scenario": config.scenario,
        "objective_version": "underwater_v9_trackability_proxy_dynamic_replay_warmstart",
        "initial_design_hash": _design_hash(benchmark.initial_design),
        "dimension": objective.dimension,
        "robot_dof": config.robot_dof,
        "intermediate_points": config.intermediate_points,
        "seed": int(config.seed),
        "pop_size": int(config.population_size),
        "population_size": int(config.population_size),
        "active_population_size": int(config.population_size),
        "initial_design_evals": int(config.initial_evals),
        # The ported surrogate-assisted baselines perform one exact infill
        # evaluation per update.  Record that granularity explicitly instead
        # of implying that they use the LLM method's 30-candidate batch.
        "update_batch_evals": 1,
        "update_protocol": "single-candidate-surrogate-infill",
        "requested_evals": config.total_evals,
        "initial_evals": config.initial_evals,
        "real_evals": evaluations,
        "generations_run": len(history) - 1,
        "stop_reason": "evaluation_budget",
        "initial_best_f": float(history[0]["best_f"]),
        "best_f": float(value),
        "wall_time_s": float(time.perf_counter() - started),
        "best_f_history": history,
        "path": path,
        "metrics": metrics,
        "mujoco_replay": {key: value for key, value in replay.items() if key != "path"},
        "dynamic_feasible": bool(
            int(replay.get("mujoco_collision_count", 0)) == 0
            and float(replay.get("mujoco_terminal_error", float("inf"))) <= 0.75
        ),
        "audit": dict(audit),
        "status": "ok",
        "error": "",
    }, np.asarray(best_vector, dtype=float)


def _write_path(path: Path, vector: np.ndarray, trajectory: np.ndarray, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vector=np.asarray(vector, dtype=float), path=np.asarray(trajectory, dtype=float), **{
        key: value for key, value in metrics.items() if np.isscalar(value)
    })


def _scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if key in {"best_f_history", "path", "metrics", "mujoco_replay"}:
            continue
        if isinstance(value, (list, tuple, dict)):
            result[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            result[key] = value
    metrics = row.get("metrics", {})
    for key in (
        "path_length",
        "travel_time",
        "energy_cost",
        "drift_error",
        "smoothness_cost",
        "collision_count",
        "minimum_clearance",
        "terminal_error",
        "feasible",
        "dynamic_proxy",
        "dynamic_penalty",
    ):
        if key in metrics:
            result[key] = metrics[key]
    replay = row.get("mujoco_replay", {})
    for key in (
        "mujoco_collision_count",
        "mujoco_collision_fraction",
        "mujoco_first_contact_time",
        "mujoco_terminal_error",
        "mujoco_energy_cost",
    ):
        if key in replay:
            result[key] = replay[key]
    if "mujoco_replay_mode" in replay:
        result["mujoco_replay_mode"] = replay["mujoco_replay_mode"]
    if "dynamic_feasible" in row:
        result["dynamic_feasible"] = row["dynamic_feasible"]
    return result


def _progress_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep structured histories so resume can rebuild workbooks safely."""

    payload = _scalar_row(row)
    payload["best_f_history"] = list(row.get("best_f_history", []))
    if "path" in row:
        payload["path"] = np.asarray(row["path"], dtype=float).tolist()
    for name in ("metrics", "mujoco_replay"):
        value = row.get(name, {})
        payload[name] = {
            key: item
            for key, item in dict(value).items()
            if np.isscalar(item) or isinstance(item, (str, bool))
        }
    return payload


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down correction for a family of pairwise tests."""

    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = float(p_values[int(index)]) * (len(p_values) - rank)
        running = max(running, value)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def _paired_effect(sample: np.ndarray, reference: np.ndarray) -> tuple[float, int, int, int]:
    """Return mean paired difference and win/tie/loss counts."""

    count = min(int(sample.size), int(reference.size))
    if count == 0:
        return float("nan"), 0, 0, 0
    difference = np.asarray(sample[:count], dtype=float) - np.asarray(reference[:count], dtype=float)
    wins = int(np.sum(difference < -1e-12))
    losses = int(np.sum(difference > 1e-12))
    ties = int(count - wins - losses)
    return float(np.mean(difference)), wins, ties, losses


def _row_metric(row: dict[str, Any], name: str, default: Any = None) -> Any:
    """Read a scalar metric from either a flattened or structured result row."""

    if name in row:
        return row[name]
    for container_name in ("metrics", "mujoco_replay"):
        container = row.get(container_name, {})
        if isinstance(container, dict) and name in container:
            return container[name]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def write_underwater_statistics(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Write reproducible paired tests and feasibility audits for the suite.

    The proposed method is the reference.  Every other method is compared on
    the same seed within each scenario using a paired Wilcoxon test, followed
    by Holm correction across the five competitors in that scenario.
    """

    from scipy.stats import wilcoxon

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_name = "LLMOperatorDE"

    # A partial LLM/API run is an execution audit item, not a valid algorithm
    # observation.  Keep those rows in a separate file and use only completed
    # rows for paired inference.  This prevents a timeout or malformed model
    # response from silently changing the ranking or the Wilcoxon sample size.
    all_rows = list(rows)
    invalid_rows = [
        row for row in all_rows
        if str(row.get("status", "ok")).strip().lower() != "ok"
    ]
    valid_rows = [
        row for row in all_rows
        if str(row.get("status", "ok")).strip().lower() == "ok"
    ]
    algorithms = sorted({str(row["algorithm"]) for row in valid_rows})
    records: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in valid_rows}):
        grouped = {
            algorithm: [
                row for row in valid_rows
                if str(row["scenario"]) == scenario and str(row["algorithm"]) == algorithm
            ]
            for algorithm in algorithms
        }
        reference_rows = {int(row["seed"]): row for row in grouped.get(reference_name, [])}
        p_rows: list[tuple[int, dict[str, Any]]] = []
        for algorithm in algorithms:
            if algorithm == reference_name:
                continue
            paired = [
                (reference_rows[int(row["seed"])], row)
                for row in grouped.get(algorithm, [])
                if int(row["seed"]) in reference_rows
            ]
            sample = np.asarray([float(pair[1]["best_f"]) for pair in paired], dtype=float)
            reference = np.asarray([float(pair[0]["best_f"]) for pair in paired], dtype=float)
            if sample.size >= 2 and not np.allclose(sample, reference):
                try:
                    p_value = float(wilcoxon(sample, reference, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
            else:
                p_value = 1.0
            difference, wins, ties, losses = _paired_effect(sample, reference)
            algorithm_feasible = [
                _as_bool(_row_metric(pair[1], "feasible", False)) for pair in paired
            ]
            reference_feasible = [
                _as_bool(_row_metric(pair[0], "feasible", False)) for pair in paired
            ]
            algorithm_collision = [
                float(_row_metric(pair[1], "mujoco_collision_count", 0.0)) > 0
                for pair in paired
            ]
            reference_collision = [
                float(_row_metric(pair[0], "mujoco_collision_count", 0.0)) > 0
                for pair in paired
            ]
            p_rows.append((len(records), {
                "scenario": scenario,
                "reference": reference_name,
                "algorithm": algorithm,
                "n": int(sample.size),
                "algorithm_mean": float(np.mean(sample)) if sample.size else float("nan"),
                "algorithm_std": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
                "algorithm_median": float(np.median(sample)) if sample.size else float("nan"),
                "reference_mean": float(np.mean(reference)) if reference.size else float("nan"),
                "reference_median": float(np.median(reference)) if reference.size else float("nan"),
                "paired_mean_difference": difference,
                "paired_median_difference": (
                    float(np.median(sample - reference)) if sample.size else float("nan")
                ),
                # This is a paired sign effect, not the unpaired Cliff's
                # delta. Lower objective values are better, so positive
                # values mean that the compared algorithm wins more seeds.
                "paired_sign_effect": (
                    float((wins - losses) / sample.size) if sample.size else float("nan")
                ),
                "wilcoxon_p_raw": p_value,
                "holm_p": 1.0,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "feasible_rate": float(np.mean(algorithm_feasible)) if paired else float("nan"),
                "reference_feasible_rate": float(np.mean(reference_feasible)) if paired else float("nan"),
                "mujoco_collision_rate": float(np.mean(algorithm_collision)) if paired else float("nan"),
                "reference_mujoco_collision_rate": float(np.mean(reference_collision)) if paired else float("nan"),
            }))
        local_p = [record[1]["wilcoxon_p_raw"] for record in p_rows]
        adjusted = _holm_adjust(local_p)
        for (_, record), value in zip(p_rows, adjusted):
            record["holm_p"] = float(value)
            records.append(record)

    csv_path = output / "underwater_statistical_tests.csv"
    fields = [
        "scenario", "reference", "algorithm", "n", "algorithm_mean", "algorithm_std",
        "algorithm_median", "reference_mean", "reference_median",
        "paired_mean_difference", "paired_median_difference", "paired_sign_effect",
        "wilcoxon_p_raw", "holm_p", "wins", "ties", "losses",
        "feasible_rate", "reference_feasible_rate", "mujoco_collision_rate",
        "reference_mujoco_collision_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    json_path = output / "underwater_statistical_tests.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    invalid_csv_path = output / "underwater_invalid_runs.csv"
    invalid_json_path = output / "underwater_invalid_runs.json"
    invalid_fields = sorted({str(key) for row in invalid_rows for key in row})
    if invalid_fields:
        with invalid_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=invalid_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(invalid_rows)
    else:
        with invalid_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status", "reason"])
            writer.writeheader()
    invalid_json_path.write_text(
        json.dumps(invalid_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # API accounting is a property of the proposed method, not a pairwise
    # performance statistic.  Keep it in a separate audit table so a zero in
    # a baseline row cannot be misread as an API failure rate.
    api_records: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in all_rows}):
        llm_rows = [
            row for row in all_rows
            if str(row["scenario"]) == scenario and str(row["algorithm"]) == reference_name
        ]
        valid_llm_rows = [
            row for row in llm_rows
            if str(row.get("status", "ok")).strip().lower() == "ok"
        ]
        attempts = sum(int(row.get("api_calls", 0) or 0) for row in llm_rows)
        successes = sum(int(row.get("api_successes", 0) or 0) for row in llm_rows)
        failures = sum(int(row.get("api_failures", 0) or 0) for row in llm_rows)
        api_records.append(
            {
                "scenario": scenario,
                "runs": len(llm_rows),
                "valid_runs": len(valid_llm_rows),
                "invalid_runs": len(llm_rows) - len(valid_llm_rows),
                "api_calls": attempts,
                "api_successes": successes,
                "api_failures": failures,
                "success_rate": float(successes / attempts) if attempts else float("nan"),
                "prompt_tokens": sum(int(row.get("api_prompt_tokens", 0) or 0) for row in llm_rows),
                "completion_tokens": sum(int(row.get("api_completion_tokens", 0) or 0) for row in llm_rows),
                "total_tokens": sum(int(row.get("api_total_tokens", 0) or 0) for row in llm_rows),
            }
        )
    api_csv_path = output / "underwater_llm_api_summary.csv"
    api_fields = list(api_records[0]) if api_records else [
        "scenario", "runs", "valid_runs", "invalid_runs", "api_calls", "api_successes", "api_failures",
        "success_rate", "prompt_tokens", "completion_tokens", "total_tokens",
    ]
    with api_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=api_fields)
        writer.writeheader()
        writer.writerows(api_records)
    api_json_path = output / "underwater_llm_api_summary.json"
    api_json_path.write_text(json.dumps(api_records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "csv": csv_path,
        "json": json_path,
        "invalid_csv": invalid_csv_path,
        "invalid_json": invalid_json_path,
        "api_csv": api_csv_path,
        "api_json": api_json_path,
    }


def run_comparison_suite(config: UnderwaterExperimentConfig = CONFIG) -> list[dict[str, Any]]:
    config.validate()
    root = Path(config.output_root) / "underwater_comparison"
    root.mkdir(parents=True, exist_ok=True)
    progress_log = root / "underwater_comparison_results.jsonl"
    rows: list[dict[str, Any]] = []
    if progress_log.exists():
        for line in progress_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    completed = {(str(row.get("scenario")), str(row.get("algorithm")), int(row.get("seed", -1))) for row in rows}
    for scenario in config.scenarios:
        for method in config.algorithms:
            for seed in config.seeds:
                key = (scenario, method, int(seed))
                if key in completed:
                    print(f"{scenario}/{method}/seed={seed:02d} already completed; resume skip", flush=True)
                    continue
                run_config = replace(config, scenario=scenario, seed=seed, output_root=root / scenario / method / f"seed_{seed:02d}")
                print(
                    f"{scenario:30s} {method:16s} D={config.trajectory_dimension:<3d} "
                    f"seed={seed:02d}/{config.seeds[-1]:02d} FE={config.total_evals}",
                    flush=True,
                )
                row, vector = _run_llm(run_config) if method == "LLMOperatorDE" else _run_baseline(method, run_config)
                path_file = Path(run_config.output_dir) / "best_trajectory.npz"
                _write_path(path_file, np.asarray(vector), np.asarray(row["path"]), dict(row.get("metrics", {})))
                row["trajectory_file"] = str(path_file)
                row["algorithm"] = method
                rows.append(row)
                completed.add(key)
                with progress_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_progress_row(row), ensure_ascii=False, default=str) + "\n")
                    handle.flush()

    (root / "raw_underwater_comparison_results.json").write_text(
        json.dumps([_progress_row(row) for row in rows], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    scalar_rows = [_scalar_row(row) for row in rows]
    fields = sorted({key for row in scalar_rows for key in row})
    with (root / "raw_underwater_comparison_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)
    write_suite_workbooks(rows, root, prefix="underwater_comparison")
    write_underwater_statistics(rows, root)
    write_protocol_manifest(config, rows, root)
    for scenario in config.scenarios:
        scenario_root = root / scenario
        if scenario_root.exists():
            plot_3d_paths(scenario_root, scenario, scenario_root / "underwater_paths_3d.png")
    print(f"saved underwater comparison outputs: {root}", flush=True)
    return rows


__all__ = [
    "UnderwaterBenchmark",
    "protocol_manifest",
    "run_comparison_suite",
    "write_protocol_manifest",
]
