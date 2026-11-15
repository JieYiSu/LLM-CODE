"""Fair single-objective expensive-optimization comparison suite.

The suite uses one true-evaluation budget for every method, including the
ported TSDDEO/ESA controllers and the three Python implementations referenced
by ``paper_surrogate_algorithms.py`` (SADE-SS, SGDLCO and BGP-SAEA).  It writes
the mean/std and convergence workbooks in the same layout family as the
provided Z1/Z2 MATLAB scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "Ackley",
    "Griewank",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "Ellipsoid",
)
DEFAULT_ALGORITHMS = (
    "LLMOperatorDE",
    "TSDDEO",
    "ESA",
    "SADE-SS",
    "SGDLCO",
    "BGP-SAEA",
)
BOUNDS = {
    "Ackley": (-32.768, 32.768),
    "Griewank": (-600.0, 600.0),
    "Rastrigin": (-5.12, 5.12),
    "Rosenbrock": (-5.0, 10.0),
    "Ellipsoid": (-5.12, 5.12),
    "Schwefel": (-500.0, 500.0),
    "Sphere": (-5.12, 5.12),
}

GLOBAL_OPTIMUM = {
    "Ackley": {"point": "[0, ..., 0]", "value": 0.0},
    "Griewank": {"point": "[0, ..., 0]", "value": 0.0},
    "Rastrigin": {"point": "[0, ..., 0]", "value": 0.0},
    "Rosenbrock": {"point": "[1, ..., 1]", "value": 0.0},
    "Schwefel": {"point": "[420.9687, ..., 420.9687]", "value": 0.0},
    "Sphere": {"point": "[0, ..., 0]", "value": 0.0},
    "Ellipsoid": {"point": "[0, ..., 0]", "value": 0.0},
}


@dataclass
class SuiteBenchmark:
    problem_id: str
    dimension: int
    problem_object: Any
    bounds: np.ndarray
    optimal_point: str
    optimal_value: float = 0.0

    @property
    def problem(self) -> Any:
        return self.problem_object

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        values = self.problem_object.evaluate(
            torch.as_tensor(np.asarray(points, dtype=np.float64), dtype=torch.float64)
        )
        return values.detach().cpu().numpy().reshape(-1)


def get_suite_benchmark(problem: str, dimension: int) -> SuiteBenchmark:
    from evox.problems.numerical import (
        Ackley,
        Ellipsoid,
        Griewank,
        Rastrigin,
        Rosenbrock,
        Schwefel,
        Sphere,
    )

    classes = {
        "Ackley": Ackley,
        "Griewank": Griewank,
        "Rastrigin": Rastrigin,
        "Rosenbrock": Rosenbrock,
        "Ellipsoid": Ellipsoid,
        "Schwefel": Schwefel,
        "Sphere": Sphere,
    }
    if problem not in classes:
        raise ValueError(f"unsupported suite problem: {problem}")
    low, high = BOUNDS[problem]
    problem_object = classes[problem]()
    return SuiteBenchmark(
        problem_id=problem,
        dimension=int(dimension),
        problem_object=problem_object,
        bounds=np.tile(np.asarray([[low, high]], dtype=float), (int(dimension), 1)),
        optimal_point=GLOBAL_OPTIMUM[problem]["point"],
        optimal_value=float(GLOBAL_OPTIMUM[problem]["value"]),
    )


@dataclass
class SuiteConfig:
    total_evals: int = 300
    initial_evals: int = 100
    population_size: int = 30
    repeats: int = 20
    dimensions: tuple[int, ...] = (30, 50, 100, 200, 500, 1000)
    problems: tuple[str, ...] = PROBLEMS
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    convergence_points: int = 25
    device: str = "cuda"
    output_dir: Path = field(
        default_factory=lambda: ROOT / "experiments" / "expensive_suite_results"
    )
    llm_config_path: Path = field(default_factory=lambda: ROOT / "config.yaml")
    force_warmup_llm: bool = True
    resume: bool = True
    checkpoint_interval: int = 1

    def validate(self) -> None:
        if self.total_evals < self.initial_evals:
            raise ValueError("total_evals must be at least initial_evals")
        if self.initial_evals < self.population_size:
            raise ValueError("initial_evals must be at least population_size")
        if self.repeats < 1 or self.convergence_points < 2:
            raise ValueError("repeats and convergence_points must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if any(problem not in PROBLEMS for problem in self.problems):
            raise ValueError("problems must be selected from the suite problem set")


SUITE_PROGRESS_VERSION = 1


def _suite_progress_path(config: SuiteConfig, mode: str) -> Path:
    filename = "suite_progress.json" if mode == "main" else "ablation_progress.json"
    return Path(config.output_dir) / filename


def _suite_result_log_path(config: SuiteConfig, mode: str) -> Path:
    filename = "suite_results.jsonl" if mode == "main" else "ablation_results.jsonl"
    return Path(config.output_dir) / filename


def _suite_protocol(config: SuiteConfig, mode: str) -> dict[str, Any]:
    algorithms = (
        tuple(config.algorithms)
        if mode == "main"
        else ("LLMOperatorDE-Full", "LLMOperatorDE-NoLLM", "LLMOperatorDE-NoSharedMating", "LLMOperatorDE-NoAdaptiveFCR")
    )
    return {
        "mode": mode,
        "total_evals": int(config.total_evals),
        "initial_evals": int(config.initial_evals),
        "population_size": int(config.population_size),
        "repeats": int(config.repeats),
        "dimensions": [int(value) for value in config.dimensions],
        "problems": [str(value) for value in config.problems],
        "algorithms": [str(value) for value in algorithms],
        "device": str(config.device),
        "convergence_points": int(config.convergence_points),
        "llm_config_path": str(Path(config.llm_config_path).resolve()),
        "force_warmup_llm": bool(config.force_warmup_llm),
    }


def _suite_protocol_signature(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(protocol), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _suite_run_key(algorithm: str, problem: str, dimension: int, seed: int) -> str:
    return f"{algorithm}|{problem}|{int(dimension)}|{int(seed)}"


def _append_suite_result(config: SuiteConfig, mode: str, row: Mapping[str, Any]) -> None:
    """Append one completed run without rewriting the whole progress file."""

    path = _suite_result_log_path(config, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist human-readable progress without exposing a half-written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Windows sync/indexing tools can hold the destination briefly.  A
        # direct replacement keeps the latest completed run recoverable.
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)


def _load_suite_progress(
    config: SuiteConfig,
    mode: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Load completed run rows after checking that the protocol is unchanged."""

    path = _suite_progress_path(config, mode)
    if not config.resume:
        path.unlink(missing_ok=True)
        _suite_result_log_path(config, mode).unlink(missing_ok=True)
        return [], False
    if not path.exists():
        return [], False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read suite progress file: {path}") from error
    protocol = _suite_protocol(config, mode)
    expected_signature = _suite_protocol_signature(protocol)
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("version", 0)) != SUITE_PROGRESS_VERSION
        or str(payload.get("signature", "")) != expected_signature
    ):
        raise ValueError(
            "suite progress parameters do not match this run; use a new output_dir "
            "or set resume=False for a fresh experiment"
        )
    raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = []
        log_path = _suite_result_log_path(config, mode)
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            # A process killed during an append can leave one
                            # incomplete final line. Earlier lines remain valid.
                            continue
                        if isinstance(item, Mapping):
                            raw_rows.append(item)
            except OSError as error:
                raise ValueError(f"cannot read suite result log: {log_path}") from error
    if not isinstance(raw_rows, list):
        raise ValueError(f"malformed suite progress rows: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        required = ("algorithm", "problem", "dimension", "seed")
        if any(field not in raw_row for field in required):
            continue
        row = dict(raw_row)
        key = _suite_run_key(
            str(row["algorithm"]),
            str(row["problem"]),
            int(row["dimension"]),
            int(row["seed"]),
        )
        if key not in seen:
            rows.append(row)
            seen.add(key)
    return rows, True


def _save_suite_progress(
    config: SuiteConfig,
    mode: str,
    rows: list[Mapping[str, Any]],
    status: str,
) -> None:
    protocol = _suite_protocol(config, mode)
    payload = {
        "version": SUITE_PROGRESS_VERSION,
        "signature": _suite_protocol_signature(protocol),
        "protocol": protocol,
        "status": status,
        "completed_runs": len(rows),
        "result_log": _suite_result_log_path(config, mode).name,
        "completed_keys": [
            _suite_run_key(
                str(row["algorithm"]),
                str(row["problem"]),
                int(row["dimension"]),
                int(row["seed"]),
            )
            for row in rows
            if all(field in row for field in ("algorithm", "problem", "dimension", "seed"))
        ],
    }
    _atomic_write_json(_suite_progress_path(config, mode), payload)


def _lhs(count: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = int(count)
    dimension = bounds.shape[0]
    if count <= 0:
        return np.empty((0, dimension), dtype=float)
    unit = np.empty((count, dimension), dtype=float)
    for column in range(dimension):
        unit[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def _shared_initial_design(count: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    """Build the deterministic design shared by all methods in one run."""

    return _lhs(int(count), np.asarray(bounds, dtype=float), np.random.default_rng(int(seed)))


class _RBF:
    def __init__(self, bounds: np.ndarray, ridge: float = 1e-8) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.ridge = float(ridge)
        self.x = np.empty((0, self.bounds.shape[0]))
        self.y = np.empty(0)
        self.coefficients = None
        self.length = 1.0

    def _normalise(self, points: np.ndarray) -> np.ndarray:
        span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        return (np.asarray(points, dtype=float) - self.bounds[:, 0]) / span

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        self.x = np.asarray(points, dtype=float)
        self.y = np.asarray(values, dtype=float).reshape(-1)
        z = self._normalise(self.x)
        if len(z) < 2:
            self.coefficients = None
            return
        distance = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        nonzero = distance[distance > 0]
        self.length = float(np.median(nonzero)) if nonzero.size else 0.25
        self.length = max(self.length, 1e-4)
        kernel = np.exp(-0.5 * (distance / self.length) ** 2)
        kernel.flat[:: len(kernel) + 1] += self.ridge
        try:
            self.coefficients = np.linalg.solve(kernel, self.y)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(kernel, self.y, rcond=None)[0]

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.coefficients is None or len(self.x) == 0:
            return np.full(len(points), np.inf)
        z = self._normalise(points)
        train = self._normalise(self.x)
        distance = np.linalg.norm(z[:, None, :] - train[None, :, :], axis=2)
        return np.exp(-0.5 * (distance / self.length) ** 2) @ self.coefficients


class _PortedController:
    """Shared exact-evaluation archive for the MATLAB-inspired Python ports."""

    def __init__(self, benchmark: SuiteBenchmark, config: SuiteConfig, seed: int) -> None:
        self.benchmark = benchmark
        self.config = config
        self.rng = np.random.default_rng(int(seed))
        self.bounds = benchmark.bounds
        self.total_evals = int(config.total_evals)
        self.archive_x: list[np.ndarray] = []
        self.archive_y: list[float] = []
        self.history: list[dict[str, Any]] = []

    @property
    def evaluations(self) -> int:
        return len(self.archive_y)

    def _initial_design(self) -> np.ndarray:
        """Use a benchmark-provided common design when a study supplies one."""

        common = getattr(self.benchmark, "initial_design", None)
        if common is not None:
            return np.asarray(common, dtype=float).copy()
        return _lhs(self.config.initial_evals, self.bounds, self.rng)

    def evaluate(self, points: np.ndarray, phase: str) -> None:
        remaining = self.total_evals - self.evaluations
        points = np.asarray(points, dtype=float).reshape(-1, self.benchmark.dimension)
        points = np.clip(points, self.bounds[:, 0], self.bounds[:, 1])[:remaining]
        if len(points) == 0:
            return
        values = self.benchmark.evaluate(points)
        for point, value in zip(points, values):
            self.archive_x.append(point.copy())
            self.archive_y.append(float(value))
            self.history.append(
                {
                    "evaluations": int(self.evaluations),
                    "best_f": float(np.min(self.archive_y)),
                    "phase": phase,
                }
            )

    def _model(self, limit: int | None = None) -> _RBF:
        order = np.argsort(np.asarray(self.archive_y, dtype=float))
        if limit is not None:
            order = order[: min(len(order), int(limit))]
        model = _RBF(self.bounds)
        model.fit(np.asarray(self.archive_x)[order], np.asarray(self.archive_y)[order])
        return model

    def _surrogate_pso(self, model: _RBF, center: np.ndarray | None = None) -> np.ndarray:
        dimension = self.benchmark.dimension
        swarm_size = min(32, max(8, self.config.population_size))
        if center is None:
            swarm = _lhs(swarm_size, self.bounds, self.rng)
        else:
            span = self.bounds[:, 1] - self.bounds[:, 0]
            swarm = center + self.rng.normal(0, 0.2, size=(swarm_size, dimension)) * span
            swarm = np.clip(swarm, self.bounds[:, 0], self.bounds[:, 1])
        velocity = np.zeros_like(swarm)
        best = swarm.copy()
        best_value = model.predict(best)
        global_best = best[int(np.argmin(best_value))].copy()
        for _ in range(12):
            r1 = self.rng.random(swarm.shape)
            r2 = self.rng.random(swarm.shape)
            velocity = 0.72 * velocity + 1.6 * r1 * (best - swarm) + 1.6 * r2 * (global_best - swarm)
            swarm = np.clip(swarm + velocity, self.bounds[:, 0], self.bounds[:, 1])
            prediction = model.predict(swarm)
            improved = prediction < best_value
            best[improved] = swarm[improved]
            best_value[improved] = prediction[improved]
            global_best = best[int(np.argmin(best_value))].copy()
        return global_best

    def result(self, method: str, initial_best: float, audit: Mapping[str, Any]) -> dict[str, Any]:
        best_index = int(np.argmin(self.archive_y))
        return {
            "algorithm": method,
            "problem": self.benchmark.problem_id,
            "dimension": self.benchmark.dimension,
            "seed": int(self.rng.bit_generator._seed_seq.entropy),
            "best_f": float(self.archive_y[best_index]),
            "initial_best_f": float(initial_best),
            "real_evals": int(self.evaluations),
            "initial_evals": int(self.config.initial_evals),
            "best_f_history": list(self.history),
            "audit": dict(audit),
        }


class TSDDEOPort(_PortedController):
    """Python port of the two-stage TSDDEO workflow.

    Stage 1 performs RBF-guided SLPSO-like search; stage 2 switches to DE with
    surrogate ranking and exact infill. The objective callback is the only
    source of true evaluations and is capped by ``total_evals``.
    """

    def run(self) -> dict[str, Any]:
        initial = self._initial_design()
        self.evaluate(initial, "initial_lhs")
        initial_best = float(np.min(self.archive_y))
        population_size = min(self.config.population_size, len(self.archive_x))
        while self.evaluations < self.total_evals:
            archive = np.asarray(self.archive_x)
            values = np.asarray(self.archive_y)
            if self.evaluations < int(0.30 * self.total_evals):
                model = self._model(limit=min(len(values), max(16, 2 * self.benchmark.dimension)))
                candidate = self._surrogate_pso(model)
                phase = "stage1_rbf_slpso"
            else:
                order = np.argsort(values)[:population_size]
                population = archive[order]
                best = population[0]
                candidates = []
                for _ in range(max(8, 4 * population_size)):
                    indices = self.rng.choice(len(population), 3, replace=len(population) < 3)
                    mutant = population[indices[0]] + 0.5 * (population[indices[1]] - population[indices[2]])
                    mask = self.rng.random(self.benchmark.dimension) < 0.9
                    mask[self.rng.integers(0, self.benchmark.dimension)] = True
                    candidates.append(np.where(mask, mutant, best))
                candidates = np.clip(np.asarray(candidates), self.bounds[:, 0], self.bounds[:, 1])
                model = self._model(limit=min(len(values), max(32, 2 * self.benchmark.dimension)))
                candidate = candidates[int(np.argmin(model.predict(candidates)))]
                phase = "stage2_de_surrogate_infill"
            self.evaluate(candidate.reshape(1, -1), phase)
        return self.result(
            "TSDDEO",
            initial_best,
            {
                "implementation": "python_port_tsddeo_two_stage_rbf_slpso_de",
                "source_matlab": str(ROOT.parents[4] / "Desktop" / "TSDDEO" / "TSDDEO_core.m"),
                "initial_evals": self.config.initial_evals,
                "stage_switch": 0.30,
                "true_evaluation_budget": self.total_evals,
            },
        )


class ESAPort(_PortedController):
    """Python port of ESA's four-action surrogate-assisted workflow."""

    def run(self) -> dict[str, Any]:
        initial = self._initial_design()
        self.evaluate(initial, "initial_lhs")
        initial_best = float(np.min(self.archive_y))
        q = np.full((8, 4), 0.25, dtype=float)
        state = 0
        while self.evaluations < self.total_evals:
            probabilities = np.exp(q[state] - np.max(q[state]))
            probabilities /= probabilities.sum()
            action = int(self.rng.choice(4, p=probabilities))
            old_best = float(np.min(self.archive_y))
            model = self._model(limit=min(len(self.archive_y), max(16, 2 * self.benchmark.dimension)))
            archive = np.asarray(self.archive_x)
            values = np.asarray(self.archive_y)
            if action == 0:
                population = archive[np.argsort(values)[: min(self.config.population_size, len(archive))]]
                candidates = []
                for _ in range(32):
                    ids = self.rng.choice(len(population), 3, replace=len(population) < 3)
                    candidates.append(population[ids[0]] + 0.5 * (population[ids[1]] - population[ids[2]]))
                candidate = np.asarray(candidates)[int(np.argmin(model.predict(np.asarray(candidates))))]
                phase = "action1_global_surrogate_screen"
            elif action == 1:
                center = archive[int(np.argmin(values))]
                candidate = self._surrogate_pso(model, center=center)
                phase = "action2_local_surrogate_search"
            elif action == 2:
                top = archive[np.argsort(values)[: min(100, len(archive))]]
                candidate = top[0].copy()
                for coordinate in self.rng.permutation(self.benchmark.dimension):
                    trial = candidate.copy()
                    trial[coordinate] = top[self.rng.integers(len(top)), coordinate]
                    if model.predict(trial.reshape(1, -1))[0] < model.predict(candidate.reshape(1, -1))[0]:
                        candidate = trial
                phase = "action3_full_crossover"
            else:
                center = archive[int(np.argmin(values))]
                span = (self.bounds[:, 1] - self.bounds[:, 0]) * max(0.05, 1.0 - self.evaluations / self.total_evals)
                pool = center + self.rng.normal(0, 0.15, size=(64, self.benchmark.dimension)) * span
                pool = np.clip(pool, self.bounds[:, 0], self.bounds[:, 1])
                candidate = pool[int(np.argmin(model.predict(pool)))]
                phase = "action4_str_trust_region"
            self.evaluate(np.asarray(candidate).reshape(1, -1), phase)
            reward = 1.0 if float(np.min(self.archive_y)) < old_best else 0.0
            next_state = min(7, 2 * action + int(reward))
            q[state, action] = 0.9 * q[state, action] + 0.1 * (reward + 0.9 * np.max(q[next_state]))
            state = next_state
        return self.result(
            "ESA",
            initial_best,
            {
                "implementation": "python_port_esa_four_action_surrogate_agent",
                "source_matlab": str(ROOT.parents[4] / "Desktop" / "ESA" / "ESA_core.m"),
                "actions": [
                    "global_surrogate_screen",
                    "local_surrogate_search",
                    "full_crossover",
                    "str_trust_region",
                ],
                "true_evaluation_budget": self.total_evals,
            },
        )


def run_paper_algorithm(
    method: str,
    problem: str,
    dimension: int,
    seed: int,
    config: SuiteConfig,
) -> dict[str, Any]:
    from .expensive_algorithms.paper_surrogates import (
        run_paper_algorithm as local_runner,
    )

    return local_runner(method, problem, dimension, seed, config)


def run_ported_algorithm(
    method: str,
    problem: str,
    dimension: int,
    seed: int,
    config: SuiteConfig,
) -> dict[str, Any]:
    benchmark = get_suite_benchmark(problem, dimension)
    benchmark.initial_design = _shared_initial_design(config.initial_evals, benchmark.bounds, seed)
    if method == "TSDDEO":
        from .expensive_algorithms.tsddeo import TSDDEOPort

        return TSDDEOPort(benchmark, config, seed).run()
    if method == "ESA":
        from .expensive_algorithms.esa import ESAPort

        return ESAPort(benchmark, config, seed).run()
    raise ValueError(f"unsupported ported algorithm: {method}")


def run_llm_algorithm(
    problem: str,
    dimension: int,
    seed: int,
    config: SuiteConfig,
    *,
    use_llm: bool = True,
    force_warmup_llm: bool | None = None,
    adaptive_parameters: bool = True,
    cross_island_rate: float = 0.35,
    global_guidance: float = 0.25,
    knowledge_path: Path | None = None,
) -> dict[str, Any]:
    """Run the current local EvoX LLMOperatorDE under the suite budget."""

    from .llm_suite_runner import run_local_llm_operator_de

    events: list[dict[str, Any]] = []
    result = run_local_llm_operator_de(
        problem=problem,
        dimension=int(dimension),
        seed=int(seed),
        config=config,
        use_llm=bool(use_llm),
        force_warmup_llm=force_warmup_llm,
        adaptive_parameters=bool(adaptive_parameters),
        cross_island_rate=float(cross_island_rate),
        global_guidance=float(global_guidance),
        island_count=3,
        migration_interval=5,
        candidate_validation_evals=20,
        llm_eval_interval=200,
        rag_top_k=2,
        knowledge_path=knowledge_path,
        progress_callback=events.append,
    )
    return result


def run_suite(config: SuiteConfig | None = None) -> list[dict[str, Any]]:
    """Run the requested methods/problems/dimensions and persist raw JSON."""

    config = config or SuiteConfig()
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, resumed = _load_suite_progress(config, "main")
    if not resumed:
        _save_suite_progress(config, "main", rows, "running")
    completed = {
        _suite_run_key(
            str(row["algorithm"]),
            str(row["problem"]),
            int(row["dimension"]),
            int(row["seed"]),
        )
        for row in rows
    }
    total_runs = len(config.problems) * len(config.dimensions) * len(config.algorithms) * config.repeats
    if resumed and rows:
        print(
            f"[suite] resumed {len(rows)}/{total_runs} completed runs from "
            f"{_suite_progress_path(config, 'main')}",
            flush=True,
        )
    try:
        for problem in config.problems:
            for dimension in config.dimensions:
                # Finish every repeat of one method before moving to the next
                # method.  Completed keys are skipped after a restart.
                for method in config.algorithms:
                    for repeat in range(config.repeats):
                        seed = repeat + 1
                        key = _suite_run_key(method, problem, dimension, seed)
                        if key in completed:
                            continue
                        if method in {"TSDDEO", "ESA"}:
                            row = run_ported_algorithm(method, problem, dimension, seed, config)
                        elif method in {"SADE-SS", "SGDLCO", "BGP-SAEA"}:
                            from .expensive_algorithms.paper_surrogates import (
                                run_paper_algorithm as separated_paper_runner,
                            )

                            row = separated_paper_runner(method, problem, dimension, seed, config)
                        elif method == "LLMOperatorDE":
                            row = run_llm_algorithm(
                                problem,
                                dimension,
                                seed,
                                config,
                                knowledge_path=(
                                    Path(config.output_dir)
                                    / "operator_knowledge"
                                    / f"{problem}_D{int(dimension)}_seed{int(seed):03d}.json"
                                ),
                            )
                        else:
                            raise ValueError(f"unsupported suite method: {method}")
                        row["repeat"] = repeat + 1
                        rows.append(row)
                        completed.add(key)
                        _append_suite_result(config, "main", row)
                        if len(rows) % int(config.checkpoint_interval) == 0:
                            _save_suite_progress(config, "main", rows, "running")
                        print(
                            f"{method:12s} {problem:10s} D={dimension:<4d} "
                            f"run={repeat + 1:02d}/{config.repeats:02d} "
                            f"suite={len(rows)}/{total_runs} "
                            f"best={float(row['best_f']):.4e} FE={row['real_evals']}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        _save_suite_progress(config, "main", rows, "interrupted")
        print(
            f"\n[suite] interrupted safely; saved {len(rows)}/{total_runs} completed runs. "
            "Rerun the same command to resume.",
            flush=True,
        )
        raise
    _save_suite_progress(config, "main", rows, "completed")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = config.output_dir / "raw_results.json"
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    import csv
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: value for key, value in row.items() if key not in {"best_f_history", "audit"}})
    csv_path = config.output_dir / "raw_results.csv"
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)
    write_suite_workbooks(rows, config.output_dir)
    return rows


ABLATION_VARIANTS = (
    "Full",
    "NoLLM",
    "NoSharedMating",
    "NoAdaptiveFCR",
)


def run_ablation_suite(config: SuiteConfig | None = None) -> list[dict[str, Any]]:
    """Run component ablations on the same five problems and FE protocol."""

    if config is None:
        config = SuiteConfig(
            algorithms=("LLMOperatorDE",),
            output_dir=ROOT / "experiments" / "expensive_suite_ablation",
        )
    else:
        from dataclasses import replace

        config = replace(
            config,
            algorithms=("LLMOperatorDE",),
            output_dir=Path(config.output_dir).parent / "expensive_suite_ablation",
        )
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, resumed = _load_suite_progress(config, "ablation")
    if not resumed:
        _save_suite_progress(config, "ablation", rows, "running")
    completed = {
        _suite_run_key(
            str(row["algorithm"]),
            str(row["problem"]),
            int(row["dimension"]),
            int(row["seed"]),
        )
        for row in rows
    }
    total_runs = len(config.problems) * len(config.dimensions) * len(ABLATION_VARIANTS) * config.repeats
    if resumed and rows:
        print(
            f"[ablation] resumed {len(rows)}/{total_runs} completed runs from "
            f"{_suite_progress_path(config, 'ablation')}",
            flush=True,
        )
    try:
        for problem in config.problems:
            for dimension in config.dimensions:
                for variant in ABLATION_VARIANTS:
                    for repeat in range(config.repeats):
                        seed = repeat + 1
                        algorithm = f"LLMOperatorDE-{variant}"
                        key = _suite_run_key(algorithm, problem, dimension, seed)
                        if key in completed:
                            continue
                        from .expensive_algorithms.llm_operator_de import run_ablation_algorithm

                        row = run_ablation_algorithm(
                            problem,
                            dimension,
                            seed,
                            config,
                            variant,
                            knowledge_path=(
                                Path(config.output_dir)
                                / "operator_knowledge"
                                / f"{problem}_D{int(dimension)}_seed{int(seed):03d}_{variant}.json"
                            ),
                        )
                        row["algorithm"] = algorithm
                        row["repeat"] = repeat + 1
                        rows.append(row)
                        completed.add(key)
                        _append_suite_result(config, "ablation", row)
                        if len(rows) % int(config.checkpoint_interval) == 0:
                            _save_suite_progress(config, "ablation", rows, "running")
                        print(
                            f"{algorithm:24s} {problem:10s} D={dimension:<4d} "
                            f"run={repeat + 1:02d}/{config.repeats:02d} "
                            f"suite={len(rows)}/{total_runs} "
                            f"best={float(row['best_f']):.4e} FE={row['real_evals']}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        _save_suite_progress(config, "ablation", rows, "interrupted")
        print(
            f"\n[ablation] interrupted safely; saved {len(rows)}/{total_runs} completed runs. "
            "Rerun the same command to resume.",
            flush=True,
        )
        raise
    _save_suite_progress(config, "ablation", rows, "completed")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "raw_ablation_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    scalar_rows = [
        {key: value for key, value in row.items() if key not in {"best_f_history", "audit"}}
        for row in rows
    ]
    import csv
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with (config.output_dir / "raw_ablation_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)
    write_suite_workbooks(rows, config.output_dir, prefix="ablation")
    write_ablation_contribution_report(rows, config.output_dir, prefix="ablation")
    return rows


def _resample_history(history: list[Mapping[str, Any]], checkpoints: np.ndarray) -> np.ndarray:
    if not history:
        return np.full(len(checkpoints), np.nan)
    x = np.asarray([int(item["evaluations"]) for item in history], dtype=float)
    y = np.minimum.accumulate(np.asarray([float(item["best_f"]) for item in history]))
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    return np.interp(checkpoints, unique, y, left=y[0], right=y[-1])


def _suite_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["problem"]), str(row["dimension"]), int(row["seed"])


def compute_ablation_contributions(
    rows: list[Mapping[str, Any]],
    reference: str = "LLMOperatorDE-Full",
    eps: float = 1e-300,
) -> list[dict[str, Any]]:
    """Compute robust leave-one-component contribution statistics.

    The loss for a component is measured after removing that component from
    the full method.  We use a base-10 log ratio of problem-dimension means so
    that Schwefel/Rosenbrock scale does not dominate Sphere/Ackley.  Positive
    loss means that removing the component made the result worse.  The share
    is the normalized mean positive loss and is explicitly a relative
    contribution under this ablation protocol, not a causal percentage.
    """
    variants = (
        ("NoAdaptiveFCR", "LLMOperatorDE-NoAdaptiveFCR"),
        ("NoLLM", "LLMOperatorDE-NoLLM"),
        ("NoSharedMating", "LLMOperatorDE-NoSharedMating"),
    )
    grouped: dict[tuple[str, int], dict[str, list[tuple[int, float]]]] = {}
    for row in rows:
        key = (str(row["problem"]), int(row["dimension"]))
        grouped.setdefault(key, {}).setdefault(str(row["algorithm"]), []).append(
            (int(row.get("seed", row.get("repeat", 0))), float(row["best_f"]))
        )

    records: list[dict[str, Any]] = []
    for component, variant in variants:
        log_losses: list[float] = []
        positive_groups = 0
        worse_groups = 0
        better_groups = 0
        tie_groups = 0
        for (problem, dimension), algorithms in sorted(grouped.items()):
            full_pairs = dict(algorithms.get(reference, []))
            variant_pairs = dict(algorithms.get(variant, []))
            common_seeds = sorted(set(full_pairs).intersection(variant_pairs))
            if not common_seeds:
                continue
            full_mean = float(np.mean([full_pairs[s] for s in common_seeds]))
            variant_mean = float(np.mean([variant_pairs[s] for s in common_seeds]))
            log_loss = math.log10(max(variant_mean, eps)) - math.log10(max(full_mean, eps))
            log_losses.append(log_loss)
            positive_groups += int(log_loss > 0.0)
            from scipy.stats import wilcoxon
            import warnings

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p_value = float(
                        wilcoxon(
                            [variant_pairs[s] for s in common_seeds],
                            [full_pairs[s] for s in common_seeds],
                            alternative="two-sided",
                        ).pvalue
                    )
            except ValueError:
                p_value = 1.0
            if not math.isfinite(p_value):
                p_value = 1.0
            if p_value > 0.05:
                tie_groups += 1
            elif variant_mean > full_mean:
                worse_groups += 1
            else:
                better_groups += 1

        if not log_losses:
            continue
        positive_losses = np.maximum(np.asarray(log_losses, dtype=float), 0.0)
        mean_log_loss = float(np.mean(log_losses))
        records.append(
            {
                "component": component,
                "variant": variant,
                "groups": len(log_losses),
                "mean_log_loss": mean_log_loss,
                "median_log_loss": float(np.median(log_losses)),
                "geometric_degradation_pct": float((10.0 ** mean_log_loss - 1.0) * 100.0),
                "positive_loss_group_rate_pct": float(positive_groups / len(log_losses) * 100.0),
                "significant_worse_groups": worse_groups,
                "significant_better_groups": better_groups,
                "no_significant_difference_groups": tie_groups,
                "positive_loss_weight": float(np.mean(positive_losses)),
                "contribution_share_pct": 0.0,
            }
        )
    denominator = sum(float(record["positive_loss_weight"]) for record in records)
    for record in records:
        record["contribution_share_pct"] = (
            float(record["positive_loss_weight"]) / denominator * 100.0
            if denominator > 0.0
            else 0.0
        )
    return records


def write_ablation_contribution_report(
    rows: list[Mapping[str, Any]],
    output_dir: str | Path,
    prefix: str = "ablation",
) -> dict[str, Path]:
    """Write contribution percentages and an embedded bar chart."""
    import csv
    import openpyxl
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Font, PatternFill

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = compute_ablation_contributions(rows)
    csv_path = output / f"{prefix}_contribution.csv"
    fieldnames = list(records[0]) if records else ["component"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    workbook_path = output / f"{prefix}_Contribution.xlsx"
    workbook = openpyxl.Workbook()
    summary = workbook.active
    summary.title = "Contribution_Summary"
    headers = fieldnames
    summary.append(headers)
    for cell in summary[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    for record in records:
        summary.append([record.get(header, "") for header in headers])
    summary.freeze_panes = "A2"
    summary.auto_filter.ref = summary.dimensions
    for column in summary.columns:
        letter = column[0].column_letter
        summary.column_dimensions[letter].width = max(
            16, min(34, max(len(str(cell.value or "")) for cell in column) + 2)
        )

    chart_ws = workbook.create_sheet("Contribution_Chart")
    chart_ws.append(["Component", "Contribution share (%)"])
    for record in records:
        chart_ws.append([record["component"], record["contribution_share_pct"]])
    chart = BarChart()
    chart.type = "bar"
    chart.style = 10
    chart.title = "Relative contribution under leave-one-component ablation"
    chart.y_axis.title = "Component"
    chart.x_axis.title = "Share (%)"
    chart.add_data(Reference(chart_ws, min_col=2, min_row=1, max_row=chart_ws.max_row), titles_from_data=True)
    chart.set_categories(Reference(chart_ws, min_col=1, min_row=2, max_row=chart_ws.max_row))
    chart.height = 7
    chart.width = 14
    chart_ws.add_chart(chart, "D2")
    workbook.save(workbook_path)
    return {"csv": csv_path, "workbook": workbook_path}


def write_suite_workbooks(
    rows: list[Mapping[str, Any]],
    output_dir: str | Path,
    prefix: str = "single_objective_expensive",
) -> dict[str, Path]:
    """Write ErrorStats-like and Convergence_data-like workbooks."""
    import json
    import openpyxl
    from copy import copy
    from openpyxl.styles import Font, PatternFill
    from scipy.stats import rankdata, wilcoxon

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # API failures and interrupted/partial runs stay in the raw CSV/JSON
    # audit trail but must not enter inferential statistics as if they were
    # successful algorithm runs.
    invalid_rows = [row for row in rows if str(row.get("status", "ok")) != "ok"]
    rows = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    if not rows:
        raise ValueError("no valid completed rows available for statistical workbooks")
    error_path = output / f"{prefix}_ErrorStats.xlsx"
    convergence_path = output / f"{prefix}_Convergence_data.xlsx"
    present_algorithms = {str(row["algorithm"]) for row in rows}
    preferred_order = ("BGP-SAEA", "ESA", "SADE-SS", "SGDLCO", "TSDDEO")
    algorithms = [name for name in preferred_order if name in present_algorithms]
    algorithms.extend(
        sorted(
            name
            for name in present_algorithms
            if name not in algorithms and not name.startswith("LLMOperatorDE")
        )
    )
    # Keep the proposed method at the right edge of the comparison table.
    algorithms.extend(sorted(name for name in present_algorithms if name.startswith("LLMOperatorDE")))
    reference = (
        "LLMOperatorDE"
        if "LLMOperatorDE" in algorithms
        else "LLMOperatorDE-Full"
        if "LLMOperatorDE-Full" in algorithms
        else algorithms[-1]
    )
    groups: dict[tuple[str, int], dict[str, list[float]]] = {}
    histories: dict[tuple[str, int, str], list[list[Mapping[str, Any]]]] = {}
    for row in rows:
        key = (str(row["problem"]), int(row["dimension"]))
        groups.setdefault(key, {}).setdefault(str(row["algorithm"]), []).append(float(row["best_f"]))
        histories.setdefault((key[0], key[1], str(row["algorithm"])), []).append(list(row["best_f_history"]))

    def set_bold(cell: Any) -> None:
        font = copy(cell.font)
        font.bold = True
        cell.font = font

    means_by_group: dict[tuple[str, int], dict[str, float]] = {}
    best_counts = {algorithm: 0 for algorithm in algorithms}
    rank_totals = {algorithm: 0.0 for algorithm in algorithms}
    comparison_counts = {
        algorithm: {"win": 0, "tie": 0, "loss": 0, "compared": 0}
        for algorithm in algorithms
        if algorithm != reference
    }
    raw_p_values: dict[tuple[str, int], dict[str, float]] = {}
    holm_p_values: dict[tuple[str, int], dict[str, float]] = {}
    effect_sizes: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}

    for key, values in groups.items():
        means = {
            algorithm: float(np.mean(sample))
            for algorithm, sample in values.items()
            if len(sample) > 0
        }
        means_by_group[key] = means
        if not means:
            continue
        best_mean = min(means.values())
        for algorithm, mean in means.items():
            if np.isclose(mean, best_mean, rtol=1e-12, atol=1e-12):
                best_counts[algorithm] += 1
            rank_totals[algorithm] += 1.0 + sum(
                other_mean < mean - 1e-12
                for other_mean in means.values()
            )

    error_wb = openpyxl.Workbook()
    error_ws = error_wb.active
    error_ws.title = "Error_Mean_Std"
    error_ws.append(["Problem", "Dimension", *algorithms])
    p_ws = error_wb.create_sheet("P_Values")
    p_ws.append(["Problem", "Dimension", *[f"{a}_vs_{reference}" for a in algorithms if a != reference]])
    holm_ws = error_wb.create_sheet("P_Values_Holm")
    holm_ws.append(["Problem", "Dimension", *[f"{a}_vs_{reference}" for a in algorithms if a != reference]])
    effect_ws = error_wb.create_sheet("Effect_Sizes")
    effect_ws.append(["Problem", "Dimension", *[f"{a}_rank_biserial" for a in algorithms if a != reference], *[f"{a}_std_delta" for a in algorithms if a != reference]])
    for cell in error_ws[1] + p_ws[1] + holm_ws[1] + effect_ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    data_row_indices: dict[tuple[str, int], int] = {}
    for key in sorted(groups):
        problem, dimension = key
        values = groups[key]
        error_row = [problem, dimension]
        p_row = [problem, dimension]
        reference_values = np.asarray(values.get(reference, []), dtype=float)
        p_by_algorithm: dict[str, float] = {}
        effect_by_algorithm: dict[str, tuple[float, float]] = {}
        for algorithm in algorithms:
            if algorithm == reference:
                continue
            sample = np.asarray(values.get(algorithm, []), dtype=float)
            if sample.size == 0 or reference_values.size == 0:
                continue
            count = min(len(sample), len(reference_values))
            try:
                p_by_algorithm[algorithm] = float(
                    wilcoxon(sample[:count], reference_values[:count], alternative="two-sided").pvalue
                )
            except ValueError:
                p_by_algorithm[algorithm] = 1.0
            paired = reference_values[:count] - sample[:count]
            nonzero = paired[np.abs(paired) > 1e-12]
            if nonzero.size:
                ranks = rankdata(np.abs(nonzero), method="average")
                signed_rank = float(np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0]))
                rank_biserial = signed_rank / max(float(np.sum(ranks)), 1.0)
            else:
                rank_biserial = 0.0
            pooled = float(np.std(np.concatenate((sample[:count], reference_values[:count])), ddof=1)) if count > 1 else 0.0
            standardized_delta = float(np.mean(paired) / max(pooled, 1e-12))
            effect_by_algorithm[algorithm] = (rank_biserial, standardized_delta)
        raw_p_values[key] = p_by_algorithm
        effect_sizes[key] = effect_by_algorithm
        ordered = sorted(p_by_algorithm.items(), key=lambda item: item[1])
        adjusted: dict[str, float] = {}
        running = 0.0
        total = len(ordered)
        for index, (algorithm, p_value) in enumerate(ordered):
            running = max(running, min(1.0, (total - index) * float(p_value)))
            adjusted[algorithm] = running
        holm_p_values[key] = adjusted
        for algorithm, adjusted_p in adjusted.items():
            if algorithm not in comparison_counts:
                continue
            sample = np.asarray(values.get(algorithm, []), dtype=float)
            count = min(len(sample), len(reference_values))
            comparison_counts[algorithm]["compared"] += 1
            if adjusted_p > 0.05:
                comparison_counts[algorithm]["tie"] += 1
            elif sample[:count].mean() < reference_values[:count].mean():
                comparison_counts[algorithm]["win"] += 1
            else:
                comparison_counts[algorithm]["loss"] += 1
        for algorithm in algorithms:
            sample = np.asarray(values.get(algorithm, []), dtype=float)
            if sample.size == 0:
                error_row.append("")
            else:
                sign = ""
                if algorithm != reference and algorithm in p_by_algorithm:
                    if holm_p_values[key].get(algorithm, 1.0) > 0.05:
                        sign = "="
                    elif sample.mean() < reference_values.mean():
                        sign = "+"
                    else:
                        sign = "-"
                error_row.append(
                    f"{sample.mean():.4e} ({sample.std(ddof=1) if sample.size > 1 else 0.0:.2e}){sign}"
                )
            if algorithm != reference:
                p_row.append(p_by_algorithm.get(algorithm, None))
        error_ws.append(error_row)
        data_row_indices[key] = error_ws.max_row
        p_ws.append(p_row)
        holm_ws.append(
            [problem, dimension]
            + [holm_p_values.get(key, {}).get(algorithm, None) for algorithm in algorithms if algorithm != reference]
        )
        effect_ws.append(
            [problem, dimension]
            + [effect_sizes.get(key, {}).get(algorithm, (None, None))[0] for algorithm in algorithms if algorithm != reference]
            + [effect_sizes.get(key, {}).get(algorithm, (None, None))[1] for algorithm in algorithms if algorithm != reference]
        )

        means = means_by_group.get(key, {})
        if means:
            best_mean = min(means.values())
            for column_index, algorithm in enumerate(algorithms, start=3):
                if algorithm in means and np.isclose(means[algorithm], best_mean, rtol=1e-12, atol=1e-12):
                    set_bold(error_ws.cell(error_ws.max_row, column_index))

    summary_metrics: list[tuple[str, dict[str, Any]]] = [
        ("Summary_BestCount", best_counts),
        (
            "Summary_MeanRank",
            {
                algorithm: (rank_totals[algorithm] / max(1, len(groups)))
                if algorithm in means_by_group or algorithm in algorithms
                else ""
                for algorithm in algorithms
            },
        ),
    ]
    for metric_name, field_name in (
        ("Summary_+_vs_" + reference, "win"),
        ("Summary_-_vs_" + reference, "loss"),
        ("Summary_=_vs_" + reference, "tie"),
        ("Summary_ComparedRows", "compared"),
    ):
        summary_metrics.append(
            (
                metric_name,
                {
                    algorithm: comparison_counts.get(algorithm, {}).get(field_name, "")
                    for algorithm in algorithms
                },
            )
        )
    summary_metrics.append(
        (
            "Summary_+/-/=_vs_" + reference,
            {
                algorithm: (
                    f"{comparison_counts[algorithm]['win']}/"
                    f"{comparison_counts[algorithm]['loss']}/"
                    f"{comparison_counts[algorithm]['tie']}"
                    if algorithm in comparison_counts
                    else ""
                )
                for algorithm in algorithms
            },
        )
    )
    for metric_name, metric_values in summary_metrics:
        error_ws.append(["Summary", metric_name, *[metric_values.get(algorithm, "") for algorithm in algorithms]])
        set_bold(error_ws.cell(error_ws.max_row, 1))
        set_bold(error_ws.cell(error_ws.max_row, 2))

    statistics_ws = error_wb.create_sheet("Statistics")
    statistics_ws.append(["Metric", "Reference", *algorithms])
    for cell in statistics_ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    for metric_name, metric_values in summary_metrics:
        statistics_ws.append(
            [metric_name, reference, *[metric_values.get(algorithm, "") for algorithm in algorithms]]
        )

    for ws in (error_ws, p_ws, holm_ws, effect_ws):
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column in ws.columns:
            letter = column[0].column_letter
            ws.column_dimensions[letter].width = max(14, min(30, max(len(str(c.value or "")) for c in column) + 2))
    for column in statistics_ws.columns:
        letter = column[0].column_letter
        statistics_ws.column_dimensions[letter].width = max(
            16, min(32, max(len(str(c.value or "")) for c in column) + 2)
        )
    statistics_ws.freeze_panes = "A2"
    statistics_ws.auto_filter.ref = statistics_ws.dimensions
    invalid_ws = error_wb.create_sheet("Invalid_Runs")
    invalid_fields = sorted({key for row in invalid_rows for key in row if key not in {"path", "metrics", "mujoco_replay", "best_f_history"}})
    invalid_ws.append(invalid_fields or ["status"])
    for row in invalid_rows:
        invalid_ws.append([json.dumps(row.get(field), ensure_ascii=False, default=str) if isinstance(row.get(field), (dict, list, tuple)) else row.get(field, "") for field in invalid_fields])
    for cell in invalid_ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="FCE4D6")
    invalid_ws.freeze_panes = "A2"
    invalid_ws.auto_filter.ref = invalid_ws.dimensions
    error_wb.save(error_path)

    conv_wb = openpyxl.Workbook()
    conv_wb.remove(conv_wb.active)
    for (problem, dimension), values in sorted(groups.items()):
        ws = conv_wb.create_sheet(f"{problem}_D{dimension}"[:31])
        checkpoints = np.linspace(
            min(int(row["initial_evals"]) for row in rows if row["problem"] == problem and int(row["dimension"]) == dimension)
            if any("initial_evals" in row for row in rows) else 1,
            max(int(row["real_evals"]) for row in rows if row["problem"] == problem and int(row["dimension"]) == dimension),
            25,
        )
        ws.append(["Number of Evaluations", *algorithms])
        for checkpoint in checkpoints:
            row = [int(round(checkpoint))]
            for algorithm in algorithms:
                run_histories = histories.get((problem, dimension, algorithm), [])
                if not run_histories:
                    row.append(None)
                else:
                    row.append(float(np.mean([_resample_history(history, np.asarray([checkpoint]))[0] for history in run_histories])))
            ws.append(row)
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
        final_row = ws.max_row
        final_values = {
            algorithm: ws.cell(final_row, column_index).value
            for column_index, algorithm in enumerate(algorithms, start=2)
            if ws.cell(final_row, column_index).value is not None
        }
        if final_values:
            best_final = min(float(value) for value in final_values.values())
            for column_index, algorithm in enumerate(algorithms, start=2):
                value = ws.cell(final_row, column_index).value
                if value is not None and np.isclose(float(value), best_final, rtol=1e-12, atol=1e-12):
                    set_bold(ws.cell(final_row, column_index))
        ws.freeze_panes = "A2"
    conv_wb.save(convergence_path)
    return {"error_stats": error_path, "convergence": convergence_path}


__all__ = [
    "DEFAULT_ALGORITHMS",
    "PROBLEMS",
    "SuiteConfig",
    "get_suite_benchmark",
    "run_paper_algorithm",
    "run_ported_algorithm",
    "run_llm_algorithm",
    "run_suite",
    "run_ablation_suite",
    "compute_ablation_contributions",
    "write_ablation_contribution_report",
    "write_suite_workbooks",
]
