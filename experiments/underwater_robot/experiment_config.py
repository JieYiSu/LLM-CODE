"""Single source of truth for the underwater comparison protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from .underwater_scenarios import SCENARIOS


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UnderwaterExperimentConfig:
    # The expensive-problem protocol is configured in Python, not through CLI flags.
    total_evals: int = 1000
    initial_evals: int = 100
    population_size: int = 30
    repeats: int = 20
    seed_start: int = 1
    seed: int = 1
    scenario: str = "uniform_cross_current"
    robot_dof: int = 6
    intermediate_points: int = 8
    island_count: int = 3
    algorithms: tuple[str, ...] = (
        "LLMOperatorDE",
        "TSDDEO",
        "ESA",
        "SADE-SS",
        "SGDLCO",
        "BGP-SAEA",
    )
    scenarios: tuple[str, ...] = tuple(SCENARIOS)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_llm: bool = True
    force_warmup_llm: bool = True
    llm_config_path: Path = PROJECT / "config.yaml"
    # Versioned because earlier results used the original cylinder-clearance,
    # replay and piecewise-linear trajectory implementations.
    # v9 starts a fresh result tree after hard endpoint anchoring and
    # path-aware shared warm starts were added to the benchmark protocol.
    output_root: Path = ROOT / "results_underwater_comparison_v9"
    migration_interval: int = 5
    # With 1000 FE, 100 initialization evaluations and a total active
    # population of 30, the generational protocol has 30 full DE updates.
    # The LLM is queried every three updates, with promotion validation inside
    # the same fixed real-evaluation budget.
    llm_interval: int = 3
    llm_eval_interval: int = 200
    # Candidate promotion uses two complete balanced-island generations
    # (the island has 30--34 members).  The evaluations are already inside
    # the fixed FE budget; this only prevents a noisy one-generation takeover.
    candidate_validation_evals: int = 60
    rag_top_k: int = 2

    @property
    def trajectory_dimension(self) -> int:
        # Each knot contains x, y, z and desired forward speed.
        return int(self.intermediate_points * 4)

    @property
    def config_path(self) -> Path:
        return self.llm_config_path

    @property
    def output_dir(self) -> Path:
        return self.output_root

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.repeats))

    def for_seed(self, seed: int, output_dir: Path | None = None) -> "UnderwaterExperimentConfig":
        return replace(
            self,
            seed=int(seed),
            output_root=Path(output_dir) if output_dir is not None else self.output_root,
        )

    def validate(self) -> None:
        if self.total_evals < self.initial_evals:
            raise ValueError("total_evals must be at least initial_evals")
        if self.initial_evals < self.population_size:
            raise ValueError("initial_evals must be at least population_size")
        if self.initial_evals // self.island_count < 4:
            raise ValueError("initial_evals must provide at least four candidates per island")
        if self.repeats < 1 or self.intermediate_points < 2:
            raise ValueError("repeats and intermediate_points must be positive")
        if any(name not in SCENARIOS for name in self.scenarios):
            raise ValueError("scenarios must be selected from SCENARIOS")
        if self.scenario not in SCENARIOS:
            raise ValueError("scenario must be selected from SCENARIOS")
        if not self.algorithms:
            raise ValueError("at least one comparison algorithm is required")


CONFIG = UnderwaterExperimentConfig()
CONFIG.validate()


__all__ = ["CONFIG", "UnderwaterExperimentConfig", "PROJECT", "ROOT"]
