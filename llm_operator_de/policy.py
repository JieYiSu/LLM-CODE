"""Bounded LLM-controlled algorithm policies.

The policy is deliberately discrete and typed.  It can change how a DE run
selects, samples, replaces, handles constraints, and migrates islands, while
the runtime still owns the objective, real-evaluation budget, safety
invariants, and tensor execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


def _bounded_float(value: Any, low: float, high: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    return number


@dataclass
class AlgorithmPolicy:
    """Safe algorithm-level choices that an LLM may propose."""

    selection_mode: str = "strict"
    selection_epsilon: float = 0.0
    survivor_mode: str = "one_to_one"
    survivor_elite_fraction: float = 0.2
    constraint_mode: str = "penalty"
    constraint_epsilon: float = 0.0
    parent_sampling_mode: str = "uniform_distinct"
    pbest_fraction: float = 0.2
    archive_probability: float = 0.2
    migration_mode: str = "best_to_worst"
    migration_interval: int = 5
    role_mode: str = "preserve"

    SELECTION_MODES = {"strict", "epsilon", "feasibility_first"}
    SURVIVOR_MODES = {"one_to_one", "elitist", "diversity_preserving"}
    CONSTRAINT_MODES = {"penalty", "feasibility_first", "epsilon_constraint"}
    PARENT_MODES = {"uniform_distinct", "pbest_biased", "archive_mixed"}
    MIGRATION_MODES = {"best_to_worst", "round_robin", "adaptive"}
    ROLE_MODES = {"preserve", "rebalance"}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AlgorithmPolicy":
        data = dict(data or {})
        selection = dict(data.get("selection", {}) or {})
        survivor = dict(data.get("survivor", {}) or {})
        constraint = dict(data.get("constraint", {}) or {})
        parent = dict(data.get("parent_sampling", {}) or {})
        island = dict(data.get("island", {}) or {})

        def mode(section: Mapping[str, Any], key: str, allowed: set[str], default: str) -> str:
            value = str(section.get(key, default))
            if value not in allowed:
                raise ValueError(f"unsupported algorithm policy mode: {value}")
            return value

        migration_interval = int(island.get("migration_interval", cls.migration_interval))
        if migration_interval < 1 or migration_interval > 100:
            raise ValueError("migration_interval must be within [1, 100]")

        return cls(
            selection_mode=mode(selection, "mode", cls.SELECTION_MODES, cls.selection_mode),
            selection_epsilon=_bounded_float(
                selection.get("epsilon", cls.selection_epsilon), 0.0, 1.0, "selection epsilon"
            ),
            survivor_mode=mode(survivor, "mode", cls.SURVIVOR_MODES, cls.survivor_mode),
            survivor_elite_fraction=_bounded_float(
                survivor.get("elite_fraction", cls.survivor_elite_fraction),
                0.05,
                0.5,
                "survivor elite_fraction",
            ),
            constraint_mode=mode(constraint, "mode", cls.CONSTRAINT_MODES, cls.constraint_mode),
            constraint_epsilon=_bounded_float(
                constraint.get("epsilon", cls.constraint_epsilon), 0.0, 1.0, "constraint epsilon"
            ),
            parent_sampling_mode=mode(
                parent, "mode", cls.PARENT_MODES, cls.parent_sampling_mode
            ),
            pbest_fraction=_bounded_float(
                parent.get("pbest_fraction", cls.pbest_fraction), 0.05, 0.5, "pbest_fraction"
            ),
            archive_probability=_bounded_float(
                parent.get("archive_probability", cls.archive_probability),
                0.0,
                0.5,
                "archive_probability",
            ),
            migration_mode=mode(
                island, "migration_mode", cls.MIGRATION_MODES, cls.migration_mode
            ),
            migration_interval=migration_interval,
            role_mode=mode(island, "role_mode", cls.ROLE_MODES, cls.role_mode),
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return {
            "selection": {
                "mode": raw.pop("selection_mode"),
                "epsilon": raw.pop("selection_epsilon"),
            },
            "survivor": {
                "mode": raw.pop("survivor_mode"),
                "elite_fraction": raw.pop("survivor_elite_fraction"),
            },
            "constraint": {
                "mode": raw.pop("constraint_mode"),
                "epsilon": raw.pop("constraint_epsilon"),
            },
            "parent_sampling": {
                "mode": raw.pop("parent_sampling_mode"),
                "pbest_fraction": raw.pop("pbest_fraction"),
                "archive_probability": raw.pop("archive_probability"),
            },
            "island": {
                "migration_mode": raw.pop("migration_mode"),
                "migration_interval": raw.pop("migration_interval"),
                "role_mode": raw.pop("role_mode"),
            },
        }

