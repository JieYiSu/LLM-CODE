"""MuJoCo-based underwater robot current-aware path-planning suite."""

from .experiment_config import CONFIG, UnderwaterExperimentConfig
from .underwater_objective import UnderwaterPathObjective

__all__ = ["CONFIG", "UnderwaterExperimentConfig", "UnderwaterPathObjective"]
