"""One code-configured entry point for the complete paper pipeline.

The stages are deliberately ordered and isolated:

1. expensive single-objective comparison;
2. the three-component LLMOperatorDE ablation (plus the full model);
3. the underwater digital-twin engineering validation.

Every underlying runner already persists a JSONL/checkpoint file.  This
wrapper adds a small phase manifest so Ctrl+C or a process restart resumes at
the unfinished phase without mixing result directories.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


PROJECT = Path(__file__).resolve().parent
EXPERIMENTS = PROJECT / "experiments"
UNIFIED_ROOT = EXPERIMENTS / "unified_experiments"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


@dataclass(frozen=True)
class UnifiedExperimentConfig:
    """Single source of truth for the three sequential phases.

    Set the three ``run_*`` flags in this file before launching the script;
    no command-line arguments are required.  The default uses the paper
    protocols already defined by ``SuiteConfig`` and ``UnderwaterExperimentConfig``.
    """

    run_comparison: bool = False
    run_ablation: bool = True
    run_engineering: bool = False
    output_root: Path = UNIFIED_ROOT
    dry_run: bool = False
    # Keep the numerical benchmark/ablation and engineering protocols explicit
    # in one place. The numerical phases use 300 FE; the engineering phase
    # keeps its separate 1000-FE digital-twin protocol.
    comparison_total_evals: int = 300
    comparison_initial_evals: int = 100
    comparison_repeats: int = 20
    comparison_dimensions: tuple[int, ...] = (30, 50, 100, 200, 500, 1000)


CONFIG = UnifiedExperimentConfig()


def _phase_manifest_path(config: UnifiedExperimentConfig) -> Path:
    return Path(config.output_root) / "pipeline_progress.json"


def _load_phase_manifest(config: UnifiedExperimentConfig) -> dict[str, Any]:
    path = _phase_manifest_path(config)
    if not path.exists():
        return {"version": 1, "phases": {}, "updated_at": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "phases": {}, "updated_at": None}
    if not isinstance(payload, dict):
        return {"version": 1, "phases": {}, "updated_at": None}
    payload.setdefault("version", 1)
    payload.setdefault("phases", {})
    return payload


def _save_phase_manifest(config: UnifiedExperimentConfig, payload: dict[str, Any]) -> None:
    path = _phase_manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _phase_plan(config: UnifiedExperimentConfig) -> list[tuple[str, Path, Callable[[], Any]]]:
    """Build the ordered phase plan without executing any experiment."""

    from experiments.expensive_suite import SuiteConfig, run_ablation_suite, run_suite
    from experiments.underwater_robot.experiment_config import CONFIG as UNDERWATER_CONFIG
    from experiments.underwater_robot.run_underwater_comparison import run_comparison_suite

    comparison_config = SuiteConfig(
        total_evals=int(config.comparison_total_evals),
        initial_evals=int(config.comparison_initial_evals),
        repeats=int(config.comparison_repeats),
        dimensions=tuple(int(value) for value in config.comparison_dimensions),
        output_dir=Path(config.output_root) / "expensive_suite_results",
        resume=True,
        checkpoint_interval=1,
    )
    engineering_config = replace(
        UNDERWATER_CONFIG,
        output_root=Path(config.output_root) / "underwater_engineering",
    )

    return [
        (
            "comparison",
            Path(comparison_config.output_dir),
            lambda: run_suite(comparison_config),
        ),
        (
            "ablation",
            Path(comparison_config.output_dir).parent / "expensive_suite_ablation",
            lambda: run_ablation_suite(comparison_config),
        ),
        (
            "engineering",
            Path(engineering_config.output_root),
            lambda: run_comparison_suite(engineering_config),
        ),
    ]


def _print_plan(config: UnifiedExperimentConfig) -> None:
    print("Unified underwater paper pipeline", flush=True)
    print(f"output_root={Path(config.output_root).resolve()}", flush=True)
    for name, output, _ in _phase_plan(config):
        enabled = {
            "comparison": config.run_comparison,
            "ablation": config.run_ablation,
            "engineering": config.run_engineering,
        }[name]
        print(f"  {'ON ' if enabled else 'OFF'} {name:12s} -> {output}", flush=True)


def run_pipeline(config: UnifiedExperimentConfig = CONFIG) -> dict[str, Any]:
    """Run enabled phases in order and return the phase manifest."""

    manifest = _load_phase_manifest(config)
    enabled = {
        "comparison": bool(config.run_comparison),
        "ablation": bool(config.run_ablation),
        "engineering": bool(config.run_engineering),
    }
    manifest["enabled_phases"] = enabled
    manifest["phase_order"] = ["comparison", "ablation", "engineering"]
    manifest["protocol"] = {
        "comparison_total_evals": int(config.comparison_total_evals),
        "comparison_initial_evals": int(config.comparison_initial_evals),
        "comparison_repeats": int(config.comparison_repeats),
        "comparison_dimensions": [int(value) for value in config.comparison_dimensions],
        "engineering_total_evals": 1000,
        "engineering_initial_evals": 100,
        "engineering_repeats": 20,
        "phase_order": ["comparison", "ablation", "engineering"],
    }
    _print_plan(config)
    if config.dry_run:
        return manifest
    config.output_root.mkdir(parents=True, exist_ok=True)
    _save_phase_manifest(config, manifest)

    for name, output, runner in _phase_plan(config):
        if not enabled[name]:
            continue
        record = manifest.setdefault("phases", {}).get(name, {})
        if record.get("status") == "completed":
            print(f"[pipeline] {name} already completed; resume skip", flush=True)
            continue
        manifest["phases"][name] = {"status": "running", "output": str(output)}
        _save_phase_manifest(config, manifest)
        print(f"[pipeline] starting {name}: {output}", flush=True)
        try:
            runner()
        except KeyboardInterrupt:
            manifest["phases"][name] = {
                "status": "interrupted",
                "output": str(output),
            }
            _save_phase_manifest(config, manifest)
            print(
                f"[pipeline] interrupted in {name}; rerun this same file to resume",
                flush=True,
            )
            raise
        except Exception as error:
            manifest["phases"][name] = {
                "status": "failed",
                "output": str(output),
                "error": f"{type(error).__name__}: {error}",
            }
            _save_phase_manifest(config, manifest)
            raise
        manifest["phases"][name] = {
            "status": "completed",
            "output": str(output),
        }
        _save_phase_manifest(config, manifest)
        print(f"[pipeline] completed {name}", flush=True)
    return manifest


def main() -> None:
    try:
        run_pipeline(CONFIG)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()


__all__ = ["CONFIG", "UnifiedExperimentConfig", "run_pipeline"]
