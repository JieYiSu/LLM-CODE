"""Code-configured entry point for the underwater comparison suite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments.underwater_robot.experiment_config import CONFIG
from experiments.underwater_robot.run_underwater_comparison import run_comparison_suite


def main() -> None:
    try:
        run_comparison_suite(CONFIG)
    except KeyboardInterrupt:
        # Each completed run is appended to the JSONL progress log. Rerunning
        # this same entry point resumes at the first incomplete run.
        print("underwater suite interrupted; rerun the same file to resume", flush=True)


if __name__ == "__main__":
    main()
