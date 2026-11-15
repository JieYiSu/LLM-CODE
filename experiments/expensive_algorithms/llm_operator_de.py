"""Ablation-only entry points for the main ``llm_operator_de`` package."""

from __future__ import annotations

from typing import Any

from ..expensive_suite import run_llm_algorithm as _run_llm_algorithm


def run_ablation_algorithm(
    problem: str,
    dimension: int,
    seed: int,
    config: Any,
    variant: str,
    knowledge_path: Any = None,
) -> dict[str, Any]:
    """Run one controlled ablation without duplicating the main algorithm."""

    if variant == "Full":
        return _run_llm_algorithm(problem, dimension, seed, config, use_llm=True, knowledge_path=knowledge_path)
    if variant == "NoLLM":
        return _run_llm_algorithm(problem, dimension, seed, config, use_llm=False, knowledge_path=knowledge_path)
    if variant == "NoSharedMating":
        return _run_llm_algorithm(
            problem,
            dimension,
            seed,
            config,
            use_llm=True,
            cross_island_rate=0.0,
            global_guidance=0.0,
            knowledge_path=knowledge_path,
        )
    if variant == "NoAdaptiveFCR":
        return _run_llm_algorithm(
            problem,
            dimension,
            seed,
            config,
            use_llm=True,
            adaptive_parameters=False,
            knowledge_path=knowledge_path,
        )
    raise ValueError(f"unknown LLMOperatorDE ablation: {variant}")


__all__ = ["run_ablation_algorithm"]
