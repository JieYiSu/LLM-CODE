"""Shared contracts for the separated expensive-algorithm modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ExpensiveAlgorithm(Protocol):
    def run(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AlgorithmMetadata:
    name: str
    source: str
    surrogate: str
    notes: str


def method_metadata(name: str) -> AlgorithmMetadata:
    records = {
        "TSDDEO": AlgorithmMetadata(
            "TSDDEO",
            "TSDDEO_core.m",
            "RBF",
            "two-stage surrogate SLPSO/DE port",
        ),
        "ESA": AlgorithmMetadata(
            "ESA",
            "ESA_core.m",
            "RBF",
            "four-action surrogate-agent port",
        ),
        "SADE-SS": AlgorithmMetadata(
            "SADE-SS",
            "experiments/expensive_algorithms/sade_ss.py",
            "Gaussian RBF",
            "local port with shared paper-surrogate core",
        ),
        "SGDLCO": AlgorithmMetadata(
            "SGDLCO",
            "experiments/expensive_algorithms/sgdlco.py",
            "Gaussian RBF",
            "unconstrained numerical-benchmark adaptation",
        ),
        "BGP-SAEA": AlgorithmMetadata(
            "BGP-SAEA",
            "experiments/expensive_algorithms/bgp_saea.py",
            "boosted GP ensemble",
            "local port with shared paper-surrogate core",
        ),
        "LLMOperatorDE": AlgorithmMetadata(
            "LLMOperatorDE",
            "llm_operator_de/core.py",
            "state-aware DE/LLM operator synthesis",
            "EvoX three-island method",
        ),
    }
    try:
        return records[name]
    except KeyError as exc:
        raise ValueError(f"unknown expensive-suite method: {name}") from exc


__all__ = ["AlgorithmMetadata", "ExpensiveAlgorithm", "method_metadata"]
