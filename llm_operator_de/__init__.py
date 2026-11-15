"""Standalone package for the LLM-assisted EvoX DE algorithm."""

from .core import (
    DEOperatorLibrary,
    DEOperatorSpec,
    LLMOperatorDE,
    LLMOperatorGenerator,
    OperatorASTCompiler,
    OperatorArchive,
    OperatorCostModel,
    OperatorPromoter,
    PromotionResult,
    ShadowEvaluation,
    ShadowEvaluator,
    constrained_pareto_better,
)
from .prompts import (
    OPERATOR_CATALOG,
    OPERATOR_SYNTHESIS_PROTOCOL,
    OPERATOR_SYNTHESIS_SYSTEM_PROMPT,
    build_operator_synthesis_prompt,
)
from .rag import OperatorRAG
from .islands import IslandCoordinator
from .policy import AlgorithmPolicy

__all__ = [
    "DEOperatorLibrary",
    "DEOperatorSpec",
    "LLMOperatorDE",
    "LLMOperatorGenerator",
    "OperatorASTCompiler",
    "OperatorArchive",
    "OperatorCostModel",
    "OperatorPromoter",
    "PromotionResult",
    "ShadowEvaluation",
    "ShadowEvaluator",
    "constrained_pareto_better",
    "OPERATOR_CATALOG",
    "OPERATOR_SYNTHESIS_PROTOCOL",
    "OPERATOR_SYNTHESIS_SYSTEM_PROMPT",
    "build_operator_synthesis_prompt",
    "OperatorRAG",
    "IslandCoordinator",
    "AlgorithmPolicy",
]
