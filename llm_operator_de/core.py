"""LLM-assisted, tensorized differential evolution.

The module deliberately separates three concerns:

* :class:`DEOperatorLibrary` contains the executable tensor operators and is
  safe to run on CPU or CUDA;
* :class:`OperatorASTCompiler` turns bounded model-produced programs into
  executable tensor specifications;
* :class:`ShadowEvaluator` and :class:`OperatorPromoter` separate cheap
  batched screening from a reserved real-evaluation budget;
* :class:`LLMOperatorGenerator` is an optional OpenAI-compatible API adapter;
* :class:`LLMOperatorDE` is the EvoX algorithm wrapper.

The model proposes a validated operator *specification*. It never controls the
objective function, the selection rule, or the evaluation budget. When the LLM
is unavailable, the algorithm remains a conventional tensorized DE run.
"""

from __future__ import annotations

import ast as python_ast
import hashlib
import json
import http.client
import math
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from evox.core import Algorithm, Mutable
from .prompts import OPERATOR_CATALOG, OPERATOR_SYNTHESIS_SYSTEM_PROMPT, build_operator_synthesis_prompt
from .policy import AlgorithmPolicy


def _device_rand(
    shape: tuple[int, ...] | list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw on CPU and transfer so CPU/CUDA runs share the same RNG stream."""

    values = torch.rand(tuple(shape), device=torch.device("cpu"), dtype=dtype)
    return values if device.type == "cpu" else values.to(device=device)


def _device_randn(
    shape: tuple[int, ...] | list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = torch.randn(tuple(shape), device=torch.device("cpu"), dtype=dtype)
    return values if device.type == "cpu" else values.to(device=device)


def _device_randint(
    low: int,
    high: int,
    shape: tuple[int, ...] | list[int],
    device: torch.device,
) -> torch.Tensor:
    values = torch.randint(low, high, tuple(shape), device=torch.device("cpu"))
    return values if device.type == "cpu" else values.to(device=device)


_BUILTIN_MUTATIONS = {
    "rand1",
    "best1",
    "best2",
    "current_to_best1",
    "current_to_rand1",
    "weighted_ensemble",
    "generated_expr",
    "elite1",
    "history1",
    "current_to_pbest1",
    "opposition1",
    "history_guided1",
}


@dataclass
class DEOperatorSpec:
    """A serializable and bounded description of one DE variation operator."""

    operator_id: str
    mutation: str = "rand1"
    crossover: str = "bin"
    F: float = 0.5
    CR: float = 0.9
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ast: Optional[dict[str, Any]] = None
    operator_code: str = ""
    F_policy: dict[str, float] = field(default_factory=dict)
    CR_policy: dict[str, float] = field(default_factory=dict)
    gating_policy: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    fallback_operator: str = "rand1"
    algorithm_policy: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate and normalize a model-produced specification in place."""

        if not self.operator_id or not isinstance(self.operator_id, str):
            raise ValueError("operator_id must be a non-empty string")
        if self.mutation not in _BUILTIN_MUTATIONS:
            raise ValueError(f"unsupported mutation: {self.mutation}")
        if self.crossover not in {"bin", "exp"}:
            raise ValueError(f"unsupported crossover: {self.crossover}")

        # The first implementation uses the stable DE interval [0, 1].
        # Larger values can be added later as a separately evaluated policy.
        self.F = float(max(0.0, min(1.0, float(self.F))))
        self.CR = float(max(0.0, min(1.0, float(self.CR))))
        self.fallback_operator = str(self.fallback_operator or "rand1")
        if self.fallback_operator not in _BUILTIN_MUTATIONS - {"weighted_ensemble", "generated_expr"}:
            self.fallback_operator = "rand1"
        self.F_policy = self._validate_policy(self.F_policy)
        self.CR_policy = self._validate_policy(self.CR_policy)
        self.gating_policy = self._validate_gate(self.gating_policy)
        self.algorithm_policy = AlgorithmPolicy.from_dict(self.algorithm_policy).to_dict()

        if self.ast is not None:
            if OperatorASTCompiler.is_vector_expression(self.ast):
                OperatorASTCompiler.validate_vector_expression(self.ast)
                self.mutation = "generated_expr"
                self.components = {}
            else:
                flattened = OperatorASTCompiler.flatten(self.ast)
                if len(flattened) == 1:
                    self.mutation = flattened[0][0]
                    self.components = {}
                else:
                    self.mutation = "weighted_ensemble"
                    self.components = {name: weight for name, weight in flattened}

        if self.mutation == "generated_expr":
            self.components = {}
        elif self.mutation == "weighted_ensemble":
            normalized: dict[str, float] = {}
            for name, weight in self.components.items():
                if name not in _BUILTIN_MUTATIONS - {"weighted_ensemble", "generated_expr"}:
                    raise ValueError(f"unsupported ensemble component: {name}")
                weight = float(weight)
                if weight > 0:
                    normalized[name] = weight
            if not normalized:
                normalized = {"rand1": 1.0}
            total = sum(normalized.values())
            self.components = {name: value / total for name, value in normalized.items()}
        else:
            self.components = {}
        return True

    @staticmethod
    def _validate_policy(policy: Mapping[str, Any] | None) -> dict[str, float]:
        policy = dict(policy or {})
        return {
            "stagnation_gain": float(max(-1.0, min(1.0, policy.get("stagnation_gain", 0.0)))),
            "diversity_gain": float(max(-1.0, min(1.0, policy.get("diversity_gain", 0.0)))),
            "budget_gain": float(max(-1.0, min(1.0, policy.get("budget_gain", 0.0)))),
        }

    @staticmethod
    def _validate_gate(policy: Mapping[str, Any] | None) -> dict[str, Any]:
        policy = dict(policy or {})
        condition = str(policy.get("condition", "always"))
        if condition not in {"always", "stagnation_ge", "diversity_le"}:
            condition = "always"
        return {"condition": condition, "threshold": float(policy.get("threshold", 0.0))}

    def effective_parameters(self, state: Mapping[str, Any]) -> tuple[float, float]:
        """Evaluate bounded state-conditioned F and CR policies."""
        stagnation = float(state.get("stagnation", 0.0))
        diversity = float(state.get("diversity", 0.0))
        budget = float(state.get("remaining_budget", state.get("remaining_evals", 0.0)))
        scale_budget = budget / max(1.0, float(state.get("initial_budget", budget or 1.0)))

        def apply(base: float, policy: Mapping[str, float]) -> float:
            value = base + float(policy.get("stagnation_gain", 0.0)) * min(1.0, stagnation / 10.0)
            value += float(policy.get("diversity_gain", 0.0)) * max(0.0, min(1.0, diversity))
            value += float(policy.get("budget_gain", 0.0)) * scale_budget
            return max(0.0, min(1.0, value))

        return apply(self.F, self.F_policy), apply(self.CR, self.CR_policy)

    def gate(self, state: Mapping[str, Any]) -> bool:
        condition = self.gating_policy.get("condition", "always")
        threshold = float(self.gating_policy.get("threshold", 0.0))
        if condition == "stagnation_ge":
            return float(state.get("stagnation", 0.0)) >= threshold
        if condition == "diversity_le":
            return float(state.get("diversity", 0.0)) <= threshold
        return True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DEOperatorSpec":
        """Construct a spec from flexible JSON-like model output.

        The executor uses one canonical representation internally, but the LLM
        does not need to memorize the field names used by that representation.
        This normalizer accepts the canonical object, a nested ``operator``
        object, a top-level expression node, or an ``expression`` alias before
        applying the same typed AST and budget checks.
        """

        if not isinstance(data, Mapping):
            raise ValueError("operator output must be a JSON object")
        data = dict(data)
        nested = data.get("operator")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            for key, value in data.items():
                if key != "operator" and key not in merged:
                    merged[key] = value
            data = merged
        elif isinstance(nested, str) and "mutation" not in data:
            data["mutation"] = nested

        if "operator_id" not in data:
            data["operator_id"] = data.get("id", data.get("name", "llm_operator"))
        parameters = data.get("parameters")
        if isinstance(parameters, Mapping):
            for key in ("F", "CR", "F_policy", "CR_policy", "gating_policy"):
                if key not in data and key in parameters:
                    data[key] = parameters[key]

        raw_ast = data.get("ast", data.get("operator_ast"))
        if isinstance(raw_ast, str):
            try:
                raw_ast = json.loads(raw_ast)
            except json.JSONDecodeError:
                raw_ast = None
        if raw_ast is None and isinstance(data.get("expression"), Mapping):
            raw_ast = {"op": "vector_expr", "expr": dict(data["expression"])}
        if raw_ast is None and isinstance(data.get("expr"), Mapping):
            raw_ast = {"op": "vector_expr", "expr": dict(data["expr"])}
        if raw_ast is None and str(data.get("op", "")) in {
            "ref", "primitive", "add", "sub", "scale", "weighted_sum", "clip", "state_scale",
        }:
            expression_node = {
                key: value
                for key, value in data.items()
                if key not in {"id", "operator_id", "operator_code", "code", "parameters"}
            }
            raw_ast = {"op": "vector_expr", "expr": expression_node}
        operator_code = str(
            data.get("operator_code", data.get("code", data.get("formula", ""))) or ""
        )
        if raw_ast is None and operator_code:
            raw_ast = OperatorASTCompiler.from_formula(operator_code)
        if raw_ast is None and operator_code:
            raw_ast = OperatorASTCompiler.from_code(operator_code)
        spec = cls(
            operator_id=str(data.get("operator_id", "llm_operator")),
            mutation=str(data.get("mutation", "rand1")),
            crossover=str(data.get("crossover", "bin")),
            F=float(data.get("F", 0.5)),
            CR=float(data.get("CR", 0.9)),
            components=dict(data.get("components", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            ast=dict(raw_ast) if isinstance(raw_ast, Mapping) else None,
            operator_code=operator_code,
            F_policy=dict(data.get("F_policy", {}) or {}),
            CR_policy=dict(data.get("CR_policy", {}) or {}),
            gating_policy=dict(data.get("gating_policy", {}) or {}),
            evidence_ids=[str(item) for item in data.get("evidence_ids", []) or []],
            fallback_operator=str(data.get("fallback_operator", "rand1")),
            algorithm_policy=dict(
                data.get("algorithm_policy", data.get("policy", {})) or {}
            ),
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def complexity(self) -> float:
        """Return a small, deterministic structural complexity estimate."""

        if self.ast is not None:
            return float(OperatorASTCompiler.node_count(self.ast))
        if self.mutation == "weighted_ensemble":
            return float(1 + len(self.components))
        return 1.0


class OperatorASTCompiler:
    """Compile a restricted expression AST into a validated DE specification."""

    PRIMITIVES = {
        "rand1",
        "best1",
        "best2",
        "current_to_best1",
        "current_to_rand1",
        "elite1",
        "history1",
        "current_to_pbest1",
        "opposition1",
        "history_guided1",
    }
    EXPR_REFS = {"x", "best", "pbest", "elite", "history", "r1", "r2", "r3", "r4"}
    EXPR_FEATURES = {"diversity", "stagnation", "remaining_budget", "improvement_rate"}

    @classmethod
    def is_vector_expression(cls, ast: Mapping[str, Any]) -> bool:
        return isinstance(ast, Mapping) and str(ast.get("op", "")) == "vector_expr"

    @classmethod
    def validate_vector_expression(cls, ast: Mapping[str, Any]) -> int:
        if not cls.is_vector_expression(ast):
            raise ValueError("generated operator AST must have op=vector_expr")
        counter = [0]
        cls._validate_expression_node(ast.get("expr", {}), depth=0, counter=counter)
        return counter[0]

    @classmethod
    def _validate_expression_node(
        cls,
        node: Mapping[str, Any],
        depth: int,
        counter: list[int],
    ) -> None:
        counter[0] += 1
        if counter[0] > 32 or depth > 8:
            raise ValueError("generated expression exceeds the depth or node limit")
        if not isinstance(node, Mapping):
            raise ValueError("generated expression nodes must be objects")
        op = str(node.get("op", ""))
        if op == "ref":
            if str(node.get("name", "")) not in cls.EXPR_REFS:
                raise ValueError("unsupported generated expression reference")
            return
        if op in cls.PRIMITIVES:
            return
        if op == "primitive":
            if str(node.get("name", "")) not in cls.PRIMITIVES:
                raise ValueError("unsupported generated expression primitive")
            return
        if op == "scale":
            factor = float(node.get("factor", node.get("weight", 1.0)))
            if not math.isfinite(factor) or abs(factor) > 4.0:
                raise ValueError("generated expression scale must be finite and bounded")
            cls._validate_expression_node(node.get("child", {}), depth + 1, counter)
            return
        if op in {"add", "sub"}:
            cls._validate_expression_node(node.get("left", {}), depth + 1, counter)
            cls._validate_expression_node(node.get("right", {}), depth + 1, counter)
            return
        if op == "weighted_sum":
            terms = node.get("terms", [])
            if not isinstance(terms, list) or not terms or len(terms) > 8:
                raise ValueError("generated weighted_sum requires 1-8 terms")
            total = 0.0
            for term in terms:
                if not isinstance(term, Mapping):
                    raise ValueError("generated weighted_sum terms must be objects")
                weight = float(term.get("weight", 1.0))
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError("generated term weights must be finite and nonnegative")
                total += weight
                cls._validate_expression_node(term.get("child", term), depth + 1, counter)
            if total <= 0.0:
                raise ValueError("generated weighted_sum needs a positive weight")
            return
        if op == "clip":
            low = float(node.get("low", -1.0))
            high = float(node.get("high", 1.0))
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("generated clip bounds are invalid")
            cls._validate_expression_node(node.get("child", {}), depth + 1, counter)
            return
        if op == "state_scale":
            # Some providers use ``factor`` for a constant state-scale node
            # and omit ``feature``.  Treat that bounded form as a constant
            # scale so the complete operator remains usable and auditable.
            if "feature" not in node:
                factor = float(node.get("factor", node.get("base", 1.0)))
                if not math.isfinite(factor) or abs(factor) > 4.0:
                    raise ValueError("generated state scale factor must be finite and bounded")
                cls._validate_expression_node(node.get("child", {}), depth + 1, counter)
                return
            feature = str(node.get("feature", ""))
            if feature not in cls.EXPR_FEATURES:
                raise ValueError("unsupported generated state feature")
            base = float(node.get("base", 1.0))
            gain = float(node.get("gain", 0.0))
            if not math.isfinite(base) or not math.isfinite(gain) or abs(base) > 4.0 or abs(gain) > 4.0:
                raise ValueError("generated state scale is unbounded")
            cls._validate_expression_node(node.get("child", {}), depth + 1, counter)
            return
        raise ValueError(f"unsupported generated expression operation: {op}")

    @classmethod
    def from_code(cls, code: str) -> Optional[dict[str, Any]]:
        """Parse the bounded DSL descriptions emitted by older LLM prompts.

        This is deliberately a recognizer for known primitive names, not a
        Python evaluator.  Unrecognized text is ignored and the normal
        validation path can reject an empty proposal.
        """
        text = str(code or "")
        terms: list[dict[str, Any]] = []
        for name in sorted(cls.PRIMITIVES, key=len, reverse=True):
            pattern = rf"(?:(\d+(?:\.\d+)?)\s*\*\s*)?{re.escape(name)}\b"
            for match in re.finditer(pattern, text):
                weight = float(match.group(1)) if match.group(1) is not None else 1.0
                terms.append({"op": "primitive", "name": name, "weight": weight})
        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return {"op": "weighted_sum", "terms": terms}

    @classmethod
    def flatten(
        cls,
        node: Mapping[str, Any],
        inherited_weight: float = 1.0,
        depth: int = 0,
        counter: Optional[list[int]] = None,
    ) -> list[tuple[str, float]]:
        if counter is None:
            counter = [0]
        counter[0] += 1
        if counter[0] > 32 or depth > 8:
            raise ValueError("operator AST exceeds the depth or node limit")
        if not isinstance(node, Mapping):
            raise ValueError("each AST node must be an object")
        op = str(node.get("op", ""))
        # Accept the compact form commonly emitted by LLMs, e.g.
        # {"op": "rand1"}, while retaining the explicit internal form.
        if op in cls.PRIMITIVES:
            return [(op, float(inherited_weight))]
        if op == "primitive":
            name = str(node.get("name", ""))
            if name not in cls.PRIMITIVES:
                raise ValueError(f"unsupported AST primitive: {name}")
            return [(name, float(inherited_weight))]
        if op == "scale":
            factor = float(node.get("weight", node.get("scale", 1.0)))
            if not math.isfinite(factor) or factor < 0:
                raise ValueError("AST scale must be a finite nonnegative number")
            return cls.flatten(node.get("child", {}), inherited_weight * factor, depth + 1, counter)
        if op in {"weighted_sum", "ensemble", "add"}:
            terms = node.get("terms", [])
            # Normalize the equivalent model-friendly representation
            # {"children": [...], "weights": [...]} to the internal terms
            # protocol before recursively flattening it.
            if not terms and isinstance(node.get("children"), list):
                children = node["children"]
                weights = node.get("weights", [])
                if not isinstance(weights, list):
                    weights = []
                terms = []
                for index, child in enumerate(children):
                    if isinstance(child, str):
                        child = {"op": child}
                    if not isinstance(child, Mapping):
                        raise ValueError("AST children must be objects or primitive names")
                    weight = weights[index] if index < len(weights) else 1.0
                    terms.append({"op": "scale", "weight": weight, "child": child})
            if not isinstance(terms, list) or not terms:
                raise ValueError("AST weighted_sum requires a non-empty terms list")
            output: list[tuple[str, float]] = []
            for term in terms:
                if not isinstance(term, Mapping):
                    raise ValueError("AST terms must be objects")
                # A scale node owns its factor; applying the term weight a
                # second time would silently square the intended coefficient.
                weight = 1.0 if term.get("op") == "scale" else float(term.get("weight", 1.0))
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("AST term weights must be finite and nonnegative")
                output.extend(cls.flatten(term, inherited_weight * weight, depth + 1, counter))
            return output
        raise ValueError(f"unsupported AST operation: {op}")

    @classmethod
    def node_count(cls, node: Mapping[str, Any]) -> int:
        if cls.is_vector_expression(node):
            return cls.validate_vector_expression(node)
        counter = [0]
        cls.flatten(node, counter=counter)
        return counter[0]

    @classmethod
    def compile(
        cls,
        ast: Mapping[str, Any],
        operator_id: str = "ast_operator",
        F: float = 0.5,
        CR: float = 0.9,
        crossover: str = "bin",
    ) -> DEOperatorSpec:
        if cls.is_vector_expression(ast):
            cls.validate_vector_expression(ast)
            return DEOperatorSpec(
                operator_id=operator_id,
                mutation="generated_expr",
                crossover=crossover,
                F=F,
                CR=CR,
                ast=dict(ast),
            )
        flattened = cls.flatten(ast)
        components: dict[str, float] = {}
        for name, weight in flattened:
            components[name] = components.get(name, 0.0) + weight
        total = sum(components.values())
        if total <= 0:
            raise ValueError("AST must contain a positive total weight")
        components = {name: value / total for name, value in components.items()}
        mutation = next(iter(components)) if len(components) == 1 else "weighted_ensemble"
        spec = DEOperatorSpec(
            operator_id=operator_id,
            mutation=mutation,
            crossover=crossover,
            F=F,
            CR=CR,
            components={} if mutation != "weighted_ensemble" else components,
            ast=dict(ast),
        )
        spec.validate()
        return spec

    @classmethod
    def from_formula(cls, formula: str) -> Optional[dict[str, Any]]:
        """Translate a tiny arithmetic vector DSL without executing code.

        This recognizer accepts expressions such as ``x + 0.4*(pbest-x)`` and
        maps only references in ``EXPR_REFS`` plus numeric ``+``, ``-``, ``*``
        and vector/scalar ``/``. It never calls ``eval``; unsupported syntax is
        rejected and the primitive-code parser can try next.
        """

        text = str(formula or "").strip()
        if not text:
            return None
        if "=" in text:
            left, right = text.split("=", 1)
            if left.strip().lower() in {"v", "vector", "trial", "x_new"}:
                text = right.strip()
        try:
            tree = python_ast.parse(text, mode="eval").body
        except (SyntaxError, ValueError):
            return None

        def convert(node: python_ast.AST) -> dict[str, Any] | None:
            if isinstance(node, python_ast.Name) and node.id in cls.EXPR_REFS:
                return {"op": "ref", "name": node.id}
            if isinstance(node, python_ast.UnaryOp) and isinstance(node.op, python_ast.USub):
                child = convert(node.operand)
                return None if child is None else {"op": "scale", "factor": -1.0, "child": child}
            if isinstance(node, python_ast.BinOp):
                if isinstance(node.op, (python_ast.Add, python_ast.Sub)):
                    left = convert(node.left)
                    right = convert(node.right)
                    if left is None or right is None:
                        return None
                    return {
                        "op": "add" if isinstance(node.op, python_ast.Add) else "sub",
                        "left": left,
                        "right": right,
                    }
                if isinstance(node.op, (python_ast.Mult, python_ast.Div)):
                    if isinstance(node.left, python_ast.Constant):
                        factor_node, child_node = node.left, node.right
                        if not isinstance(factor_node.value, (int, float)):
                            return None
                        factor = float(factor_node.value)
                        if isinstance(node.op, python_ast.Div):
                            return None
                    elif isinstance(node.right, python_ast.Constant):
                        factor_node, child_node = node.right, node.left
                        if not isinstance(factor_node.value, (int, float)):
                            return None
                        factor = float(factor_node.value)
                        if isinstance(node.op, python_ast.Div):
                            if factor == 0.0:
                                return None
                            factor = 1.0 / factor
                    else:
                        return None
                    child = convert(child_node)
                    return None if child is None else {"op": "scale", "factor": factor, "child": child}
            return None

        expression = convert(tree)
        if expression is None:
            return None
        ast = {"op": "vector_expr", "expr": expression}
        try:
            cls.validate_vector_expression(ast)
        except (TypeError, ValueError, OverflowError):
            return None
        return ast


class OperatorCostModel:
    """Deterministic proxy for operator structure and tensor arithmetic cost."""

    @staticmethod
    def estimate(spec: DEOperatorSpec, pop_size: int, dim: int) -> float:
        primitive_terms = len(spec.components) if spec.mutation == "weighted_ensemble" else 1
        # The proxy is proportional to vector arithmetic and remains cheap to
        # compute during prompt-time screening.
        return float(max(1, pop_size) * max(1, dim) * (2.0 + primitive_terms) + spec.complexity())


class DEOperatorLibrary:
    """Tensorized DE primitives plus bounded LLM-created compositions."""

    def __init__(self, operators: Optional[list[DEOperatorSpec]] = None):
        self._operators: dict[str, DEOperatorSpec] = {}
        defaults = operators or [
            DEOperatorSpec("rand1", "rand1", F=0.5, CR=0.9),
            DEOperatorSpec("best1", "best1", F=0.5, CR=0.9),
            DEOperatorSpec("current_to_best1", "current_to_best1", F=0.5, CR=0.9),
            DEOperatorSpec("current_to_rand1", "current_to_rand1", F=0.5, CR=0.9),
        ]
        for spec in defaults:
            self.add(spec)

    def add(self, spec: DEOperatorSpec) -> None:
        spec.validate()
        self._operators[spec.operator_id] = spec

    def get(self, operator_id: str) -> DEOperatorSpec:
        return self._operators[operator_id]

    def all(self) -> list[DEOperatorSpec]:
        return list(self._operators.values())

    @staticmethod
    def _sample_pbest_references(
        populations: torch.Tensor,
        fitnesses: torch.Tensor,
        p_fraction: float = 0.2,
    ) -> torch.Tensor:
        """Sample p-best references from each current population.

        ``current_to_pbest1`` is defined against the best fraction of the
        *current* population.  Historical elite archives remain available to
        ``elite1`` and the generated DSL, but using them as p-best references
        can pull a low-budget run toward stale points from another phase or
        another task.
        """

        if populations.ndim != 3 or fitnesses.ndim < 2:
            raise ValueError("p-best sampling expects [batch, population, dimension] tensors")
        batch, pop_size, dim = populations.shape
        scalar = fitnesses
        if scalar.ndim > 2:
            scalar = scalar.reshape(batch, pop_size, -1).sum(dim=2)
        p_count = max(2, int(math.ceil(max(0.0, float(p_fraction)) * pop_size)))
        p_count = min(pop_size, p_count)
        order = torch.argsort(scalar, dim=1)[:, :p_count]
        pool = torch.gather(
            populations,
            1,
            order[:, :, None].expand(-1, -1, dim),
        )
        sample_index = _device_randint(0, p_count, (batch, pop_size), populations.device)
        return torch.gather(
            pool,
            1,
            sample_index[:, :, None].expand(-1, -1, dim),
        )

    @staticmethod
    def _sample_parent_indices(pop_size: int, device: torch.device) -> torch.Tensor:
        if pop_size < 4:
            raise ValueError("DE requires a population of at least four individuals")
        # Draw one random key per possible population member, mask the current
        # vector, then select the four smallest keys. The old implementation
        # sorted only four keys per row, so its indices were always 0..3 and
        # every DE difference came from the first four individuals. ``topk``
        # keeps the operation tensorized on CUDA and returns distinct parents;
        # for pop_size == 4 the fourth parent is necessarily a repeated one.
        parent_count = min(4, pop_size - 1)
        keys = _device_rand((pop_size, pop_size), device)
        current = torch.arange(pop_size, device=device)
        keys[current, current] = torch.inf
        parents = torch.topk(
            keys, k=parent_count, dim=1, largest=False, sorted=True
        ).indices
        if parent_count < 4:
            parents = torch.cat(
                [parents, parents[:, :1].expand(-1, 4 - parent_count)], dim=1
            )
        return parents

    @staticmethod
    def sample_shared_parent_indices(
        populations: torch.Tensor,
        fitnesses: torch.Tensor,
        cross_island_rate: float = 0.35,
    ) -> torch.Tensor:
        """Sample DE parents from a shared multi-island mating pool.

        The returned tensor contains ``(island, individual)`` pairs and has
        shape ``[I, N, 4, 2]``.  A target vector is always excluded.  The
        global pool is important when each island is small: with three
        populations of ten, a local four-parent draw wastes most of the
        evaluated information and makes the islands behave like three weak
        independent optimizers.
        """

        if populations.ndim != 3:
            raise ValueError("populations must have shape [islands, population, dimension]")
        islands, pop_size, _ = populations.shape
        if islands < 2 or pop_size < 4:
            raise ValueError("shared mating requires at least two islands of size four")
        if fitnesses.shape[0] != islands or fitnesses.shape[1] != pop_size:
            raise ValueError("fitnesses must match populations in the first two dimensions")
        rate = max(0.0, min(1.0, float(cross_island_rate)))
        device = populations.device
        total = islands * pop_size
        keys = _device_rand((islands, pop_size, total), device)
        target_island = torch.arange(islands, device=device)[:, None, None]
        target_individual = torch.arange(pop_size, device=device)[None, :, None]
        target_global = target_island * pop_size + target_individual
        keys.scatter_(2, target_global.expand(-1, -1, -1), torch.inf)
        global_index = torch.topk(keys, k=4, dim=2, largest=False, sorted=True).indices
        global_pairs = torch.stack(
            [global_index // pop_size, global_index % pop_size], dim=-1
        )

        # Keep local mating available for the exploration island; use a global
        # draw with the configured probability for every target/parent slot.
        local_keys = _device_rand((islands, pop_size, pop_size), device)
        local_keys.scatter_(
            2,
            torch.arange(pop_size, device=device)[None, :, None].expand(islands, -1, -1),
            torch.inf,
        )
        local_index = torch.topk(
            local_keys, k=min(4, pop_size - 1), dim=2, largest=False, sorted=True
        ).indices
        if pop_size - 1 < 4:
            local_index = torch.cat(
                [local_index, local_index[:, :, :1].expand(-1, -1, 4 - (pop_size - 1))],
                dim=2,
            )
        local_pairs = torch.stack(
            [torch.arange(islands, device=device)[:, None, None].expand(-1, pop_size, 4), local_index],
            dim=-1,
        )
        use_global = _device_rand((islands, pop_size, 4, 1), device) < rate
        return torch.where(use_global, global_pairs, local_pairs)

    @staticmethod
    def _bias_parent_indices(
        parents: torch.Tensor,
        scalar_fitness: torch.Tensor,
        mode: str,
        pbest_fraction: float,
        archive_probability: float,
    ) -> torch.Tensor:
        """Apply a policy bias while preserving self-exclusion and distinctness."""

        if mode == "uniform_distinct":
            return parents
        if scalar_fitness.ndim != 1:
            scalar_fitness = scalar_fitness.reshape(-1)
        pop_size = int(scalar_fitness.shape[0])
        p_count = min(pop_size, max(2, int(math.ceil(pbest_fraction * pop_size))))
        pbest_pool = torch.argsort(scalar_fitness)[:p_count]
        batch, population, parent_count = parents.shape
        candidate = pbest_pool[
            _device_randint(0, p_count, (batch, population), parents.device)
        ]
        if mode == "archive_mixed":
            use_bias = _device_rand((batch, population), parents.device) < archive_probability
            candidate = torch.where(use_bias, candidate, parents[:, :, 0])
        current = torch.arange(population, device=parents.device)[None, :]
        collision = candidate == current
        if parent_count > 1:
            collision = collision | (candidate[:, :, None] == parents[:, :, 1:]).any(dim=2)
        parents = parents.clone()
        parents[:, :, 0] = torch.where(collision, parents[:, :, 0], candidate)
        return parents

    @staticmethod
    def _mutation(
        name: str,
        population: torch.Tensor,
        parents: torch.Tensor,
        best: torch.Tensor,
        F: float,
    ) -> torch.Tensor:
        x = population
        r1, r2, r3, r4 = (population[parents[:, i]] for i in range(4))
        if name == "rand1":
            return r1 + F * (r2 - r3)
        if name == "best1":
            return best.unsqueeze(0) + F * (r1 - r2)
        if name == "current_to_best1":
            return x + F * (best.unsqueeze(0) - x) + F * (r1 - r2)
        if name == "current_to_rand1":
            return x + F * (r1 - x) + F * (r2 - r3)
        if name == "best2":
            return best.unsqueeze(0) + F * (r1 - r2) + F * (r3 - r4)
        raise ValueError(f"unsupported mutation: {name}")

    @staticmethod
    def _evaluate_vector_expression(
        node: Mapping[str, Any],
        refs: Mapping[str, torch.Tensor],
        primitives: Mapping[str, torch.Tensor],
        state: Mapping[str, Any],
    ) -> torch.Tensor:
        op = str(node.get("op", ""))
        if op == "ref":
            return refs[str(node["name"])]
        if op in _BUILTIN_MUTATIONS - {"weighted_ensemble", "generated_expr"}:
            return primitives[op]
        if op == "primitive":
            return primitives[str(node["name"])]
        if op == "scale":
            factor = float(node.get("factor", node.get("weight", 1.0)))
            return factor * DEOperatorLibrary._evaluate_vector_expression(
                node["child"], refs, primitives, state
            )
        if op == "add":
            return DEOperatorLibrary._evaluate_vector_expression(node["left"], refs, primitives, state) + DEOperatorLibrary._evaluate_vector_expression(node["right"], refs, primitives, state)
        if op == "sub":
            return DEOperatorLibrary._evaluate_vector_expression(node["left"], refs, primitives, state) - DEOperatorLibrary._evaluate_vector_expression(node["right"], refs, primitives, state)
        if op == "weighted_sum":
            result = torch.zeros_like(next(iter(refs.values())))
            total = 0.0
            for term in node["terms"]:
                weight = float(term.get("weight", 1.0))
                child = term.get("child", term)
                result = result + weight * DEOperatorLibrary._evaluate_vector_expression(child, refs, primitives, state)
                total += weight
            return result / max(total, 1e-12)
        if op == "clip":
            child = DEOperatorLibrary._evaluate_vector_expression(node["child"], refs, primitives, state)
            return child.clamp(float(node.get("low", -1.0)), float(node.get("high", 1.0)))
        if op == "state_scale":
            if "feature" not in node:
                factor = float(node.get("factor", node.get("base", 1.0)))
                return factor * DEOperatorLibrary._evaluate_vector_expression(
                    node["child"], refs, primitives, state
                )
            feature = str(node["feature"])
            value = float(state.get(feature, 0.0))
            if feature == "stagnation":
                value = min(1.0, max(0.0, value / 10.0))
            elif feature == "remaining_budget":
                value = value / max(1.0, float(state.get("initial_budget", value or 1.0)))
                value = min(1.0, max(0.0, value))
            else:
                value = min(1.0, max(0.0, value))
            factor = float(node.get("base", 1.0)) + float(node.get("gain", 0.0)) * value
            return factor * DEOperatorLibrary._evaluate_vector_expression(node["child"], refs, primitives, state)
        raise ValueError(f"unsupported generated expression operation: {op}")

    def apply_batch(
        self,
        specs: list[DEOperatorSpec],
        population: torch.Tensor,
        fitness: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        state: Optional[Mapping[str, Any]] = None,
        elite_archive: Optional[torch.Tensor] = None,
        history_archive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply several operators in one tensorized GPU batch.

        All candidates share one parent-index tensor. This makes shadow
        evaluation comparable and avoids a Python loop over individuals or
        candidate operators.
        """

        if not specs:
            return torch.empty((0, *population.shape), device=population.device, dtype=population.dtype)
        for spec in specs:
            spec.validate()
        batch = len(specs)
        pop_size, dim = population.shape
        state = state or {}
        scalar_fitness = fitness
        if scalar_fitness.ndim > 1:
            scalar_fitness = scalar_fitness.reshape(scalar_fitness.shape[0], -1).sum(dim=1)
        best = population[torch.argmin(scalar_fitness)]

        parent_count = min(4, pop_size - 1)
        keys = _device_rand((batch, pop_size, pop_size), population.device)
        current = torch.arange(pop_size, device=population.device)[None, :, None]
        keys.scatter_(2, current.expand(batch, -1, -1), torch.inf)
        parents = torch.topk(
            keys, k=parent_count, dim=2, largest=False, sorted=True
        ).indices
        if parent_count < 4:
            parents = torch.cat(
                [parents, parents[:, :, :1].expand(-1, -1, 4 - parent_count)],
                dim=2,
            )
        parents = self._bias_parent_indices(
            parents,
            scalar_fitness,
            str(state.get("parent_sampling_mode", "uniform_distinct")),
            float(state.get("pbest_fraction", 0.2)),
            float(state.get("archive_probability", 0.2)),
        )
        selected = population[parents]
        r1, r2, r3, r4 = (selected[:, :, i, :] for i in range(4))
        x = population[None, :, :]
        best = best[None, None, :]
        parameters = [spec.effective_parameters(state) for spec in specs]
        F = torch.tensor([item[0] for item in parameters], device=population.device, dtype=population.dtype)[:, None, None]
        f_values = state.get("_F_values")
        cr_values = state.get("_CR_values")
        if isinstance(f_values, torch.Tensor) and f_values.numel() == pop_size:
            F = f_values.to(device=population.device, dtype=population.dtype).reshape(1, pop_size, 1)

        def sample_archive(archive: Optional[torch.Tensor]) -> torch.Tensor:
            if archive is None or archive.numel() == 0:
                return best.expand(batch, pop_size, dim)
            archive = archive.to(device=population.device, dtype=population.dtype)
            indexes = _device_randint(0, archive.shape[0], (batch, pop_size), population.device)
            return archive[indexes]

        elite_reference = sample_archive(elite_archive)
        history_reference = sample_archive(history_archive)
        pbest_reference = self._sample_pbest_references(
            population.unsqueeze(0).expand(batch, -1, -1),
            scalar_fitness.unsqueeze(0).expand(batch, -1),
        )

        primitive_stack = torch.stack(
            [
                r1 + F * (r2 - r3),
                best + F * (r1 - r2),
                best + F * (r1 - r2) + F * (r3 - r4),
                x + F * (best - x) + F * (r1 - r2),
                x + F * (r1 - x) + F * (r2 - r3),
                elite_reference + F * (r1 - r2),
                history_reference + F * (r1 - r2),
                x + F * (pbest_reference - x) + F * (r1 - r2),
                best + F * (best - x),
                x + F * (history_reference - x) + F * (r2 - r3),
            ],
            dim=1,
        )
        primitive_names = [
            "rand1",
            "best1",
            "best2",
            "current_to_best1",
            "current_to_rand1",
            "elite1",
            "history1",
            "current_to_pbest1",
            "opposition1",
            "history_guided1",
        ]
        weights = torch.zeros((batch, len(primitive_names)), device=population.device, dtype=population.dtype)
        name_to_index = {name: i for i, name in enumerate(primitive_names)}
        for row, spec in enumerate(specs):
            if spec.mutation == "weighted_ensemble":
                for name, weight in spec.components.items():
                    weights[row, name_to_index[name]] = float(weight)
            elif spec.mutation != "generated_expr":
                weights[row, name_to_index[spec.mutation]] = 1.0
        mutant = torch.einsum("kp,kpnd->knd", weights, primitive_stack)
        for row, spec in enumerate(specs):
            if spec.mutation != "generated_expr":
                continue
            primitive_values = {
                name: primitive_stack[row, index]
                for name, index in name_to_index.items()
            }
            refs = {
                "x": x[0],
                "best": best[0].expand(pop_size, dim),
                "pbest": pbest_reference[row],
                "elite": elite_reference[row],
                "history": history_reference[row],
                "r1": r1[row],
                "r2": r2[row],
                "r3": r3[row],
                "r4": r4[row],
            }
            generated = self._evaluate_vector_expression(
                spec.ast["expr"], refs, primitive_values, state
            )
            mutant[row] = torch.nan_to_num(generated)

        CR = torch.tensor([item[1] for item in parameters], device=population.device, dtype=population.dtype)[:, None, None]
        if isinstance(cr_values, torch.Tensor) and cr_values.numel() == pop_size:
            CR = cr_values.to(device=population.device, dtype=population.dtype).reshape(1, pop_size, 1)
        binomial = _device_rand((batch, pop_size, dim), population.device) < CR
        forced = _device_randint(0, dim, (batch, pop_size), population.device)
        binomial.scatter_(2, forced[:, :, None], True)
        exp_length = torch.ceil(
            torch.log(_device_rand((batch, pop_size), population.device).clamp_min(1e-7))
            / torch.log(CR[:, 0, 0].clamp_min(1e-7))[:, None]
        ).clamp(1, dim).to(torch.long)
        columns = torch.arange(dim, device=population.device)[None, None, :]
        start = _device_randint(0, dim, (batch, pop_size), population.device)
        exponential = ((columns - start[:, :, None]) % dim) < exp_length[:, :, None]
        is_exp = torch.tensor([spec.crossover == "exp" for spec in specs], device=population.device)[:, None, None]
        mask = torch.where(is_exp, exponential, binomial)
        trial = torch.where(mask, mutant, x)
        return torch.maximum(torch.minimum(trial, ub[None, None, :]), lb[None, None, :])

    def apply_islands(
        self,
        specs: list[DEOperatorSpec],
        populations: torch.Tensor,
        fitnesses: torch.Tensor,
        lbs: torch.Tensor,
        ubs: torch.Tensor,
        states: Optional[list[Mapping[str, Any]]] = None,
        elite_archives: Optional[list[Optional[torch.Tensor]]] = None,
        history_archives: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Apply one operator per equally sized island in one GPU batch.

        ``populations`` has shape ``[islands, population, dimension]``.  The
        parent keys and crossover masks are generated for all islands in one
        tensor operation, while each island keeps its own best point, bounds,
        state-conditioned parameters, and archives.  The caller can group
        islands with different population sizes before calling this method.
        """

        if populations.ndim != 3:
            raise ValueError("island populations must have shape [I, N, D]")
        batch, pop_size, dim = populations.shape
        if len(specs) != batch:
            raise ValueError("one operator specification is required per island")
        if fitnesses.shape[0] != batch or fitnesses.shape[1] != pop_size:
            raise ValueError("island fitnesses must have shape [I, N, ...]")
        if lbs.shape != (batch, dim) or ubs.shape != (batch, dim):
            raise ValueError("island bounds must have shape [I, D]")
        for spec in specs:
            spec.validate()

        states = states or [{} for _ in range(batch)]
        if len(states) != batch:
            raise ValueError("one state mapping is required per island")
        elite_archives = elite_archives or [None for _ in range(batch)]
        history_archives = history_archives or [None for _ in range(batch)]
        if len(elite_archives) != batch or len(history_archives) != batch:
            raise ValueError("one archive entry is required per island")

        scalar_fitness = fitnesses
        if scalar_fitness.ndim > 2:
            scalar_fitness = scalar_fitness.reshape(batch, pop_size, -1).sum(dim=2)
        best_indices = torch.argmin(scalar_fitness, dim=1)
        island_indices = torch.arange(batch, device=populations.device)
        best = populations[island_indices, best_indices]

        # Sample four distinct local parents first.  The shared-mating draw
        # below replaces a configurable fraction with parents from the other
        # islands, so the three small islands exchange evaluated directions on
        # every generation instead of waiting for a single elite migration.
        parent_count = min(4, pop_size - 1)
        keys = _device_rand((batch, pop_size, pop_size), populations.device)
        current = torch.arange(pop_size, device=populations.device)[None, :, None]
        keys.scatter_(2, current.expand(batch, -1, -1), torch.inf)
        parents = torch.topk(
            keys, k=parent_count, dim=2, largest=False, sorted=True
        ).indices
        if parent_count < 4:
            parents = torch.cat(
                [parents, parents[:, :, :1].expand(-1, -1, 4 - parent_count)],
                dim=2,
            )
        for row, policy_state in enumerate(states):
            parents[row : row + 1] = self._bias_parent_indices(
                parents[row : row + 1],
                scalar_fitness[row],
                str(policy_state.get("parent_sampling_mode", "uniform_distinct")),
                float(policy_state.get("pbest_fraction", 0.2)),
                float(policy_state.get("archive_probability", 0.2)),
            )
        shared_rate = max(
            0.0,
            min(
                1.0,
                sum(float(state.get("cross_island_rate", 0.35)) for state in states)
                / max(1, len(states)),
            ),
        )
        local_pairs = torch.stack(
            [
                torch.arange(batch, device=populations.device)[:, None, None].expand(-1, pop_size, 4),
                parents,
            ],
            dim=-1,
        )
        # ``IslandCoordinator`` groups islands by population size before
        # calling this method.  When the initial FE budget is not divisible
        # by the island count (for example sizes 34, 33, 33), one group can
        # contain a single island.  Shared mating is undefined for that
        # group, so retain the already sampled local parents and continue with
        # the same tensorized DE path.
        if batch >= 2:
            shared_pairs = self.sample_shared_parent_indices(
                populations, scalar_fitness, cross_island_rate=shared_rate
            )
        else:
            shared_pairs = local_pairs
            shared_rate = 0.0
        # For shared_rate=0 the local draw is retained exactly.  At nonzero
        # rates, use the same sampled mask as the helper so the operation is
        # still a single indexed tensor gather on CUDA.
        shared_choice = _device_rand(
            (batch, pop_size, 4, 1), populations.device
        ) < shared_rate
        parent_pairs = torch.where(shared_choice, shared_pairs, local_pairs)
        selected = populations[
            parent_pairs[..., 0], parent_pairs[..., 1]
        ]
        r1, r2, r3, r4 = (selected[:, :, i, :] for i in range(4))
        x = populations
        global_best = populations.reshape(batch * pop_size, dim)[
            torch.argmin(scalar_fitness.reshape(-1))
        ]
        guidance = torch.tensor(
            [
                max(0.0, min(1.0, float(state.get("global_guidance", 0.25))))
                for state in states
            ],
            device=populations.device,
            dtype=populations.dtype,
        )[:, None]
        best = ((1.0 - guidance) * best + guidance * global_best[None, :])[:, None, :]

        parameters = [spec.effective_parameters(state) for spec, state in zip(specs, states)]
        F = torch.tensor(
            [item[0] for item in parameters],
            device=populations.device,
            dtype=populations.dtype,
        )[:, None, None]
        adaptive_f: list[torch.Tensor] = []
        adaptive_cr: list[torch.Tensor] = []
        for row, state in enumerate(states):
            f_values = state.get("_F_values")
            cr_values = state.get("_CR_values")
            if isinstance(f_values, torch.Tensor) and f_values.numel() == pop_size:
                adaptive_f.append(f_values.to(device=populations.device, dtype=populations.dtype).reshape(pop_size))
            else:
                adaptive_f.append(F[row, 0, 0].expand(pop_size))
            if isinstance(cr_values, torch.Tensor) and cr_values.numel() == pop_size:
                adaptive_cr.append(cr_values.to(device=populations.device, dtype=populations.dtype).reshape(pop_size))
            else:
                adaptive_cr.append(torch.tensor(parameters[row][1], device=populations.device, dtype=populations.dtype).expand(pop_size))
        F = torch.stack(adaptive_f, dim=0)[:, :, None]

        def sample_archives(archives: list[Optional[torch.Tensor]]) -> torch.Tensor:
            references: list[torch.Tensor] = []
            for index, archive in enumerate(archives):
                if archive is None or archive.numel() == 0:
                    references.append(best[index].expand(pop_size, dim))
                    continue
                values = archive.to(device=populations.device, dtype=populations.dtype)
                sample_index = _device_randint(
                    0, values.shape[0], (pop_size,), populations.device
                )
                references.append(values[sample_index])
            return torch.stack(references, dim=0)

        elite_reference = sample_archives(elite_archives)
        history_reference = sample_archives(history_archives)
        pbest_reference = self._sample_pbest_references(
            populations,
            scalar_fitness,
        )
        primitive_stack = torch.stack(
            [
                r1 + F * (r2 - r3),
                best + F * (r1 - r2),
                best + F * (r1 - r2) + F * (r3 - r4),
                x + F * (best - x) + F * (r1 - r2),
                x + F * (r1 - x) + F * (r2 - r3),
                elite_reference + F * (r1 - r2),
                history_reference + F * (r1 - r2),
                x + F * (pbest_reference - x) + F * (r1 - r2),
                best + F * (best - x),
                x + F * (history_reference - x) + F * (r2 - r3),
            ],
            dim=1,
        )
        primitive_names = [
            "rand1",
            "best1",
            "best2",
            "current_to_best1",
            "current_to_rand1",
            "elite1",
            "history1",
            "current_to_pbest1",
            "opposition1",
            "history_guided1",
        ]
        weights = torch.zeros(
            (batch, len(primitive_names)),
            device=populations.device,
            dtype=populations.dtype,
        )
        name_to_index = {name: index for index, name in enumerate(primitive_names)}
        for row, spec in enumerate(specs):
            if spec.mutation == "weighted_ensemble":
                for name, weight in spec.components.items():
                    weights[row, name_to_index[name]] = float(weight)
            elif spec.mutation != "generated_expr":
                weights[row, name_to_index[spec.mutation]] = 1.0
        mutant = torch.einsum("bp,bpnd->bnd", weights, primitive_stack)
        for row, spec in enumerate(specs):
            if spec.mutation != "generated_expr":
                continue
            primitive_values = {
                name: primitive_stack[row, index]
                for name, index in name_to_index.items()
            }
            refs = {
                "x": x[row],
                "best": best[row].expand(pop_size, dim),
                "pbest": pbest_reference[row],
                "elite": elite_reference[row],
                "history": history_reference[row],
                "r1": r1[row],
                "r2": r2[row],
                "r3": r3[row],
                "r4": r4[row],
            }
            generated = self._evaluate_vector_expression(
                spec.ast["expr"], refs, primitive_values, states[row]
            )
            mutant[row] = torch.nan_to_num(generated)

        CR = torch.stack(adaptive_cr, dim=0)[:, :, None]
        binomial = _device_rand((batch, pop_size, dim), populations.device) < CR
        forced = _device_randint(0, dim, (batch, pop_size), populations.device)
        binomial.scatter_(2, forced[:, :, None], True)
        exp_length = torch.ceil(
            torch.log(_device_rand((batch, pop_size), populations.device).clamp_min(1e-7))
            / torch.log(CR[:, 0, 0].clamp_min(1e-7))[:, None]
        ).clamp(1, dim).to(torch.long)
        columns = torch.arange(dim, device=populations.device)[None, None, :]
        start = _device_randint(0, dim, (batch, pop_size), populations.device)
        exponential = ((columns - start[:, :, None]) % dim) < exp_length[:, :, None]
        is_exp = torch.tensor(
            [spec.crossover == "exp" for spec in specs],
            device=populations.device,
        )[:, None, None]
        mask = torch.where(is_exp, exponential, binomial)
        trial = torch.where(mask, mutant, x)
        return torch.maximum(
            torch.minimum(trial, ubs[:, None, :]), lbs[:, None, :]
        )

    def apply(
        self,
        spec: DEOperatorSpec,
        population: torch.Tensor,
        fitness: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        state: Optional[Mapping[str, Any]] = None,
        elite_archive: Optional[torch.Tensor] = None,
        history_archive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one operator to a whole population on its current device."""

        return self.apply_batch(
            [spec], population, fitness, lb, ub,
            state=state,
            elite_archive=elite_archive,
            history_archive=history_archive,
        )[0]


class OperatorArchive:
    """Small in-memory RAG store for generated operators and their outcomes."""

    def __init__(self, max_size: int = 128, storage_path: str | os.PathLike[str] | None = None):
        self.max_size = max(1, int(max_size))
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.records: list[dict[str, Any]] = []
        self.last_persist_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self.records = [item for item in payload if isinstance(item, dict)][-self.max_size :]
        except (OSError, json.JSONDecodeError):
            self.records = []

    def _persist(self) -> None:
        if self.storage_path is None:
            return
        temporary = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        payload = json.dumps(self.records, ensure_ascii=False, default=str, indent=2)
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(payload, encoding="utf-8")
            try:
                os.replace(temporary, self.storage_path)
            except PermissionError:
                # Windows can briefly deny replacement when an editor, sync
                # service, or an earlier experiment still has the destination
                # open.  A direct overwrite preserves the knowledge record and
                # keeps optional RAG persistence from aborting optimization.
                self.storage_path.write_text(payload, encoding="utf-8")
                temporary.unlink(missing_ok=True)
            self.last_persist_error = None
        except OSError as exc:
            self.last_persist_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)

    def add(
        self,
        spec: DEOperatorSpec,
        state: Mapping[str, Any],
        score: float = 0.0,
        task_id: Optional[str] = None,
    ) -> None:
        record = {
            "operator": spec.to_dict(),
            "state": dict(state),
            "score": float(score),
            "task_id": task_id or state.get("task_id", "default"),
            "complexity": spec.complexity(),
        }
        self.records.append(record)
        self.records.sort(key=lambda x: x["score"], reverse=True)
        del self.records[self.max_size :]
        self._persist()

    def retrieve(self, state: Mapping[str, Any], top_k: int = 4) -> list[dict[str, Any]]:
        # A small state-aware score is enough for the first RAG implementation.
        # The archive interface intentionally leaves room for a vector index.
        numeric_keys = ("diversity", "stagnation", "active_operator_success")
        query_task = str(state.get("task_id", "") or "")

        def task_affinity(record: Mapping[str, Any]) -> float:
            """Prefer same-task evidence without discarding transfer cases."""

            if not query_task:
                return 0.0
            record_task = str(
                record.get("task_id", record.get("state", {}).get("task_id", ""))
                or ""
            )
            if not record_task:
                return -0.05
            if record_task == query_task:
                return 0.20
            query_problem = query_task.split(":", 1)[0].lower()
            record_problem = record_task.split(":", 1)[0].lower()
            if query_problem and query_problem == record_problem:
                return 0.08
            return -0.08

        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in self.records:
            distance = 0.0
            for key in numeric_keys:
                current = float(state.get(key, 0.0))
                previous = float(record["state"].get(key, 0.0))
                distance += abs(current - previous) / (1.0 + abs(current))
            ranked.append((float(record["score"]) - distance + task_affinity(record), record))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in ranked[: max(0, int(top_k))]]

    def update_score(self, operator_id: str, increment: float) -> None:
        for record in self.records:
            if record["operator"].get("operator_id") == operator_id:
                record["score"] = float(record["score"]) + float(increment)
        self.records.sort(key=lambda x: x["score"], reverse=True)
        self._persist()

    def context(self, state: Mapping[str, Any], top_k: int = 4) -> str:
        return json.dumps(self.retrieve(state, top_k), ensure_ascii=False, default=str)


def _scalarize_objectives(fitness: torch.Tensor, constraints: Optional[torch.Tensor] = None, penalty: float = 1e6) -> torch.Tensor:
    values = fitness.reshape(fitness.shape[0], -1).sum(dim=1) if fitness.ndim > 1 else fitness
    if constraints is not None:
        violation = torch.relu(constraints).reshape(constraints.shape[0], -1).sum(dim=1)
        values = values + penalty * violation
    return values


def constrained_pareto_better(
    new_fitness: torch.Tensor,
    old_fitness: torch.Tensor,
    new_constraints: Optional[torch.Tensor] = None,
    old_constraints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-individual constrained Pareto improvement masks."""

    new_values = new_fitness.reshape(new_fitness.shape[0], -1)
    old_values = old_fitness.reshape(old_fitness.shape[0], -1)
    new_cv = (
        torch.relu(new_constraints).reshape(new_constraints.shape[0], -1).sum(dim=1)
        if new_constraints is not None
        else torch.zeros(new_values.shape[0], device=new_values.device, dtype=new_values.dtype)
    )
    old_cv = (
        torch.relu(old_constraints).reshape(old_constraints.shape[0], -1).sum(dim=1)
        if old_constraints is not None
        else torch.zeros(old_values.shape[0], device=old_values.device, dtype=old_values.dtype)
    )
    lower_violation = new_cv < old_cv
    equal_violation = torch.isclose(new_cv, old_cv, rtol=1e-6, atol=1e-8)
    dominates = (new_values <= old_values).all(dim=1) & (new_values < old_values).any(dim=1)
    return lower_violation | (equal_violation & dominates)


@dataclass
class ShadowEvaluation:
    trials: torch.Tensor
    scores: torch.Tensor
    costs: torch.Tensor
    used_evaluations: int
    latency: float


class ShadowEvaluator:
    """Evaluate many operator proposals in one batched tensor call."""

    def __init__(self, library: DEOperatorLibrary, constraint_penalty: float = 1e6):
        self.library = library
        self.constraint_penalty = float(constraint_penalty)

    def evaluate(
        self,
        specs: list[DEOperatorSpec],
        population: torch.Tensor,
        fitness: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        objective: Any,
        state: Optional[Mapping[str, Any]] = None,
        elite_archive: Optional[torch.Tensor] = None,
        history_archive: Optional[torch.Tensor] = None,
    ) -> ShadowEvaluation:
        started = time.perf_counter()
        trials = self.library.apply_batch(
            specs, population, fitness, lb, ub,
            state=state,
            elite_archive=elite_archive,
            history_archive=history_archive,
        )
        batch, pop_size, dim = trials.shape
        raw = objective(trials.reshape(batch * pop_size, dim))
        if isinstance(raw, tuple):
            raw_fit, raw_cons = raw
        else:
            raw_fit, raw_cons = raw, None
        candidate_scores = _scalarize_objectives(raw_fit, raw_cons, self.constraint_penalty).reshape(batch, pop_size)
        baseline = _scalarize_objectives(fitness).reshape(1, pop_size)
        gains = torch.relu(baseline - candidate_scores).mean(dim=1)
        costs = torch.tensor(
            [OperatorCostModel.estimate(spec, pop_size, dim) for spec in specs],
            device=population.device,
            dtype=population.dtype,
        )
        return ShadowEvaluation(
            trials=trials,
            scores=gains,
            costs=costs,
            used_evaluations=batch * pop_size,
            latency=time.perf_counter() - started,
        )


@dataclass
class PromotionResult:
    accepted: bool
    operator_id: Optional[str]
    score: float
    used_evaluations: int
    diagnostics: list[dict[str, float]]


class OperatorPromoter:
    """Promote a generated operator only after a bounded real-FE trial."""

    def __init__(
        self,
        library: DEOperatorLibrary,
        evaluation_budget: int,
        complexity_penalty: float = 0.0,
        cost_penalty: float = 0.0,
        min_improvement: float = 0.0,
        constraint_penalty: float = 1e6,
    ):
        self.library = library
        self.evaluation_budget = max(0, int(evaluation_budget))
        self.complexity_penalty = float(complexity_penalty)
        self.cost_penalty = float(cost_penalty)
        self.min_improvement = float(min_improvement)
        self.constraint_penalty = float(constraint_penalty)
        self.used_evaluations = 0

    def promote(
        self,
        specs: list[DEOperatorSpec],
        population: torch.Tensor,
        fitness: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        objective: Any,
        state: Optional[Mapping[str, Any]] = None,
        elite_archive: Optional[torch.Tensor] = None,
        history_archive: Optional[torch.Tensor] = None,
    ) -> PromotionResult:
        remaining = self.evaluation_budget - self.used_evaluations
        if not specs or remaining <= 0:
            return PromotionResult(False, None, float("-inf"), 0, [])
        specs = specs[:remaining]
        per_spec = max(1, min(population.size(0), remaining // len(specs)))
        specs = specs[: max(1, min(len(specs), remaining // per_spec))]
        trials = self.library.apply_batch(
            specs, population, fitness, lb, ub,
            state=state,
            elite_archive=elite_archive,
            history_archive=history_archive,
        )[:, :per_spec]
        raw = objective(trials.reshape(-1, population.size(1)))
        if isinstance(raw, tuple):
            raw_fit, raw_cons = raw
        else:
            raw_fit, raw_cons = raw, None
        candidate_scores = _scalarize_objectives(raw_fit, raw_cons, self.constraint_penalty).reshape(len(specs), per_spec)
        baseline = _scalarize_objectives(fitness)[:per_spec]
        gains = torch.relu(baseline[None, :] - candidate_scores).mean(dim=1)
        diagnostics: list[dict[str, float]] = []
        for index, spec in enumerate(specs):
            complexity = spec.complexity()
            cost = OperatorCostModel.estimate(spec, per_spec, population.size(1))
            score = float(gains[index].detach().cpu()) - self.complexity_penalty * complexity - self.cost_penalty * cost
            diagnostics.append({"score": score, "complexity": complexity, "cost": cost})
        best_index = max(range(len(specs)), key=lambda index: diagnostics[index]["score"])
        best_score = diagnostics[best_index]["score"]
        used = len(specs) * per_spec
        self.used_evaluations += used
        accepted = best_score >= self.min_improvement
        return PromotionResult(
            accepted=accepted,
            operator_id=specs[best_index].operator_id if accepted else None,
            score=best_score,
            used_evaluations=used,
            diagnostics=diagnostics,
        )


class LLMOperatorGenerator:
    """OpenAI-compatible adapter that proposes a bounded DE operator spec."""

    OPERATOR_CATALOG = OPERATOR_CATALOG
    SYSTEM_PROMPT = OPERATOR_SYNTHESIS_SYSTEM_PROMPT

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_tokens: int = 256,
        max_prompt_chars: int = 8000,
        json_mode: bool = False,
        thinking: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        response_log_path: str | os.PathLike[str] | None = None,
        response_log_max_chars: int = 200000,
    ):
        self.base_url = base_url or os.getenv("EVOX_LLM_BASE_URL")
        self.api_key = api_key or os.getenv("EVOX_LLM_API_KEY")
        self.model = model or os.getenv("EVOX_LLM_MODEL")
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.max_tokens = max(64, int(max_tokens))
        self.max_prompt_chars = max(512, int(max_prompt_chars))
        self.json_mode = bool(json_mode)
        thinking_value = None if thinking is None else str(thinking).strip().lower()
        self.thinking = thinking_value if thinking_value in {"auto", "enabled", "disabled"} else None
        effort_value = None if reasoning_effort is None else str(reasoning_effort).strip().lower()
        self.reasoning_effort = (
            effort_value
            if effort_value in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
            else None
        )
        self.response_log_path = (
            Path(response_log_path) if response_log_path is not None else None
        )
        self.response_log_max_chars = max(4096, int(response_log_max_chars))
        self.response_log_error: Optional[str] = None
        self.last_error: Optional[str] = None
        self.successful_generations = 0
        self.failed_generations = 0
        self.last_response_chars = 0
        self.last_response_complete = False
        self.last_response_preview = ""
        self.last_finish_reason = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @classmethod
    def from_config(cls, config_path: str | os.PathLike[str]) -> "LLMOperatorGenerator":
        """Load the ``openai``-compatible provider block from a local file.

        PyYAML is kept optional so EvoX can still be imported without the LLM
        integration installed. The algorithm itself does not require a network
        client; a missing or invalid configuration simply leaves generation
        disabled when the caller chooses to handle the exception.
        """

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("PyYAML is required to read an LLM config file") from exc
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        openai_config = config.get("openai", {})
        if not isinstance(openai_config, Mapping):
            raise ValueError("config.openai must be a mapping")
        return cls(
            base_url=openai_config.get("url"),
            api_key=openai_config.get("api_key"),
            model=openai_config.get("model"),
            temperature=openai_config.get("temperature", 0.2),
            timeout=openai_config.get("timeout", 60.0),
            max_tokens=openai_config.get("max_tokens", 256),
            max_prompt_chars=openai_config.get("max_prompt_chars", 8000),
            json_mode=openai_config.get("json_mode", False),
            thinking=openai_config.get("thinking"),
            reasoning_effort=openai_config.get("reasoning_effort"),
            response_log_path=(
                (path.parent / str(openai_config["response_log_path"]))
                if openai_config.get("response_log_path")
                and not Path(str(openai_config["response_log_path"])).is_absolute()
                else openai_config.get("response_log_path")
            ),
            response_log_max_chars=openai_config.get("response_log_max_chars", 200000),
        )

    def _log_response_payload(self, payload: Mapping[str, Any]) -> None:
        """Append a bounded raw API response record without request secrets."""

        if self.response_log_path is None:
            return
        try:
            serialized = json.dumps(payload, ensure_ascii=False, default=str)
            if len(serialized) > self.response_log_max_chars:
                serialized = serialized[: self.response_log_max_chars] + "...[truncated]"
            record = {
                "timestamp": time.time(),
                "model": self.model,
                "response": serialized,
            }
            self.response_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.response_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.response_log_error = None
        except OSError as exc:
            self.response_log_error = f"{type(exc).__name__}: {exc}"

    def build_prompt(self, state: Mapping[str, Any], context: str) -> str:
        context = str(context or "")
        if len(context) > self.max_prompt_chars:
            context = context[: self.max_prompt_chars] + "\n...[RAG context truncated]"
        return build_operator_synthesis_prompt(state, context)

    @staticmethod
    def parse_response(content: str | Mapping[str, Any]) -> DEOperatorSpec:
        if isinstance(content, Mapping):
            if "operator_ast" in content and "ast" not in content:
                content = dict(content)
                content["ast"] = content["operator_ast"]
            return DEOperatorSpec.from_dict(content)
        text = str(content).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        decoder = json.JSONDecoder()
        # Reasoning models often place prose, Markdown, or an invalid draft
        # before the final object. Try every opening brace independently rather
        # than passing the whole first-to-last-brace span to json.loads().
        best_spec: DEOperatorSpec | None = None
        best_priority = -1
        for start, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            if not any(
                key in value
                for key in (
                    "operator_id", "mutation", "operator_ast", "ast", "operator_code",
                    "operator", "expression", "expr", "op", "id", "name",
                )
            ):
                continue
            try:
                spec = DEOperatorSpec.from_dict(value)
            except (TypeError, ValueError):
                continue
            # A reasoning trace may contain a valid nested expression such as
            # {"op":"ref","name":"x"} before the complete operator object.
            # Prefer the complete protocol object so it does not become an
            # identity operator with id="x".
            complete_keys = {
                "operator_id",
                "operator_ast",
                "ast",
                "mutation",
                "operator_code",
                "code",
                "formula",
                "operator",
                "id",
                "expression",
            }
            priority = 2 if any(key in value for key in complete_keys) else 1
            if priority >= best_priority:
                best_spec = spec
                best_priority = priority

        if best_spec is not None:
            return best_spec

        # The transport format may be plain bounded DSL code instead of a
        # complete JSON object.  This keeps the external LLM protocol
        # flexible while still compiling through the same restricted AST
        # parser; arbitrary Python is never executed.
        plain_code: list[str] = []
        for line in text.splitlines():
            candidate = line.strip().strip("`")
            if not re.search(r"(?:^|\b)(?:v|vector|trial|x_new)\s*=", candidate, flags=re.IGNORECASE):
                continue
            candidate = candidate.split("#", 1)[0].strip().rstrip(";")
            if candidate:
                plain_code.append(candidate)
        for candidate in plain_code:
            try:
                digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
                return DEOperatorSpec.from_dict(
                    {
                        "operator_id": f"llm_text_operator_{digest}",
                        "operator_code": candidate,
                    }
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        raise ValueError("LLM response does not contain a valid JSON operator object")

    @staticmethod
    def _extract_response_contents(payload: Mapping[str, Any]) -> list[Any]:
        """Collect possible answer fields from provider response variants."""

        candidates: list[Any] = []

        def add(value: Any) -> None:
            if value is None or value == "":
                return
            if isinstance(value, str) and value in {
                item for item in candidates if isinstance(item, str)
            }:
                return
            candidates.append(value)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError("response choice must be an object")
            message = choice.get("message")
            if isinstance(message, Mapping):
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if isinstance(call, Mapping):
                            function = call.get("function")
                            if isinstance(function, Mapping):
                                add(function.get("arguments"))
                function_call = message.get("function_call")
                if isinstance(function_call, Mapping):
                    add(function_call.get("arguments"))
                content = message.get("content")
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, Mapping):
                            value = part.get("text", part.get("content", ""))
                        else:
                            value = part
                        if value is not None:
                            parts.append(str(value))
                    content = "".join(parts)
                add(content)
                for key in ("reasoning_content", "reasoning", "text"):
                    add(message.get(key))
            for key in ("text", "content", "reasoning_content"):
                add(choice.get(key))
        else:
            for key in ("output_text", "text", "content", "reasoning_content"):
                add(payload.get(key))

        if not candidates:
            raise ValueError("LLM response does not contain text content")
        return candidates

    @classmethod
    def _extract_response_content(cls, payload: Mapping[str, Any]) -> Any:
        """Return the first content candidate for backwards compatibility."""

        return cls._extract_response_contents(payload)[0]

    def generate(self, state: Mapping[str, Any], context: str = "") -> Optional[DEOperatorSpec]:
        if not self.enabled:
            return None
        self.last_finish_reason = ""
        self.last_response_preview = ""
        self.last_response_complete = False
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        request_body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(state, context)},
            ],
        }
        if self.json_mode:
            request_body["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            request_body["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None:
            request_body["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("LLM response payload must be an object")
            self._log_response_payload(payload)
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self.total_tokens += int(
                    usage.get("total_tokens", int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0))
                    or 0
                )
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                self.last_finish_reason = str(choices[0].get("finish_reason", "") or "")
            candidates = self._extract_response_contents(payload)
            parse_error: Exception | None = None
            spec: DEOperatorSpec | None = None
            for content in candidates:
                self.last_response_chars = len(str(content))
                self.last_response_complete = str(content).rstrip().endswith("}")
                compact = re.sub(r"\s+", " ", str(content))
                self.last_response_preview = (
                    compact
                    if len(compact) <= 320
                    else compact[:160] + " ... [middle omitted] ... " + compact[-140:]
                )
                try:
                    spec = self.parse_response(content)
                    break
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    parse_error = exc
            if spec is None:
                if parse_error is not None:
                    raise parse_error
                raise ValueError("LLM response does not contain a valid JSON operator object")
            self.last_response_complete = True
            self.last_error = None
            self.successful_generations += 1
            return spec
        # A provider or proxy can close a chunked response before sending the
        # terminating chunk.  ``http.client`` reports that as ``IncompleteRead``
        # (an ``HTTPException``), which must be treated like any other optional
        # LLM-generation failure so the DE run can continue with its current
        # validated operator.
        except (
            OSError,
            TimeoutError,
            http.client.HTTPException,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if self.last_response_preview and isinstance(exc, ValueError):
                detail += f" | response_preview={self.last_response_preview}"
            if self.last_finish_reason:
                detail += f" | finish_reason={self.last_finish_reason}"
            if self.total_tokens:
                detail += f" | tokens={self.total_tokens}"
            if self.response_log_path is not None:
                detail += f" | response_log={self.response_log_path}"
            self.last_error = detail
            self.failed_generations += 1
            return None


class LLMOperatorDE(Algorithm):
    """EvoX-compatible DE with optional online LLM operator synthesis."""

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        max_gen: int = 100,
        llm_interval: int = 10,
        stagnation_window: int = 10,
        max_llm_calls: int | None = None,
        operator_patience: int = 3,
        operator_generator: Optional[LLMOperatorGenerator] = None,
        operator_library: Optional[DEOperatorLibrary] = None,
        operator_archive: Optional[OperatorArchive] = None,
        knowledge_path: str | os.PathLike[str] | None = None,
        task_id: str = "default",
        pareto_mode: bool = False,
        constraint_penalty: float = 1e6,
        promotion_budget: int = 0,
        promotion_complexity_penalty: float = 0.0,
        promotion_cost_penalty: float = 0.0,
        promotion_min_improvement: float = 0.0,
        shadow_evaluator: Optional[ShadowEvaluator] = None,
        shadow_objective: Any = None,
        algorithm_policy: Optional[Mapping[str, Any]] = None,
        adaptive_parameters: bool = True,
        cross_island_rate: float = 0.35,
        global_guidance: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if pop_size < 4:
            raise ValueError("pop_size must be at least 4 for differential evolution")
        if lb.ndim != 1 or ub.ndim != 1 or lb.shape != ub.shape:
            raise ValueError("lb and ub must be one-dimensional tensors with equal shape")
        if device is None:
            device = lb.device
        self.pop_size = int(pop_size)
        self.dim = int(lb.numel())
        self.max_gen = int(max_gen)
        self.llm_interval = max(1, int(llm_interval))
        self.stagnation_window = max(1, int(stagnation_window))
        self.max_llm_calls = None if max_llm_calls is None else max(0, int(max_llm_calls))
        self.operator_patience = max(1, int(operator_patience))
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.library = operator_library or DEOperatorLibrary()
        self.operator_generator = operator_generator
        archive_path = knowledge_path or os.getenv("EVOX_OPERATOR_KB")
        self.operator_archive = operator_archive or OperatorArchive(storage_path=archive_path)
        self.task_id = str(task_id)
        self.pareto_mode = bool(pareto_mode)
        self.constraint_penalty = float(constraint_penalty)
        self.shadow_evaluator = shadow_evaluator
        self.shadow_objective = shadow_objective
        self.algorithm_policy = AlgorithmPolicy.from_dict(algorithm_policy).to_dict()
        self.adaptive_parameters = bool(adaptive_parameters)
        self.cross_island_rate = max(0.0, min(1.0, float(cross_island_rate)))
        self.global_guidance = max(0.0, min(1.0, float(global_guidance)))
        self.promoter = OperatorPromoter(
            self.library,
            evaluation_budget=promotion_budget,
            complexity_penalty=promotion_complexity_penalty,
            cost_penalty=promotion_cost_penalty,
            min_improvement=promotion_min_improvement,
            constraint_penalty=self.constraint_penalty,
        )
        self.last_shadow_evaluation: Optional[ShadowEvaluation] = None
        self.last_promotion: Optional[PromotionResult] = None
        population = self.lb + _device_rand((self.pop_size, self.dim), device) * (self.ub - self.lb)
        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.best = Mutable(torch.zeros(self.dim, device=device))
        self.best_fit = Mutable(torch.tensor(torch.inf, device=device))
        self.cons: Optional[torch.Tensor] = None
        self.gen = 0
        # Number of individual trial vectors that have actually been sent to
        # the objective.  In generational mode this increases by pop_size;
        # steady-state mode increases it by one, so expensive budgets can be
        # controlled in exact real-evaluation units.
        self.trial_evals = 0
        self.stagnation = 0
        self.active_operator_id = "rand1"
        self.previous_operator_id: Optional[str] = None
        self.operator_failures = 0
        # A stagnation trigger remains active for several generations.  Keep a
        # cooldown so an unavailable or slow LLM is not queried on every one of
        # those generations.
        self.last_llm_generation = -self.llm_interval
        self.llm_attempts = 0
        self.operator_success: dict[str, int] = {}
        self.operator_gain: dict[str, float] = {}
        self.improvement_history: list[float] = []
        self.feasible_history: list[float] = []
        # Success-history adaptation (JADE/SHADE style).  Fixed F/CR values
        # are a poor default for high-dimensional expensive functions; the
        # LLM still chooses the operator and nominal parameters, while the
        # optimizer learns a small, reproducible distribution around them from
        # accepted trials.
        self.mu_f = 0.5
        self.mu_cr = 0.9
        self.learning_rate = 0.1
        self._pending_f: Optional[torch.Tensor] = None
        self._pending_cr: Optional[torch.Tensor] = None
        self.elite_archive = torch.empty((0, self.dim), device=device)
        self.history_archive = torch.empty((0, self.dim), device=device)
        self.initial_budget = max(1, self.max_gen * self.pop_size)
        self._seed_operator_knowledge()

    def _seed_operator_knowledge(self) -> None:
        """Register the known DE primitives as neutral RAG seed cases.

        These records have zero score and ``source=builtin``; they are
        evidence candidates for the prompt, not claimed successful trials.
        Generated operators are added later with their measured score.
        """
        known = {
            str(record.get("operator", {}).get("operator_id"))
            for record in self.operator_archive.records
        }
        for spec in self.library.all():
            if spec.operator_id not in known:
                self.operator_archive.add(
                    spec,
                    {
                        "task_id": self.task_id,
                        "source": "builtin",
                        "diversity": 0.0,
                        "stagnation": 0,
                        "improvement_rate": 0.0,
                    },
                    score=0.0,
                    task_id=self.task_id,
                )

    @staticmethod
    def _split_result(
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    def _score(self, fitness: torch.Tensor, constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _scalarize_objectives(fitness, constraints, self.constraint_penalty)

    def set_algorithm_policy(self, policy: Mapping[str, Any] | None) -> None:
        """Install a validated policy after an LLM candidate is accepted."""

        self.algorithm_policy = AlgorithmPolicy.from_dict(policy).to_dict()

    def _better_mask(
        self,
        new_fitness: torch.Tensor,
        old_fitness: torch.Tensor,
        new_constraints: Optional[torch.Tensor] = None,
        old_constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the validated selection and constraint policy."""

        constraint = self.algorithm_policy.get("constraint", {})
        selection = self.algorithm_policy.get("selection", {})
        constraint_mode = constraint.get("mode", "penalty")
        if self.pareto_mode:
            return constrained_pareto_better(
                new_fitness, old_fitness, new_constraints, old_constraints
            )
        if constraint_mode in {"feasibility_first", "epsilon_constraint"}:
            epsilon = float(constraint.get("epsilon", 0.0))
            new_cv = (
                torch.relu(new_constraints).reshape(new_constraints.shape[0], -1).sum(dim=1)
                if new_constraints is not None
                else torch.zeros(new_fitness.shape[0], device=new_fitness.device)
            )
            old_cv = (
                torch.relu(old_constraints).reshape(old_constraints.shape[0], -1).sum(dim=1)
                if old_constraints is not None
                else torch.zeros(old_fitness.shape[0], device=old_fitness.device)
            )
            if constraint_mode == "epsilon_constraint":
                new_feasible = new_cv <= epsilon
                old_feasible = old_cv <= epsilon
            else:
                new_feasible = new_cv <= 0
                old_feasible = old_cv <= 0
            objective_new = _scalarize_objectives(new_fitness)
            objective_old = _scalarize_objectives(old_fitness)
            return (
                (new_feasible & ~old_feasible)
                | ((new_feasible == old_feasible) & (new_cv < old_cv))
                | ((new_feasible == old_feasible) & torch.isclose(new_cv, old_cv) &
                   (objective_new < objective_old - float(selection.get("epsilon", 0.0))))
            )
        new_score = self._score(new_fitness, new_constraints)
        old_score = self._score(old_fitness, old_constraints)
        epsilon = float(selection.get("epsilon", 0.0)) if selection.get("mode") == "epsilon" else 0.0
        return new_score < old_score - epsilon

    def init_step(self):
        fitness, constraints = self._split_result(self.evaluate(self.pop))
        self.trial_evals += self.pop_size
        self.fit = fitness
        self.cons = constraints
        index = torch.argmin(self._score(self.fit, self.cons))
        self.best = self.pop[index]
        self.best_fit = self._score(self.fit, self.cons)[index]
        self._update_archives()

    def _state_summary(self) -> dict[str, Any]:
        diversity = torch.std(self.pop, dim=0).mean()
        best_value = float(self.best_fit.detach().cpu())
        success = self.operator_success.get(self.active_operator_id, 0)
        improvement_rate = float(sum(self.improvement_history[-5:]) / max(1, len(self.improvement_history[-5:])))
        feasible_rate = float(sum(self.feasible_history[-5:]) / max(1, len(self.feasible_history[-5:]))) if self.feasible_history else 1.0
        return {
            "task_id": self.task_id,
            "generation": self.gen,
            "trial_evals": self.trial_evals,
            "best_fitness": best_value,
            "improvement_rate": improvement_rate,
            "diversity": float(diversity.detach().cpu()),
            "stagnation": self.stagnation,
            "feasible_rate": feasible_rate,
            "active_operator": self.active_operator_id,
            "active_operator_success": success,
            "operator_failures": int(self.operator_failures),
            "operator_success_rate": success / max(1, self.gen),
            "remaining_generations": max(0, self.max_gen - self.gen),
            "remaining_budget": max(0, (self.max_gen - self.gen) * self.pop_size),
            "initial_budget": self.initial_budget,
            "elite_archive_size": int(self.elite_archive.shape[0]),
            "history_archive_size": int(self.history_archive.shape[0]),
            "selection_mode": self.algorithm_policy["selection"]["mode"],
            "constraint_mode": self.algorithm_policy["constraint"]["mode"],
            "parent_sampling_mode": self.algorithm_policy["parent_sampling"]["mode"],
            "pbest_fraction": self.algorithm_policy["parent_sampling"]["pbest_fraction"],
            "archive_probability": self.algorithm_policy["parent_sampling"]["archive_probability"],
            "adaptive_f": float(self.mu_f),
            "adaptive_cr": float(self.mu_cr),
            "adaptive_parameters": bool(self.adaptive_parameters),
            "cross_island_rate": float(self.cross_island_rate),
            "global_guidance": float(self.global_guidance),
            "algorithm_policy": dict(self.algorithm_policy),
            "adaptive_parameters": bool(self.adaptive_parameters),
            "cross_island_rate": float(self.cross_island_rate),
            "global_guidance": float(self.global_guidance),
        }

    def record_success_history(
        self,
        f_values: torch.Tensor,
        cr_values: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Update F/CR memories from successful trial vectors only."""

        f_values = f_values.detach().reshape(-1).to(dtype=torch.float32)
        cr_values = cr_values.detach().reshape(-1).to(dtype=torch.float32)
        weights = weights.detach().reshape(-1).to(dtype=torch.float32)
        mask = torch.isfinite(f_values) & torch.isfinite(cr_values) & (weights > 0)
        if not bool(mask.any()):
            return
        f_values = f_values[mask].clamp(0.05, 1.0)
        cr_values = cr_values[mask].clamp(0.0, 1.0)
        weights = weights[mask].clamp_min(1e-8)
        norm = weights.sum().clamp_min(1e-8)
        lehmer_f = (weights * f_values.square()).sum() / (weights * f_values).sum().clamp_min(1e-8)
        mean_cr = (weights * cr_values).sum() / norm
        alpha = max(0.0, min(1.0, float(self.learning_rate)))
        self.mu_f = float((1.0 - alpha) * self.mu_f + alpha * lehmer_f.item())
        self.mu_cr = float((1.0 - alpha) * self.mu_cr + alpha * mean_cr.item())
        self.mu_f = max(0.05, min(1.0, self.mu_f))
        self.mu_cr = max(0.0, min(1.0, self.mu_cr))

    def _sample_adaptive_parameters(
        self, spec: DEOperatorSpec, state: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw bounded per-target F/CR values for one generation."""

        base_f, base_cr = spec.effective_parameters(state)
        if not self.adaptive_parameters:
            return (
                torch.full((self.pop_size,), float(base_f), device=self.pop.device),
                torch.full((self.pop_size,), float(base_cr), device=self.pop.device),
            )
        center_f = 0.5 * float(base_f) + 0.5 * float(self.mu_f)
        center_cr = 0.5 * float(base_cr) + 0.5 * float(self.mu_cr)
        # Cauchy tails occasionally produce a large differential step; this is
        # the useful part of SHADE for escaping a collapsed Ackley population.
        f = center_f + 0.1 * torch.tan(
            torch.pi * (_device_rand((self.pop_size,), self.pop.device) - 0.5)
        )
        f = f.nan_to_num(center_f).clamp(0.05, 1.0)
        cr = (center_cr + 0.1 * _device_randn((self.pop_size,), self.pop.device)).clamp(0.0, 1.0)
        return f, cr

    def _update_archives(self) -> None:
        """Maintain bounded elite and historical decision archives for the DSL."""
        scores = self._score(self.fit, self.cons)
        elite_count = max(1, self.pop_size // 5)
        elite = self.pop[torch.argsort(scores)[:elite_count]].detach()
        history = self.pop.detach()
        self.elite_archive = torch.cat([self.elite_archive, elite], dim=0)[-max(1, self.pop_size) :]
        self.history_archive = torch.cat([self.history_archive, history], dim=0)[-max(2 * self.pop_size, 32) :]

    def _maybe_generate_operator(self) -> None:
        if self.operator_generator is None:
            return
        if self.max_llm_calls is not None and self.llm_attempts >= self.max_llm_calls:
            return
        trigger = (self.gen > 0 and self.gen % self.llm_interval == 0) or self.stagnation >= self.stagnation_window
        if not trigger or self.gen - self.last_llm_generation < self.llm_interval:
            return
        self.last_llm_generation = self.gen
        self.llm_attempts += 1
        state = self._state_summary()
        spec = self.operator_generator.generate(state, self.operator_archive.context(state))
        if spec is None:
            return
        try:
            spec.validate()
        except (TypeError, ValueError):
            return

        if self.shadow_evaluator is not None and self.shadow_objective is not None:
            current = self.library.get(self.active_operator_id)
            self.last_shadow_evaluation = self.shadow_evaluator.evaluate(
                [current, spec], self.pop, self.fit, self.lb, self.ub, self.shadow_objective,
                state=state, elite_archive=self.elite_archive, history_archive=self.history_archive,
            )
            if self.last_shadow_evaluation.scores[1] < self.last_shadow_evaluation.scores[0]:
                self.last_promotion = PromotionResult(False, None, float("-inf"), 0, [])
                return

        self.library.add(spec)
        if self.promoter.evaluation_budget > 0:
            self.last_promotion = self.promoter.promote(
                [spec], self.pop, self.fit, self.lb, self.ub, self.evaluate,
                state=state, elite_archive=self.elite_archive, history_archive=self.history_archive,
            )
            if not self.last_promotion.accepted:
                return
        self.previous_operator_id = self.active_operator_id
        self.active_operator_id = spec.operator_id
        self.operator_failures = 0
        self.operator_archive.add(spec, {**state, "task_id": self.task_id}, score=0.0, task_id=self.task_id)

    def prepare_operator(self) -> tuple[DEOperatorSpec, str, dict[str, Any]]:
        """Select the operator and state used for one generation.

        Separating preparation from evaluation lets the island coordinator
        generate trials for all islands first and submit one concatenated batch
        to the objective without changing the single-island semantics.
        """

        self._maybe_generate_operator()
        active_operator_id = self.active_operator_id
        spec = self.library.get(active_operator_id)
        state = self._state_summary()
        if not spec.gate(state):
            spec = self.library.get(spec.fallback_operator)
        actual_operator_id = spec.operator_id
        self._pending_f, self._pending_cr = self._sample_adaptive_parameters(spec, state)
        # Private tensor entries are consumed by the tensorized library and
        # never serialized into the LLM prompt or the persistent RAG record.
        state["_F_values"] = self._pending_f
        state["_CR_values"] = self._pending_cr
        return spec, actual_operator_id, state

    def prepare_trial(self) -> tuple[torch.Tensor, str]:
        """Build one unevaluated trial population and return its score owner."""

        spec, active_operator_id, state = self.prepare_operator()
        trial = self.library.apply(
            spec,
            self.pop,
            self.fit,
            self.lb,
            self.ub,
            state=state,
            elite_archive=self.elite_archive,
            history_archive=self.history_archive,
        )
        return trial, active_operator_id

    def prepare_single_trial(self, index: Optional[int] = None) -> tuple[int, torch.Tensor, str]:
        """Build one trial vector for steady-state evaluation.

        The complete tensorized mutation is still generated from the current
        population so parent sampling and operator semantics remain identical
        to generational DE; only the selected target vector is submitted to
        the expensive objective.
        """

        spec, active_operator_id, state = self.prepare_operator()
        trial = self.library.apply(
            spec,
            self.pop,
            self.fit,
            self.lb,
            self.ub,
            state=state,
            elite_archive=self.elite_archive,
            history_archive=self.history_archive,
        )
        target = self.gen % self.pop_size if index is None else int(index) % self.pop_size
        return target, trial[target], active_operator_id

    def finish_trial(
        self,
        trial: torch.Tensor,
        evaluated: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        operator_id: Optional[str] = None,
    ) -> None:
        """Apply selection and accounting after an objective evaluation."""

        trial_fit, trial_cons = self._split_result(evaluated)
        score_operator_id = operator_id or self.active_operator_id
        old_scores = self._score(self.fit, self.cons)
        improved = self._better_mask(trial_fit, self.fit, trial_cons, self.cons)
        if self.algorithm_policy["survivor"]["mode"] == "diversity_preserving":
            scale = (self.ub - self.lb).clamp_min(1e-12)
            displacement = ((trial - self.pop) / scale).norm(dim=1)
            improved = improved & (displacement >= 1e-3)
        self.pop = torch.where(improved[:, None], trial, self.pop)
        if self.fit.ndim > 1:
            self.fit = torch.where(improved[:, None], trial_fit, self.fit)
        else:
            self.fit = torch.where(improved, trial_fit, self.fit)
        if trial_cons is not None:
            if self.cons is None:
                self.cons = torch.zeros_like(trial_cons)
            self.cons = torch.where(improved[:, None], trial_cons, self.cons)

        old_best_fit = self.best_fit
        current_scores = self._score(self.fit, self.cons)
        if self._pending_f is not None and self._pending_cr is not None:
            gains = (old_scores - current_scores).clamp_min(0.0)
            self.record_success_history(self._pending_f, self._pending_cr, gains)
            self._pending_f = None
            self._pending_cr = None
        new_best_index = torch.argmin(current_scores)
        new_best_fit = current_scores[new_best_index]
        if bool(new_best_fit < self.best_fit) or bool(improved.any()):
            self.best = self.pop[new_best_index]
            self.best_fit = new_best_fit
            self.stagnation = 0
            self.operator_success[score_operator_id] = self.operator_success.get(score_operator_id, 0) + 1
            self.operator_failures = 0
            improvement = float((old_best_fit - new_best_fit).detach().cpu())
            self.operator_gain[score_operator_id] = self.operator_gain.get(score_operator_id, 0.0) + max(0.0, improvement)
            self.operator_archive.update_score(score_operator_id, max(0.0, improvement))
            self.improvement_history.append(max(0.0, improvement))
        else:
            self.stagnation += 1
            self.operator_failures += 1
            self.improvement_history.append(0.0)

        if trial_cons is None:
            self.feasible_history.append(1.0)
        else:
            self.feasible_history.append(float((torch.relu(trial_cons).sum(dim=1) <= 0).float().mean().detach().cpu()))
        self._update_archives()

        # A generated operator that repeatedly fails is rolled back to the
        # previous operator; the deterministic DE loop remains the authority.
        if self.previous_operator_id and self.operator_failures >= self.operator_patience:
            self.active_operator_id = self.previous_operator_id
            self.previous_operator_id = None
            self.operator_failures = 0
        self.gen += 1
        self.trial_evals += self.pop_size

    def finish_single_trial(
        self,
        index: int,
        trial: torch.Tensor,
        evaluated: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        operator_id: Optional[str] = None,
    ) -> None:
        """Apply selection and accounting for exactly one real evaluation."""

        index = int(index) % self.pop_size
        trial_fit, trial_cons = self._split_result(evaluated)
        score_operator_id = operator_id or self.active_operator_id
        old_score_value = self._score(self.fit[index].reshape(1, -1), None if self.cons is None else self.cons[index].reshape(1, -1))[0]
        trial_fit = trial_fit.reshape(1, -1)
        old_fit = self.fit[index].reshape(1, -1)
        old_cons = None if self.cons is None else self.cons[index].reshape(1, -1)
        new_cons = None if trial_cons is None else trial_cons.reshape(1, -1)
        improved = bool(self._better_mask(trial_fit, old_fit, new_cons, old_cons)[0])
        if improved and self.algorithm_policy["survivor"]["mode"] == "diversity_preserving":
            scale = (self.ub - self.lb).clamp_min(1e-12)
            improved = bool((((trial - self.pop[index]) / scale).norm() >= 1e-3))

        old_best_fit = self.best_fit
        if improved:
            self.pop[index] = trial
            if self.fit.ndim > 1:
                self.fit[index] = trial_fit[0]
            else:
                self.fit[index] = trial_fit[0, 0]
            if new_cons is not None:
                if self.cons is None:
                    self.cons = torch.zeros(
                        (self.pop_size, new_cons.shape[1]),
                        device=new_cons.device,
                        dtype=new_cons.dtype,
                    )
                self.cons[index] = new_cons[0]

        current_scores = self._score(self.fit, self.cons)
        if improved and self._pending_f is not None and self._pending_cr is not None:
            gain = (old_score_value - current_scores[index]).clamp_min(0.0)
            self.record_success_history(
                self._pending_f[index : index + 1],
                self._pending_cr[index : index + 1],
                gain.reshape(1),
            )
        self._pending_f = None
        self._pending_cr = None
        new_best_index = torch.argmin(current_scores)
        new_best_fit = current_scores[new_best_index]
        if bool(new_best_fit < self.best_fit) or improved:
            self.best = self.pop[new_best_index]
            self.best_fit = new_best_fit
            self.stagnation = 0
            self.operator_success[score_operator_id] = self.operator_success.get(score_operator_id, 0) + 1
            self.operator_failures = 0
            improvement = float((old_best_fit - new_best_fit).detach().cpu())
            self.operator_gain[score_operator_id] = self.operator_gain.get(score_operator_id, 0.0) + max(0.0, improvement)
            self.operator_archive.update_score(score_operator_id, max(0.0, improvement))
            self.improvement_history.append(max(0.0, improvement))
        else:
            self.stagnation += 1
            self.operator_failures += 1
            self.improvement_history.append(0.0)

        if new_cons is None:
            self.feasible_history.append(1.0)
        else:
            self.feasible_history.append(
                float((torch.relu(new_cons).sum(dim=1) <= 0).float().mean().detach().cpu())
            )
        self._update_archives()
        if self.previous_operator_id and self.operator_failures >= self.operator_patience:
            self.active_operator_id = self.previous_operator_id
            self.previous_operator_id = None
            self.operator_failures = 0
        self.gen += 1
        self.trial_evals += 1

    def step(self):
        trial, operator_id = self.prepare_trial()
        self.finish_trial(trial, self.evaluate(trial), operator_id)

    def record_step(self):
        result = {"pop": self.pop, "fit": self.fit}
        if self.cons is not None:
            result["cons"] = self.cons
        return result

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Return the complete mutable DE state needed for an exact resume.

        The tensors are moved to CPU so a checkpoint written on CUDA can be
        resumed on either CUDA or CPU.  Generated operators are included
        because the active operator may not exist in the built-in library.
        """

        def cpu_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.detach().cpu().clone()

        return {
            "version": 1,
            "pop": cpu_tensor(self.pop),
            "fit": cpu_tensor(self.fit),
            "best": cpu_tensor(self.best),
            "best_fit": cpu_tensor(self.best_fit),
            "cons": cpu_tensor(self.cons),
            "gen": int(self.gen),
            "trial_evals": int(self.trial_evals),
            "stagnation": int(self.stagnation),
            "active_operator_id": self.active_operator_id,
            "previous_operator_id": self.previous_operator_id,
            "operator_failures": int(self.operator_failures),
            "last_llm_generation": int(self.last_llm_generation),
            "llm_attempts": int(self.llm_attempts),
            "operator_success": dict(self.operator_success),
            "operator_gain": dict(self.operator_gain),
            "improvement_history": list(self.improvement_history),
            "feasible_history": list(self.feasible_history),
            "mu_f": float(self.mu_f),
            "mu_cr": float(self.mu_cr),
            "learning_rate": float(self.learning_rate),
            "elite_archive": cpu_tensor(self.elite_archive),
            "history_archive": cpu_tensor(self.history_archive),
            "initial_budget": int(self.initial_budget),
            "algorithm_policy": dict(self.algorithm_policy),
            "promoter_used_evaluations": int(self.promoter.used_evaluations),
            "library": [spec.to_dict() for spec in self.library.all()],
            "operator_archive_records": list(self.operator_archive.records),
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`checkpoint_state_dict`."""

        if not isinstance(state, Mapping) or int(state.get("version", 0)) != 1:
            raise ValueError("unsupported or malformed LLMOperatorDE checkpoint state")

        def restore_tensor(name: str, optional: bool = False) -> Optional[torch.Tensor]:
            value = state.get(name)
            if value is None and optional:
                return None
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"checkpoint field {name!r} is not a tensor")
            return value.to(device=self.pop.device)

        self.pop = restore_tensor("pop")
        self.fit = restore_tensor("fit")
        self.best = restore_tensor("best")
        self.best_fit = restore_tensor("best_fit")
        self.cons = restore_tensor("cons", optional=True)
        self.elite_archive = restore_tensor("elite_archive")
        self.history_archive = restore_tensor("history_archive")
        self.gen = int(state.get("gen", 0))
        self.trial_evals = int(state.get("trial_evals", self.gen * self.pop_size))
        self.stagnation = int(state.get("stagnation", 0))
        self.active_operator_id = str(state.get("active_operator_id", "rand1"))
        previous = state.get("previous_operator_id")
        self.previous_operator_id = None if previous is None else str(previous)
        self.operator_failures = int(state.get("operator_failures", 0))
        self.last_llm_generation = int(state.get("last_llm_generation", -self.llm_interval))
        self.llm_attempts = int(state.get("llm_attempts", 0))
        self.operator_success = {
            str(key): int(value) for key, value in dict(state.get("operator_success", {})).items()
        }
        self.operator_gain = {
            str(key): float(value) for key, value in dict(state.get("operator_gain", {})).items()
        }
        self.improvement_history = [float(value) for value in state.get("improvement_history", [])]
        self.feasible_history = [float(value) for value in state.get("feasible_history", [])]
        self.mu_f = max(0.05, min(1.0, float(state.get("mu_f", self.mu_f))))
        self.mu_cr = max(0.0, min(1.0, float(state.get("mu_cr", self.mu_cr))))
        self.learning_rate = max(0.0, min(1.0, float(state.get("learning_rate", self.learning_rate))))
        self._pending_f = None
        self._pending_cr = None
        self.initial_budget = int(state.get("initial_budget", self.initial_budget))
        self.set_algorithm_policy(state.get("algorithm_policy", self.algorithm_policy))
        self.adaptive_parameters = bool(state.get("adaptive_parameters", self.adaptive_parameters))
        self.cross_island_rate = max(0.0, min(1.0, float(state.get("cross_island_rate", self.cross_island_rate))))
        self.global_guidance = max(0.0, min(1.0, float(state.get("global_guidance", self.global_guidance))))
        self.promoter.used_evaluations = int(state.get("promoter_used_evaluations", 0))
        archive_records = state.get("operator_archive_records")
        if isinstance(archive_records, list):
            self.operator_archive.records = [
                record for record in archive_records if isinstance(record, Mapping)
            ][-self.operator_archive.max_size :]

        for raw_spec in state.get("library", []):
            try:
                self.library.add(DEOperatorSpec.from_dict(raw_spec))
            except (TypeError, ValueError, KeyError):
                continue
        if self.active_operator_id not in {spec.operator_id for spec in self.library.all()}:
            raise ValueError(f"checkpoint active operator is unavailable: {self.active_operator_id}")

