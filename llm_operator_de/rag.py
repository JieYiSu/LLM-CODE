"""Explicit operator-performance RAG interface.

The first implementation uses the validated JSON-backed state-aware archive.
It stores operator structure, search state, budget, gains, cost and failure
metadata instead of retrieving generic paper text.
"""

from __future__ import annotations

from typing import Any, Mapping

from .core import DEOperatorSpec, OperatorArchive


class OperatorRAG(OperatorArchive):
    """Persistent state-aware knowledge base for successful and failed operators."""

    def retrieve_cases(self, state: Mapping[str, Any], top_k: int = 4) -> list[dict[str, Any]]:
        return self.retrieve(state, top_k=top_k)

    def add_case(
        self,
        spec: DEOperatorSpec,
        state: Mapping[str, Any],
        score: float = 0.0,
        task_id: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        enriched = dict(state)
        if diagnostics:
            enriched["diagnostics"] = dict(diagnostics)
        self.add(spec, enriched, score=score, task_id=task_id)

    def build_context(self, state: Mapping[str, Any], top_k: int = 4) -> str:
        return self.context(state, top_k=top_k)
