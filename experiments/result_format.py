"""Formatting helpers for reproducible experiment result files."""

from __future__ import annotations

from typing import Any


def csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return CSV-safe copies with best_f rendered in scientific notation."""
    formatted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        value = item.get("best_f")
        if value is not None:
            item["best_f"] = f"{float(value):.6e}"
        formatted.append(item)
    return formatted
