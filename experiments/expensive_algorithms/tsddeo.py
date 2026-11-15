"""TSDDEO algorithm entry point for the expensive-suite runner."""

from __future__ import annotations

from ..expensive_suite import TSDDEOPort as _TSDDEOPort


class TSDDEOPort(_TSDDEOPort):
    """Separated public entry point for the Python TSDDEO port."""


__all__ = ["TSDDEOPort"]
