"""ESA algorithm entry point for the expensive-suite runner."""

from __future__ import annotations

from ..expensive_suite import ESAPort as _ESAPort


class ESAPort(_ESAPort):
    """Separated public entry point for the Python ESA port."""


__all__ = ["ESAPort"]
