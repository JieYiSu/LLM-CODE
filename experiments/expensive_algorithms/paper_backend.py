"""Small local numerical backend used by the paper-baseline ports.

The supplied paper implementations use a much larger XDE-SINDy backend for
optional GPU dispatch.  The three comparison methods in this suite are run on
the CPU and only need the small common surface implemented here.  Keeping this
adapter in the local ``expensive_algorithms`` package makes the comparison
suite self-contained and removes an import-time dependency on another
checkout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ComputeBackend:
    """NumPy implementation of the paper-port numerical backend."""

    def __init__(
        self,
        requested: str = "cpu",
        device: int = 0,
        min_elements: int = 1,
    ) -> None:
        requested = str(requested).strip().lower()
        if requested not in {"cpu", "auto"}:
            raise ValueError(
                "the local paper-baseline backend is CPU-only; use requested='cpu'"
            )
        if int(device) < 0:
            raise ValueError("device must be non-negative")
        if int(min_elements) < 1:
            raise ValueError("min_elements must be positive")
        self.requested = requested
        self.device_index = int(device)
        self.min_elements = int(min_elements)
        self.resolved = "cpu"
        self.device_name = "CPU (local paper baseline backend)"
        self.array_module = np

    @property
    def cuda_enabled(self) -> bool:
        return False

    def to_device(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def matmul(
        self,
        left: Any,
        right: Any,
        *,
        operation: str = "matmul",
        return_device: bool = False,
    ) -> np.ndarray:
        result = np.asarray(left) @ np.asarray(right)
        return result if return_device else np.asarray(result)

    def ridge_fit(
        self,
        x: Any,
        y: Any,
        alpha: float,
        *,
        return_device: bool = False,
    ) -> np.ndarray:
        design = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        if design.ndim != 2 or design.shape[1] == 0:
            result = np.empty(0, dtype=float)
        else:
            gram = design.T @ design
            gram.flat[:: gram.shape[0] + 1] += float(alpha)
            try:
                result = np.linalg.solve(gram, design.T @ target)
            except np.linalg.LinAlgError:
                result = np.linalg.pinv(gram, rcond=1e-10) @ (design.T @ target)
        return result if return_device else np.asarray(result, dtype=float)
