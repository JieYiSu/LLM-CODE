"""Atomic experiment checkpoints and reproducible random-state handling."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch and CUDA random states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write a checkpoint atomically, retaining the old file on interruption."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), temporary)
    try:
        os.replace(temporary, destination)
    except PermissionError:
        # Windows can briefly deny replacement when an editor or sync service
        # has the destination open.  A direct write still keeps restart support
        # available; the temporary file is removed afterwards.
        torch.save(dict(payload), destination)
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a local checkpoint with an explicit compatibility fallback."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch versions lack ``weights_only``
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported or malformed checkpoint: {source}")
    return payload
