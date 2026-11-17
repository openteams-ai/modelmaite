"""Helpers shared by the torch-based wrappers.

The torch module is passed in by each wrapper, so this module imports torch only for type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as _torch_types


def select_torch_device(torch: Any, requested: str | int | None) -> _torch_types.device:
    """Return the requested ``torch.device``, or the best available one (CUDA, then MPS, then CPU)."""
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except (RuntimeError, TypeError, ValueError) as e:
        raise ValueError(f"Unsupported torch device {requested!r}.") from e
