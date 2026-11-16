"""Shared object-detection types."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass
class DetectionTarget:
    """MAITE-compatible object-detection target.

    Parameters
    ----------
    boxes
        Coordinates of bounding boxes in ``xyxy`` format with shape ``(n_boxes, 4)``.
    labels
        Integer labels for the detected objects with shape ``(n_boxes,)``.
    scores
        Detection confidence scores with shape ``(n_boxes,)`` or ``(n_boxes, n_classes)``. MAITE 0.10 requires
        each score to be at most 1, and each row of ``(n_boxes, n_classes)`` scores to sum to at most 1.
    """

    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
