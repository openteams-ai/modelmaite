"""Shared multi-object-tracking types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass
class MOTFrameTarget:
    """MAITE-compatible tracked detections for one video frame.

    Parameters
    ----------
    boxes
        Bounding boxes in ``xyxy`` format with shape ``(n_detections, 4)``.
    labels
        Integer labels with shape ``(n_detections,)``.
    scores
        Detection confidence scores with shape ``(n_detections,)``.
    track_ids
        Track identifiers with shape ``(n_detections,)``. ``-1`` denotes a
        returned detection that is unconfirmed or untracked; the upstream
        tracker may omit other unmatched detections according to its confidence
        thresholds.
    """

    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
    track_ids: ArrayLike


@dataclass
class MOTTarget:
    """MAITE-compatible tracks over a complete video stream."""

    frame_tracks: Sequence[MOTFrameTarget]
