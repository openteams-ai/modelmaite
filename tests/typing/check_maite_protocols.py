"""Static checks that ByteTrackMOTModel accepts MAITE detectors, and of ObjectDetectorBase's output type.

Nothing here runs. CI type-checks this file with pyright in the ``pyright-internal``
job, where maite is installed. Conformance of each wrapper to its own MAITE
protocol is checked through its entry point by
``scripts/statically_verify_maite_entrypoints.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from maite.protocols import object_detection as od
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from modelmaite import ByteTrackMOTModel, OnnxODModel, TorchvisionODModel, VisdroneODModel
from modelmaite.base import ObjectDetectorBase


def bytetrack_accepts_any_maite_detector(detector: od.Model) -> ByteTrackMOTModel:
    return ByteTrackMOTModel(detector=detector)


def bytetrack_accepts_modelmaite_detectors(
    onnx: OnnxODModel, torchvision: TorchvisionODModel, visdrone: VisdroneODModel
) -> list[ByteTrackMOTModel]:
    return [ByteTrackMOTModel(detector=detector) for detector in (onnx, torchvision, visdrone)]


# A detector on ObjectDetectorBase may return its own target type, as long as it has boxes,
# labels and scores. Extra fields are kept, and the wrapper is still a MAITE detector.


@dataclass(frozen=True)
class _TargetWithMasks:
    boxes: NDArray[np.float32]
    labels: NDArray[np.int64]
    scores: NDArray[np.float32]
    masks: NDArray[np.bool_]


class _MaskDetector(ObjectDetectorBase[_TargetWithMasks, list[ArrayLike], list[_TargetWithMasks]]):
    @property
    def name(self) -> str:
        return "masks"

    @override
    def preprocess(self, images: list[ArrayLike]) -> list[ArrayLike]:
        return images

    @override
    def forward(self, prepared: list[ArrayLike]) -> list[_TargetWithMasks]:
        return []

    @override
    def postprocess(self, raw: list[_TargetWithMasks], prepared: list[ArrayLike]) -> list[_TargetWithMasks]:
        return raw


def own_target_type_is_kept(detector: _MaskDetector, images: list[ArrayLike]) -> NDArray[np.bool_]:
    return detector(images)[0].masks


def own_target_type_is_a_maite_detector(detector: _MaskDetector) -> od.Model:
    return detector


def own_target_type_works_with_bytetrack(detector: _MaskDetector) -> ByteTrackMOTModel:
    return ByteTrackMOTModel(detector=detector)


@dataclass(frozen=True)
class _TargetWithoutScores:
    boxes: NDArray[np.float32]
    labels: NDArray[np.int64]


# Rejected: the output type must have boxes, labels and scores.
_NoScores = ObjectDetectorBase[_TargetWithoutScores, int, int]  # pyright: ignore[reportInvalidTypeArguments]
