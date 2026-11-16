"""Static check that ByteTrackMOTModel accepts MAITE detectors.

Nothing here runs. CI type-checks this file with pyright in the ``pyright-internal``
job, where maite is installed. Conformance of each wrapper to its own MAITE
protocol is checked through its entry point by
``scripts/statically_verify_maite_entrypoints.py``.
"""

from __future__ import annotations

from maite.protocols import object_detection as od

from modelmaite import ByteTrackMOTModel, OnnxODModel, TorchvisionODModel, VisdroneODModel


def bytetrack_accepts_any_maite_detector(detector: od.Model) -> ByteTrackMOTModel:
    return ByteTrackMOTModel(detector=detector)


def bytetrack_accepts_modelmaite_detectors(
    onnx: OnnxODModel, torchvision: TorchvisionODModel, visdrone: VisdroneODModel
) -> list[ByteTrackMOTModel]:
    return [ByteTrackMOTModel(detector=detector) for detector in (onnx, torchvision, visdrone)]
