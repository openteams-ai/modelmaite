"""MAITE-compliant object-detection model wrappers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike

from modelmaite._onnx import (
    IMAGE_OBJECT_DETECTION_INTERFACE,
    ONNX_INSTALL_HINT,
    UNIT_INTERVAL_TOLERANCE,
    get_onnx_providers,
    id_hash,
    import_onnxruntime,
    load_jatic_onnx_metadata,
    optional_output_int,
    prepare_jatic_onnx_image_batch,
    validate_jatic_onnx_session,
)
from modelmaite.object_detection.types import DetectionTarget

SUPPORTED_ONNX_MODELS = {"jatic_onnx"}


class OnnxODModel:
    """A MAITE-compliant wrapper for JATIC_ONNX object-detection models."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        config_path: str | Path,
        device: object | None = None,
        index2label_key: str = "index2label",
        model_id: str | None = None,
        batch_size: int | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        validate_onnx: bool = False,
    ) -> None:
        """Initialize a JATIC_ONNX object-detection model.

        Parameters
        ----------
        weights_path
            Path to the ONNX model file.
        config_path
            Path to JATIC_ONNX metadata JSON, including ``index2label``.
        device
            Device/provider request (``cpu``, ``cuda``, or ``mps``/CoreML). If omitted, the best available provider is
            selected from the installed ONNX Runtime package.
        index2label_key
            Metadata key for mapping class indices to labels.
        model_id
            Optional model identifier.
        batch_size
            Optional runtime batch-size override.
        image_height
            Optional runtime input-height override.
        image_width
            Optional runtime input-width override.
        validate_onnx
            If ``True``, run ``onnx.checker.check_model`` before creating the ONNX Runtime session. This parses the
            model separately from ONNX Runtime, so it is disabled by default for large models.
        """
        ort = import_onnxruntime()

        if validate_onnx:
            try:
                onnx = importlib.import_module("onnx")
            except ImportError:
                raise ImportError(
                    f"validate_onnx=True requires optional dependency 'onnx'. {ONNX_INSTALL_HINT}"
                ) from None
            onnx.checker.check_model(str(weights_path))

        self.device, providers = get_onnx_providers(device, ort=ort)
        self.model = ort.InferenceSession(str(weights_path), providers=providers)
        validate_jatic_onnx_session(self.model, expected_outputs={"boxes", "scores"})

        self.config, self.index2label = load_jatic_onnx_metadata(
            config_path,
            expected_io_interface=IMAGE_OBJECT_DETECTION_INTERFACE,
            index2label_key=index2label_key,
        )
        output_meta = self.config.get("io", {}).get("output", {})
        if not self.index2label:
            raise ValueError("ONNX metadata index2label must contain at least one class.")
        self._n_classes = len(self.index2label)
        n_classes = optional_output_int(output_meta.get("nClasses"), "nClasses", minimum=1)
        if n_classes is not None and n_classes != self._n_classes:
            raise ValueError(
                f"ONNX metadata io.output.nClasses={n_classes} does not match len(index2label)={self._n_classes}."
            )

        self._model_name = "jatic_onnx"
        self._batch_size = batch_size
        self._image_height = image_height
        self._image_width = image_width
        self._n_boxes = optional_output_int(output_meta.get("nBoxes"), "nBoxes", minimum=0)
        if model_id is None:
            model_id = f"{self._model_name}_{id_hash(weights_path=weights_path, config_path=config_path)}"
        self.metadata = {"id": model_id, "index2label": self.index2label}

    def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[DetectionTarget]:
        """Make object-detection predictions for a CHW image batch.

        Every JATIC_ONNX output slot is returned after coordinate scaling; this wrapper does not apply NMS,
        confidence thresholding, or background filtering.
        """
        batch, original_sizes = prepare_jatic_onnx_image_batch(
            input_batch,
            self.config,
            batch_size=self._batch_size,
            image_height=self._image_height,
            image_width=self._image_width,
        )
        outputs = self.model.run(["boxes", "scores"], {"image": batch})
        boxes_batch = np.asarray(outputs[0], dtype=np.float32)
        scores_batch = np.asarray(outputs[1], dtype=np.float32)
        _validate_output_shapes(
            boxes_batch,
            scores_batch,
            expected_batch_size=len(original_sizes),
            expected_num_boxes=self._n_boxes,
            expected_num_classes=self._n_classes,
        )

        output_batch: list[DetectionTarget] = []
        for boxes, class_scores, (height, width) in zip(boxes_batch, scores_batch, original_sizes, strict=True):
            labels = np.argmax(class_scores, axis=-1).astype(np.int64)
            scores = np.max(class_scores, axis=-1).astype(np.float32)
            pixel_boxes = np.asarray(boxes, dtype=np.float32).copy()
            pixel_boxes[:, [0, 2]] *= float(width)
            pixel_boxes[:, [1, 3]] *= float(height)
            output_batch.append(DetectionTarget(boxes=pixel_boxes, labels=labels, scores=scores))

        return output_batch

    @property
    def name(self) -> str:
        """Human-readable name for JATIC_ONNX object-detection model."""
        return self._model_name


class ModelSpecification(TypedDict):
    """Model metadata required for loading modelmaite wrappers."""

    model_weights_path: str | Path
    model_config_path: str | Path
    model_type: Literal["jatic_onnx"]


def load_models(
    models: Mapping[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, OnnxODModel]:
    """Load object-detection models from model specification dictionaries."""
    loaded = {}
    for name, meta_dict in models.items():
        model_type = meta_dict.get("model_type")
        if model_type is None:
            raise ValueError("JATIC_ONNX model specifications require model_type.")
        if model_type not in SUPPORTED_ONNX_MODELS:
            raise ValueError(
                f"Unsupported model_type {model_type!r}; supported model types: {sorted(SUPPORTED_ONNX_MODELS)}."
            )

        weights_path = meta_dict.get("model_weights_path")
        config_path = meta_dict.get("model_config_path")
        if weights_path is None or config_path is None:
            raise ValueError("JATIC_ONNX models require model_weights_path and model_config_path.")

        loaded[name] = OnnxODModel(
            weights_path=weights_path,
            config_path=config_path,
            **kwargs,
        )

    return loaded


def _validate_output_shapes(
    boxes_batch: np.ndarray,
    scores_batch: np.ndarray,
    *,
    expected_batch_size: int,
    expected_num_boxes: int | None,
    expected_num_classes: int,
) -> None:
    if boxes_batch.ndim != 3 or boxes_batch.shape[-1] != 4:
        raise ValueError(
            f"JATIC_ONNX object-detection 'boxes' output must have shape (N, D, 4), got {boxes_batch.shape}."
        )
    if scores_batch.ndim != 3:
        raise ValueError(
            f"JATIC_ONNX object-detection 'scores' output must have shape (N, D, C), got {scores_batch.shape}."
        )
    if boxes_batch.shape[:2] != scores_batch.shape[:2]:
        raise ValueError(
            "JATIC_ONNX object-detection 'boxes' and 'scores' outputs must agree on batch and detection dimensions, "
            f"got {boxes_batch.shape} and {scores_batch.shape}."
        )
    if boxes_batch.shape[0] != expected_batch_size:
        raise ValueError(
            f"JATIC_ONNX object-detection outputs contain batch size {boxes_batch.shape[0]}, "
            f"expected {expected_batch_size}."
        )
    if expected_num_boxes is not None and boxes_batch.shape[1] != expected_num_boxes:
        raise ValueError(
            f"JATIC_ONNX object-detection outputs contain {boxes_batch.shape[1]} detections per image, "
            f"but ONNX metadata io.output.nBoxes is {expected_num_boxes}."
        )
    if scores_batch.shape[-1] != expected_num_classes:
        raise ValueError(
            f"JATIC_ONNX object-detection 'scores' output has {scores_batch.shape[-1]} classes, "
            f"but index2label contains {expected_num_classes}."
        )
    _validate_output_values(boxes_batch, scores_batch)


def _validate_output_values(boxes_batch: np.ndarray, scores_batch: np.ndarray) -> None:
    if not np.all(np.isfinite(boxes_batch)):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must contain only finite values.")
    if boxes_batch.size and (
        np.min(boxes_batch) < -UNIT_INTERVAL_TOLERANCE or np.max(boxes_batch) > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must be normalized to the range [0, 1].")
    if boxes_batch.size and (
        np.any(boxes_batch[..., 0] > boxes_batch[..., 2]) or np.any(boxes_batch[..., 1] > boxes_batch[..., 3])
    ):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must use xyxy order with x0 <= x1 and y0 <= y1.")
    if not np.all(np.isfinite(scores_batch)):
        raise ValueError("JATIC_ONNX object-detection 'scores' output must contain only finite values.")
    if scores_batch.size and (
        np.min(scores_batch) < -UNIT_INTERVAL_TOLERANCE or np.max(scores_batch) > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("JATIC_ONNX object-detection 'scores' output must be in the range [0, 1].")
