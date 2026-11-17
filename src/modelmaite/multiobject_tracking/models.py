"""MAITE-compliant multi-object-tracking model wrappers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from modelmaite._typing import ObjectDetectionModel, ObjectDetectionTarget, VideoFrame
from modelmaite.base import TrackerBase
from modelmaite.multiobject_tracking.types import MOTFrameTarget


class _Detections(Protocol):
    xyxy: ArrayLike
    class_id: ArrayLike | None
    confidence: ArrayLike | None
    tracker_id: ArrayLike | None


class _DetectionsFactory(Protocol):
    def __call__(
        self,
        *,
        xyxy: NDArray[np.float32],
        class_id: NDArray[np.int64],
        confidence: NDArray[np.float32],
    ) -> _Detections: ...


class _SupervisionModule(Protocol):
    Detections: _DetectionsFactory


class _Tracker(Protocol):
    def update(self, detections: _Detections, frame: NDArray[np.uint8] | None = None) -> _Detections: ...


class _TrackerFactory(Protocol):
    def __call__(self, **kwargs: object) -> _Tracker: ...


class ByteTrackConfig(TypedDict, total=False):
    """Options forwarded to ``trackers.ByteTrackTracker``."""

    lost_track_buffer: int
    frame_rate: float
    track_activation_threshold: float
    minimum_consecutive_frames: int
    minimum_iou_threshold: float
    high_conf_det_threshold: float
    state_estimator_class: type[object]
    iou: object


MOT_INSTALL_HINT = 'Install multi-object-tracking support with `uv add "modelmaite[mot]"`.'


class ByteTrackMOTModel(TrackerBase[_Tracker]):
    """Turn a MAITE object-detection model into a multi-object-tracking model.

    The wrapped detector runs on each video frame and Roboflow's ByteTrack
    implementation associates its detections over time. A fresh tracker is used
    for every video, so track state never leaks between batch elements or calls.

    Parameters
    ----------
    detector
        A MAITE-compatible object-detection model.
    detector_batch_size
        Number of consecutive frames sent to the detector at once. Tracker
        updates remain sequential and preserve video-frame order.
    tracker_kwargs
        Optional keyword arguments forwarded to ``trackers.ByteTrackTracker``. They are copied,
        and the ``tracker_kwargs`` attribute is a read-only view of the copy.
    model_id
        Optional model identifier. By default, it is
        ``f"bytetrack_{detector id}_{hash of tracker_kwargs and detector_batch_size}"``.

    Notes
    -----
    ByteTrack associates detections using geometry and confidence, not class
    labels. Consequently, one track ID may carry different labels across frames.
    The returned targets also follow the upstream tracker's filtering behavior:
    unmatched detections between ``high_conf_det_threshold`` and
    ``track_activation_threshold`` may be omitted rather than returned with a
    track ID of ``-1``.
    """

    tracker_name = "bytetrack"

    def __init__(
        self,
        *,
        detector: ObjectDetectionModel,
        detector_batch_size: int = 1,
        tracker_kwargs: ByteTrackConfig | None = None,
        model_id: str | None = None,
    ) -> None:
        # A read-only copy: the kwargs are part of the id, so they must not change after construction.
        self._tracker_kwargs: Mapping[str, object] = MappingProxyType(dict(tracker_kwargs or {}))
        super().__init__(detector=detector, detector_batch_size=detector_batch_size, model_id=model_id)
        self._supervision, self._tracker_type = _import_tracking_dependencies()

    @property
    def tracker_kwargs(self) -> Mapping[str, object]:
        """A read-only copy of the keyword arguments forwarded to ``trackers.ByteTrackTracker``."""
        return self._tracker_kwargs

    @override
    def _id_fields(self) -> dict[str, object]:
        # detector_batch_size decides which frames share a detector call, which can change a
        # batch-sensitive detector's detections.
        return {"tracker_kwargs": self.tracker_kwargs, "detector_batch_size": self.detector_batch_size}

    @override
    def _new_state(self) -> _Tracker:
        return self._tracker_type(**self.tracker_kwargs)

    @override
    def _update(self, state: _Tracker, detections: ObjectDetectionTarget, frame: VideoFrame) -> MOTFrameTarget:
        return self._to_frame_target(state.update(self._to_detections(detections)))

    def _to_detections(self, prediction: ObjectDetectionTarget) -> _Detections:
        boxes = np.asarray(prediction.boxes, dtype=np.float32)
        labels = np.asarray(prediction.labels, dtype=np.int64)
        scores = _confidence_scores(prediction.scores, labels)
        _validate_detection_arrays(boxes=boxes, labels=labels, scores=scores)
        return self._supervision.Detections(xyxy=boxes, class_id=labels, confidence=scores)

    @staticmethod
    def _to_frame_target(detections: _Detections) -> MOTFrameTarget:
        if detections.class_id is None or detections.confidence is None or detections.tracker_id is None:
            raise RuntimeError("ByteTrack returned detections without class IDs, confidence scores, or tracker IDs.")
        return MOTFrameTarget(
            boxes=np.asarray(detections.xyxy, dtype=np.float32),
            labels=np.asarray(detections.class_id, dtype=np.int64),
            scores=np.asarray(detections.confidence, dtype=np.float32),
            track_ids=np.asarray(detections.tracker_id, dtype=np.int64),
        )


def _import_tracking_dependencies() -> tuple[_SupervisionModule, _TrackerFactory]:
    try:
        supervision = importlib.import_module("supervision")
        trackers = importlib.import_module("trackers")
    except ModuleNotFoundError as e:
        if e.name in {"supervision", "trackers"}:
            raise ImportError(f"ByteTrack MOT models require optional dependency 'trackers'. {MOT_INSTALL_HINT}") from e
        raise
    return cast(_SupervisionModule, supervision), cast(_TrackerFactory, trackers.ByteTrackTracker)


def _confidence_scores(scores: ArrayLike, labels: NDArray[np.int64]) -> NDArray[np.float32]:
    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.ndim == 1:
        return score_array
    if score_array.ndim == 2 and score_array.shape[1] > 0:
        if labels.shape != (score_array.shape[0],):
            raise ValueError("Object-detection labels and per-class scores must contain the same number of detections.")
        if np.any(labels < 0) or np.any(labels >= score_array.shape[1]):
            raise ValueError("Object-detection labels must index a column in the per-class scores array.")
        return score_array[np.arange(score_array.shape[0]), labels]
    raise ValueError(
        "Object-detection scores must have shape (n_detections,) or (n_detections, n_classes) with at least one class."
    )


def _validate_detection_arrays(
    *,
    boxes: NDArray[np.float32],
    labels: NDArray[np.int64],
    scores: NDArray[np.float32],
) -> None:
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Object-detection boxes must have shape (n_detections, 4), got {boxes.shape}.")
    expected = len(boxes)
    if labels.shape != (expected,):
        raise ValueError(f"Object-detection labels must have shape ({expected},), got {labels.shape}.")
    if scores.shape != (expected,):
        raise ValueError(f"Object-detection scores must have shape ({expected},), got {scores.shape}.")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("Object-detection boxes must contain only finite values.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Object-detection scores must contain only finite values.")
