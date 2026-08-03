"""MAITE-compliant multi-object-tracking model wrappers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol, TypedDict, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modelmaite.multiobject_tracking.types import MOTFrameTarget, MOTTarget

_ModelMetadataValue = str | dict[int, str]
_T = TypeVar("_T")


class _ObjectDetectionTarget(Protocol):
    @property
    def boxes(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


class _ObjectDetectionModel(Protocol):
    @property
    def metadata(self) -> Mapping[str, _ModelMetadataValue]: ...

    def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[_ObjectDetectionTarget]: ...


class _VideoFrame(Protocol):
    @property
    def pixels(self) -> ArrayLike: ...

    @property
    def time_s(self) -> float: ...

    @property
    def pts(self) -> int: ...

    @property
    def frame_index(self) -> int: ...


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


class ByteTrackMOTModel:
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
        Optional keyword arguments forwarded to ``trackers.ByteTrackTracker``.
    model_id
        Optional model identifier. By default, it is derived from the detector's
        metadata ID.

    Notes
    -----
    ByteTrack associates detections using geometry and confidence, not class
    labels. Consequently, one track ID may carry different labels across frames.
    The returned targets also follow the upstream tracker's filtering behavior:
    unmatched detections between ``high_conf_det_threshold`` and
    ``track_activation_threshold`` may be omitted rather than returned with a
    track ID of ``-1``.
    """

    def __init__(
        self,
        *,
        detector: _ObjectDetectionModel,
        detector_batch_size: int = 1,
        tracker_kwargs: ByteTrackConfig | None = None,
        model_id: str | None = None,
    ) -> None:
        if detector_batch_size < 1:
            raise ValueError("detector_batch_size must be at least 1.")

        detector_metadata = getattr(detector, "metadata", None)
        if not isinstance(detector_metadata, Mapping) or not isinstance(detector_metadata.get("id"), str):
            raise TypeError("detector must expose MAITE model metadata containing a string 'id'.")

        self.detector = detector
        self.detector_batch_size = detector_batch_size
        self.tracker_kwargs: ByteTrackConfig = tracker_kwargs.copy() if tracker_kwargs is not None else {}
        self._supervision, self._tracker_type = _import_tracking_dependencies()

        metadata: dict[str, _ModelMetadataValue] = {"id": model_id or detector_metadata["id"]}
        if "index2label" in detector_metadata:
            metadata["index2label"] = _validate_index2label(detector_metadata["index2label"])
        if model_id is None:
            metadata["id"] = f"bytetrack_{metadata['id']}"
        self.metadata = metadata

    def __call__(self, input_batch: Sequence[Iterable[_VideoFrame]]) -> Sequence[MOTTarget]:
        """Track objects in a batch of MAITE video streams."""
        return [self._track_stream(stream) for stream in input_batch]

    def _track_stream(self, stream: Iterable[_VideoFrame]) -> MOTTarget:
        tracker = self._tracker_type(**self.tracker_kwargs)
        frame_tracks: list[MOTFrameTarget] = []

        for frames in _batched(stream, self.detector_batch_size):
            predictions = list(self.detector([frame.pixels for frame in frames]))
            if len(predictions) != len(frames):
                raise ValueError(
                    f"Object detector returned {len(predictions)} predictions for a batch of {len(frames)} frames."
                )
            for prediction in predictions:
                detections = self._to_detections(prediction)
                tracked = tracker.update(detections)
                frame_tracks.append(self._to_frame_target(tracked))

        return MOTTarget(frame_tracks=frame_tracks)

    def _to_detections(self, prediction: _ObjectDetectionTarget) -> _Detections:
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


def _batched(stream: Iterable[_T], batch_size: int) -> Iterator[list[_T]]:
    batch: list[_T] = []
    for frame in stream:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _validate_index2label(value: object) -> dict[int, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(index, int) and isinstance(label, str) for index, label in value.items()
    ):
        raise TypeError("detector metadata 'index2label' must map integer indices to string labels.")
    return dict(value)


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
