from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from maite.protocols import multiobject_tracking as mot
from maite.tasks import predict

import modelmaite.multiobject_tracking.models as mot_models
from modelmaite.multiobject_tracking import ByteTrackMOTModel, MOTFrameTarget, MOTTarget
from modelmaite.multiobject_tracking.models import _confidence_scores, _validate_detection_arrays
from modelmaite.object_detection import DetectionTarget, OnnxODModel

ROOT = Path(__file__).parent / "data" / "jatic_onnx_od"
HAS_MOT_RUNTIME = (
    importlib.util.find_spec("supervision") is not None and importlib.util.find_spec("trackers") is not None
)
HAS_ONNX_RUNTIME = importlib.util.find_spec("onnxruntime") is not None
REQUIRE_MOT = os.environ.get("MODELMAITE_REQUIRE_MOT") == "1"
requires_mot_runtime = pytest.mark.skipif(
    not HAS_MOT_RUNTIME and not REQUIRE_MOT,
    reason="MOT dependencies are optional.",
)
requires_mot_and_onnx_runtime = pytest.mark.skipif(
    not (HAS_MOT_RUNTIME and HAS_ONNX_RUNTIME) and not REQUIRE_MOT,
    reason="MOT and ONNX Runtime dependencies are optional.",
)


@dataclass
class _VideoFrame:
    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int


class _MovingDetector:
    """Small deterministic MAITE OD model used with the real ByteTrack implementation."""

    metadata = {"id": "moving-detector", "index2label": {0: "background", 1: "object"}}

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, input_batch):
        self.batch_sizes.append(len(input_batch))
        return [
            DetectionTarget(
                boxes=np.array([[1, 2, 5, 7]], dtype=np.float32),
                labels=np.array([1], dtype=np.int64),
                scores=np.array([[0.1, 0.9]], dtype=np.float32),
            )
            for _ in input_batch
        ]


class _BadDetector(_MovingDetector):
    def __call__(self, input_batch):
        return []


def _frames(count=3):
    return [
        _VideoFrame(
            pixels=np.zeros((3, 10, 20), dtype=np.uint8),
            time_s=index / 30,
            pts=index,
            frame_index=index,
        )
        for index in range(count)
    ]


def _assert_maite_mot_target(target):
    assert hasattr(target, "frame_tracks")
    for frame_target in target.frame_tracks:
        assert all(hasattr(frame_target, field) for field in ("boxes", "labels", "scores", "track_ids"))


@requires_mot_runtime
def test_bytetrack_mot_model_tracks_any_maite_detector_and_resets_per_video():
    detector = _MovingDetector()
    model = ByteTrackMOTModel(
        detector=detector,
        detector_batch_size=2,
        tracker_kwargs={"track_activation_threshold": 0.5},
    )

    first, second = model([_frames(), _frames(3)])

    assert isinstance(model, mot.Model)
    assert model.metadata == {
        "id": "bytetrack_moving-detector",
        "index2label": {0: "background", 1: "object"},
    }
    assert detector.batch_sizes == [2, 1, 2, 1]
    _assert_maite_mot_target(first)
    _assert_maite_mot_target(second)
    assert len(first.frame_tracks) == 3
    assert len(second.frame_tracks) == 3
    np.testing.assert_array_equal(first.frame_tracks[0].track_ids, np.array([-1]))
    np.testing.assert_array_equal(first.frame_tracks[1].track_ids, np.array([0]))
    np.testing.assert_array_equal(first.frame_tracks[2].track_ids, np.array([0]))
    np.testing.assert_array_equal(second.frame_tracks[0].track_ids, np.array([-1]))
    np.testing.assert_array_equal(second.frame_tracks[1].track_ids, np.array([0]))
    np.testing.assert_allclose(first.frame_tracks[0].scores, np.array([0.9], dtype=np.float32))


def test_bytetrack_mot_model_rejects_invalid_detector_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        ByteTrackMOTModel(detector=_MovingDetector(), detector_batch_size=0)


def test_bytetrack_mot_model_requires_detector_metadata():
    with pytest.raises(TypeError, match="metadata"):
        ByteTrackMOTModel(detector=object())


def test_bytetrack_mot_model_has_actionable_optional_dependency_error(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(mot_models.importlib, "import_module", missing_import)

    with pytest.raises(ImportError, match=r"modelmaite\[mot\]"):
        ByteTrackMOTModel(detector=_MovingDetector())


@requires_mot_runtime
def test_bytetrack_mot_model_requires_one_prediction_per_frame():
    model = ByteTrackMOTModel(detector=_BadDetector())

    with pytest.raises(ValueError, match="returned 0 predictions"):
        model([_frames(1)])


def test_confidence_scores_accepts_per_detection_and_selects_labeled_class_scores():
    labels = np.array([1, 1], dtype=np.int64)

    np.testing.assert_allclose(_confidence_scores(np.array([0.3, 0.7]), labels), np.array([0.3, 0.7]))
    np.testing.assert_allclose(
        _confidence_scores(np.array([[0.1, 0.9], [0.6, 0.4]]), labels),
        np.array([0.9, 0.4]),
    )


def test_confidence_scores_rejects_labels_outside_class_score_columns():
    with pytest.raises(ValueError, match="index a column"):
        _confidence_scores(np.array([[0.1, 0.9]]), np.array([2], dtype=np.int64))


@pytest.mark.parametrize(
    ("boxes", "labels", "scores", "message"),
    [
        (np.zeros((1, 5)), np.zeros(1), np.zeros(1), "boxes"),
        (np.zeros((1, 4)), np.zeros(2), np.zeros(1), "labels"),
        (np.zeros((1, 4)), np.zeros(1), np.zeros(2), "scores"),
        (np.array([[0.0, 0.0, np.nan, 1.0]]), np.zeros(1), np.ones(1), "finite"),
        (np.zeros((1, 4)), np.zeros(1), np.array([np.inf]), "finite"),
    ],
)
def test_validate_detection_arrays_rejects_invalid_values(boxes, labels, scores, message):
    with pytest.raises(ValueError, match=message):
        _validate_detection_arrays(boxes=boxes, labels=labels, scores=scores)


class _TinyMOTDataset:
    metadata = {"id": "tiny-video", "index2label": {0: "background", 1: "cat", 2: "dog"}}

    def __init__(self):
        empty_frame = MOTFrameTarget(
            boxes=np.empty((0, 4), dtype=np.float32),
            labels=np.empty(0, dtype=np.int64),
            scores=np.empty(0, dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
        )
        self.target = MOTTarget(frame_tracks=[empty_frame for _ in range(3)])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return _frames(), self.target, {"id": "video-0"}


@requires_mot_and_onnx_runtime
def test_bytetrack_mot_model_runs_end_to_end_with_real_od_model_and_maite_predict():
    detector = OnnxODModel(
        weights_path=ROOT / "constant_detector.onnx",
        config_path=ROOT / "model-metadata.json",
        device="cpu",
    )
    model = ByteTrackMOTModel(
        detector=detector,
        tracker_kwargs={"track_activation_threshold": 0.5},
    )

    prediction_batches, _ = predict(model=model, dataset=_TinyMOTDataset())

    (prediction,) = prediction_batches[0]
    _assert_maite_mot_target(prediction)
    assert len(prediction.frame_tracks) == 3
    np.testing.assert_array_equal(prediction.frame_tracks[0].track_ids, np.array([-1, -1]))
    np.testing.assert_array_equal(prediction.frame_tracks[1].track_ids, np.array([0, 1]))
    np.testing.assert_array_equal(prediction.frame_tracks[2].track_ids, np.array([0, 1]))
    np.testing.assert_array_equal(prediction.frame_tracks[1].labels, np.array([1, 2]))
