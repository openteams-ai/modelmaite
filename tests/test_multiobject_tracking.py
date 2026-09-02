from __future__ import annotations

import importlib.util
import json
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
HAS_DATAMAITE = importlib.util.find_spec("datamaite") is not None
REQUIRE_MOT = os.environ.get("MODELMAITE_REQUIRE_MOT") == "1"
# datamaite gets its own flag: the test-wheel job already sets MODELMAITE_REQUIRE_MOT=1
# and installs no datamaite, so reusing REQUIRE_MOT would fail that job instead of
# skipping. Set MODELMAITE_REQUIRE_DATAMAITE=1 only where datamaite is installed, so a
# forgotten install there fails loudly rather than skipping green.
REQUIRE_DATAMAITE = os.environ.get("MODELMAITE_REQUIRE_DATAMAITE") == "1"
requires_mot_runtime = pytest.mark.skipif(
    not HAS_MOT_RUNTIME and not REQUIRE_MOT,
    reason="MOT dependencies are optional.",
)
requires_mot_and_onnx_runtime = pytest.mark.skipif(
    not (HAS_MOT_RUNTIME and HAS_ONNX_RUNTIME) and not REQUIRE_MOT,
    reason="MOT and ONNX Runtime dependencies are optional.",
)
requires_datamaite = pytest.mark.skipif(
    not HAS_DATAMAITE and not REQUIRE_DATAMAITE,
    reason="datamaite is an optional test dependency.",
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


# ── datamaite image-sequence readers ──────────────────────────────────────────
# MOTChallenge, TAO, and VisDrone-video datasets store each video as a folder of
# images. datamaite streams those as MAITE VideoFrames, so ByteTrackMOTModel needs
# no reader of its own. These tests pin that contract end to end.

FRAME_SIZE = 32
# Written as pure red at increasing brightness, so one assertion covers both BGR->RGB
# conversion and frame ordering. PNG keeps it lossless, so the values are exact.
RED_LEVELS = (40, 100, 160)
# Every fixture annotates one track at xywh (4, 4, 8, 8) on every frame.
GT_BOX_XYXY = np.array([[4.0, 4.0, 12.0, 12.0]], dtype=np.float32)


def _write_frames(directory, names):
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for name, level in zip(names, RED_LEVELS, strict=True):
        frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frame[:, :, 2] = level  # OpenCV writes BGR, so channel 2 is red.
        assert cv2.imwrite(str(directory / name), frame)


def _build_motchallenge(root, gt_frames=(1, 2, 3)):
    sequence = root / "train" / "MOT17-02"
    _write_frames(sequence / "img1", [f"{frame:06d}.png" for frame in (1, 2, 3)])
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=MOT17-02",
                "imDir=img1",
                "frameRate=30",
                "seqLength=3",
                f"imWidth={FRAME_SIZE}",
                f"imHeight={FRAME_SIZE}",
                "imExt=.png",
                "",
            ]
        ),
        encoding="utf-8",
    )
    gt_dir = sequence / "gt"
    gt_dir.mkdir()
    # frame,id,x,y,w,h,conf,class,visibility -- class 1 is pedestrian.
    (gt_dir / "gt.txt").write_text(
        "".join(f"{frame},1,4,4,8,8,1,1,1\n" for frame in gt_frames),
        encoding="utf-8",
    )
    return {"dataset_format": "motchallenge"}


def _build_tao(root):
    # TAO resolves images.file_name under <root>/frames, so the frames must live
    # there -- not beside the annotations.
    _write_frames(root / "frames" / "train" / "video-a", [f"{frame:06d}.png" for frame in (1, 2, 3)])
    payload = {
        "videos": [{"id": 10, "name": "video-a", "width": FRAME_SIZE, "height": FRAME_SIZE, "fps": 30}],
        # One video with a dense frame_index: a sparse table would leave None holes
        # that the image-sequence stream cannot resolve.
        "images": [
            {
                "id": 100 + index,
                "video_id": 10,
                "file_name": f"train/video-a/{index + 1:06d}.png",
                "frame_index": index,
                "width": FRAME_SIZE,
                "height": FRAME_SIZE,
            }
            for index in range(3)
        ],
        "tracks": [{"id": 5, "video_id": 10, "category_id": 1}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 900 + index, "image_id": 100 + index, "track_id": 5, "category_id": 1, "bbox": [4, 4, 8, 8]}
            for index in range(3)
        ],
    }
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "train.json").write_text(json.dumps(payload), encoding="utf-8")
    return {"dataset_format": "tao"}


def _build_visdrone_video(root):
    split = root / "VisDrone2019-VID-train"
    _write_frames(split / "sequences" / "uav0000013_00000_v", [f"{frame:07d}.png" for frame in (1, 2, 3)])
    annotations = split / "annotations"
    annotations.mkdir(parents=True)
    # frame,track,x,y,w,h,score,category,truncation,occlusion -- category 4 is car.
    (annotations / "uav0000013_00000_v.txt").write_text(
        "".join(f"{frame},1,4,4,8,8,1,4,0,0\n" for frame in (1, 2, 3)),
        encoding="utf-8",
    )
    # VisDrone carries no frame rate of its own; without fps= every frame reports
    # time_s == 0.0.
    return {"dataset_format": "visdrone_video", "frame_ext": ".png", "fps": 30.0}


@requires_mot_runtime
@requires_datamaite
@pytest.mark.parametrize(
    ("build", "expected_first_frame", "expected_meta_size"),
    [
        # MOTChallenge reads dimensions from seqinfo.ini and TAO from the videos
        # entry. VisDrone has no dimension metadata and only counts frame files
        # unless probe_images=True, so its datum metadata falls back to 0.
        (_build_motchallenge, "000001.png", FRAME_SIZE),
        (_build_tao, "000001.png", FRAME_SIZE),
        (_build_visdrone_video, "0000001.png", 0),
    ],
    ids=["motchallenge", "tao", "visdrone_video"],
)
def test_bytetrack_mot_model_tracks_datamaite_image_sequence_datasets(
    tmp_path, build, expected_first_frame, expected_meta_size
):
    from datamaite import load_mot

    dataset = load_mot(tmp_path, **build(tmp_path))

    # Image-folder sequences became MAITE items in datamaite 0.5.0; 0.4.1 and
    # earlier report 0 here.
    assert len(dataset) == 1
    sequence = dataset.sequences[0]
    # Source frame numbering differs from the model's 0-based index (MOTChallenge and
    # VisDrone both name the first frame 1).
    assert sequence.frame_filename(0) == expected_first_frame

    stream, target, metadata = dataset[0]
    frames = list(stream)
    # The invariant everything else rests on: a dropped frame would silently pair
    # every later prediction against the wrong ground truth.
    assert len(frames) == len(target.frame_tracks) == 3
    for frame, level in zip(frames, RED_LEVELS, strict=True):
        assert frame.pixels.shape == (3, FRAME_SIZE, FRAME_SIZE)
        assert frame.pixels.dtype == np.uint8
        red, green, blue = (round(channel.mean()) for channel in frame.pixels)
        assert (red, green, blue) == (level, 0, 0)

    # Loaders store xywh; the MAITE view must hand models pixel xyxy.
    np.testing.assert_allclose(target.frame_tracks[0].boxes, GT_BOX_XYXY)
    assert metadata["height"] == expected_meta_size
    assert metadata["width"] == expected_meta_size

    model = ByteTrackMOTModel(
        detector=_MovingDetector(),
        tracker_kwargs={"track_activation_threshold": 0.5},
    )
    prediction_batches, _ = predict(model=model, dataset=dataset)

    (prediction,) = prediction_batches[0]
    _assert_maite_mot_target(prediction)
    assert len(prediction.frame_tracks) == len(target.frame_tracks)
    # ByteTrack reports the first detection as unconfirmed, then keeps the track.
    np.testing.assert_array_equal(prediction.frame_tracks[0].track_ids, np.array([-1]))
    np.testing.assert_array_equal(prediction.frame_tracks[1].track_ids, np.array([0]))
    np.testing.assert_array_equal(prediction.frame_tracks[2].track_ids, np.array([0]))


@requires_mot_runtime
@requires_datamaite
def test_empty_frame_policy_all_streams_the_frames_the_docs_recommend(tmp_path):
    """The docs tell users to prefer ``"all"``; pin what that changes.

    Frame 2 carries no annotation, so ``"annotated"`` (the default) drops it and
    hands ByteTrack a temporal gap. Fixtures that annotate every frame make the
    two policies identical and prove nothing.
    """
    from datamaite import load_mot

    dataset = load_mot(tmp_path, **_build_motchallenge(tmp_path, gt_frames=(1, 3)))
    every_frame = dataset.with_mot_options(empty_frame_policy="all")

    stream, target, _ = dataset[0]
    assert len(list(stream)) == len(target.frame_tracks) == 2

    stream, target, _ = every_frame[0]
    frames = list(stream)
    assert len(frames) == len(target.frame_tracks) == 3
    # seqinfo.ini's seqLength makes the count exact, so "all" is not downgraded.
    assert [round(frame.pixels[0].mean()) for frame in frames] == list(RED_LEVELS)
    assert len(np.asarray(target.frame_tracks[1].boxes)) == 0  # the unannotated frame
    np.testing.assert_allclose(target.frame_tracks[2].boxes, GT_BOX_XYXY)

    model = ByteTrackMOTModel(
        detector=_MovingDetector(),
        tracker_kwargs={"track_activation_threshold": 0.5},
    )
    prediction_batches, _ = predict(model=model, dataset=every_frame)

    (prediction,) = prediction_batches[0]
    assert len(prediction.frame_tracks) == 3
    np.testing.assert_array_equal(prediction.frame_tracks[2].track_ids, np.array([0]))
