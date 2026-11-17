"""Tests for the shared wrapper base classes, using small numpy-only subclasses."""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass

import numpy as np
import pytest

from modelmaite._identity import id_hash
from modelmaite.base import ImageClassifierBase, ImageModelBase, ObjectDetectorBase, TrackerBase, normalize_device
from modelmaite.multiobject_tracking import MOTFrameTarget, MOTTarget
from modelmaite.object_detection import DetectionTarget


class _MeanModel(ImageModelBase[float, np.ndarray, np.ndarray]):
    """Return the mean pixel value of each image and record how it was called."""

    def __init__(self, *, batch_size=None, device=None, model_id=None, index2label=None, offset=0.0):
        super().__init__(device=device, batch_size=batch_size)
        self.offset = offset
        self.calls = []
        self._init_metadata(model_id=model_id, index2label=index2label)

    @property
    def name(self):
        return "mean"

    def _id_fields(self):
        return {"offset": self.offset}

    @contextlib.contextmanager
    def _inference_context(self):
        self.calls.append("enter")
        yield
        self.calls.append("exit")

    def preprocess(self, images):
        self.calls.append(("preprocess", len(images)))
        return np.stack([np.asarray(image, dtype=np.float64).mean() for image in images])

    def forward(self, prepared):
        self.calls.append("forward")
        return prepared + self.offset

    def postprocess(self, raw, prepared):
        self.calls.append("postprocess")
        return [float(value) for value in raw]


def _images(count, shape=(3, 4, 5)):
    return [np.full(shape, index, dtype=np.uint8) for index in range(count)]


def test_empty_batch_returns_empty_list_without_running_the_model():
    model = _MeanModel()

    assert model([]) == []
    assert model.calls == []


def test_batch_can_be_passed_by_keyword():
    assert _MeanModel()(input_batch=_images(1)) == [0.0]


def test_batches_are_split_into_chunks_and_outputs_keep_input_order():
    model = _MeanModel(batch_size=2)

    outputs = model(_images(5))

    assert outputs == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [call[1] for call in model.calls if isinstance(call, tuple)] == [2, 2, 1]


def test_whole_batch_runs_at_once_without_batch_size():
    model = _MeanModel()

    model(_images(5))

    assert model.calls == [("preprocess", 5), "enter", "forward", "exit", "postprocess"]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        _MeanModel(batch_size=batch_size)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((4, 5), dtype=np.uint8), "CHW-ordering"),
        (np.zeros((6, 5, 3), dtype=np.uint8), "CHW images with 1, 2, 3 or 4 channels"),
        (np.zeros((3, 4, 5), dtype=bool), "integer or floating-point"),
        (np.zeros((3, 4, 5), dtype=object), "integer or floating-point"),
    ],
)
def test_invalid_images_are_rejected_before_the_model_runs(image, message):
    model = _MeanModel()

    with pytest.raises(ValueError, match=message):
        model([image])
    assert model.calls == []


@pytest.mark.parametrize(
    ("bad_image", "message"),
    [
        (np.zeros((3, 0, 5), dtype=np.uint8), "image 1 has zero height or width"),
        (np.zeros((3, 4, 0), dtype=np.float32), "image 1 has zero height or width"),
        (np.full((3, 4, 5), np.nan, dtype=np.float32), "image 1 contains NaN or infinite"),
        (np.full((3, 4, 5), np.inf, dtype=np.float64), "image 1 contains NaN or infinite"),
    ],
)
def test_every_image_is_checked_before_the_first_chunk_runs(bad_image, message):
    model = _MeanModel(batch_size=1)

    with pytest.raises(ValueError, match=message):
        model([np.zeros((3, 4, 5), dtype=np.uint8), bad_image])
    assert model.calls == []


def test_input_range_is_checked_for_every_image_before_the_model_runs(monkeypatch):
    monkeypatch.setattr(_MeanModel, "input_range", (0.0, 1.0))
    model = _MeanModel(batch_size=1)

    with pytest.raises(ValueError, match=r"image 1 has values in \[2.0, 2.0\], outside the expected input range"):
        model([np.zeros((3, 4, 5), dtype=np.float32), np.full((3, 4, 5), 2.0, dtype=np.float32)])
    assert model.calls == []
    assert model([np.ones((3, 4, 5), dtype=np.float32)]) == [1.0]


def test_channel_counts_come_from_input_channels(monkeypatch):
    model = _MeanModel()
    monkeypatch.setattr(_MeanModel, "input_channels", (3,))
    with pytest.raises(ValueError, match="exactly 3 channels"):
        model([np.zeros((1, 4, 5), dtype=np.uint8)])

    monkeypatch.setattr(_MeanModel, "input_channels", (1, 3, 4))
    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        model([np.zeros((2, 4, 5), dtype=np.uint8)])


def test_input_channels_can_allow_more_than_four_channels(monkeypatch):
    monkeypatch.setattr(_MeanModel, "input_channels", (8,))

    assert _MeanModel()([np.ones((8, 4, 5), dtype=np.uint8)]) == [1.0]


def test_mixed_image_sizes_are_allowed_unless_same_shape_batches_is_set(monkeypatch):
    model = _MeanModel()
    batch = [np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 6, 7), dtype=np.uint8)]

    assert model(batch) == [0.0, 1.0]

    monkeypatch.setattr(_MeanModel, "same_shape_batches", True)
    model.calls.clear()
    with pytest.raises(ValueError, match=r"must have the same shape; got \(3, 4, 5\) and \(3, 6, 7\)"):
        model(batch)
    assert model.calls == []


def test_same_shape_batches_applies_to_each_forward_call(monkeypatch):
    monkeypatch.setattr(_MeanModel, "same_shape_batches", True)
    small, large = np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 6, 7), dtype=np.uint8)

    assert _MeanModel(batch_size=1)([small, large]) == [0.0, 1.0]

    model = _MeanModel(batch_size=2)
    with pytest.raises(ValueError, match="same shape"):
        model([small, small, small, large])
    assert model.calls == []


def test_output_count_must_match_input_count(monkeypatch):
    model = _MeanModel()
    monkeypatch.setattr(model, "_predict", lambda items: [0.0])

    with pytest.raises(ValueError, match="returned 1 outputs for 2 inputs"):
        model(_images(2))


def test_output_count_is_checked_for_each_chunk(monkeypatch):
    model = _MeanModel(batch_size=2)
    chunk_outputs = iter([[], [2.0, 2.0, 3.0, 3.0]])
    monkeypatch.setattr(model, "postprocess", lambda raw, prepared: next(chunk_outputs))

    with pytest.raises(ValueError, match="returned 0 outputs for a chunk of 2 images"):
        model(_images(4))


def test_a_chunks_prepared_input_and_raw_output_are_released_before_the_next_chunk(monkeypatch):
    model = _MeanModel(batch_size=1)
    previous = []
    alive_at_preprocess = []
    preprocess, forward = model.preprocess, model.forward

    def tracking_preprocess(images):
        alive_at_preprocess.append([ref() is not None for ref in previous])
        prepared = preprocess(images)
        previous.append(weakref.ref(prepared))
        return prepared

    def tracking_forward(prepared):
        raw = forward(prepared)
        previous.append(weakref.ref(raw))
        return raw

    monkeypatch.setattr(model, "preprocess", tracking_preprocess)
    monkeypatch.setattr(model, "forward", tracking_forward)

    assert model(_images(3)) == [0.0, 1.0, 2.0]
    assert alive_at_preprocess == [[], [False, False], [False, False, False, False]]


def test_outputs_are_checked_before_the_next_chunk_runs(monkeypatch):
    model = _MeanModel(batch_size=1)

    def reject_one(output):
        if output == 1.0:
            raise ValueError("mean scores must not be 1.")

    monkeypatch.setattr(model, "_validate_output", reject_one)

    with pytest.raises(ValueError, match=r"must not be 1\. \(image 1 of the batch\)"):
        model(_images(4))
    assert model.calls.count("forward") == 2


def test_a_chunk_plan_groups_images_and_outputs_keep_input_order(monkeypatch):
    model = _MeanModel(batch_size=2)
    # Even indices first, then odd: as a model grouping images by shape might.
    monkeypatch.setattr(model, "_plan_chunks", lambda images, size: [[0, 2], [1, 3]])

    assert model(_images(4)) == [0.0, 1.0, 2.0, 3.0]
    assert [call for call in model.calls if isinstance(call, tuple)] == [("preprocess", 2), ("preprocess", 2)]


def test_a_chunk_plan_names_the_input_index_in_output_errors(monkeypatch):
    model = _MeanModel(batch_size=2)
    monkeypatch.setattr(model, "_plan_chunks", lambda images, size: [[3, 1], [0, 2]])

    def reject_one(output):
        if output == 1.0:
            raise ValueError("mean scores must not be 1.")

    monkeypatch.setattr(model, "_validate_output", reject_one)

    with pytest.raises(ValueError, match=r"\(image 1 of the batch\)"):
        model(_images(4))


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ([[0, 1], [1, 2]], "exactly once"),
        ([[0, 1, 2]], "chunk of 3 images; limit 2"),
        ([[0], [], [1, 2]], "chunk of 0 images"),
    ],
)
def test_a_bad_chunk_plan_is_rejected_before_the_model_runs(monkeypatch, plan, message):
    model = _MeanModel(batch_size=2)
    monkeypatch.setattr(model, "_plan_chunks", lambda images, size: plan)

    with pytest.raises(ValueError, match=message):
        model(_images(3))
    assert model.calls == []


def test_default_id_hashes_id_fields_and_model_id_wins():
    assert _MeanModel(offset=1.0).metadata == {"id": f"mean_{id_hash(offset=1.0)}"}
    assert _MeanModel(offset=2.0).metadata["id"] != _MeanModel(offset=1.0).metadata["id"]
    assert _MeanModel(model_id="custom").metadata == {"id": "custom"}

    with pytest.raises(TypeError, match="model_id"):
        _MeanModel(model_id=3)


class _IoU:
    """Stands in for a tracker's IoU option: an object whose attributes are its settings."""

    def __init__(self, buffer_ratio=0.1):
        self.buffer_ratio = buffer_ratio


class _OtherIoU(_IoU):
    pass


def test_id_hash_canonicalises_objects_by_class_and_attributes():
    assert id_hash(iou=_IoU()) == id_hash(iou=_IoU())
    assert id_hash(iou=_IoU(buffer_ratio=0.2)) != id_hash(iou=_IoU())
    assert id_hash(iou=_OtherIoU()) != id_hash(iou=_IoU())
    assert id_hash(estimator=_IoU) == id_hash(estimator=_IoU)
    assert id_hash(estimator=_IoU) != id_hash(estimator=_OtherIoU)
    assert len(id_hash(offset=1.0)) == 16


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (lambda: None, "'fn'"),
        (object(), "'fn'"),
        ({1: "one"}, "'fn'"),
        ([_IoU(buffer_ratio=object())], r"'fn\[0\]\.buffer_ratio'"),
    ],
)
def test_id_hash_rejects_values_without_a_canonical_form(value, field):
    with pytest.raises(TypeError, match=rf"setting {field}.*Pass model_id"):
        id_hash(fn=value)


def test_without_id_fields_model_id_is_required(monkeypatch):
    monkeypatch.setattr(_MeanModel, "_id_fields", lambda self: {})

    with pytest.raises(ValueError, match="needs model_id, or an _id_fields"):
        _MeanModel()
    assert _MeanModel(model_id="custom").metadata == {"id": "custom"}


def test_index2label_is_checked_and_kept_as_the_same_dict():
    index2label = {0: "dark", 1: "bright"}

    model = _MeanModel(index2label=index2label)

    assert model.metadata["index2label"] is index2label
    with pytest.raises(TypeError, match="index2label must map integer indices to string labels"):
        _MeanModel(index2label={"0": "dark"})


def test_extra_metadata_keys_are_added_but_cannot_replace_maite_keys():
    model = _MeanModel(model_id="m")
    model._init_metadata(model_id="m", index2label={0: "dark"}, extra={"framework": "numpy", "version": 2})

    assert model.metadata == {"id": "m", "index2label": {0: "dark"}, "framework": "numpy", "version": 2}
    for key in ("id", "index2label"):
        with pytest.raises(ValueError, match=f"cannot set '{key}': ModelMetadata defines it"):
            model._init_metadata(model_id="m", extra={key: "x"})
    with pytest.raises(TypeError, match="keys must be strings"):
        model._init_metadata(model_id="m", extra={1: "x"})


class _TorchLikeDevice:
    def __str__(self):
        return "cuda:1"


@pytest.mark.parametrize(
    ("device", "expected"),
    [(None, None), ("cpu", "cpu"), ("CUDA:1", "cuda:1"), (_TorchLikeDevice(), "cuda:1"), (0, 0)],
)
def test_normalize_device(device, expected):
    assert normalize_device(device) == expected
    assert _MeanModel(device=device).device == expected


def test_normalize_device_rejects_bool_and_empty_string():
    with pytest.raises(TypeError, match="device"):
        normalize_device(True)
    with pytest.raises(ValueError, match="device"):
        normalize_device("")


class _FixedClassifier(ImageClassifierBase[int, np.ndarray]):
    """Return the same scores for every image."""

    def __init__(self, scores, *, index2label=None):
        super().__init__()
        self.scores = np.asarray(scores)
        self._init_metadata(model_id="fixed", index2label=index2label)

    @property
    def name(self):
        return "fixed"

    def preprocess(self, images):
        return len(images)

    def forward(self, prepared):
        return np.stack([self.scores] * prepared)

    def postprocess(self, raw, prepared):
        return list(raw)


def test_classifier_requires_index2label():
    with pytest.raises(ValueError, match="_FixedClassifier needs index2label"):
        _FixedClassifier(np.array([0.5, 0.5], dtype=np.float32))


@pytest.mark.parametrize("index2label", [{1: "cat", 2: "dog"}, {0: "cat", 2: "dog"}])
def test_classifier_index2label_keys_must_be_score_positions(index2label):
    with pytest.raises(ValueError, match=r"keys must be exactly 0..1, got \[\d, 2\]: score i is the score of label i"):
        _FixedClassifier(np.array([0.5, 0.5], dtype=np.float32), index2label=index2label)


def test_classifier_index2label_keys_may_be_in_any_order():
    model = _FixedClassifier(np.array([0.5, 0.5], dtype=np.float32), index2label={1: "dog", 0: "cat"})

    assert model.metadata["index2label"] == {0: "cat", 1: "dog"}


def test_detector_index2label_keys_need_not_be_score_positions():
    assert _FixedDetector(_detections(np.zeros((0, 4)), [], []), index2label={1: "cat", 2: "dog"})


def test_detector_requires_index2label():
    with pytest.raises(ValueError, match="_FixedDetector needs index2label"):
        _FixedDetector(_detections(np.zeros((0, 4)), [], []))


def test_requires_index2label_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(_FixedClassifier, "requires_index2label", False)
    monkeypatch.setattr(_FixedDetector, "requires_index2label", False)
    target = _detections(np.zeros((1, 4)), [0], [[0.4, 0.6]])

    assert len(_FixedClassifier(np.array([0.25, 0.75], dtype=np.float32))(_images(1))) == 1
    assert _FixedDetector(target)(_images(1)) == [target]


def test_classifier_returns_one_score_array_per_image():
    model = _FixedClassifier(np.array([0.25, 0.75], dtype=np.float32), index2label={0: "a", 1: "b"})

    outputs = model(_images(2))

    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[1], np.array([0.25, 0.75], dtype=np.float32))


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.array([[0.5, 0.5]], dtype=np.float32), "1-D float array"),
        (np.array([0, 1], dtype=np.int64), "1-D float array"),
        (np.array([0.2, 0.3, 0.5], dtype=np.float32), "index2label has 2 classes"),
        (np.array([np.nan, 0.5], dtype=np.float32), "finite"),
        (np.array([1.5, 0.5], dtype=np.float32), r"range \[0, 1\]"),
        (np.array([-0.5, 0.5], dtype=np.float32), r"range \[0, 1\]"),
    ],
)
def test_classifier_rejects_invalid_scores(scores, message):
    model = _FixedClassifier(scores, index2label={0: "a", 1: "b"})

    with pytest.raises(ValueError, match=message):
        model(_images(1))


class _FixedDetector(ObjectDetectorBase[DetectionTarget, int, DetectionTarget]):
    """Return the same detections for every image."""

    def __init__(self, target, *, index2label=None):
        super().__init__()
        self.target = target
        self._init_metadata(model_id="fixed", index2label=index2label)

    @property
    def name(self):
        return "fixed"

    def preprocess(self, images):
        return len(images)

    def forward(self, prepared):
        return self.target

    def postprocess(self, raw, prepared):
        return [raw] * prepared


def _detections(boxes, labels, scores):
    return DetectionTarget(
        boxes=np.asarray(boxes, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "target",
    [
        _detections(np.zeros((2, 4)), [0, 1], [0.5, 1.0]),
        _detections(np.zeros((0, 4)), [], []),
        _detections(np.zeros((1, 4)), [1], [[0.4, 0.6]]),
        _detections(np.zeros((1, 4)), [1], [[0.1, 0.3, 0.6]]),
    ],
    ids=["per-box", "empty", "per-class", "per-class-with-background"],
)
def test_detector_accepts_maite_detection_targets(target):
    assert _FixedDetector(target, index2label={0: "a", 1: "b"})(_images(2)) == [target, target]


@dataclass(frozen=True)
class _TargetWithMasks:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    masks: np.ndarray


def test_detector_accepts_its_own_target_type():
    target = _TargetWithMasks(np.zeros((1, 4)), np.array([1]), np.array([0.5]), np.ones((1, 4, 5), dtype=bool))

    assert _FixedDetector(target, index2label={0: "a", 1: "b"})(_images(1)) == [target]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (_detections(np.zeros((1, 5)), [0], [0.5]), r"boxes must have shape \(n_boxes, 4\)"),
        (_detections(np.zeros((2, 4)), [0], [0.5, 0.5]), r"labels must have shape \(2,\)"),
        (_detections(np.zeros((2, 4)), [0, 1], [0.5]), r"scores must have shape \(2,\)"),
        (_detections(np.zeros((1, 4)), [0], [1.5]), "at most 1"),
        (_detections(np.zeros((1, 4)), [0], [[0.7, 0.6]]), "sum to at most 1"),
        (_detections(np.zeros((2, 4)), [0, 1], [np.nan, 1.5]), "scores must be finite"),
        (_detections(np.zeros((1, 4)), [0], [[np.nan, 0.2]]), "scores must be finite"),
        (_detections([[0, 0, np.inf, 1]], [0], [0.5]), "boxes must be finite"),
        (_detections(np.zeros((1, 4)), [0], [[0.9]]), "1 per-class scores, but index2label has 2"),
    ],
)
def test_detector_rejects_outputs_maite_rejects(target, message):
    with pytest.raises(ValueError, match=message):
        _FixedDetector(target, index2label={0: "a", 1: "b"})(_images(1))


@dataclass
class _Frame:
    pixels: np.ndarray
    time_s: float | None = None
    pts: int | None = None
    frame_index: int = 0


class _CountingDetector:
    metadata = {"id": "counting", "index2label": {0: "thing"}}

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(len(batch))
        return [_detections(np.zeros((1, 4)), [0], [0.5]) for _ in batch]


class _FrameCounter(TrackerBase[list]):
    """Give every detection the number of frames seen so far in its stream as track id."""

    tracker_name = "counter"

    def _id_fields(self):
        return {"start": 1}

    def _new_state(self):
        return []

    def _update(self, state, detections, frame):
        state.append(frame)
        n = len(np.asarray(detections.labels))
        return MOTFrameTarget(
            boxes=detections.boxes, labels=detections.labels, scores=detections.scores, track_ids=np.full(n, len(state))
        )


def _stream(n_frames):
    return [_Frame(pixels=np.zeros((3, 4, 5), dtype=np.uint8), frame_index=index) for index in range(n_frames)]


def test_tracker_batches_frames_and_starts_fresh_state_per_stream():
    detector = _CountingDetector()
    model = _FrameCounter(detector=detector, detector_batch_size=2)

    first, second = model([_stream(3), _stream(2)])

    assert isinstance(first, MOTTarget)
    assert detector.batch_sizes == [2, 1, 2]
    assert [int(frame.track_ids[0]) for frame in first.frame_tracks] == [1, 2, 3]
    assert [int(frame.track_ids[0]) for frame in second.frame_tracks] == [1, 2]
    assert model([]) == []


def test_tracker_metadata_comes_from_the_detector():
    detector = _CountingDetector()

    model = _FrameCounter(detector=detector)

    assert model.metadata == {"id": f"counter_counting_{id_hash(start=1)}", "index2label": {0: "thing"}}
    assert model.metadata["index2label"] is not detector.metadata["index2label"]
    assert _FrameCounter(detector=detector, model_id="mine").metadata["id"] == "mine"


def test_tracker_id_covers_its_id_fields_and_requires_them(monkeypatch):
    detector = _CountingDetector()
    first = _FrameCounter(detector=detector).metadata["id"]
    monkeypatch.setattr(_FrameCounter, "_id_fields", lambda self: {"start": 2})

    assert _FrameCounter(detector=detector).metadata["id"] != first

    monkeypatch.setattr(_FrameCounter, "_id_fields", lambda self: {})
    with pytest.raises(ValueError, match="needs model_id"):
        _FrameCounter(detector=detector)


def test_batch_settings_are_read_only_after_construction():
    model = _MeanModel(batch_size=2)
    tracker = _FrameCounter(detector=_CountingDetector(), detector_batch_size=2)

    with pytest.raises(AttributeError):
        model.batch_size = 3
    with pytest.raises(AttributeError):
        tracker.detector_batch_size = 3
    assert (model.batch_size, tracker.detector_batch_size) == (2, 2)


def test_tracker_checks_its_arguments():
    with pytest.raises(ValueError, match="detector_batch_size"):
        _FrameCounter(detector=_CountingDetector(), detector_batch_size=0)
    with pytest.raises(TypeError, match="metadata"):
        _FrameCounter(detector=object())


class _FixedTracks(_FrameCounter):
    """Return the given frame target for every frame."""

    frame_target = None

    def _update(self, state, detections, frame):
        return self.frame_target


@pytest.mark.parametrize(
    ("frame_target", "message"),
    [
        (
            MOTFrameTarget(boxes=np.zeros((1, 5)), labels=[0], scores=[0.5], track_ids=[1]),
            r"frame 0: boxes must have shape \(n_boxes, 4\)",
        ),
        (
            MOTFrameTarget(boxes=np.zeros((2, 4)), labels=[0, 1], scores=[0.5, 0.5], track_ids=[1]),
            "frame 0: track_ids must have 2 entries",
        ),
        (
            MOTFrameTarget(boxes=np.zeros((1, 4)), labels=[0], scores=[np.nan], track_ids=[1]),
            "frame 0: scores must be finite",
        ),
        (
            MOTFrameTarget(boxes=[[0, 0, np.nan, 1]], labels=[0], scores=[0.5], track_ids=[1]),
            "frame 0: boxes must be finite",
        ),
    ],
)
def test_tracker_rejects_frame_tracks_maite_rejects(monkeypatch, frame_target, message):
    monkeypatch.setattr(_FixedTracks, "frame_target", frame_target)

    with pytest.raises(ValueError, match=message):
        _FixedTracks(detector=_CountingDetector())([_stream(1)])


def test_tracker_checks_each_frame_before_running_the_next(monkeypatch):
    monkeypatch.setattr(
        _FixedTracks,
        "frame_target",
        MOTFrameTarget(boxes=np.zeros((1, 4)), labels=[0], scores=[np.nan], track_ids=[1]),
    )
    detector = _CountingDetector()

    with pytest.raises(ValueError, match="stream 0, frame 0: scores must be finite"):
        _FixedTracks(detector=detector)([_stream(3), _stream(3)])
    assert detector.batch_sizes == [1]


def test_tracker_accepts_empty_frames():
    empty = MOTFrameTarget(
        boxes=np.zeros((0, 4)), labels=np.zeros(0), scores=np.zeros(0), track_ids=np.zeros(0, dtype=np.int64)
    )
    model = _FixedTracks(detector=_CountingDetector())
    model.frame_target = empty

    (output,) = model([_stream(2)])

    assert len(output.frame_tracks) == 2


def test_tracker_requires_one_prediction_per_frame():
    model = _FrameCounter(detector=_CountingDetector())
    model.detector = lambda batch: []

    with pytest.raises(ValueError, match="returned 0 predictions for a batch of 1 frames"):
        model([_stream(1)])


def test_fixed_batch_size_sets_the_chunk_size_and_rejects_other_batch_sizes_up_front(monkeypatch):
    model = _MeanModel(batch_size=3)
    monkeypatch.setattr(model, "_fixed_batch_size", lambda: 2)

    assert model(_images(4)) == [0.0, 1.0, 2.0, 3.0]
    assert [call[1] for call in model.calls if isinstance(call, tuple)] == [2, 2]

    model.calls.clear()
    with pytest.raises(ValueError, match="multiple of 2; got 3 images"):
        model(_images(3))
    assert model.calls == []
