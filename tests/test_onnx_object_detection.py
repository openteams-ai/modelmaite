import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from maite.protocols import object_detection as od

from modelmaite.object_detection import DetectionTarget, OnnxODModel, load_models

ROOT = Path(__file__).parent / "data" / "jatic_onnx_od"
HAS_ONNX_RUNTIME = importlib.util.find_spec("onnxruntime") is not None
REQUIRE_ONNX_RUNTIME = os.environ.get("MODELMAITE_REQUIRE_ONNX_RUNTIME") == "1"
requires_onnx_runtime = pytest.mark.skipif(
    not HAS_ONNX_RUNTIME and not REQUIRE_ONNX_RUNTIME,
    reason="ONNX Runtime is optional.",
)


def _metadata():
    return {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": 1,
            "interface": "IMAGE_OBJECT_DETECTION",
            "input": {"channels": "RGB", "height": 10, "width": 20},
            "output": {"nBoxes": 2, "nClasses": 3},
        },
        "index2label": {"0": "background", "1": "cat", "2": "dog"},
    }


class _FakeOnnxSession:
    def run(self, output_names, feeds):
        assert output_names == ["boxes", "scores"]
        assert feeds["image"].shape == (1, 3, 10, 20)
        boxes = np.array([[[0.0, 0.1, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
        scores = np.array([[[0.1, 0.9, 0.2], [0.1, 0.3, 0.7]]], dtype=np.float32)
        return [boxes, scores]


class _BadShapeOnnxSession:
    def run(self, output_names, feeds):
        return [np.zeros((1, 2, 5), dtype=np.float32), np.zeros((1, 2, 3), dtype=np.float32)]


class _WrongBoxCountOnnxSession:
    def run(self, output_names, feeds):
        return [np.zeros((1, 3, 4), dtype=np.float32), np.zeros((1, 3, 3), dtype=np.float32)]


class _WrongClassCountOnnxSession:
    def run(self, output_names, feeds):
        return [np.zeros((1, 2, 4), dtype=np.float32), np.zeros((1, 2, 4), dtype=np.float32)]


class _NanBoxOnnxSession:
    def run(self, output_names, feeds):
        boxes = np.array([[[0.0, 0.0, np.nan, 1.0], [0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
        scores = np.array([[[0.1, 0.9, 0.2], [0.1, 0.3, 0.7]]], dtype=np.float32)
        return [boxes, scores]


class _OutOfRangeScoreOnnxSession:
    def run(self, output_names, feeds):
        boxes = np.array([[[0.0, 0.1, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
        scores = np.array([[[0.1, 1.2, 0.2], [0.1, 0.3, 0.7]]], dtype=np.float32)
        return [boxes, scores]


def _fake_model(session, *, n_boxes=2, n_classes=3):
    model = OnnxODModel.__new__(OnnxODModel)
    model.config = _metadata()
    model.index2label = {0: "background", 1: "cat", 2: "dog"}
    model.model = session
    model._model_name = "jatic_onnx"
    model._batch_size = None
    model._image_height = None
    model._image_width = None
    model._n_boxes = n_boxes
    model._n_classes = n_classes
    model.metadata = {"id": "fake", "index2label": model.index2label}
    return model


def test_onnx_od_model_converts_jatic_outputs_to_detection_targets_without_runtime():
    model = _fake_model(_FakeOnnxSession())

    (prediction,) = model([np.zeros((3, 10, 20), dtype=np.uint8)])

    assert isinstance(prediction, DetectionTarget)
    np.testing.assert_allclose(prediction.boxes, np.array([[0.0, 1.0, 10.0, 10.0], [10.0, 5.0, 20.0, 10.0]]))
    np.testing.assert_array_equal(prediction.labels, np.array([1, 2]))
    np.testing.assert_allclose(prediction.scores, np.array([0.9, 0.7]))


def test_onnx_od_model_satisfies_maite_protocol():
    model = _fake_model(_FakeOnnxSession())

    assert isinstance(model, od.Model)


def test_detection_target_satisfies_maite_protocol():
    prediction = DetectionTarget(
        boxes=np.zeros((1, 4), dtype=np.float32),
        labels=np.zeros(1, dtype=np.int64),
        scores=np.ones(1, dtype=np.float32),
    )

    assert isinstance(prediction, od.ObjectDetectionTarget)


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (_BadShapeOnnxSession(), "boxes"),
        (_WrongBoxCountOnnxSession(), "nBoxes"),
        (_WrongClassCountOnnxSession(), "index2label"),
        (_NanBoxOnnxSession(), "finite"),
        (_OutOfRangeScoreOnnxSession(), "range"),
    ],
)
def test_onnx_od_model_rejects_invalid_outputs(session, message):
    model = _fake_model(session)

    with pytest.raises(ValueError, match=message):
        model([np.zeros((3, 10, 20), dtype=np.uint8)])


@requires_onnx_runtime
def test_onnx_od_model_runs_constant_fixture():
    model = OnnxODModel(
        weights_path=ROOT / "constant_detector.onnx",
        config_path=ROOT / "model-metadata.json",
        device="cpu",
    )

    (prediction,) = model([np.zeros((3, 10, 20), dtype=np.uint8)])

    np.testing.assert_allclose(prediction.boxes, np.array([[0.0, 1.0, 10.0, 10.0], [10.0, 5.0, 20.0, 10.0]]))
    np.testing.assert_array_equal(prediction.labels, np.array([1, 2]))
    np.testing.assert_allclose(prediction.scores, np.array([0.9, 0.7]))
    assert model.name == "jatic_onnx"
    assert model.metadata["index2label"] == {0: "background", 1: "cat", 2: "dog"}


@requires_onnx_runtime
def test_load_models_dispatches_to_onnx_wrapper():
    loaded = load_models(
        {
            "onnx_model": {
                "model_type": "jatic_onnx",
                "model_weights_path": ROOT / "constant_detector.onnx",
                "model_config_path": ROOT / "model-metadata.json",
            }
        },
        device="cpu",
    )

    assert isinstance(loaded["onnx_model"], OnnxODModel)


def test_load_models_requires_model_type():
    with pytest.raises(ValueError, match="model_type"):
        load_models({"onnx_model": {}})


def test_load_models_requires_paths():
    with pytest.raises(ValueError, match="model_weights_path"):
        load_models({"onnx_model": {"model_type": "jatic_onnx"}})
