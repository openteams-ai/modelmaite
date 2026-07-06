from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from modelmaite.object_detection import DetectionTarget, TorchvisionODModel
from modelmaite.object_detection.models import load_models


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def to(self, device):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def __array__(self, dtype=None):
        if dtype is None:
            return self.array
        return self.array.astype(dtype)


class _NoGrad:
    entered = False

    def __enter__(self):
        _NoGrad.entered = True
        return

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeDetectionModel:
    instances = []

    def __init__(self, *, weights, weights_backbone="default", num_classes=None, **kwargs):
        self.weights = weights
        self.weights_backbone = weights_backbone
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.loaded_state_dict = None
        self.eval_called = False
        _FakeDetectionModel.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def eval(self):
        self.eval_called = True

    def __call__(self, batch):
        return [
            {
                "boxes": _FakeTensor(np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)),
                "labels": _FakeTensor(np.array([1], dtype=np.int64)),
                "scores": _FakeTensor(np.array([0.9], dtype=np.float32)),
            }
            for _ in range(len(batch))
        ]


def _fake_detection_transform(image):
    if isinstance(image, list):
        raise TypeError("Torchvision detection transforms accept one image tensor at a time.")
    return image


class _FakeWeightsDefault:
    meta = {"categories": ["background", "cat", "dog"]}

    @staticmethod
    def transforms():
        return _fake_detection_transform


class _FakeWeights:
    DEFAULT = _FakeWeightsDefault()


def _install_fake_torchvision(monkeypatch):
    _FakeDetectionModel.instances = []
    _NoGrad.entered = False

    torch = types.ModuleType("torch")
    torch.device = lambda device: str(device)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.as_tensor = lambda obj, device=None: _FakeTensor(obj)
    torch.stack = lambda tensors: _FakeTensor(np.stack([np.asarray(tensor) for tensor in tensors]))
    torch.load = lambda path, map_location=None, weights_only=True: {"loaded_from": str(path)}
    torch.no_grad = _NoGrad

    torchvision = types.ModuleType("torchvision")
    torchvision_models = types.ModuleType("torchvision.models")
    detection_models = types.ModuleType("torchvision.models.detection")
    detection_models.SSD300_VGG16_Weights = _FakeWeights
    detection_models.SSDLite320_MobileNet_V3_Large_Weights = _FakeWeights
    detection_models.ssd300_vgg16 = lambda **kwargs: _FakeDetectionModel(**kwargs)
    detection_models.ssdlite320_mobilenet_v3_large = lambda **kwargs: _FakeDetectionModel(**kwargs)
    torchvision_models.detection = detection_models
    torchvision.models = torchvision_models

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)
    monkeypatch.setitem(sys.modules, "torchvision.models.detection", detection_models)


def test_torchvision_od_model_returns_detection_targets(monkeypatch):
    _install_fake_torchvision(monkeypatch)
    model = TorchvisionODModel(model_name="ssd300_vgg16", device="cpu")

    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert model.name == "ssd300_vgg16"
    assert model.index2label == {0: "background", 1: "cat", 2: "dog"}
    assert model.metadata["id"].startswith("ssd300_vgg16_")
    assert _FakeDetectionModel.instances[-1].eval_called
    assert _NoGrad.entered
    assert len(predictions) == 2
    assert all(isinstance(prediction, DetectionTarget) for prediction in predictions)
    assert isinstance(predictions[0].boxes, np.ndarray)
    assert predictions[0].boxes.dtype == np.float32
    assert predictions[0].labels.dtype == np.int64
    assert predictions[0].scores.dtype == np.float32
    np.testing.assert_allclose(predictions[0].boxes, np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32))
    np.testing.assert_array_equal(predictions[0].labels, np.array([1]))
    np.testing.assert_allclose(predictions[0].scores, np.array([0.9], dtype=np.float32))


def test_torchvision_od_model_loads_user_weights(monkeypatch, tmp_path: Path):
    _install_fake_torchvision(monkeypatch)
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "weights.pt"
    config_path.write_text('{"index2label": {"0": "zero", "1": "one"}, "num_classes": 2}', encoding="utf-8")
    weights_path.write_text("unused by fake torch.load", encoding="utf-8")

    model = TorchvisionODModel(
        model_name="ssd300_vgg16", weights_path=weights_path, config_path=config_path, device="cpu"
    )

    assert model.index2label == {0: "zero", 1: "one"}
    assert _FakeDetectionModel.instances[-1].weights is None
    assert _FakeDetectionModel.instances[-1].weights_backbone is None
    assert _FakeDetectionModel.instances[-1].num_classes == 2
    assert _FakeDetectionModel.instances[-1].loaded_state_dict == {"loaded_from": str(weights_path)}


def test_torchvision_od_model_requires_config_for_user_weights(tmp_path: Path):
    weights_path = tmp_path / "weights.pt"
    weights_path.write_text("unused", encoding="utf-8")

    with pytest.raises(ValueError, match="config_path"):
        TorchvisionODModel(model_name="ssd300_vgg16", weights_path=weights_path)


def test_torchvision_od_model_requires_three_channel_rgb(monkeypatch):
    _install_fake_torchvision(monkeypatch)
    model = TorchvisionODModel(model_name="ssd300_vgg16", device="cpu")

    with pytest.raises(ValueError, match="exactly 3 channels"):
        model([np.zeros((1, 4, 5), dtype=np.uint8)])


def test_torchvision_od_model_rejects_unsupported_model_name():
    with pytest.raises(ValueError, match="not currently supported"):
        TorchvisionODModel(model_name="not_a_model")


def test_torchvision_od_real_integration_contract():
    if os.environ.get("MODELMAITE_RUN_TORCHVISION_INTEGRATION") != "1":
        pytest.skip("Set MODELMAITE_RUN_TORCHVISION_INTEGRATION=1 to run the real torchvision detector.")
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"torchvision optional dependencies are unavailable: {exc}")

    model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device="cpu")
    (prediction,) = model([np.zeros((3, 64, 64), dtype=np.uint8)])

    assert prediction.boxes.shape[1:] == (4,)
    if len(prediction.boxes):
        assert np.all(prediction.boxes[:, 0] <= prediction.boxes[:, 2])
        assert np.all(prediction.boxes[:, 1] <= prediction.boxes[:, 3])
        assert set(prediction.labels.tolist()).issubset(model.index2label)


def test_load_models_dispatches_to_torchvision_wrapper(monkeypatch):
    _install_fake_torchvision(monkeypatch)

    loaded = load_models({"tv_model": {"model_type": "ssd300_vgg16"}}, device="cpu")

    assert isinstance(loaded["tv_model"], TorchvisionODModel)
