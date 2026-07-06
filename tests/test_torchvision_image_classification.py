from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from modelmaite.image_classification import TorchvisionICModel
from modelmaite.image_classification.models import load_models


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self.array.shape

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.array = self.array.astype(np.float32)
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.array

    def __array__(self, dtype=None):
        if dtype is None:
            return self.array
        return self.array.astype(dtype)

    def __iter__(self):
        return iter(_FakeTensor(row) for row in self.array)


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeModel:
    instances = []

    def __init__(self, *, weights, num_classes=None, **kwargs):
        self.weights = weights
        self.num_classes = num_classes or 3
        self.kwargs = kwargs
        self.loaded_state_dict = None
        self.eval_called = False
        _FakeModel.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def eval(self):
        self.eval_called = True

    def __call__(self, batch):
        logits = np.arange(batch.shape[0] * self.num_classes, dtype=np.float32).reshape(
            batch.shape[0], self.num_classes
        )
        return _FakeTensor(logits)


class _FakeWeightsDefault:
    meta = {"categories": ["background", "cat", "dog"]}

    @staticmethod
    def transforms():
        return lambda batch: batch


class _FakeWeights:
    DEFAULT = _FakeWeightsDefault()


def _install_fake_torchvision(monkeypatch):
    _FakeModel.instances = []

    torch = types.ModuleType("torch")
    torch.device = lambda device: str(device)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.as_tensor = lambda obj, device=None: _FakeTensor(obj)
    torch.stack = lambda tensors: _FakeTensor(np.stack([np.asarray(tensor) for tensor in tensors]))
    torch.no_grad = _NoGrad
    torch.float32 = "float32"
    torch.load = lambda path, map_location=None, weights_only=True: {"loaded_from": str(path)}

    def softmax(**kwargs):
        arr = np.asarray(kwargs["input"])
        dim = kwargs["dim"]
        exp = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
        return _FakeTensor(exp / np.sum(exp, axis=dim, keepdims=True))

    torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(softmax=softmax))

    torchvision = types.ModuleType("torchvision")
    torchvision_models = types.ModuleType("torchvision.models")
    torchvision_models.AlexNet_Weights = _FakeWeights
    torchvision_models.ResNeXt50_32X4D_Weights = _FakeWeights
    torchvision_models.alexnet = lambda **kwargs: _FakeModel(**kwargs)
    torchvision_models.resnext50_32x4d = lambda **kwargs: _FakeModel(**kwargs)
    torchvision.models = torchvision_models

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)


def test_torchvision_ic_model_returns_softmax_scores(monkeypatch):
    _install_fake_torchvision(monkeypatch)
    model = TorchvisionICModel(model_name="alexnet", device="cpu")

    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert model.name == "alexnet"
    assert model.index2label == {0: "background", 1: "cat", 2: "dog"}
    assert model.metadata["id"].startswith("alexnet_")
    assert _FakeModel.instances[-1].eval_called
    assert len(predictions) == 2
    assert isinstance(predictions[0], np.ndarray)
    assert predictions[0].dtype == np.float32
    np.testing.assert_allclose(predictions[0].sum(), 1.0)
    np.testing.assert_allclose(predictions[1].sum(), 1.0)


def test_torchvision_ic_model_loads_user_weights(monkeypatch, tmp_path: Path):
    _install_fake_torchvision(monkeypatch)
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "weights.pt"
    config_path.write_text('{"index2label": {"0": "zero", "1": "one"}, "num_classes": 2}', encoding="utf-8")
    weights_path.write_text("unused by fake torch.load", encoding="utf-8")

    model = TorchvisionICModel(model_name="alexnet", weights_path=weights_path, config_path=config_path, device="cpu")

    assert model.index2label == {0: "zero", 1: "one"}
    assert _FakeModel.instances[-1].weights is None
    assert _FakeModel.instances[-1].num_classes == 2
    assert _FakeModel.instances[-1].loaded_state_dict == {"loaded_from": str(weights_path)}


def test_torchvision_ic_model_rejects_num_class_mismatch(monkeypatch, tmp_path: Path):
    _install_fake_torchvision(monkeypatch)
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "weights.pt"
    config_path.write_text('{"index2label": {"0": "zero", "1": "one"}, "num_classes": 3}', encoding="utf-8")
    weights_path.write_text("unused by fake torch.load", encoding="utf-8")

    with pytest.raises(ValueError, match="num_classes"):
        TorchvisionICModel(model_name="alexnet", weights_path=weights_path, config_path=config_path, device="cpu")


def test_torchvision_ic_model_requires_config_for_user_weights(tmp_path: Path):
    weights_path = tmp_path / "weights.pt"
    weights_path.write_text("unused", encoding="utf-8")

    with pytest.raises(ValueError, match="config_path"):
        TorchvisionICModel(model_name="alexnet", weights_path=weights_path)


def test_torchvision_ic_real_preprocessing_contract():
    try:
        import torch
        import torchvision.models as torchvision_models
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"torchvision optional dependencies are unavailable: {exc}")

    batch = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
    processed = torchvision_models.AlexNet_Weights.DEFAULT.transforms()(batch)

    assert tuple(processed.shape) == (2, 3, 224, 224)
    assert processed.dtype == torch.float32


def test_torchvision_ic_model_rejects_unsupported_model_name():
    with pytest.raises(ValueError, match="not currently supported"):
        TorchvisionICModel(model_name="not_a_model")


def test_load_models_dispatches_to_torchvision_wrapper(monkeypatch):
    _install_fake_torchvision(monkeypatch)

    loaded = load_models({"tv_model": {"model_type": "alexnet"}}, device="cpu")

    assert isinstance(loaded["tv_model"], TorchvisionICModel)
