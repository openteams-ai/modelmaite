from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

from modelmaite.object_detection import DetectionTarget, VisdroneODModel
from modelmaite.object_detection.models import load_models


class _FakeTorchDevice(str):
    pass


def _install_fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.device = lambda device: _FakeTorchDevice(str(device))
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)


VISDRONE_LABELS = [
    "ignored regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]


class _FakeCenterNetVisdrone:
    instances = []
    class_names = VISDRONE_LABELS
    label_map = {"car": 0.8, "bus": 0.2}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeCenterNetVisdrone.instances.append(self)

    def detect_objects(self, array_batch):
        assert array_batch[0].shape == (5, 7, 3)
        return [[(_FakeBBox(), self.label_map)], []]


class _FakeBBox:
    min_vertex = (1.0, 2.0)
    max_vertex = (3.0, 4.0)


class _FakeStreamResponse:
    def __init__(self, chunks=(b"weights",)):
        self.chunks = chunks
        self.headers = {"content-length": str(sum(len(chunk) for chunk in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield from self.chunks


class _FailingStreamResponse(_FakeStreamResponse):
    def iter_bytes(self):
        yield b"partial"
        raise RuntimeError("download interrupted")


def _install_fake_smqtk(monkeypatch):
    _FakeCenterNetVisdrone.instances = []
    _FakeCenterNetVisdrone.class_names = VISDRONE_LABELS
    _FakeCenterNetVisdrone.label_map = {"car": 0.8, "bus": 0.2}
    monkeypatch.setitem(
        sys.modules,
        "smqtk_detection.impls.detect_image_objects.centernet",
        types.SimpleNamespace(CenterNetVisdrone=_FakeCenterNetVisdrone),
    )


def _install_fake_httpx(monkeypatch, response=None):
    if response is None:
        response = _FakeStreamResponse()
    httpx = types.ModuleType("httpx")
    httpx.stream = lambda *args, **kwargs: response
    monkeypatch.setitem(sys.modules, "httpx", httpx)


def _patch_resnet18_weight_metadata(monkeypatch, payload=b"weights"):
    monkeypatch.setitem(VisdroneODModel._weights_sizes, "ResNet18_Weights", len(payload))
    monkeypatch.setitem(VisdroneODModel._weights_sha512, "ResNet18_Weights", hashlib.sha512(payload).hexdigest())


@pytest.fixture
def fake_model_location(tmp_path):
    model_name = "fake_model"
    fake_file_path = tmp_path / f"{model_name}.pth"
    fake_file_path.write_bytes(b"dummy content")
    return tmp_path, model_name


def test_visdrone_invalid_arch_name():
    with pytest.raises(ValueError, match="not currently supported"):
        VisdroneODModel(arch="invalid_arch", device="cpu")


def test_visdrone_downloads_missing_default_weights(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    _install_fake_httpx(monkeypatch)
    _patch_resnet18_weight_metadata(monkeypatch)

    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=tmp_path)

    assert model.name == "visdrone-centernet-resnet18"
    assert (tmp_path / "centernet-resnet18.pth").read_bytes() == b"weights"
    assert _FakeCenterNetVisdrone.instances[-1].kwargs["arch"] == "resnet18"


def test_visdrone_failed_download_does_not_leave_partial_weights(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    _install_fake_httpx(monkeypatch, _FailingStreamResponse())
    _patch_resnet18_weight_metadata(monkeypatch)

    with pytest.raises(RuntimeError, match="interrupted"):
        VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=tmp_path)

    assert not (tmp_path / "centernet-resnet18.pth").exists()
    assert not list(tmp_path.glob(".centernet-resnet18.pth.*.tmp"))


def test_visdrone_rejects_checksum_mismatch(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    _install_fake_httpx(monkeypatch)
    _patch_resnet18_weight_metadata(monkeypatch, payload=b"WEIGHTS")

    with pytest.raises(RuntimeError, match="SHA-512"):
        VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=tmp_path)

    assert not (tmp_path / "centernet-resnet18.pth").exists()
    assert not list(tmp_path.glob(".centernet-resnet18.pth.*.tmp"))


def test_visdrone_valid_model_initialization(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    dir_, fname = fake_model_location

    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)

    assert model.name == f"visdrone-{fname}"
    assert model.metadata["id"] == "visdrone_resnet18_kitware"
    assert model.metadata["index2label"][4] == "car"
    assert _FakeCenterNetVisdrone.instances[-1].kwargs["model_file"] == str(dir_ / f"{fname}.pth")


def test_visdrone_rejects_unknown_smqtk_label_metadata(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    monkeypatch.setattr(_FakeCenterNetVisdrone, "class_names", ["car", "hovercraft"])
    dir_, fname = fake_model_location

    with pytest.raises(ValueError, match="unknown label"):
        VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)


def test_visdrone_call_invalid_shape(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)

    with pytest.raises(ValueError, match="CHW-ordering"):
        model([np.zeros((224, 224, 3), dtype=np.uint8)])


def test_visdrone_wrapper_returns_one_detection_target_per_input(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)

    outputs = model([np.zeros((3, 5, 7), dtype=np.uint8), np.ones((3, 5, 7), dtype=np.uint8)])

    assert len(outputs) == 2
    assert isinstance(outputs[0], DetectionTarget)
    np.testing.assert_allclose(outputs[0].boxes, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(outputs[0].labels, np.array([4]))
    np.testing.assert_allclose(outputs[0].scores, np.array([0.8], dtype=np.float32))
    assert len(outputs[1].boxes) == 0


def test_visdrone_rejects_unknown_prediction_label(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    monkeypatch.setattr(_FakeCenterNetVisdrone, "label_map", {"hovercraft": 0.9})
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)

    with pytest.raises(ValueError, match="unknown label"):
        model([np.zeros((3, 5, 7), dtype=np.uint8)])


def test_load_models_dispatches_to_visdrone_wrapper(fake_model_location, monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_smqtk(monkeypatch)
    dir_, fname = fake_model_location

    loaded = load_models(
        {"visdrone_model": {"model_type": "resnet18", "model_pickle_dir": dir_}},
        device="cpu",
        model_name=fname,
    )

    assert isinstance(loaded["visdrone_model"], VisdroneODModel)
