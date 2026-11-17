"""Golden-output tests: every wrapper must keep giving the results it gives today.

Each wrapper runs on fixed inputs and the outputs, ``metadata["id"]``, ``index2label``
and ``name`` are compared with JSON files in ``tests/data/golden``. The goldens were
recorded from modelmaite 0.2.0 (identical to ``main`` when they were added), so a
refactor of the wrappers must leave these tests passing unchanged.

Inputs are ``uint8`` images and floating-point images in ``[0, 1]``. Other integer
dtypes are left out on purpose: how they are scaled is still being decided.

Models are built locally so nothing is downloaded:

- ONNX: small input-dependent graphs written at test time from fixed weights.
- Torchvision classification: ``alexnet`` with seeded random weights.
- Torchvision detection: ``ssdlite320_mobilenet_v3_large`` with seeded random weights
  is built and loaded through the user-weights path (this pins the id). Random
  detectors give near-identical scores, so their top-k and NMS choices depend on
  the platform. The outputs are therefore produced by a small deterministic torch
  module set on ``model.model``, which keeps the real torchvision transforms and the
  wrapper's own conversion code under test.
- ByteTrack: a numpy detector that finds coloured boxes in each frame, plus the real
  ``trackers`` package.
- VisDrone: real Kitware weights, downloaded on first use. Skipped unless
  ``MODELMAITE_RUN_VISDRONE_INTEGRATION=1``.

To re-record after an intended change, run with ``MODELMAITE_UPDATE_GOLDENS=1`` and
explain the change in the CHANGELOG.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modelmaite.image_classification import OnnxICModel, TorchvisionICModel
from modelmaite.multiobject_tracking import ByteTrackMOTModel, MOTTarget
from modelmaite.object_detection import DetectionTarget, OnnxODModel, TorchvisionODModel, VisdroneODModel

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
UPDATE_GOLDENS = os.environ.get("MODELMAITE_UPDATE_GOLDENS") == "1"

HAS_ONNX_DEPS = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
HAS_TORCHVISION = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
HAS_MOT_RUNTIME = (
    importlib.util.find_spec("supervision") is not None and importlib.util.find_spec("trackers") is not None
)
HAS_SMQTK = importlib.util.find_spec("smqtk_detection") is not None

requires_onnx_deps = pytest.mark.skipif(
    not HAS_ONNX_DEPS and os.environ.get("MODELMAITE_REQUIRE_ONNX_RUNTIME") != "1",
    reason="ONNX golden tests require the optional ONNX dependencies.",
)
requires_torchvision = pytest.mark.skipif(
    not HAS_TORCHVISION and os.environ.get("MODELMAITE_REQUIRE_TORCHVISION") != "1",
    reason="Torchvision golden tests require the optional torchvision dependencies.",
)
requires_mot_runtime = pytest.mark.skipif(
    not HAS_MOT_RUNTIME and os.environ.get("MODELMAITE_REQUIRE_MOT") != "1",
    reason="MOT dependencies are optional.",
)
requires_visdrone_download = pytest.mark.skipif(
    os.environ.get("MODELMAITE_RUN_VISDRONE_INTEGRATION") != "1" or not HAS_SMQTK,
    reason="Set MODELMAITE_RUN_VISDRONE_INTEGRATION=1 with the visdrone extra installed to download the weights.",
)

INDEX2LABEL = {0: "background", 1: "cat", 2: "dog"}

# Pure numpy pipelines and ONNX Runtime on CPU agree to float32 rounding across
# platforms. Torch kernels differ a little more between CPUs.
NUMPY_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
TORCH_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}


# ── Fixed inputs ──────────────────────────────────────────────────────────────


def _pattern(channels: int, height: int, width: int, offset: int) -> np.ndarray:
    """Return a deterministic uint8 CHW image with structure in every direction."""
    c, y, x = np.indices((channels, height, width))
    return ((c * 71 + y * 31 + x * 17 + offset * 13 + (x * y) % 7) % 256).astype(np.uint8)


def _uint8_batch(channels: int, sizes: list[tuple[int, int]], offset: int = 0) -> list[np.ndarray]:
    return [_pattern(channels, h, w, offset + i) for i, (h, w) in enumerate(sizes)]


def _float_batch(channels: int, sizes: list[tuple[int, int]], offset: int = 50) -> list[np.ndarray]:
    return [image.astype(np.float32) / np.float32(255.0) for image in _uint8_batch(channels, sizes, offset)]


def _fixed_weights(shape: tuple[int, ...], salt: int, scale: float) -> np.ndarray:
    """Deterministic weights from integer arithmetic, so no RNG or libm is involved."""
    values = (np.arange(int(np.prod(shape)), dtype=np.int64) * 7919 + salt * 104729) % 201 - 100
    return (values.astype(np.float64) * (scale / 100.0)).astype(np.float32).reshape(shape)


# ── Golden file handling ──────────────────────────────────────────────────────


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, DetectionTarget):
        return {
            "boxes": _to_jsonable(value.boxes),
            "labels": _to_jsonable(value.labels),
            "scores": _to_jsonable(value.scores),
        }
    if isinstance(value, MOTTarget):
        return {"frame_tracks": [_to_jsonable(frame) for frame in value.frame_tracks]}
    if hasattr(value, "track_ids"):
        return {
            "boxes": _to_jsonable(value.boxes),
            "labels": _to_jsonable(value.labels),
            "scores": _to_jsonable(value.scores),
            "track_ids": _to_jsonable(value.track_ids),
        }
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            data = [float(f"{item:.7g}") for item in value.ravel().tolist()]
        else:
            data = value.ravel().tolist()
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": data}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _assert_matches(expected: Any, actual: Any, tolerance: dict[str, float], path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {actual!r}"
        assert sorted(actual) == sorted(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        if {"dtype", "shape", "data"} == set(expected):
            assert actual["dtype"] == expected["dtype"], f"{path}: dtype {actual['dtype']} != {expected['dtype']}"
            assert actual["shape"] == expected["shape"], f"{path}: shape {actual['shape']} != {expected['shape']}"
            if expected["dtype"].startswith("float"):
                np.testing.assert_allclose(
                    np.asarray(actual["data"], dtype=np.float64),
                    np.asarray(expected["data"], dtype=np.float64),
                    err_msg=path,
                    **tolerance,
                )
            else:
                assert actual["data"] == expected["data"], f"{path}: {actual['data']} != {expected['data']}"
            return
        for key in expected:
            _assert_matches(expected[key], actual[key], tolerance, f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {actual!r}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_matches(exp_item, act_item, tolerance, f"{path}[{index}]")
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _check_golden(file_name: str, section: str, actual: Any, tolerance: dict[str, float] | None = None) -> None:
    """Compare ``actual`` with one section of a golden file, or record it when updating."""
    path = GOLDEN_DIR / file_name
    actual_json = _to_jsonable(actual)
    if UPDATE_GOLDENS:
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        golden[section] = actual_json
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(golden, indent=1, sort_keys=True)
        # One line per array keeps the files short and their diffs readable.
        text = re.sub(r"\[[-+0-9.eE,\s]*\]", lambda match: re.sub(r"\s+", "", match.group(0)).replace(",", ", "), text)
        path.write_text(text + "\n", encoding="utf-8")
        return
    assert path.exists(), f"Missing golden file {path}; record it with MODELMAITE_UPDATE_GOLDENS=1."
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert section in golden, f"Missing section {section!r} in {path}; record it with MODELMAITE_UPDATE_GOLDENS=1."
    # Round-trip through JSON so tuples, int keys and floats compare the same way as the file.
    _assert_matches(golden[section], json.loads(json.dumps(actual_json)), tolerance or NUMPY_TOLERANCE, section)


def _model_facts(model: Any) -> dict[str, Any]:
    facts = {"metadata": dict(model.metadata)}
    if hasattr(model, "name"):
        facts["name"] = model.name
    return facts


# ── ONNX ──────────────────────────────────────────────────────────────────────

ONNX_INPUT_SIZE = (8, 8)
ONNX_IMAGE_SIZES = [(12, 10), (12, 10)]
ONNX_N_BOXES = 3


def _onnx_metadata(interface: str, output: dict[str, int]) -> dict[str, Any]:
    height, width = ONNX_INPUT_SIZE
    return {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": -1,
            "interface": interface,
            "input": {"channels": "RGB", "height": height, "width": width},
            "output": output,
        },
        "index2label": {str(key): value for key, value in INDEX2LABEL.items()},
    }


def _onnx_save(path: Path, nodes: list[Any], initializers: dict[str, np.ndarray], outputs: list[str]) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    height, width = ONNX_INPUT_SIZE
    graph = helper.make_graph(
        nodes,
        "modelmaite_golden",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, height, width])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs],
        initializer=[numpy_helper.from_array(value, name=name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _write_onnx_classifier(directory: Path) -> tuple[str, str]:
    from onnx import helper

    n_features = 3 * ONNX_INPUT_SIZE[0] * ONNX_INPUT_SIZE[1]
    _onnx_save(
        directory / "classifier.onnx",
        [
            helper.make_node("Flatten", ["image"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["logits_raw"]),
            helper.make_node("Add", ["logits_raw", "bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["scores"], axis=1),
        ],
        {
            "weight": _fixed_weights((n_features, len(INDEX2LABEL)), salt=1, scale=0.2),
            "bias": _fixed_weights((len(INDEX2LABEL),), salt=2, scale=0.5),
        },
        ["scores"],
    )
    metadata = _onnx_metadata("IMAGE_CLASSIFICATION", {"nClasses": len(INDEX2LABEL)})
    (directory / "classifier.json").write_text(json.dumps(metadata), encoding="utf-8")
    return "classifier.onnx", "classifier.json"


def _write_onnx_detector(directory: Path) -> tuple[str, str]:
    from onnx import helper

    n_features = 3 * ONNX_INPUT_SIZE[0] * ONNX_INPUT_SIZE[1]
    n_classes = len(INDEX2LABEL)
    _onnx_save(
        directory / "detector.onnx",
        [
            helper.make_node("Flatten", ["image"], ["flat"], axis=1),
            # Boxes: sigmoid features mapped so x0, y0 lie in [0, 0.5] and x1, y1 in [0.5, 1].
            helper.make_node("MatMul", ["flat", "box_weight"], ["box_raw"]),
            helper.make_node("Sigmoid", ["box_raw"], ["box_unit"]),
            helper.make_node("Reshape", ["box_unit", "box_shape"], ["box_grid"]),
            helper.make_node("Mul", ["box_grid", "box_scale"], ["box_scaled"]),
            helper.make_node("Add", ["box_scaled", "box_offset"], ["boxes"]),
            helper.make_node("MatMul", ["flat", "score_weight"], ["score_raw"]),
            helper.make_node("Reshape", ["score_raw", "score_shape"], ["score_grid"]),
            helper.make_node("Softmax", ["score_grid"], ["scores"], axis=-1),
        ],
        {
            "box_weight": _fixed_weights((n_features, ONNX_N_BOXES * 4), salt=3, scale=0.1),
            "box_shape": np.array([-1, ONNX_N_BOXES, 4], dtype=np.int64),
            "box_scale": np.full(4, 0.5, dtype=np.float32),
            "box_offset": np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
            "score_weight": _fixed_weights((n_features, ONNX_N_BOXES * n_classes), salt=4, scale=0.2),
            "score_shape": np.array([-1, ONNX_N_BOXES, n_classes], dtype=np.int64),
        },
        ["boxes", "scores"],
    )
    metadata = _onnx_metadata("IMAGE_OBJECT_DETECTION", {"nBoxes": ONNX_N_BOXES, "nClasses": n_classes})
    (directory / "detector.json").write_text(json.dumps(metadata), encoding="utf-8")
    return "detector.onnx", "detector.json"


def _onnx_cases() -> dict[str, list[np.ndarray]]:
    return {
        "uint8_rgb": _uint8_batch(3, ONNX_IMAGE_SIZES),
        "float_rgb": _float_batch(3, ONNX_IMAGE_SIZES),
        "uint8_gray": _uint8_batch(1, ONNX_IMAGE_SIZES, offset=7),
        "uint8_rgba": _uint8_batch(4, ONNX_IMAGE_SIZES, offset=11),
    }


@requires_onnx_deps
def test_onnx_ic_model_matches_golden_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Relative paths keep metadata["id"] (a hash of the paths) independent of tmp_path.
    monkeypatch.chdir(tmp_path)
    weights, config = _write_onnx_classifier(tmp_path)
    model = OnnxICModel(weights_path=weights, config_path=config, device="cpu")

    _check_golden("onnx_ic.json", "model", _model_facts(model))
    for case, batch in _onnx_cases().items():
        _check_golden("onnx_ic.json", f"outputs.{case}", list(model(batch)))


@requires_onnx_deps
def test_onnx_od_model_matches_golden_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    weights, config = _write_onnx_detector(tmp_path)
    model = OnnxODModel(weights_path=weights, config_path=config, device="cpu")

    _check_golden("onnx_od.json", "model", _model_facts(model))
    for case, batch in _onnx_cases().items():
        _check_golden("onnx_od.json", f"outputs.{case}", list(model(batch)))


# ── Torchvision ───────────────────────────────────────────────────────────────


def _seed_torch() -> Any:
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return torch


@pytest.fixture
def _deterministic_torch():
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


def _write_torch_weights(directory: Path, model: Any, stem: str) -> tuple[str, str]:
    import torch

    torch.save(model.state_dict(), directory / f"{stem}.pt")
    config = {"index2label": {str(key): value for key, value in INDEX2LABEL.items()}, "num_classes": len(INDEX2LABEL)}
    (directory / f"{stem}.json").write_text(json.dumps(config), encoding="utf-8")
    return f"{stem}.pt", f"{stem}.json"


@requires_torchvision
@pytest.mark.usefixtures("_deterministic_torch")
def test_torchvision_ic_model_matches_golden_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = _seed_torch()
    import torchvision.models

    network = torchvision.models.alexnet(weights=None, num_classes=len(INDEX2LABEL))
    with torch.no_grad():
        # Random alexnet logits are almost flat; widen them so the scores move with the input.
        network.classifier[-1].weight.mul_(50.0)
    monkeypatch.chdir(tmp_path)
    weights, config = _write_torch_weights(tmp_path, network, "alexnet")
    model = TorchvisionICModel(model_name="alexnet", weights_path=weights, config_path=config, device="cpu")

    _check_golden("torchvision_ic.json", "model", _model_facts(model))
    sizes = [(64, 64), (64, 64)]
    cases = {"uint8_rgb": _uint8_batch(3, sizes), "float_rgb": _float_batch(3, sizes)}
    for case, batch in cases.items():
        _check_golden("torchvision_ic.json", f"outputs.{case}", list(model(batch)), TORCH_TOLERANCE)


def _pixel_statistics_detector() -> Any:
    import torch

    class PixelStatisticsDetector(torch.nn.Module):
        """Deterministic stand-in for a torchvision detector.

        It returns one box per channel. Each box is centred on that channel's
        intensity centroid and sized from its maximum, and scored by its mean.
        The output therefore depends on the image size, on how the wrapper
        scaled the pixels, and on where they are: a flipped or transposed
        input moves the boxes.
        """

        def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            predictions = []
            for image in images:
                _, height, width = image.shape
                ys = (torch.arange(height, dtype=image.dtype) + 0.5).view(1, height, 1)
                xs = (torch.arange(width, dtype=image.dtype) + 0.5).view(1, 1, width)
                mass = image.sum(dim=(1, 2)).clamp_min(1e-6)
                centre_x = (image * xs).sum(dim=(1, 2)) / mass
                centre_y = (image * ys).sum(dim=(1, 2)) / mass
                maxima = image.amax(dim=(1, 2))
                half_width = width * (0.1 + 0.2 * maxima)
                half_height = height * (0.1 + 0.2 * maxima)
                boxes = torch.stack(
                    [
                        (centre_x - half_width).clamp(0, width),
                        (centre_y - half_height).clamp(0, height),
                        (centre_x + half_width).clamp(0, width),
                        (centre_y + half_height).clamp(0, height),
                    ],
                    dim=1,
                )
                predictions.append(
                    {
                        "boxes": boxes,
                        "labels": torch.tensor([1, 2, 1], dtype=torch.int64),
                        "scores": image.mean(dim=(1, 2)),
                    }
                )
            return predictions

    return PixelStatisticsDetector()


@requires_torchvision
@pytest.mark.usefixtures("_deterministic_torch")
def test_torchvision_od_model_matches_golden_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_torch()
    import torchvision.models.detection

    network = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(INDEX2LABEL)
    )
    monkeypatch.chdir(tmp_path)
    weights, config = _write_torch_weights(tmp_path, network, "ssdlite")
    model = TorchvisionODModel(
        model_name="ssdlite320_mobilenet_v3_large", weights_path=weights, config_path=config, device="cpu"
    )
    _check_golden("torchvision_od.json", "model", _model_facts(model))

    model.model = _pixel_statistics_detector()
    # Torchvision detection accepts mixed image sizes in one batch.
    sizes = [(64, 64), (48, 80)]
    cases = {"uint8_rgb": _uint8_batch(3, sizes), "float_rgb": _float_batch(3, sizes)}
    for case, batch in cases.items():
        _check_golden("torchvision_od.json", f"outputs.{case}", list(model(batch)), TORCH_TOLERANCE)


@requires_torchvision
@pytest.mark.usefixtures("_deterministic_torch")
def test_torchvision_od_real_network_runs_without_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real network, which the golden test above replaces, runs through the wrapper.

    Random-weight detectors give near-tied scores, so only the structure of the
    output is checked, not its values.
    """
    _seed_torch()
    import torchvision.models.detection

    network = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(INDEX2LABEL)
    )
    monkeypatch.chdir(tmp_path)
    weights, config = _write_torch_weights(tmp_path, network, "ssdlite")
    model = TorchvisionODModel(
        model_name="ssdlite320_mobilenet_v3_large", weights_path=weights, config_path=config, device="cpu"
    )

    batch = _uint8_batch(3, [(64, 64), (48, 80)])
    predictions = model(batch)

    assert len(predictions) == len(batch)
    for prediction in predictions:
        assert isinstance(prediction, DetectionTarget)
        boxes = np.asarray(prediction.boxes)
        labels = np.asarray(prediction.labels)
        scores = np.asarray(prediction.scores)
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert len(labels) == len(scores) == len(boxes)
        assert set(labels.tolist()) <= set(model.metadata["index2label"])
        assert np.all((scores >= 0) & (scores <= 1))


# ── VisDrone (downloads weights) ──────────────────────────────────────────────


@requires_visdrone_download
def test_visdrone_od_model_matches_golden_outputs():
    model = VisdroneODModel(arch="resnet18", device="cpu")

    _check_golden("visdrone_od.json", "model", _model_facts(model))
    # The wrapped smqtk detector expects 0-255 data, so only uint8 inputs are pinned.
    batch = _uint8_batch(3, [(96, 128), (96, 128), (96, 128), (96, 128)])
    _check_golden("visdrone_od.json", "outputs.uint8_rgb", list(model(batch)), TORCH_TOLERANCE)


# ── ByteTrack ─────────────────────────────────────────────────────────────────


@dataclass
class _VideoFrame:
    pixels: np.ndarray
    time_s: float | None
    pts: int | None
    frame_index: int


class _ColourBoxDetector:
    """Find the red and green boxes in a frame with numpy only."""

    metadata = {"id": "colour-box-detector", "index2label": {0: "background", 1: "red", 2: "green"}}

    def __call__(self, input_batch):
        outputs = []
        for image in input_batch:
            pixels = np.asarray(image)
            boxes, labels, scores = [], [], []
            for channel, label in ((0, 1), (1, 2)):
                ys, xs = np.nonzero(pixels[channel])
                if len(xs) == 0:
                    continue
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
                labels.append(label)
                scores.append(float(pixels[channel][ys, xs].mean()) / 255.0)
            outputs.append(
                DetectionTarget(
                    boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.asarray(labels, dtype=np.int64),
                    scores=np.asarray(scores, dtype=np.float32),
                )
            )
        return outputs


def _video(n_frames: int, *, green_gap: int | None) -> list[_VideoFrame]:
    frames = []
    for index in range(n_frames):
        pixels = np.zeros((3, 40, 60), dtype=np.uint8)
        pixels[0, 5:15, 2 + 3 * index : 12 + 3 * index] = 240  # red box moves right
        if index != green_gap:
            pixels[1, 10 + 2 * index : 22 + 2 * index, 30:40] = 200  # green box moves down
        frames.append(_VideoFrame(pixels=pixels, time_s=None, pts=None, frame_index=index))
    return frames


@requires_mot_runtime
def test_bytetrack_mot_model_matches_golden_outputs():
    model = ByteTrackMOTModel(detector=_ColourBoxDetector(), detector_batch_size=2)

    _check_golden("bytetrack_mot.json", "model", _model_facts(model))
    streams = [_video(6, green_gap=3), list(reversed(_video(5, green_gap=None)))]
    _check_golden("bytetrack_mot.json", "outputs.two_streams", list(model(streams)))


# ── Default ids (no backend needed) ───────────────────────────────────────────


def test_default_ids_match_goldens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Pin the default ``metadata["id"]`` formulas that the backend tests above don't reach."""
    from tests import test_torchvision_image_classification as tv_ic
    from tests import test_torchvision_object_detection as tv_od
    from tests import test_visdrone_object_detection as visdrone

    ids = {}
    tv_ic._install_fake_torchvision(monkeypatch)
    ids["TorchvisionICModel(alexnet)"] = TorchvisionICModel(model_name="alexnet", device="cpu").metadata["id"]
    tv_od._install_fake_torchvision(monkeypatch)
    ids["TorchvisionODModel(ssd300_vgg16)"] = TorchvisionODModel(model_name="ssd300_vgg16", device="cpu").metadata["id"]
    visdrone._install_fake_torch(monkeypatch)
    visdrone._install_fake_smqtk(monkeypatch)
    (tmp_path / "centernet-resnet18.pth").write_bytes(b"unused")
    ids["VisdroneODModel(resnet18)"] = VisdroneODModel(arch="resnet18", model_pickle_dir=tmp_path).metadata["id"]

    # ByteTrack builds its id from the detector id and tracker_kwargs unless model_id is given.
    detector = _ColourBoxDetector()
    monkeypatch.setattr(
        "modelmaite.multiobject_tracking.models._import_tracking_dependencies", lambda: (object(), object())
    )
    ids["ByteTrackMOTModel()"] = ByteTrackMOTModel(detector=detector).metadata["id"]
    ids["ByteTrackMOTModel(model_id='custom')"] = ByteTrackMOTModel(detector=detector, model_id="custom").metadata["id"]

    _check_golden("default_ids.json", "ids", ids)
