import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from maite.protocols import image_classification as ic
from maite.spotcheck import SpotcheckError, spotcheck_component

from modelmaite.image_classification import OnnxICModel, load_models

HAS_ONNX_DEPS = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
REQUIRE_ONNX_DEPS = os.environ.get("MODELMAITE_REQUIRE_ONNX_RUNTIME") == "1"
requires_onnx_deps = pytest.mark.skipif(
    not HAS_ONNX_DEPS and not REQUIRE_ONNX_DEPS,
    reason="ONNX wrapper tests require the optional ONNX dependencies.",
)


def _metadata(*, batch_size=2, n_classes=3):
    return {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": batch_size,
            "interface": "IMAGE_CLASSIFICATION",
            "input": {"channels": "RGB", "height": 4, "width": 5},
            "output": {"nClasses": n_classes},
        },
        "index2label": {"0": "background", "1": "cat", "2": "dog"},
    }


class _FakeOnnxSession:
    def run(self, output_names, feeds):
        assert output_names == ["scores"]
        assert feeds["image"].shape == (2, 3, 4, 5)
        return [np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)]


class _BadShapeOnnxSession:
    def run(self, output_names, feeds):
        return [np.zeros((2, 3, 1), dtype=np.float32)]


class _WrongClassCountOnnxSession:
    def run(self, output_names, feeds):
        return [np.zeros((2, 4), dtype=np.float32)]


class _NanScoresOnnxSession:
    def run(self, output_names, feeds):
        return [np.array([[0.1, np.nan, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)]


class _OutOfRangeScoresOnnxSession:
    def run(self, output_names, feeds):
        return [np.array([[0.1, 1.2, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)]


def _fake_model(session):
    model = OnnxICModel.__new__(OnnxICModel)
    model.config = _metadata()
    model.index2label = {0: "background", 1: "cat", 2: "dog"}
    model.model = session
    model._model_name = "jatic_onnx"
    model._batch_size = None
    model._image_height = None
    model._image_width = None
    model._n_classes = 3
    model.metadata = {"id": "fake", "index2label": model.index2label}
    return model


def _save_constant_onnx_model(path: Path, *, input_shape: list[int], outputs: dict[str, np.ndarray]) -> None:
    """Create a tiny ONNX model that ignores its image input and returns fixed tensors."""
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, input_shape)
    output_infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)) for name, value in outputs.items()
    ]
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[name],
            value=helper.make_tensor(
                name=f"{name}_value",
                data_type=TensorProto.FLOAT,
                dims=list(value.shape),
                vals=value.astype(np.float32).ravel().tolist(),
            ),
        )
        for name, value in outputs.items()
    ]
    graph = helper.make_graph(nodes, "constant_jatic_onnx_test_model", [input_info], output_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _save_input_dependent_onnx_model(path: Path) -> None:
    """Create a tiny classifier whose output changes with the image pixels."""
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [2, 3, 4, 5])
    output_info = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2, 3])
    nodes = [
        helper.make_node("GlobalAveragePool", inputs=["image"], outputs=["pooled"]),
        helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1),
        helper.make_node("Softmax", inputs=["flat"], outputs=["scores"], axis=1),
    ]
    graph = helper.make_graph(nodes, "pixel_dependent_jatic_onnx_test_model", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _write_onnx_metadata(path: Path, *, n_classes=3, include_index2label=True) -> None:
    metadata = _metadata(n_classes=n_classes)
    if not include_index2label:
        del metadata["index2label"]
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_onnx_ic_model_returns_jatic_scores_without_runtime():
    model = _fake_model(_FakeOnnxSession())

    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert len(predictions) == 2
    np.testing.assert_allclose(predictions[0], np.array([0.1, 0.7, 0.2], dtype=np.float32))
    np.testing.assert_allclose(predictions[1], np.array([0.8, 0.1, 0.1], dtype=np.float32))


def test_onnx_ic_model_satisfies_maite_protocol():
    model = _fake_model(_FakeOnnxSession())

    assert isinstance(model, ic.Model)


def test_onnx_ic_model_passes_maite_spotcheck():
    model = spotcheck_component(ic.Model)(_fake_model(_FakeOnnxSession()))

    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert len(predictions) == 2
    with pytest.raises(SpotcheckError):
        model([np.zeros((4, 5), dtype=np.uint8)])


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (_BadShapeOnnxSession(), "shape"),
        (_WrongClassCountOnnxSession(), "index2label"),
        (_NanScoresOnnxSession(), "finite"),
        (_OutOfRangeScoresOnnxSession(), "range"),
    ],
)
def test_onnx_ic_model_rejects_invalid_outputs(session, message):
    model = _fake_model(session)

    with pytest.raises(ValueError, match=message):
        model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])


@requires_onnx_deps
def test_onnx_ic_model_returns_jatic_scores(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    expected_scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)
    _save_constant_onnx_model(model_path, input_shape=[2, 3, 4, 5], outputs={"scores": expected_scores})
    _write_onnx_metadata(config_path)

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")
    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert model.name == "jatic_onnx"
    assert model.index2label == {0: "background", 1: "cat", 2: "dog"}
    assert len(predictions) == 2
    np.testing.assert_allclose(predictions[0], expected_scores[0])
    np.testing.assert_allclose(predictions[1], expected_scores[1])


@requires_onnx_deps
def test_onnx_ic_model_consumes_pixels_for_scores(tmp_path: Path):
    model_path = tmp_path / "ic_pixel_dependent.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_input_dependent_onnx_model(model_path)
    _write_onnx_metadata(config_path)

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")
    image0 = np.zeros((3, 4, 5), dtype=np.float32)
    image1 = np.zeros((3, 4, 5), dtype=np.float32)
    image1[0] = 1.0
    predictions = model([image0, image1])

    logits = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    expected1 = np.exp(logits) / np.sum(np.exp(logits))
    np.testing.assert_allclose(predictions[0], np.full(3, 1.0 / 3.0, dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(predictions[1], expected1, rtol=1e-6)


@requires_onnx_deps
def test_onnx_ic_load_models_dispatch(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path,
        input_shape=[2, 3, 4, 5],
        outputs={"scores": np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)},
    )
    _write_onnx_metadata(config_path)

    loaded = load_models(
        {
            "onnx_model": {
                "model_type": "jatic_onnx",
                "model_weights_path": model_path,
                "model_config_path": config_path,
            }
        },
        device="cpu",
    )

    assert isinstance(loaded["onnx_model"], OnnxICModel)


@requires_onnx_deps
def test_onnx_ic_model_requires_index2label(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path,
        input_shape=[1, 3, 4, 5],
        outputs={"scores": np.array([[1.0]], dtype=np.float32)},
    )
    _write_onnx_metadata(config_path, n_classes=1, include_index2label=False)

    with pytest.raises(ValueError, match="index2label"):
        OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")


def test_load_models_requires_model_type():
    with pytest.raises(ValueError, match="model_type"):
        load_models({"onnx_model": {}})


def test_load_models_requires_paths():
    with pytest.raises(ValueError, match="model_weights_path"):
        load_models({"onnx_model": {"model_type": "jatic_onnx"}})


class _RecordingOnnxSession:
    def __init__(self):
        self.batch_sizes = []

    def run(self, output_names, feeds):
        n_images = feeds["image"].shape[0]
        self.batch_sizes.append(n_images)
        return [np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (n_images, 1))]


def test_onnx_ic_model_splits_batches_by_metadata_batch_size():
    session = _RecordingOnnxSession()
    model = _fake_model(session)

    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8)] * 5)

    assert len(predictions) == 5
    assert session.batch_sizes == [2, 2, 1]


def test_onnx_ic_model_batch_size_overrides_metadata_batch_size():
    session = _RecordingOnnxSession()
    model = _fake_model(session)
    model._batch_size = 3

    model([np.zeros((3, 4, 5), dtype=np.uint8)] * 5)

    assert session.batch_sizes == [3, 2]


def test_onnx_ic_model_returns_empty_list_for_empty_batch():
    session = _RecordingOnnxSession()

    assert _fake_model(session)([]) == []
    assert session.batch_sizes == []


def test_onnx_ic_model_rejects_two_channel_images_before_the_model_runs():
    session = _RecordingOnnxSession()

    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        _fake_model(session)([np.zeros((2, 4, 5), dtype=np.uint8)])
    assert session.batch_sizes == []


class _FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_device_index_zero_is_accepted_and_other_indices_are_rejected():
    from modelmaite._onnx import get_onnx_providers

    assert get_onnx_providers("cuda:0", ort=_FakeOrt())[0] == "cuda"
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        get_onnx_providers("cuda:1", ort=_FakeOrt())


@requires_onnx_deps
def test_onnx_ic_model_runs_a_fixed_batch_graph_exactly_that_many_images_at_a_time(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)
    _save_constant_onnx_model(model_path, input_shape=[2, 3, 4, 5], outputs={"scores": scores})
    config_path.write_text(json.dumps(_metadata(batch_size=-1)), encoding="utf-8")
    images = [np.zeros((3, 4, 5), dtype=np.uint8)] * 4

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")

    assert len(model(images)) == 4
    with pytest.raises(ValueError, match="multiple of 2; got 3 images"):
        model(images[:3])


@requires_onnx_deps
@pytest.mark.parametrize(
    ("metadata_batch_size", "batch_size", "message"),
    [
        (3, None, "ONNX metadata batchSize=3 does not match"),
        (-1, 1, "batch_size=1 is smaller than"),
    ],
)
def test_onnx_ic_model_rejects_batch_sizes_that_contradict_a_fixed_batch_graph(
    tmp_path: Path, metadata_batch_size, batch_size, message
):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path, input_shape=[2, 3, 4, 5], outputs={"scores": np.full((2, 3), 1 / 3, dtype=np.float32)}
    )
    config_path.write_text(json.dumps(_metadata(batch_size=metadata_batch_size)), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{message} the ONNX graph's fixed batch dimension 2"):
        OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu", batch_size=batch_size)


@requires_onnx_deps
def test_onnx_ic_model_accepts_a_batch_size_above_a_fixed_batch_graph(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path, input_shape=[2, 3, 4, 5], outputs={"scores": np.full((2, 3), 1 / 3, dtype=np.float32)}
    )
    config_path.write_text(json.dumps(_metadata(batch_size=-1)), encoding="utf-8")

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu", batch_size=8)

    assert len(model([np.zeros((3, 4, 5), dtype=np.uint8)] * 4)) == 4


@requires_onnx_deps
def test_onnx_ic_model_runs_mixed_image_sizes_one_at_a_time(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path, input_shape=[1, 3, 4, 5], outputs={"scores": np.array([[0.1, 0.7, 0.2]], dtype=np.float32)}
    )
    config_path.write_text(json.dumps(_metadata(batch_size=-1)), encoding="utf-8")
    images = [np.zeros((3, 4, 5), dtype=np.uint8), np.zeros((3, 8, 10), dtype=np.uint8)]

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")

    assert len(model(images)) == 2


def test_onnx_ic_model_id_covers_the_resize_overrides():
    model = _fake_model(session=None)
    model._weights_path = "model.onnx"
    model._config_path = "model.json"
    default = model._default_model_id()

    model._image_height = 32
    resized = model._default_model_id()
    model._image_width = 48

    assert len({default, resized, model._default_model_id()}) == 3
