import json
from pathlib import Path

import numpy as np
import pytest

from modelmaite._onnx import (
    IMAGE_OBJECT_DETECTION_INTERFACE,
    _normalize_image,
    load_jatic_onnx_metadata,
    prepare_jatic_onnx_image_batch,
)

ROOT = Path(__file__).parent / "data" / "jatic_onnx_od"


def _metadata(*, height=-1, width=-1, batch_size=-1, channels="RGB"):
    return {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": batch_size,
            "interface": "IMAGE_OBJECT_DETECTION",
            "input": {"channels": channels, "height": height, "width": width},
            "output": {"nBoxes": 2, "nClasses": 3},
        },
        "index2label": {"0": "background", "1": "cat", "2": "dog"},
    }


def test_load_jatic_onnx_metadata_normalizes_index2label():
    metadata, index2label = load_jatic_onnx_metadata(
        ROOT / "model-metadata.json",
        expected_io_interface=IMAGE_OBJECT_DETECTION_INTERFACE,
    )

    assert metadata["interface"]["name"] == "JATIC_ONNX"
    assert index2label == {0: "background", 1: "cat", 2: "dog"}


def test_load_jatic_onnx_metadata_rejects_missing_index2label(tmp_path):
    metadata = _metadata()
    del metadata["index2label"]
    config_path = tmp_path / "model-metadata.json"
    config_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="index2label"):
        load_jatic_onnx_metadata(config_path, expected_io_interface=IMAGE_OBJECT_DETECTION_INTERFACE)


def test_normalize_image_scales_uint8_to_unit_interval():
    image = np.array([[[0, 127, 255]]], dtype=np.uint8)

    normalized = _normalize_image(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, np.array([[[0.0, 127 / 255, 1.0]]], dtype=np.float32))


def test_normalize_image_rejects_float_images_above_unit_interval():
    image = np.array([[[0.0, 1.01]]], dtype=np.float32)

    with pytest.raises(ValueError, match="range"):
        _normalize_image(image)


def test_normalize_image_rejects_non_uint8_integer_images():
    image = np.array([[[0, 127, 255]]], dtype=np.int32)

    with pytest.raises(TypeError, match="uint8"):
        _normalize_image(image)


def test_prepare_jatic_onnx_image_batch_rejects_non_chw_input():
    hwc_image = np.zeros((10, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="CHW-ordering"):
        prepare_jatic_onnx_image_batch([hwc_image], _metadata())


def test_prepare_jatic_onnx_image_batch_resizes_to_target_height_and_width():
    image = np.zeros((3, 10, 20), dtype=np.uint8)
    image[:, 2:8, 5:15] = 255

    batch, original_sizes = prepare_jatic_onnx_image_batch([image], _metadata(height=5, width=7))

    assert batch.shape == (1, 3, 5, 7)
    assert batch.dtype == np.float32
    assert original_sizes == [(10, 20)]
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0


def test_prepare_jatic_onnx_image_batch_converts_rgb_inputs_to_grayscale():
    image = np.array(
        [
            [[0, 255], [64, 128]],
            [[255, 0], [64, 128]],
            [[0, 0], [255, 128]],
        ],
        dtype=np.uint8,
    )
    rgb = image.astype(np.float32) / 255
    expected = np.tensordot(np.array([0.299, 0.587, 0.114], dtype=np.float32), rgb, axes=(0, 0))[None, ...]

    batch, original_sizes = prepare_jatic_onnx_image_batch([image], _metadata(channels="GRAYSCALE"))

    assert batch.shape == (1, 1, 2, 2)
    assert batch.dtype == np.float32
    assert original_sizes == [(2, 2)]
    np.testing.assert_allclose(batch[0], expected, rtol=1e-6)


def test_prepare_jatic_onnx_image_batch_enforces_batch_size():
    image = np.zeros((3, 10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="batchSize"):
        prepare_jatic_onnx_image_batch([image, image], _metadata(batch_size=1))
