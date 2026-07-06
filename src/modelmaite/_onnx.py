"""Utilities for JATIC_ONNX model wrappers."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

JaticOnnxIoInterface = Literal["IMAGE_CLASSIFICATION", "IMAGE_OBJECT_DETECTION"]

JATIC_ONNX_INTERFACE_NAME = "JATIC_ONNX"
JATIC_ONNX_INTERFACE_VERSION = "v1"
IMAGE_CLASSIFICATION_INTERFACE: JaticOnnxIoInterface = "IMAGE_CLASSIFICATION"
IMAGE_OBJECT_DETECTION_INTERFACE: JaticOnnxIoInterface = "IMAGE_OBJECT_DETECTION"
ONNX_INSTALL_HINT = 'Install ONNX support with `poetry add "modelmaite[onnx]"`.'
UNIT_INTERVAL_TOLERANCE = 1e-6


def id_hash(**kwargs: Any) -> str:
    """Generate a stable eight-character hash from keyword arguments."""
    return hashlib.sha256(json.dumps(kwargs, default=str, sort_keys=True).encode()).hexdigest()[:8]


def get_index2label_from_model_config(
    config_path: str | Path,
    model_config: dict[str, Any],
    index2label_key: str,
) -> dict[int, str]:
    """Extract and normalize an index-to-label mapping from model metadata."""
    if index2label_key not in model_config:
        raise ValueError(f"The config_file at {config_path} is missing a {index2label_key} key.")

    index2label = model_config[index2label_key]
    if isinstance(index2label, list):
        return dict(enumerate(str(label) for label in index2label))
    if isinstance(index2label, dict):
        try:
            return {int(key): str(label) for key, label in index2label.items()}
        except (TypeError, ValueError):
            raise ValueError(f"index2label keys in {config_path} must be integer-like strings.") from None
    raise TypeError(f"index2label should be provided as a dict or list, not {type(index2label)}")


def validate_input_batch(input_batch: Sequence[ArrayLike]) -> None:
    """Validate that a batch contains same-shaped CHW image arrays."""
    if len(input_batch) == 0:
        raise ValueError("Input batch must contain at least one image.")

    first_image = np.asarray(input_batch[0])
    if first_image.ndim != 3:
        raise ValueError(f"Input data must follow CHW-ordering, current shape: {first_image.shape}")

    total_channels, image_height, image_width = first_image.shape
    # Channels can be used as a proxy to confirm CHW ordering.
    if not 1 <= total_channels <= 4:
        raise ValueError(
            f"Input data must follow CHW-ordering, current shape: {total_channels, image_height, image_width}"
        )

    expected_shape = (total_channels, image_height, image_width)
    for image in input_batch:
        image_array = np.asarray(image)
        if image_array.shape != expected_shape:
            raise ValueError(
                f"All input images currently required to have identical shape, {image_array.shape} "
                f"not equal to {expected_shape}. Please contact modelmaite team if your use case requires unevenly "
                "shaped images."
            )


def load_jatic_onnx_metadata(
    config_path: str | Path,
    *,
    expected_io_interface: JaticOnnxIoInterface,
    index2label_key: str = "index2label",
) -> tuple[dict[str, Any], dict[int, str]]:
    """Load and validate a JATIC_ONNX metadata JSON file.

    The JATIC Interoperability Requirements specify that ONNX model input/output metadata should be provided alongside
    the model in a metadata file such as ``model-metadata.json``. The standard fields identify the JATIC_ONNX interface
    version, the CV task interface, input channel/size constraints, and output dimensions. modelmaite additionally
    requires model wrappers to expose ``index2label`` metadata, so this loader requires that mapping in the same JSON
    file.
    """
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {config_path}") from None

    if not isinstance(metadata, dict):
        raise TypeError(f"Configuration file at {config_path} must contain a JSON object.")

    interface = _require_dict(metadata, "interface", config_path)
    if interface.get("name") != JATIC_ONNX_INTERFACE_NAME:
        raise ValueError(
            f"ONNX metadata interface.name must be {JATIC_ONNX_INTERFACE_NAME!r}, got {interface.get('name')!r}."
        )
    if interface.get("version") != JATIC_ONNX_INTERFACE_VERSION:
        raise ValueError(
            f"ONNX metadata interface.version must be {JATIC_ONNX_INTERFACE_VERSION!r}, "
            f"got {interface.get('version')!r}."
        )

    io = _require_dict(metadata, "io", config_path)
    if io.get("interface") != expected_io_interface:
        raise ValueError(f"ONNX metadata io.interface must be {expected_io_interface!r}, got {io.get('interface')!r}.")
    _require_dict(io, "input", config_path)
    _require_dict(io, "output", config_path)

    index2label = get_index2label_from_model_config(config_path, metadata, index2label_key)
    return metadata, index2label


def validate_jatic_onnx_session(session: Any, *, expected_outputs: set[str]) -> None:
    """Validate ONNX Runtime input/output names against the JATIC_ONNX contract."""
    input_names = [inp.name for inp in session.get_inputs()]
    if input_names != ["image"]:
        raise ValueError(f"JATIC ONNX models must have exactly one input named 'image', got {input_names}.")

    output_names = {out.name for out in session.get_outputs()}
    if output_names != expected_outputs:
        raise ValueError(
            f"JATIC ONNX model outputs must be exactly {sorted(expected_outputs)}, got {sorted(output_names)}."
        )


def import_onnxruntime() -> Any:
    """Import ONNX Runtime with the modelmaite ONNX install hint."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError:
        raise ImportError(
            f"JATIC_ONNX model wrappers require optional dependency 'onnxruntime'. {ONNX_INSTALL_HINT}"
        ) from None


def get_onnx_providers(device: object | None, *, ort: Any | None = None) -> tuple[str, list[str]]:
    """Translate a device request into ONNX Runtime execution providers."""
    runtime = import_onnxruntime() if ort is None else ort
    available = set(runtime.get_available_providers())

    if device is None:
        if "CUDAExecutionProvider" in available:
            return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in available:
            return "mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return "cpu", ["CPUExecutionProvider"]

    requested = str(device).lower().split(":", maxsplit=1)[0]
    if requested == "cpu":
        return "cpu", ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested for ONNX inference, but CUDAExecutionProvider is not available. "
                f"Available ONNX Runtime providers: {sorted(available)}"
            )
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested in {"mps", "coreml"}:
        if "CoreMLExecutionProvider" not in available:
            raise RuntimeError(
                "MPS/CoreML was requested for ONNX inference, but CoreMLExecutionProvider is not available. "
                f"Available ONNX Runtime providers: {sorted(available)}"
            )
        return "mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    raise ValueError(f"Unsupported ONNX inference device: {device}")


def prepare_jatic_onnx_image_batch(
    input_batch: Sequence[ArrayLike],
    metadata: dict[str, Any],
    *,
    batch_size: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert a CHW image batch into the JATIC_ONNX input tensor.

    JATIC_ONNX v1 requires a single input named ``image`` containing an FP32 NCHW batch with pixel values normalized to
    ``[0, 1]``. The metadata file declares whether the model expects RGB or grayscale images and whether height, width,
    or batch size are fixed. Optional keyword arguments support the runtime overrides allowed by the standard.
    """
    validate_input_batch(input_batch)

    io = _require_dict(metadata, "io", "metadata")
    input_meta = _require_dict(io, "input", "metadata")

    metadata_batch_size = int(batch_size if batch_size is not None else io.get("batchSize", -1))
    if metadata_batch_size != -1 and len(input_batch) > metadata_batch_size:
        raise ValueError(
            f"Input batch has {len(input_batch)} images, but ONNX metadata batchSize is {metadata_batch_size}."
        )

    channels = str(input_meta.get("channels", "RGB")).upper()
    if channels not in {"RGB", "GRAYSCALE"}:
        raise ValueError(f"ONNX metadata io.input.channels must be 'RGB' or 'GRAYSCALE', got {channels!r}.")

    target_height = int(image_height if image_height is not None else input_meta.get("height", -1))
    target_width = int(image_width if image_width is not None else input_meta.get("width", -1))
    height_source = "image_height override" if image_height is not None else "ONNX metadata io.input.height"
    width_source = "image_width override" if image_width is not None else "ONNX metadata io.input.width"
    _validate_size_override(height_source, target_height)
    _validate_size_override(width_source, target_width)

    arrays = []
    original_sizes = []
    for image in input_batch:
        arr = np.asarray(image)
        _, orig_h, orig_w = arr.shape
        original_sizes.append((orig_h, orig_w))
        # Normalize before channel conversion: RGB/RGBA -> grayscale luminance emits float32, so integer images
        # must be scaled to [0, 1] before that conversion.
        arrays.append(_convert_channels(_normalize_image(arr), channels))

    batch = np.stack(arrays).astype(np.float32, copy=False)

    if target_height != -1 or target_width != -1:
        _, _, current_height, current_width = batch.shape
        resize_height = current_height if target_height == -1 else target_height
        resize_width = current_width if target_width == -1 else target_width
        batch = _resize_nchw_bilinear(batch, resize_height, resize_width)

    return batch.astype(np.float32, copy=False), original_sizes


def _require_dict(obj: dict[str, Any], key: str, source: str | Path) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"ONNX metadata at {source} must include object field {key!r}.")
    return value


def optional_output_int(value: object, field_name: str, *, minimum: int) -> int | None:
    """Validate an optional integer field in ONNX output metadata."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ONNX metadata io.output.{field_name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"ONNX metadata io.output.{field_name} must be >= {minimum}, got {value}.")
    return value


def _validate_size_override(source: str, value: int) -> None:
    if value == -1 or value > 0:
        return
    raise ValueError(f"{source} must be -1 or a positive integer, got {value}.")


def _convert_channels(arr: np.ndarray, channels: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Input image must have CHW shape, got {arr.shape}.")

    channel_count = arr.shape[0]
    if channels == "RGB":
        if channel_count == 3:
            return arr
        if channel_count == 4:
            return arr[:3]
        if channel_count == 1:
            return np.repeat(arr, 3, axis=0)
    elif channels == "GRAYSCALE":
        if channel_count == 1:
            return arr
        if channel_count in {3, 4}:
            # Convert RGB/RGBA to luminance using the standard luma coefficients for red, green, and blue.
            rgb = arr[:3].astype(np.float32, copy=False)
            return np.tensordot(np.array([0.299, 0.587, 0.114], dtype=np.float32), rgb, axes=(0, 0))[None, ...]

    raise ValueError(f"Cannot convert input with {channel_count} channel(s) to {channels}.")


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype != np.uint8:
            raise TypeError(
                "Integer image inputs for JATIC_ONNX must use dtype uint8. Convert other integer dtypes to "
                "float32 arrays in the range [0, 1] before calling the wrapper."
            )
        return arr.astype(np.float32) / 255.0

    out = arr.astype(np.float32, copy=False)
    if out.size and (not np.all(np.isfinite(out)) or np.min(out) < 0.0 or np.max(out) > 1.0):
        raise ValueError("Float image inputs for JATIC_ONNX must contain finite values in the range [0, 1].")
    return out


def _resize_nchw_bilinear(batch: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"Output size must be positive, got {(output_height, output_width)}.")

    if batch.ndim != 4:
        raise ValueError(f"Expected an NCHW batch, got shape {batch.shape}.")

    _, _, input_height, input_width = batch.shape
    if (input_height, input_width) == (output_height, output_width):
        return batch

    y0, y1, wy = _interpolation_indices(input_height, output_height)
    x0, x1, wx = _interpolation_indices(input_width, output_width)

    top_left = batch[:, :, y0[:, None], x0[None, :]]
    top_right = batch[:, :, y0[:, None], x1[None, :]]
    bottom_left = batch[:, :, y1[:, None], x0[None, :]]
    bottom_right = batch[:, :, y1[:, None], x1[None, :]]

    wy_b = wy.reshape(1, 1, output_height, 1)
    wx_b = wx.reshape(1, 1, 1, output_width)
    resized = (
        top_left * (1.0 - wy_b) * (1.0 - wx_b)
        + top_right * (1.0 - wy_b) * wx_b
        + bottom_left * wy_b * (1.0 - wx_b)
        + bottom_right * wy_b * wx_b
    )
    return resized.astype(np.float32, copy=False)


def _interpolation_indices(input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Half-pixel centers match the coordinate transform used by torch.interpolate(..., align_corners=False).
    scale = input_size / output_size
    coordinates = (np.arange(output_size, dtype=np.float32) + 0.5) * scale - 0.5
    coordinates = np.clip(coordinates, 0.0, float(input_size - 1))
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, input_size - 1)
    weights = (coordinates - lower).astype(np.float32)
    return lower, upper, weights
