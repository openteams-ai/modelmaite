"""MAITE-compliant image-classification model wrappers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, override  # noqa: UP035

from modelmaite._onnx import (
    IMAGE_CLASSIFICATION_INTERFACE,
    ONNX_INSTALL_HINT,
    UNIT_INTERVAL_TOLERANCE,
    JaticOnnxImageInputs,
    get_index2label_from_model_config,
    import_onnxruntime,
    load_jatic_onnx_metadata,
    optional_output_int,
    validate_jatic_onnx_session,
)
from modelmaite._torch import select_torch_device
from modelmaite._typing import Image
from modelmaite.base import ImageClassifierBase

if TYPE_CHECKING:
    import torch

SUPPORTED_TORCHVISION_MODELS = {
    "alexnet": "AlexNet_Weights",
    "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
}
SUPPORTED_ONNX_MODELS = {"jatic_onnx"}
TORCHVISION_INSTALL_HINT = 'Install TorchVision support with `uv add "modelmaite[torchvision]"`.'


class TorchvisionICModel(ImageClassifierBase["torch.Tensor", "torch.Tensor"]):
    """A MAITE-compliant wrapper for torchvision image-classification models."""

    input_channels = (3,)

    def __init__(
        self,
        *,
        model_name: str,
        device: object | None = None,
        weights_path: str | Path | None = None,
        config_path: str | Path | None = None,
        index2label_key: str = "index2label",
        model_id: str | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a torchvision image-classification model.

        Parameters
        ----------
        model_name
            Name of the torchvision model to instantiate.
        device
            Device to use (for example, ``cpu`` or ``cuda``). If omitted, the best available torch device is used.
        weights_path
            Optional path to a torch state-dict file with user-supplied weights.
        config_path
            Optional path to a metadata JSON file for user-supplied weights. Required when ``weights_path`` is used.
            User-supplied weights still use the default torchvision preprocessing transforms for ``model_name``.
        index2label_key
            Metadata key for mapping class indices to labels.
        model_id
            Optional model identifier.
        batch_size
            Most images per forward pass. By default the whole batch runs at once.
        **kwargs
            Additional keyword arguments forwarded to the torchvision model constructor.
        """
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise ValueError(f"Model {model_name} is not currently supported by modelmaite.")
        if weights_path is not None and config_path is None:
            raise ValueError("Torchvision models with user weights require config_path.")

        self._torch = torch = _import_torch()
        torchvision_models = _import_torchvision_models()
        self._model_name = model_name
        self._constructor_kwargs = dict(kwargs)
        self._weights_path = weights_path
        self._config_path = config_path
        super().__init__(device=device, batch_size=batch_size)

        try:
            model_constructor = getattr(torchvision_models, model_name)
            weights_constructor = getattr(torchvision_models, SUPPORTED_TORCHVISION_MODELS[model_name])
        except Exception as e:
            raise ImportError(f"There was an error importing {model_name} from torchvision.") from e

        self.transforms = weights_constructor.DEFAULT.transforms()
        if weights_path is not None:
            config_path_for_weights = cast(str | Path, config_path)
            with Path(config_path_for_weights).open(encoding="utf-8") as f:
                model_config = json.load(f)
            if not isinstance(model_config, dict):
                raise TypeError(f"Configuration file at {config_path_for_weights} must contain a JSON object.")
            self.index2label = get_index2label_from_model_config(config_path_for_weights, model_config, index2label_key)
            num_classes = _get_config_num_classes(model_config, len(self.index2label), config_path_for_weights)
            self.model = model_constructor(weights=None, num_classes=num_classes, **kwargs).to(self.device)
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise RuntimeError(f"Error loading data from state_dict from {weights_path}") from e
        else:
            default_weights = weights_constructor.DEFAULT
            self.index2label = _index2label_from_default_weights(default_weights, model_name)
            self.model = model_constructor(weights=default_weights, **kwargs).to(self.device)

        self._init_metadata(model_id=model_id, index2label=self.index2label)
        self.model.eval()

    @property
    def name(self) -> str:
        """Human-readable name for the torchvision image-classification model."""
        return self._model_name

    @override
    def preprocess(self, images: list[Image]) -> torch.Tensor:
        """Apply the torchvision transforms to each image on the CPU, stack them, then move the batch to ``device``.

        Each image is resized and cropped before stacking, so images of different sizes can share a
        batch and a batch of large images never exists at full resolution. Inputs should be ``uint8``
        CHW images or floating-point CHW images already scaled to ``[0, 1]``.
        """
        torch = self._torch
        batch = torch.stack([self.transforms(torch.as_tensor(image)) for image in images])
        return batch.to(device=self.device, dtype=torch.float32)

    @override
    def forward(self, prepared: torch.Tensor) -> torch.Tensor:
        """Return the model's logits."""
        return self.model(prepared)

    @override
    def postprocess(self, raw: torch.Tensor, prepared: torch.Tensor) -> list[NDArray[np.float32]]:
        """Apply softmax to the logits and return one score array per image."""
        scores_batch = self._torch.nn.functional.softmax(input=raw, dim=1).cpu().detach()
        return list(scores_batch.numpy().astype(np.float32, copy=False))

    @override
    def _resolve_device(self, requested: str | int | None) -> torch.device:
        return select_torch_device(self._torch, requested)

    @override
    def _inference_context(self) -> AbstractContextManager[object]:
        return self._torch.no_grad()

    @override
    def _id_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"weights_path": self._weights_path, "config_path": self._config_path}
        # Constructor arguments change the network, so they are hashed, but only when given,
        # so a model built without them keeps the same id.
        if self._constructor_kwargs:
            fields["constructor_kwargs"] = self._constructor_kwargs
        return fields


class OnnxICModel(JaticOnnxImageInputs, ImageClassifierBase[tuple[np.ndarray, list[tuple[int, int]]], np.ndarray]):
    """A MAITE-compliant wrapper for JATIC_ONNX image-classification models."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        config_path: str | Path,
        device: object | None = None,
        index2label_key: str = "index2label",
        model_id: str | None = None,
        batch_size: int | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        validate_onnx: bool = False,
    ) -> None:
        """Initialize a JATIC_ONNX image-classification model.

        Parameters
        ----------
        weights_path
            Path to the ONNX model file.
        config_path
            Path to JATIC_ONNX metadata JSON, including ``index2label``.
        device
            Device/provider request (``cpu``, ``cuda``, or ``mps``/CoreML). If omitted, the best available provider is
            selected from the installed ONNX Runtime package.
        index2label_key
            Metadata key for mapping class indices to labels.
        model_id
            Optional model identifier.
        batch_size
            Most images per ONNX Runtime call. Overrides the metadata ``batchSize``. If neither is set, the whole
            batch runs at once.
        image_height
            Optional runtime input-height override.
        image_width
            Optional runtime input-width override.
        validate_onnx
            If ``True``, run ``onnx.checker.check_model`` before creating the ONNX Runtime session. This parses the
            model separately from ONNX Runtime, so it is disabled by default for large models.
        """
        self._ort = ort = import_onnxruntime()

        if validate_onnx:
            try:
                onnx = importlib.import_module("onnx")
            except ImportError:
                raise ImportError(
                    f"validate_onnx=True requires optional dependency 'onnx'. {ONNX_INSTALL_HINT}"
                ) from None
            onnx.checker.check_model(str(weights_path))

        super().__init__(device=device, batch_size=batch_size)
        self.model = ort.InferenceSession(str(weights_path), providers=self._providers)
        validate_jatic_onnx_session(self.model, expected_outputs={"scores"})

        self.config, self.index2label = load_jatic_onnx_metadata(
            config_path,
            expected_io_interface=IMAGE_CLASSIFICATION_INTERFACE,
            index2label_key=index2label_key,
        )
        if not self.index2label:
            raise ValueError("ONNX metadata index2label must contain at least one class.")
        self._n_classes = len(self.index2label)
        output_meta = self.config.get("io", {}).get("output", {})
        n_classes = optional_output_int(output_meta.get("nClasses"), "nClasses", minimum=1)
        if n_classes is not None and n_classes != self._n_classes:
            raise ValueError(
                f"ONNX metadata io.output.nClasses={n_classes} does not match len(index2label)={self._n_classes}."
            )

        self._check_graph_batch_size()
        self._model_name = "jatic_onnx"
        self._image_height = image_height
        self._image_width = image_width
        self._weights_path = weights_path
        self._config_path = config_path
        self._init_metadata(model_id=model_id, index2label=self.index2label)

    @property
    def name(self) -> str:
        """Human-readable name for JATIC_ONNX image-classification model."""
        return self._model_name

    @override
    def forward(self, prepared: tuple[np.ndarray, list[tuple[int, int]]]) -> np.ndarray:
        """Run the ONNX session and return its ``scores`` output."""
        batch, _ = prepared
        return self.model.run(["scores"], {"image": batch})[0]

    @override
    def postprocess(
        self, raw: np.ndarray, prepared: tuple[np.ndarray, list[tuple[int, int]]]
    ) -> list[NDArray[np.float32]]:
        """Check the ``(N, C)`` probabilities and return one row per image."""
        scores_batch = np.asarray(raw, dtype=np.float32)
        _validate_scores(scores_batch, expected_batch_size=len(prepared[1]), expected_num_classes=self._n_classes)
        return list(scores_batch)

    @override
    def _id_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"weights_path": self._weights_path, "config_path": self._config_path}
        # The resize overrides change every output. They are hashed only when given, so a model
        # that uses the ONNX metadata's size keeps the same id.
        if self._image_height is not None:
            fields["image_height"] = self._image_height
        if self._image_width is not None:
            fields["image_width"] = self._image_width
        return fields


class TorchvisionModelSpecification(TypedDict):
    """Torchvision image-classification model metadata."""

    model_type: Literal["alexnet", "resnext50_32x4d"]
    model_weights_path: NotRequired[str | Path]
    model_config_path: NotRequired[str | Path]


class OnnxModelSpecification(TypedDict):
    """JATIC_ONNX image-classification model metadata."""

    model_type: Literal["jatic_onnx"]
    model_weights_path: str | Path
    model_config_path: str | Path


ModelSpecification: TypeAlias = TorchvisionModelSpecification | OnnxModelSpecification


def load_models(
    models: Mapping[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, TorchvisionICModel | OnnxICModel]:
    """Load image-classification models from model specification dictionaries.

    Keyword arguments are forwarded to every loaded model and must be valid for each selected wrapper.
    """
    loaded = {}
    for name, meta_dict in models.items():
        model_type = meta_dict.get("model_type")
        if model_type is None:
            raise ValueError("Image-classification model specifications require model_type.")
        if model_type in SUPPORTED_TORCHVISION_MODELS:
            loaded[name] = TorchvisionICModel(
                model_name=model_type,
                weights_path=meta_dict.get("model_weights_path"),
                config_path=meta_dict.get("model_config_path"),
                **kwargs,
            )
            continue
        if model_type in SUPPORTED_ONNX_MODELS:
            weights_path = meta_dict.get("model_weights_path")
            config_path = meta_dict.get("model_config_path")
            if weights_path is None or config_path is None:
                raise ValueError("JATIC_ONNX models require model_weights_path and model_config_path.")
            loaded[name] = OnnxICModel(
                weights_path=weights_path,
                config_path=config_path,
                **kwargs,
            )
            continue

        supported = sorted(set(SUPPORTED_TORCHVISION_MODELS) | SUPPORTED_ONNX_MODELS)
        raise ValueError(f"Unsupported model_type {model_type!r}; supported model types: {supported}.")

    return loaded


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        raise ImportError(
            f"Torchvision image-classification wrappers require optional dependency 'torch'. {TORCHVISION_INSTALL_HINT}"
        ) from None


def _import_torchvision_models() -> Any:
    try:
        return importlib.import_module("torchvision.models")
    except ImportError:
        raise ImportError(
            "Torchvision image-classification wrappers require optional dependency 'torchvision'. "
            f"{TORCHVISION_INSTALL_HINT}"
        ) from None


def _get_config_num_classes(model_config: dict[str, Any], expected_num_classes: int, config_path: str | Path) -> int:
    num_classes = int(model_config.get("num_classes", expected_num_classes))
    if num_classes != expected_num_classes:
        raise ValueError(
            f"Torchvision model config at {config_path} has num_classes={num_classes}, "
            f"but index2label contains {expected_num_classes} classes."
        )
    return num_classes


def _index2label_from_default_weights(default_weights: Any, model_name: str) -> dict[int, str]:
    meta = getattr(default_weights, "meta", {})
    categories = meta.get("categories") if isinstance(meta, dict) else None
    if categories is None:
        raise ValueError(f"Torchvision weights for {model_name} do not declare category metadata.")
    return dict(enumerate(str(label) for label in categories))


def _validate_scores(
    scores_batch: np.ndarray,
    *,
    expected_batch_size: int,
    expected_num_classes: int,
) -> None:
    if scores_batch.ndim != 2:
        raise ValueError(
            f"JATIC_ONNX image-classification 'scores' output must have shape (N, C), got {scores_batch.shape}."
        )
    if scores_batch.shape[0] != expected_batch_size:
        raise ValueError(
            f"JATIC_ONNX image-classification output contains batch size {scores_batch.shape[0]}, "
            f"expected {expected_batch_size}."
        )
    if scores_batch.shape[1] != expected_num_classes:
        raise ValueError(
            f"JATIC_ONNX image-classification 'scores' output has {scores_batch.shape[1]} classes, "
            f"but index2label contains {expected_num_classes}."
        )
    if not np.all(np.isfinite(scores_batch)):
        raise ValueError("JATIC_ONNX image-classification 'scores' output must contain only finite values.")
    # JATIC_ONNX v1 image-classification `scores` are normalized per-class probabilities, not raw logits.
    if scores_batch.size and (
        np.min(scores_batch) < -UNIT_INTERVAL_TOLERANCE or np.max(scores_batch) > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("JATIC_ONNX image-classification 'scores' output must be in the range [0, 1].")
