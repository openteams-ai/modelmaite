"""MAITE-compliant object-detection model wrappers."""

from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import NotRequired  # noqa: UP035

from modelmaite._onnx import (
    IMAGE_OBJECT_DETECTION_INTERFACE,
    ONNX_INSTALL_HINT,
    UNIT_INTERVAL_TOLERANCE,
    get_index2label_from_model_config,
    get_onnx_providers,
    id_hash,
    import_onnxruntime,
    load_jatic_onnx_metadata,
    optional_output_int,
    prepare_jatic_onnx_image_batch,
    validate_input_batch,
    validate_jatic_onnx_session,
)
from modelmaite.object_detection.types import DetectionTarget

SUPPORTED_TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "maskrcnn_resnet50_fpn_v2": "MaskRCNN_ResNet50_FPN_V2_Weights",
    "maskrcnn_resnet50_fpn": "MaskRCNN_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
    "keypointrcnn_resnet50_fpn": "KeypointRCNN_ResNet50_FPN_Weights",
    "ssd300_vgg16": "SSD300_VGG16_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}
SUPPORTED_VISDRONE_MODELS = {
    "res2net50": "Res2Net50_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet18": "ResNet18_Weights",
}
SUPPORTED_ONNX_MODELS = {"jatic_onnx"}
TORCHVISION_INSTALL_HINT = 'Install TorchVision support with `poetry add "modelmaite[torchvision]"`.'
VISDRONE_INSTALL_HINT = 'Install VisDrone support with `poetry add "modelmaite[visdrone]"`.'


class TorchvisionODModel:
    """A MAITE-compliant wrapper for torchvision object-detection models."""

    def __init__(
        self,
        *,
        model_name: str,
        device: object | None = None,
        weights_path: str | Path | None = None,
        config_path: str | Path | None = None,
        index2label_key: str = "index2label",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a torchvision object-detection model.

        Parameters
        ----------
        model_name
            Name of the torchvision detection model to instantiate.
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
        **kwargs
            Additional keyword arguments forwarded to the torchvision model constructor.
        """
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise ValueError(f"Model {model_name} is not currently supported by modelmaite.")
        if weights_path is not None and config_path is None:
            raise ValueError("Torchvision models with user weights require config_path.")

        torch = _import_torch()
        detection_models = _import_torchvision_detection_models()
        self._model_name = model_name
        self.device = _set_torch_device(torch, device)

        try:
            model_constructor = getattr(detection_models, model_name)
            weights_constructor = getattr(detection_models, SUPPORTED_TORCHVISION_MODELS[model_name])
        except Exception as e:
            raise ImportError(f"There was an error importing {model_name} from torchvision.") from e

        self.preprocess = weights_constructor.DEFAULT.transforms()
        if weights_path is not None:
            config_path_for_weights = cast(str | Path, config_path)
            with Path(config_path_for_weights).open(encoding="utf-8") as f:
                model_config = json.load(f)
            if not isinstance(model_config, dict):
                raise TypeError(f"Configuration file at {config_path_for_weights} must contain a JSON object.")
            self.index2label = get_index2label_from_model_config(config_path_for_weights, model_config, index2label_key)
            num_classes = int(model_config.get("num_classes", len(self.index2label)))
            self.model = model_constructor(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
                **kwargs,
            ).to(self.device)
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise RuntimeError(f"Error loading data from state_dict from {weights_path}") from e
        else:
            default_weights = weights_constructor.DEFAULT
            self.index2label = dict(enumerate(str(label) for label in default_weights.meta["categories"]))
            self.model = model_constructor(weights=default_weights, **kwargs).to(self.device)

        if model_id is None:
            model_id = f"{self._model_name}_{id_hash(weights_path=weights_path, config_path=config_path)}"
        self.metadata = {"id": model_id, "index2label": self.index2label}
        self.model.eval()

    def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[DetectionTarget]:
        """Make object-detection predictions for a CHW image batch.

        Inputs should be 3-channel RGB ``uint8`` CHW images or floating-point CHW images already scaled to ``[0, 1]``.
        """
        torch = _import_torch()
        images = _to_torch_images(torch, input_batch)
        preprocessed = [self.preprocess(image) for image in images]
        batch_on_device = _move_to_device(preprocessed, self.device)
        with torch.no_grad():
            predictions_batch = self.model(batch_on_device)
        return [
            DetectionTarget(
                boxes=np.asarray(prediction["boxes"].detach().cpu(), dtype=np.float32),
                labels=np.asarray(prediction["labels"].detach().cpu(), dtype=np.int64),
                scores=np.asarray(prediction["scores"].detach().cpu(), dtype=np.float32),
            )
            for prediction in predictions_batch
        ]

    @property
    def name(self) -> str:
        """Human-readable name for the torchvision object-detection model."""
        return self._model_name


class VisdroneODModel:
    """A MAITE-compatible wrapper for Kitware CenterNet VisDrone detectors."""

    _weights_urls = {
        "Res2Net50_Weights": "https://data.kitware.com/api/v1/item/623e18464acac99f42f40a4e/download",
        "ResNet50_Weights": "https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download",
        "ResNet18_Weights": "https://data.kitware.com/api/v1/item/623de4744acac99f42f05fb1/download",
    }
    _weights_sizes = {
        "Res2Net50_Weights": 113258852,
        "ResNet50_Weights": 112520312,
        "ResNet18_Weights": 60001986,
    }
    _weights_sha512 = {
        "Res2Net50_Weights": (
            "ab587d8eba848dccfe9d63bb1169bd554bfac2bb335e5b22fc1f9d37f0eb66a7"
            "c61441d1b656f695b23f63dc914660b09da82954a34695e42a1cb589e8ae3755"
        ),
        "ResNet50_Weights": (
            "a0083ec55d46c420d06c414e5ecc1863d6ad9b6a1732acff5c9dba28158a4c5"
            "a04f43541415d503fa776031a7329e3912864ae2348b3bee035df0d1e7acefa49"
        ),
        "ResNet18_Weights": (
            "e8b0b9fd685d7ef8aa493917a1696c0b85e370689db99e9bb8882136c43eb8f"
            "246bcabb737cefe95fc7098a75b17900a9ba6f746467fe3847c42327f452b97b2"
        ),
    }

    def __init__(
        self,
        *,
        arch: str,
        model_pickle_dir: str | Path | None = None,
        model_name: str | None = None,
        device: object | None = None,
        batch_size: int = 3,
        num_workers: int = 0,
        max_dets: int = 500,
        model_id: str | None = None,
    ) -> None:
        """Initialize a VisDrone object-detection model.

        Parameters
        ----------
        arch
            Model backbone architecture: ``res2net50``, ``resnet50``, or ``resnet18``.
        model_pickle_dir
            Directory where Kitware weights are stored or should be downloaded.
        model_name
            File stem for the weights file. Defaults to ``centernet-{arch}``.
        device
            Device to use (for example, ``cpu`` or ``cuda``). If omitted, the best available torch device is used.
        batch_size
            Number of images processed by the wrapped detector per batch.
        num_workers
            Number of worker subprocesses used by the wrapped detector.
        max_dets
            Maximum detections returned per image.
        model_id
            Optional model identifier.
        """
        if arch not in SUPPORTED_VISDRONE_MODELS:
            raise ValueError(f"Model with backbone {arch} is not currently supported by the VisDrone wrapper.")

        try:
            centernet = importlib.import_module("smqtk_detection.impls.detect_image_objects.centernet")
        except ImportError:
            raise ImportError(
                "VisDrone object-detection wrappers require optional dependency 'smqtk-detection'. "
                f"{VISDRONE_INSTALL_HINT}"
            ) from None

        torch = _import_torch(install_hint=VISDRONE_INSTALL_HINT, wrapper_name="VisDrone object-detection")
        self.device = _set_torch_device(torch, device)
        self._model_name = f"centernet-{arch}" if model_name is None else model_name

        if model_pickle_dir is None:
            model_pickle_dir = Path.home() / ".cache" / "modelmaite" / "visdrone"
        model_file = Path(model_pickle_dir) / f"{self._model_name}.pth"
        weights_key = SUPPORTED_VISDRONE_MODELS[arch]
        if not model_file.is_file():
            model_file.parent.mkdir(parents=True, exist_ok=True)
            _download_file(
                self._weights_urls[weights_key],
                model_file,
                expected_size=self._weights_sizes[weights_key],
                expected_sha512=self._weights_sha512[weights_key],
            )

        self.model = centernet.CenterNetVisdrone(
            arch=arch,
            model_file=str(model_file),
            device=str(self.device),
            max_dets=max_dets,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.index2label = {
            0: "ignored regions",
            1: "pedestrian",
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor",
            11: "others",
        }
        _validate_smqtk_labels(self.model, set(self.index2label.values()))
        if model_id is None:
            model_id = f"visdrone_{arch}_kitware"
        self.metadata = {"id": model_id, "index2label": self.index2label}

    def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[DetectionTarget]:
        """Make object-detection predictions for a CHW image batch."""
        validate_input_batch(input_batch)
        array_batch = [np.asarray(inp).transpose(1, 2, 0) for inp in input_batch]
        predictions_batch = self.model.detect_objects(array_batch)
        label2index = {label: index for index, label in self.index2label.items()}

        output_batch = []
        for prediction in predictions_batch:
            num_boxes = len(prediction)
            boxes = np.empty((num_boxes, 4), dtype=np.float32)
            labels = np.empty(num_boxes, dtype=np.int64)
            scores = np.empty(num_boxes, dtype=np.float32)
            for i, (bbox, label_map) in enumerate(prediction):
                (min_x, min_y), (max_x, max_y) = (bbox.min_vertex, bbox.max_vertex)
                boxes[i] = [min_x, min_y, max_x, max_y]
                label, score = max(label_map.items(), key=lambda x: x[1])
                if label not in label2index:
                    expected = sorted(label2index)
                    raise ValueError(f"VisDrone detector returned unknown label {label!r}; expected one of {expected}.")
                labels[i] = label2index[label]
                scores[i] = score
            output_batch.append(DetectionTarget(boxes=boxes, labels=labels, scores=scores))

        return output_batch

    @property
    def name(self) -> str:
        """Human-readable name for the VisDrone object-detection model."""
        return f"visdrone-{self._model_name}"


class OnnxODModel:
    """A MAITE-compliant wrapper for JATIC_ONNX object-detection models."""

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
        """Initialize a JATIC_ONNX object-detection model.

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
            Optional runtime batch-size override.
        image_height
            Optional runtime input-height override.
        image_width
            Optional runtime input-width override.
        validate_onnx
            If ``True``, run ``onnx.checker.check_model`` before creating the ONNX Runtime session. This parses the
            model separately from ONNX Runtime, so it is disabled by default for large models.
        """
        ort = import_onnxruntime()

        if validate_onnx:
            try:
                onnx = importlib.import_module("onnx")
            except ImportError:
                raise ImportError(
                    f"validate_onnx=True requires optional dependency 'onnx'. {ONNX_INSTALL_HINT}"
                ) from None
            onnx.checker.check_model(str(weights_path))

        self.device, providers = get_onnx_providers(device, ort=ort)
        self.model = ort.InferenceSession(str(weights_path), providers=providers)
        validate_jatic_onnx_session(self.model, expected_outputs={"boxes", "scores"})

        self.config, self.index2label = load_jatic_onnx_metadata(
            config_path,
            expected_io_interface=IMAGE_OBJECT_DETECTION_INTERFACE,
            index2label_key=index2label_key,
        )
        output_meta = self.config.get("io", {}).get("output", {})
        if not self.index2label:
            raise ValueError("ONNX metadata index2label must contain at least one class.")
        self._n_classes = len(self.index2label)
        n_classes = optional_output_int(output_meta.get("nClasses"), "nClasses", minimum=1)
        if n_classes is not None and n_classes != self._n_classes:
            raise ValueError(
                f"ONNX metadata io.output.nClasses={n_classes} does not match len(index2label)={self._n_classes}."
            )

        self._model_name = "jatic_onnx"
        self._batch_size = batch_size
        self._image_height = image_height
        self._image_width = image_width
        self._n_boxes = optional_output_int(output_meta.get("nBoxes"), "nBoxes", minimum=0)
        if model_id is None:
            model_id = f"{self._model_name}_{id_hash(weights_path=weights_path, config_path=config_path)}"
        self.metadata = {"id": model_id, "index2label": self.index2label}

    def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[DetectionTarget]:
        """Make object-detection predictions for a CHW image batch.

        Every JATIC_ONNX output slot is returned after coordinate scaling; this wrapper does not apply NMS,
        confidence thresholding, or background filtering.
        """
        batch, original_sizes = prepare_jatic_onnx_image_batch(
            input_batch,
            self.config,
            batch_size=self._batch_size,
            image_height=self._image_height,
            image_width=self._image_width,
        )
        outputs = self.model.run(["boxes", "scores"], {"image": batch})
        boxes_batch = np.asarray(outputs[0], dtype=np.float32)
        scores_batch = np.asarray(outputs[1], dtype=np.float32)
        _validate_output_shapes(
            boxes_batch,
            scores_batch,
            expected_batch_size=len(original_sizes),
            expected_num_boxes=self._n_boxes,
            expected_num_classes=self._n_classes,
        )

        output_batch: list[DetectionTarget] = []
        for boxes, class_scores, (height, width) in zip(boxes_batch, scores_batch, original_sizes, strict=True):
            labels = np.argmax(class_scores, axis=-1).astype(np.int64)
            scores = np.max(class_scores, axis=-1).astype(np.float32)
            pixel_boxes = np.asarray(boxes, dtype=np.float32).copy()
            pixel_boxes[:, [0, 2]] *= float(width)
            pixel_boxes[:, [1, 3]] *= float(height)
            output_batch.append(DetectionTarget(boxes=pixel_boxes, labels=labels, scores=scores))

        return output_batch

    @property
    def name(self) -> str:
        """Human-readable name for JATIC_ONNX object-detection model."""
        return self._model_name


class TorchvisionModelSpecification(TypedDict):
    """Torchvision object-detection model metadata."""

    model_type: Literal[
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "maskrcnn_resnet50_fpn_v2",
        "maskrcnn_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "fcos_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ]
    model_weights_path: NotRequired[str | Path]
    model_config_path: NotRequired[str | Path]


class VisdroneModelSpecification(TypedDict):
    """VisDrone object-detection model metadata."""

    model_type: Literal["res2net50", "resnet50", "resnet18"]
    model_pickle_dir: NotRequired[str | Path]


class OnnxModelSpecification(TypedDict):
    """JATIC_ONNX object-detection model metadata."""

    model_type: Literal["jatic_onnx"]
    model_weights_path: str | Path
    model_config_path: str | Path


ModelSpecification: TypeAlias = TorchvisionModelSpecification | VisdroneModelSpecification | OnnxModelSpecification


def load_models(
    models: Mapping[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, TorchvisionODModel | VisdroneODModel | OnnxODModel]:
    """Load object-detection models from model specification dictionaries.

    Keyword arguments are forwarded to every loaded model and must be valid for each selected wrapper.
    """
    loaded = {}
    for name, meta_dict in models.items():
        model_type = meta_dict.get("model_type")
        if model_type is None:
            raise ValueError("Object-detection model specifications require model_type.")
        if model_type in SUPPORTED_TORCHVISION_MODELS:
            loaded[name] = TorchvisionODModel(
                model_name=model_type,
                weights_path=meta_dict.get("model_weights_path"),
                config_path=meta_dict.get("model_config_path"),
                **kwargs,
            )
            continue
        if model_type in SUPPORTED_VISDRONE_MODELS:
            loaded[name] = VisdroneODModel(
                arch=model_type,
                model_pickle_dir=meta_dict.get("model_pickle_dir"),
                **kwargs,
            )
            continue
        if model_type in SUPPORTED_ONNX_MODELS:
            weights_path = meta_dict.get("model_weights_path")
            config_path = meta_dict.get("model_config_path")
            if weights_path is None or config_path is None:
                raise ValueError("JATIC_ONNX models require model_weights_path and model_config_path.")
            loaded[name] = OnnxODModel(
                weights_path=weights_path,
                config_path=config_path,
                **kwargs,
            )
            continue

        supported = sorted(set(SUPPORTED_TORCHVISION_MODELS) | set(SUPPORTED_VISDRONE_MODELS) | SUPPORTED_ONNX_MODELS)
        raise ValueError(f"Unsupported model_type {model_type!r}; supported model types: {supported}.")

    return loaded


def _import_torch(
    *,
    install_hint: str = TORCHVISION_INSTALL_HINT,
    wrapper_name: str = "Torchvision object-detection",
) -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        raise ImportError(f"{wrapper_name} wrappers require optional dependency 'torch'. {install_hint}") from None


def _import_torchvision_detection_models() -> Any:
    try:
        return importlib.import_module("torchvision.models.detection")
    except ImportError:
        raise ImportError(
            "Torchvision object-detection wrappers require optional dependency 'torchvision'. "
            f"{TORCHVISION_INSTALL_HINT}"
        ) from None


def _set_torch_device(torch: Any, device: object | None) -> Any:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _to_torch_images(torch: Any, input_batch: Sequence[ArrayLike]) -> list[Any]:
    if len(input_batch) == 0:
        raise ValueError("Input batch must contain at least one image.")

    images = []
    for image in input_batch:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Input data must follow CHW-ordering, current shape: {arr.shape}")
        channel_count = arr.shape[0]
        if channel_count != 3:
            raise ValueError(
                "Torchvision object-detection inputs must be RGB CHW images with exactly 3 channels, "
                f"got shape {arr.shape}."
            )
        images.append(torch.as_tensor(image))
    return images


def _move_to_device(batch: Any, device: Any) -> Any:
    if isinstance(batch, list):
        return [item.to(device) for item in batch]
    return batch.to(device)


def _download_file(url: str, destination: Path, *, expected_size: int, expected_sha512: str) -> None:
    try:
        httpx = importlib.import_module("httpx")
    except ImportError:
        raise ImportError(
            "VisDrone object-detection wrappers require optional dependency 'httpx'. " f"{VISDRONE_INSTALL_HINT}"
        ) from None

    hasher = hashlib.sha512()
    bytes_written = 0
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temp_file:
        temp_destination = Path(temp_file.name)
        try:
            with httpx.stream("GET", url, timeout=10, follow_redirects=True) as response:
                response.raise_for_status()
                content_length = getattr(response, "headers", {}).get("content-length")
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    hasher.update(chunk)
                    temp_file.write(chunk)
            if content_length is not None and int(content_length) != bytes_written:
                raise RuntimeError(
                    f"Downloaded VisDrone weights from {url} were {bytes_written} bytes, "
                    f"but Content-Length was {content_length}."
                )
            if bytes_written != expected_size:
                raise RuntimeError(
                    f"Downloaded VisDrone weights from {url} were {bytes_written} bytes, expected {expected_size}."
                )
            if hasher.hexdigest() != expected_sha512:
                raise RuntimeError(f"Downloaded VisDrone weights from {url} failed SHA-512 verification.")
            temp_destination.replace(destination)
        except Exception:
            temp_destination.unlink(missing_ok=True)
            raise


def _validate_smqtk_labels(model: Any, expected_labels: set[str]) -> None:
    for attr_name in ("class_names", "classes", "labels", "CLASS_NAMES", "CLASSES", "LABELS"):
        observed_labels = getattr(model, attr_name, None)
        if observed_labels is None:
            continue
        unknown = {str(label) for label in observed_labels} - expected_labels
        if unknown:
            expected = sorted(expected_labels)
            raise ValueError(
                f"VisDrone detector exposes unknown label(s) {sorted(unknown)}; expected one of {expected}."
            )
        return


def _validate_output_shapes(
    boxes_batch: np.ndarray,
    scores_batch: np.ndarray,
    *,
    expected_batch_size: int,
    expected_num_boxes: int | None,
    expected_num_classes: int,
) -> None:
    if boxes_batch.ndim != 3 or boxes_batch.shape[-1] != 4:
        raise ValueError(
            f"JATIC_ONNX object-detection 'boxes' output must have shape (N, D, 4), got {boxes_batch.shape}."
        )
    if scores_batch.ndim != 3:
        raise ValueError(
            f"JATIC_ONNX object-detection 'scores' output must have shape (N, D, C), got {scores_batch.shape}."
        )
    if boxes_batch.shape[:2] != scores_batch.shape[:2]:
        raise ValueError(
            "JATIC_ONNX object-detection 'boxes' and 'scores' outputs must agree on batch and detection dimensions, "
            f"got {boxes_batch.shape} and {scores_batch.shape}."
        )
    if boxes_batch.shape[0] != expected_batch_size:
        raise ValueError(
            f"JATIC_ONNX object-detection outputs contain batch size {boxes_batch.shape[0]}, "
            f"expected {expected_batch_size}."
        )
    if expected_num_boxes is not None and boxes_batch.shape[1] != expected_num_boxes:
        raise ValueError(
            f"JATIC_ONNX object-detection outputs contain {boxes_batch.shape[1]} detections per image, "
            f"but ONNX metadata io.output.nBoxes is {expected_num_boxes}."
        )
    if scores_batch.shape[-1] != expected_num_classes:
        raise ValueError(
            f"JATIC_ONNX object-detection 'scores' output has {scores_batch.shape[-1]} classes, "
            f"but index2label contains {expected_num_classes}."
        )
    _validate_output_values(boxes_batch, scores_batch)


def _validate_output_values(boxes_batch: np.ndarray, scores_batch: np.ndarray) -> None:
    if not np.all(np.isfinite(boxes_batch)):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must contain only finite values.")
    if boxes_batch.size and (
        np.min(boxes_batch) < -UNIT_INTERVAL_TOLERANCE or np.max(boxes_batch) > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must be normalized to the range [0, 1].")
    if boxes_batch.size and (
        np.any(boxes_batch[..., 0] > boxes_batch[..., 2]) or np.any(boxes_batch[..., 1] > boxes_batch[..., 3])
    ):
        raise ValueError("JATIC_ONNX object-detection 'boxes' output must use xyxy order with x0 <= x1 and y0 <= y1.")
    if not np.all(np.isfinite(scores_batch)):
        raise ValueError("JATIC_ONNX object-detection 'scores' output must contain only finite values.")
    if scores_batch.size and (
        np.min(scores_batch) < -UNIT_INTERVAL_TOLERANCE or np.max(scores_batch) > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("JATIC_ONNX object-detection 'scores' output must be in the range [0, 1].")
