"""Base classes shared by the modelmaite wrappers.

Every wrapper derives from :class:`ModelBase`, which owns ``metadata`` and the
batch-level ``__call__``. Image wrappers derive from :class:`ImageModelBase`
through a task base, and run each batch as::

    check inputs -> split into chunks of batch_size -> for each chunk:
        preprocess -> forward (inside the inference context) -> postprocess
    -> check outputs

This module imports neither torch, onnxruntime nor maite. Framework code lives in
the wrappers and is imported lazily there.

The base classes are new and may still change before modelmaite 1.0.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from modelmaite._identity import id_hash
from modelmaite._typing import (
    Image,
    ModelMetadata,
    ObjectDetectionModel,
    ObjectDetectionTarget,
    VideoFrame,
    VideoStream,
)

if TYPE_CHECKING:
    # Imported for type checkers only: the task packages import this module, so a
    # runtime import of their types would be circular.
    from modelmaite.multiobject_tracking.types import MOTFrameTarget, MOTTarget

__all__ = [
    "ImageClassifierBase",
    "ImageModelBase",
    "ModelBase",
    "ObjectDetectorBase",
    "TrackerBase",
    "normalize_device",
]

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
PreparedT = TypeVar("PreparedT")
RawT = TypeVar("RawT")
StateT = TypeVar("StateT")
DetectionT = TypeVar("DetectionT", bound=ObjectDetectionTarget)
_T = TypeVar("_T")

_MODEL_METADATA_KEYS = ModelMetadata.__required_keys__ | ModelMetadata.__optional_keys__

# Same tolerance as MAITE 0.10's score predicates.
SCORE_TOLERANCE = 1e-6


class ModelBase(ABC, Generic[InputT, OutputT]):
    """Root of every wrapper: MAITE ``metadata`` and ``__call__`` on a batch.

    An empty batch returns an empty list without running the model. Otherwise
    ``_predict`` runs and the number of outputs is checked. The task bases check
    each output as soon as it is produced, so a bad output stops the call early.
    """

    metadata: ModelMetadata
    # MAITE makes index2label optional, but CheckMAITE's conventions require it of
    # models, and classifier scores and detection labels cannot be read without it.
    requires_index2label: ClassVar[bool] = False
    # True when outputs are read by position, so index2label keys must be exactly 0..n-1.
    positional_index2label: ClassVar[bool] = False

    def __call__(self, input_batch: Sequence[InputT]) -> Sequence[OutputT]:
        """Run the model on a batch and return one output per input."""
        items = list(input_batch)
        if not items:
            return []
        outputs = list(self._predict(items))
        if len(outputs) != len(items):
            raise ValueError(f"{type(self).__name__} returned {len(outputs)} outputs for {len(items)} inputs.")
        return outputs

    @abstractmethod
    def _predict(self, items: list[InputT]) -> Sequence[OutputT]:
        """Return one output per item. ``items`` is never empty."""

    def _init_metadata(
        self,
        *,
        model_id: str | None,
        index2label: Mapping[int, str] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Set ``self.metadata``. Call it at the end of ``__init__``, once the model is loaded.

        ``model_id=None`` uses ``_default_model_id()``. A plain ``dict`` label map is
        stored as given, so it stays the same object as ``self.index2label``. ``extra``
        adds other keys to ``metadata``; it may not set a key ``ModelMetadata`` defines.
        Type checkers still see ``metadata`` as ``ModelMetadata``, so read extra keys
        with ``.get()`` or a cast.
        """
        extra = dict(extra or {})
        for key in extra:
            if not isinstance(key, str):
                raise TypeError(f"extra metadata keys must be strings, got {key!r}.")
            if key in _MODEL_METADATA_KEYS:
                raise ValueError(
                    f"extra metadata cannot set {key!r}: ModelMetadata defines it. "
                    "Pass model_id or index2label to _init_metadata instead."
                )
        if index2label is None and self.requires_index2label:
            raise ValueError(f"{type(self).__name__} needs index2label: its outputs are read through it.")
        if model_id is None:
            model_id = self._default_model_id()
        if not isinstance(model_id, str):
            raise TypeError(f"model_id must be a string, got {type(model_id).__name__}.")
        if not model_id.strip():
            # Callers such as CheckMAITE key cached results by id, so every model with an
            # empty id would share them.
            raise ValueError("model_id must not be empty; pass None to use the default id.")
        if index2label is None:
            metadata: ModelMetadata = {"id": model_id}
        else:
            index2label = _check_index2label(index2label)
            if self.positional_index2label and sorted(index2label) != list(range(len(index2label))):
                raise ValueError(
                    f"{type(self).__name__} index2label keys must be exactly 0..{len(index2label) - 1}, "
                    f"got {sorted(index2label)}: score i is the score of label i."
                )
            metadata = {"id": model_id, "index2label": index2label}
        # ModelMetadata is not closed, so extra keys are allowed at runtime; the TypedDict can't name them.
        self.metadata = cast(ModelMetadata, {**metadata, **extra})

    def _default_model_id(self) -> str:
        """Return the id used when ``model_id`` is not given."""
        raise NotImplementedError(f"{type(self).__name__} needs model_id or a _default_model_id() method.")

    def _id_fields(self) -> Mapping[str, object]:
        """Return the settings hashed into the default id.

        Values may be JSON values, paths, classes, or objects whose attributes are such values;
        an object is hashed by its class and attributes. Anything else makes the default id
        raise ``TypeError``, and the wrapper then needs ``model_id``.

        The default is empty, and an empty mapping gives no default id: the wrapper then
        needs ``model_id``, so two differently configured models never share an id by accident.
        """
        return {}

    def _id_hash(self) -> str:
        """Hash ``_id_fields()``, or raise if the wrapper declares none."""
        fields = self._id_fields()
        if not fields:
            raise ValueError(
                f"{type(self).__name__} needs model_id, or an _id_fields() method returning the settings "
                "that change its outputs."
            )
        return id_hash(**fields)


class ImageModelBase(ModelBase[Image, OutputT], Generic[OutputT, PreparedT, RawT]):
    """Base for wrappers that take a batch of CHW images.

    Subclasses implement ``name``, ``preprocess``, ``forward`` and ``postprocess``,
    and may override the class attributes and the underscored hooks below.

    Attributes
    ----------
    input_channels
        Channel counts the wrapper accepts. Other counts are rejected before the model runs.
    same_shape_batches
        If ``True``, the images in each ``forward`` call must have the same shape, for
        wrappers that stack them. Images in different calls may differ.
    input_range
        ``(low, high)`` bounds every pixel value must lie in, such as ``(0, 255)`` for a
        model that divides by 255 itself. ``None`` (the default) checks no range. Like the
        other input checks, it runs on every image before any chunk is sent to the model.
    batch_size
        Most images sent to ``forward`` at once. ``None`` sends the whole batch.
        It bounds each model call, not memory: ``__call__`` holds every input and
        output of the batch, so split large datasets into batches before calling.
        Read-only after construction, since it may be part of the id.
    device
        The device chosen by ``_resolve_device``, or ``None`` if the wrapper has none.
    """

    input_channels: ClassVar[tuple[int, ...]] = (1, 2, 3, 4)
    same_shape_batches: ClassVar[bool] = False
    input_range: ClassVar[tuple[float, float] | None] = None
    _batch_size: int | None = None
    device: Any = None

    def __init__(self, *, device: object | None = None, batch_size: int | None = None) -> None:
        """Store the batch size and resolve the device. Call it before loading the model."""
        self._batch_size = _check_batch_size(batch_size)
        self.device = self._resolve_device(normalize_device(device))

    @property
    def batch_size(self) -> int | None:
        """Most images sent to ``forward`` at once; ``None`` sends the whole batch."""
        return self._batch_size

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def preprocess(self, images: list[Image]) -> PreparedT:
        """Turn one chunk of checked images into model input."""

    @abstractmethod
    def forward(self, prepared: PreparedT) -> RawT:
        """Run the model. Keep this to the model call itself."""

    @abstractmethod
    def postprocess(self, raw: RawT, prepared: PreparedT) -> Sequence[OutputT]:
        """Turn raw model output into one output per image. ``prepared`` is what ``preprocess`` returned."""

    def _resolve_device(self, requested: str | int | None) -> Any:
        """Pick the device from the normalised request. The default keeps the request as is."""
        return requested

    def _inference_context(self) -> contextlib.AbstractContextManager[object]:
        """Return the context ``forward`` runs in, such as ``torch.no_grad()``.

        It covers ``forward`` only, not ``preprocess`` or ``postprocess``.
        """
        return contextlib.nullcontext()

    def _chunk_size(self) -> int | None:
        """Return the most images per ``forward`` call. ``None`` means the whole batch."""
        return self.batch_size

    def _fixed_batch_size(self) -> int | None:
        """Return the exact number of images every ``forward`` call needs, or ``None`` if any number works.

        When set, it replaces ``_chunk_size()``, and a batch that is not a multiple of it
        is rejected before the model runs.
        """
        return None

    def _plan_chunks(self, images: list[Image], chunk_size: int | None) -> list[list[int]]:
        """Return the indices of the images each ``forward`` call gets, in the order the calls run.

        The default sends consecutive images, ``chunk_size`` at a time (``None``: all at
        once). Override it to group images differently, for example by aspect ratio for
        a model that pads each batch to its largest image. Outputs are returned in input
        order whatever the plan. Every index must appear exactly once, and no chunk may
        be larger than ``chunk_size``. If the plan can change the outputs, as batch
        composition can for a padding model, put what decides it in ``_id_fields()``.
        """
        return list(_batched(range(len(images)), chunk_size))

    def _default_model_id(self) -> str:
        return f"{self.name}_{self._id_hash()}"

    def _validate_output(self, output: OutputT) -> None:
        """Check one output. Task bases add their own checks."""

    def _predict(self, items: list[Image]) -> list[OutputT]:
        self._validate_inputs(items)
        fixed = self._fixed_batch_size()
        if fixed is not None and len(items) % fixed:
            raise ValueError(
                f"{type(self).__name__} runs exactly {fixed} images at a time, "
                f"so a batch must be a multiple of {fixed}; got {len(items)} images."
            )
        chunk_size = fixed if fixed is not None else self._chunk_size()
        plan = self._plan_chunks(items, chunk_size)
        self._check_plan(plan, len(items), chunk_size, exact=fixed is not None)
        if self.same_shape_batches:
            # Checked for every chunk before any runs, so a bad batch fails before the model does.
            for indices in plan:
                self._check_same_shape([items[i] for i in indices])
        outputs: dict[int, OutputT] = {}
        for indices in plan:
            outputs.update(zip(indices, self._predict_chunk([items[i] for i in indices], indices), strict=True))
        return [outputs[i] for i in range(len(items))]

    def _check_plan(self, plan: list[list[int]], n_images: int, chunk_size: int | None, *, exact: bool) -> None:
        name = type(self).__name__
        if sorted(i for indices in plan for i in indices) != list(range(n_images)):
            raise ValueError(f"{name}._plan_chunks must use each of the {n_images} image indices exactly once.")
        for indices in plan:
            if not indices or (chunk_size is not None and len(indices) > chunk_size):
                raise ValueError(f"{name}._plan_chunks returned a chunk of {len(indices)} images; limit {chunk_size}.")
            if exact and len(indices) != chunk_size:
                raise ValueError(
                    f"{name}._plan_chunks returned a chunk of {len(indices)} images; the model needs {chunk_size}."
                )

    def _predict_chunk(self, chunk: list[Image], indices: list[int]) -> list[OutputT]:
        """Run one chunk through the model and check its outputs before the next chunk runs.

        A separate method so that ``prepared`` and ``raw`` are released when it returns,
        before the next chunk is preprocessed. Outputs that are views into them keep
        what they need alive.
        """
        prepared = self.preprocess(chunk)
        with self._inference_context():
            raw = self.forward(prepared)
        outputs = list(self.postprocess(raw, prepared))
        # Checked per chunk: a total check alone would let one chunk's outputs fill in for another's.
        if len(outputs) != len(chunk):
            raise ValueError(
                f"{type(self).__name__}.postprocess returned {len(outputs)} outputs for a chunk of {len(chunk)} images."
            )
        for offset, output in enumerate(outputs):
            try:
                self._validate_output(output)
            except ValueError as e:
                raise ValueError(f"{e} (image {indices[offset]} of the batch)") from None
        return outputs

    def _check_same_shape(self, chunk: list[Image]) -> None:
        first = np.shape(chunk[0])
        for image in chunk[1:]:
            if np.shape(image) != first:
                raise ValueError(
                    f"{type(self).__name__} stacks the images it sends to the model together, so they must have "
                    f"the same shape; got {first} and {np.shape(image)}. A smaller batch_size sends fewer images "
                    "together."
                )

    def _validate_inputs(self, images: Sequence[Image]) -> None:
        """Check every image before any of them reaches the model."""
        name = type(self).__name__
        for index, image in enumerate(images):
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"Input data must follow CHW-ordering, current shape: {array.shape}")
            # input_channels alone decides the channel count, so a subclass can accept more than 4 bands.
            if array.shape[0] not in self.input_channels:
                raise ValueError(
                    f"{name} inputs must be CHW images with {_describe_channels(self.input_channels)}, "
                    f"got shape {array.shape}."
                )
            if array.dtype.kind not in "uif":
                raise ValueError(f"{name} inputs must be integer or floating-point, got {array.dtype}.")
            if array.shape[1] == 0 or array.shape[2] == 0:
                raise ValueError(f"{name} image {index} has zero height or width: shape {array.shape}.")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"{name} image {index} contains NaN or infinite values.")
            if self.input_range is not None:
                low, high = self.input_range
                if array.min() < low or array.max() > high:
                    raise ValueError(
                        f"{name} image {index} has values in [{array.min()}, {array.max()}], "
                        f"outside the expected input range [{low}, {high}]."
                    )


class ImageClassifierBase(ImageModelBase[NDArray[np.floating], PreparedT, RawT]):
    """Base for image classifiers: one 1-D array of class scores per image.

    A classifier needs ``index2label`` (see ``ModelBase.requires_index2label``), with
    keys exactly ``0..n-1``: the score at position ``i`` is the score of label ``i``.
    Scores must be finite and in ``[0, 1]``, with one score per label.
    """

    requires_index2label: ClassVar[bool] = True
    positional_index2label: ClassVar[bool] = True

    def _validate_output(self, output: NDArray[np.floating]) -> None:
        scores = np.asarray(output)
        name = type(self).__name__
        if scores.ndim != 1 or not np.issubdtype(scores.dtype, np.floating):
            raise ValueError(f"{name} scores must be a 1-D float array, got {scores.dtype} {scores.shape}.")
        # None only if a subclass opts out with requires_index2label = False.
        index2label = self.metadata.get("index2label")
        if index2label is not None and len(scores) != len(index2label):
            raise ValueError(f"{name} returned {len(scores)} scores, but index2label has {len(index2label)} classes.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} scores must be finite.")
        if scores.size and (scores.min() < -SCORE_TOLERANCE or scores.max() > 1 + SCORE_TOLERANCE):
            raise ValueError(f"{name} scores must be in the range [0, 1].")


class ObjectDetectorBase(ImageModelBase[DetectionT, PreparedT, RawT]):
    """Base for object detectors: one detection target per image.

    ``DetectionT`` is the output type: any type with ``boxes``, ``labels`` and ``scores``
    (the ``ObjectDetectionTarget`` protocol), such as ``DetectionTarget`` or a wrapper's
    own dataclass with extra fields.

    Outputs are checked against MAITE 0.10's rules: ``(N, 4)`` boxes, ``N`` labels,
    ``N`` scores or ``(N, C)`` per-class scores, every score at most 1, and each
    row of per-class scores summing to at most 1. Boxes and scores must also be
    finite, and with ``index2label`` in the metadata, per-class scores need at
    least one column per label (more are allowed, for example a background class).
    Columns beyond the labels have no declared meaning, and the base does not say
    which column belongs to which key. ``ByteTrackMOTModel`` reads each label as a
    column index into per-class scores.
    A detector needs ``index2label`` (see ``ModelBase.requires_index2label``).
    """

    requires_index2label: ClassVar[bool] = True

    def _validate_output(self, output: DetectionT) -> None:
        boxes = np.asarray(output.boxes)
        labels = np.asarray(output.labels)
        scores = np.asarray(output.scores)
        name = type(self).__name__
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"{name} boxes must have shape (n_boxes, 4), got {boxes.shape}.")
        n_boxes = boxes.shape[0]
        if labels.shape != (n_boxes,):
            raise ValueError(f"{name} labels must have shape ({n_boxes},), got {labels.shape}.")
        if scores.ndim not in (1, 2) or scores.shape[0] != n_boxes:
            raise ValueError(
                f"{name} scores must have shape ({n_boxes},) or ({n_boxes}, n_classes), got {scores.shape}."
            )
        # NaN fails every comparison, so the range checks below would let it through.
        if not np.all(np.isfinite(boxes)):
            raise ValueError(f"{name} boxes must be finite.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} scores must be finite.")
        if scores.size and scores.max() > 1 + SCORE_TOLERANCE:
            raise ValueError(f"{name} scores must be at most 1.")
        if scores.ndim == 2 and np.any(scores.sum(axis=1) > 1 + SCORE_TOLERANCE):
            raise ValueError(f"{name} per-class scores must sum to at most 1 for each box.")
        # None only if a subclass opts out with requires_index2label = False.
        index2label = self.metadata.get("index2label")
        # At least, not exactly: CheckMAITE allows a detector head with more classes than
        # index2label, such as a background slot, and does not define how columns map to keys.
        if scores.ndim == 2 and index2label is not None and scores.shape[1] < len(index2label):
            raise ValueError(
                f"{name} returned {scores.shape[1]} per-class scores, "
                f"but index2label has {len(index2label)} classes."
            )


class TrackerBase(ModelBase[VideoStream, "MOTTarget"], Generic[StateT]):
    """Base for trackers that turn an object detector into a multi-object tracker.

    For each video stream, a fresh state comes from ``_new_state``. Frames go to
    the detector ``detector_batch_size`` at a time, then ``_update`` runs once per
    frame, in order. The id defaults to ``f"{tracker_name}_{detector id}_{hash}"``,
    where the hash covers ``_id_fields()``, and the label map is copied from the
    detector. ``__init__`` builds the id, so set the attributes ``_id_fields()``
    reads before calling it, and don't change them afterwards.

    Each frame's tracks are checked as soon as ``_update`` returns them, against
    MAITE 0.10's rules: ``(N, 4)`` boxes, and ``N`` labels, scores and track ids.
    Boxes and scores must also be finite.
    """

    tracker_name: ClassVar[str]

    def __init__(
        self,
        *,
        detector: ObjectDetectionModel,
        detector_batch_size: int = 1,
        model_id: str | None = None,
    ) -> None:
        """Check and store the detector, then set the metadata."""
        if isinstance(detector_batch_size, bool) or not isinstance(detector_batch_size, int) or detector_batch_size < 1:
            raise ValueError("detector_batch_size must be at least 1.")
        detector_metadata = getattr(detector, "metadata", None)
        if not isinstance(detector_metadata, Mapping) or not isinstance(detector_metadata.get("id"), str):
            raise TypeError("detector must expose MAITE model metadata containing a string 'id'.")
        self.detector = detector
        self._detector_batch_size = detector_batch_size
        index2label = detector_metadata.get("index2label")
        if index2label is not None:
            index2label = dict(_check_index2label(index2label, source="detector metadata 'index2label'"))
        self._init_metadata(model_id=model_id, index2label=index2label)

    @property
    def detector_batch_size(self) -> int:
        """Frames sent to the detector at once. Read-only, since it may be part of the id."""
        return self._detector_batch_size

    @abstractmethod
    def _new_state(self) -> StateT:
        """Return fresh tracker state for one stream."""

    @abstractmethod
    def _update(self, state: StateT, detections: ObjectDetectionTarget, frame: VideoFrame) -> MOTFrameTarget:
        """Update ``state`` with one frame's detections and return that frame's tracks."""

    def _default_model_id(self) -> str:
        return f"{self.tracker_name}_{self.detector.metadata['id']}_{self._id_hash()}"

    def _validate_frame(self, frame: MOTFrameTarget, where: str) -> None:
        """Check one frame's tracks. ``where`` names the stream and frame in errors."""
        name = type(self).__name__
        boxes = np.asarray(frame.boxes)
        if boxes.size == 0 and all(np.asarray(a).size == 0 for a in (frame.labels, frame.scores, frame.track_ids)):
            return
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"{name} {where}: boxes must have shape (n_boxes, 4), got {boxes.shape}.")
        n_boxes = boxes.shape[0]
        for field in ("labels", "scores", "track_ids"):
            values = np.asarray(getattr(frame, field))
            if values.ndim == 0 or values.shape[0] != n_boxes:
                raise ValueError(
                    f"{name} {where}: {field} must have {n_boxes} entries, one per box, got shape {values.shape}."
                )
        # NaN fails every comparison, so check finiteness explicitly.
        if not np.all(np.isfinite(boxes)):
            raise ValueError(f"{name} {where}: boxes must be finite.")
        if not np.all(np.isfinite(np.asarray(frame.scores))):
            raise ValueError(f"{name} {where}: scores must be finite.")

    def _predict(self, items: list[VideoStream]) -> list[MOTTarget]:
        from modelmaite.multiobject_tracking.types import MOTTarget

        return [MOTTarget(frame_tracks=self._track(stream, index)) for index, stream in enumerate(items)]

    def _track(self, stream: VideoStream, stream_index: int) -> list[MOTFrameTarget]:
        state = self._new_state()
        frame_tracks: list[MOTFrameTarget] = []
        for frames in _batched(stream, self.detector_batch_size):
            predictions = list(self.detector([frame.pixels for frame in frames]))
            if len(predictions) != len(frames):
                raise ValueError(
                    f"Object detector returned {len(predictions)} predictions for a batch of {len(frames)} frames."
                )
            for frame, prediction in zip(frames, predictions, strict=True):
                tracks = self._update(state, prediction, frame)
                self._validate_frame(tracks, f"stream {stream_index}, frame {len(frame_tracks)}")
                frame_tracks.append(tracks)
        return frame_tracks


def normalize_device(device: object | None) -> str | int | None:
    """Normalise a ``device`` argument the same way for every wrapper.

    ``None`` means "let the wrapper choose". An ``int`` is passed through as a device
    index. Anything else, such as ``"cuda:1"`` or a ``torch.device``, becomes its
    lower-case string form.
    """
    if device is None:
        return None
    if isinstance(device, bool):
        raise TypeError("device must be a string, an integer index or None.")
    if isinstance(device, int):
        return device
    text = str(device).lower()
    if not text:
        raise ValueError("device must not be empty. Use None to let the wrapper choose.")
    return text


def _check_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be None or a positive integer, got {batch_size!r}.")
    return batch_size


def _check_index2label(index2label: object, *, source: str = "index2label") -> dict[int, str]:
    if not isinstance(index2label, Mapping) or not all(
        isinstance(index, int) and not isinstance(index, bool) and isinstance(label, str)
        for index, label in index2label.items()
    ):
        raise TypeError(f"{source} must map integer indices to string labels.")
    # A dict is kept as the same object, so a wrapper's index2label attribute stays in sync with its metadata.
    return index2label if isinstance(index2label, dict) else dict(index2label)


def _describe_channels(channels: tuple[int, ...]) -> str:
    if len(channels) == 1:
        return f"exactly {channels[0]} channels"
    return f"{', '.join(str(count) for count in channels[:-1])} or {channels[-1]} channels"


def _batched(items: Iterable[_T], size: int | None) -> Iterator[list[_T]]:
    """Yield lists of up to ``size`` items. ``None`` yields everything in one list."""
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if size is not None and len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
