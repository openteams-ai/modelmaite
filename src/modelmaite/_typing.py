"""Types shared by the model wrappers.

They mirror the MAITE 0.10 protocol types, so type checkers accept the wrappers as
MAITE models while modelmaite itself never imports maite.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol, TypeAlias

from numpy.typing import ArrayLike
from typing_extensions import NotRequired, ReadOnly, Required, TypedDict  # noqa: UP035

Image: TypeAlias = ArrayLike
"""A single image in CHW order, as in MAITE's ``Image``."""


class ModelMetadata(TypedDict):
    """Model metadata with the same fields and types as MAITE's ``ModelMetadata``.

    Attributes
    ----------
    id
        Identifier for the model.
    index2label
        Optional mapping from integer class indices to label names.
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class ObjectDetectionTarget(Protocol):
    """Detections for one image, as in MAITE's ``ObjectDetectionTarget``."""

    @property
    def boxes(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


class ObjectDetectionModel(Protocol):
    """Any MAITE object-detection model."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def __call__(self, input_batch: Sequence[Image], /) -> Sequence[ObjectDetectionTarget]: ...


class VideoFrame(Protocol):
    """One video frame, as in MAITE's ``VideoFrame``. ``pixels`` is a CHW image."""

    @property
    def pixels(self) -> ArrayLike: ...

    @property
    def time_s(self) -> float | None: ...

    @property
    def pts(self) -> int | None: ...

    @property
    def frame_index(self) -> int: ...


VideoStream: TypeAlias = Iterable[VideoFrame]
"""A video as an iterable of frames, as in MAITE's ``VideoStream``."""
