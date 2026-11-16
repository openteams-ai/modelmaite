"""Types shared by the model wrappers.

They mirror the MAITE 0.10 protocol types, so type checkers accept the wrappers as
MAITE models while modelmaite itself never imports maite.
"""

from typing import TypeAlias

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
