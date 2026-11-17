"""A small image classifier built on modelmaite's ImageClassifierBase."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from modelmaite.base import ImageClassifierBase


class BrightnessClassifier(ImageClassifierBase[NDArray[np.float32], NDArray[np.float32]]):
    """Score each image as "dark" or "bright" from its mean pixel value."""

    # Grayscale or RGB. Other channel counts are rejected before preprocess runs.
    input_channels = (1, 3)

    def __init__(self, *, sharpness: float = 10.0, model_id: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(batch_size=batch_size)
        # Load or build the model here. This one is a single number.
        self.sharpness = sharpness
        self._init_metadata(model_id=model_id, index2label={0: "dark", 1: "bright"})

    @property
    def name(self) -> str:
        return "brightness"

    @override
    def preprocess(self, images: list[ArrayLike]) -> NDArray[np.float32]:
        # One mean per image, scaled to [0, 1]. uint8 images are divided by 255.
        means = [np.asarray(image).mean() / (255.0 if np.asarray(image).dtype == np.uint8 else 1.0) for image in images]
        return np.asarray(means, dtype=np.float32)

    @override
    def forward(self, prepared: NDArray[np.float32]) -> NDArray[np.float32]:
        # The "model": a logistic curve centred on mid-grey.
        return (1.0 / (1.0 + np.exp(-self.sharpness * (prepared - 0.5)))).astype(np.float32)

    @override
    def postprocess(self, raw: NDArray[np.float32], prepared: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        # One score per label in index2label, in key order.
        return [np.array([1.0 - bright, bright], dtype=np.float32) for bright in raw]

    @override
    def _id_fields(self) -> dict[str, object]:
        # Every setting that changes the outputs, so differently configured models get different ids.
        return {"sharpness": self.sharpness}
