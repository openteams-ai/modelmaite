"""Run the example wrapper from docs/write_a_wrapper.md, so the page stays correct."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from maite.protocols import image_classification as ic
from maite.spotcheck import SpotcheckError, spotcheck_component

from modelmaite._identity import id_hash

EXAMPLE = Path(__file__).parents[1] / "docs" / "examples" / "brightness_classifier.py"


@pytest.fixture(scope="module")
def brightness_classifier():
    spec = importlib.util.spec_from_file_location("brightness_classifier", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module.BrightnessClassifier
    del sys.modules[spec.name]


def test_example_classifier_scores_images(brightness_classifier):
    model = brightness_classifier()
    dark = np.zeros((3, 8, 8), dtype=np.uint8)
    bright = np.full((1, 4, 6), 1.0, dtype=np.float32)

    scores = model([dark, bright])

    assert model.metadata == {"id": f"brightness_{id_hash(sharpness=10.0)}", "index2label": {0: "dark", 1: "bright"}}
    assert [int(np.argmax(score)) for score in scores] == [0, 1]
    np.testing.assert_allclose([score.sum() for score in scores], [1.0, 1.0], rtol=1e-6)
    assert model([]) == []


def test_example_classifier_uses_the_shared_checks(brightness_classifier):
    model = brightness_classifier(batch_size=2)

    assert len(model([np.zeros((3, 8, 8), dtype=np.uint8)] * 5)) == 5
    with pytest.raises(ValueError, match="1 or 3 channels"):
        model([np.zeros((4, 8, 8), dtype=np.uint8)])
    assert brightness_classifier(sharpness=5.0).metadata["id"] != model.metadata["id"]


def test_example_classifier_is_a_maite_model(brightness_classifier):
    model = spotcheck_component(ic.Model)(brightness_classifier())

    assert isinstance(model, ic.Model)
    assert len(model([np.zeros((3, 8, 8), dtype=np.uint8)])) == 1
    with pytest.raises(SpotcheckError):
        model([np.zeros((8, 8), dtype=np.uint8)])
