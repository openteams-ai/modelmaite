"""Every model modelmaite exports is advertised as a MAITE entry point.

``scripts/statically_verify_maite_entrypoints.py`` type-checks each advertised
model against its MAITE protocol, so a model missing here would go unchecked.
"""

from importlib.metadata import entry_points

import modelmaite

PROTOCOL_GROUPS = (
    "maite.protocols.image_classification.Model",
    "maite.protocols.object_detection.Model",
    "maite.protocols.multiobject_tracking.Model",
)


def _advertised() -> dict[str, str]:
    """Map each advertised class name to its entry point group."""
    advertised = {}
    for group in PROTOCOL_GROUPS:
        for ep in entry_points(group=group):
            if ep.dist is not None and ep.dist.name == "modelmaite":
                advertised[ep.attr] = group
    return advertised


def test_every_exported_model_is_advertised():
    exported_models = {name for name in modelmaite.__all__ if name.endswith("Model")}
    assert set(_advertised()) == exported_models


def test_entry_points_load_the_exported_classes():
    for group in PROTOCOL_GROUPS:
        for ep in entry_points(group=group):
            if ep.dist is not None and ep.dist.name == "modelmaite":
                assert ep.load() is getattr(modelmaite, ep.attr)
                assert ep.name == f"modelmaite_{ep.attr}"
