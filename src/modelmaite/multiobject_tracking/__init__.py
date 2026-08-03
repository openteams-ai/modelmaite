"""MAITE-compliant multi-object-tracking model wrappers."""

from modelmaite.multiobject_tracking.models import ByteTrackConfig, ByteTrackMOTModel
from modelmaite.multiobject_tracking.types import MOTFrameTarget, MOTTarget

__all__ = ["ByteTrackConfig", "ByteTrackMOTModel", "MOTFrameTarget", "MOTTarget"]
