"""MAITE-compliant object-detection model wrappers."""

from modelmaite.object_detection.models import ModelSpecification, OnnxODModel, load_models
from modelmaite.object_detection.types import DetectionTarget

__all__ = ["DetectionTarget", "ModelSpecification", "OnnxODModel", "load_models"]
