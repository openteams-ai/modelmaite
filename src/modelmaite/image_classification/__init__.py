"""MAITE-compliant image-classification model wrappers."""

from modelmaite.image_classification.models import ModelSpecification, OnnxICModel, TorchvisionICModel, load_models

__all__ = ["ModelSpecification", "OnnxICModel", "TorchvisionICModel", "load_models"]
