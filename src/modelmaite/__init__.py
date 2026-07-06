"""modelmaite: model utilities built on the maite protocols."""

from modelmaite.image_classification import OnnxICModel, TorchvisionICModel
from modelmaite.object_detection import DetectionTarget, OnnxODModel

__all__ = ["DetectionTarget", "OnnxICModel", "OnnxODModel", "TorchvisionICModel"]
