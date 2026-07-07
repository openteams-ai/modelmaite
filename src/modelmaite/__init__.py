"""modelmaite: model utilities built on the maite protocols."""

from importlib.metadata import version

from modelmaite.image_classification import OnnxICModel, TorchvisionICModel
from modelmaite.object_detection import DetectionTarget, OnnxODModel, TorchvisionODModel, VisdroneODModel

__version__ = version("modelmaite")

__all__ = [
    "DetectionTarget",
    "OnnxICModel",
    "OnnxODModel",
    "TorchvisionICModel",
    "TorchvisionODModel",
    "VisdroneODModel",
    "__version__",
]
