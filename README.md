# modelmaite - MAITE-compliant model wrappers

`modelmaite` is a Python package providing model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

**THIS PACKAGE IS CURRENTLY UNDER CONSTRUCTION**

## Model wrappers

`modelmaite.image_classification.TorchvisionICModel` wraps torchvision image-classification
models as MAITE-compatible image-classification models.

`modelmaite.image_classification.OnnxICModel` wraps JATIC_ONNX v1 image-classification
models as MAITE-compatible image-classification models.

`modelmaite.object_detection.TorchvisionODModel` wraps torchvision object-detection
models as MAITE-compatible object-detection models.

`modelmaite.object_detection.VisdroneODModel` wraps Kitware CenterNet VisDrone
models as MAITE-compatible object-detection models.

`modelmaite.object_detection.OnnxODModel` wraps JATIC_ONNX v1 object-detection
models as MAITE-compatible object-detection models.

Install the optional torchvision dependencies with Poetry:

```bash
poetry add "modelmaite[torchvision]"
```

Install the optional VisDrone dependencies with Poetry:

```bash
poetry add "modelmaite[visdrone]"
```

Install the optional ONNX Runtime dependencies with Poetry:

```bash
poetry add "modelmaite[onnx]"
```
