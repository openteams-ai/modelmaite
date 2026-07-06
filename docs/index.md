# modelmaite

`modelmaite` provides model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

## Installation

```bash
poetry add modelmaite
```

Install torchvision support with Poetry:

```bash
poetry add "modelmaite[torchvision]"
```

Install VisDrone support with Poetry:

```bash
poetry add "modelmaite[visdrone]"
```

Install ONNX support with Poetry:

```bash
poetry add "modelmaite[onnx]"
```

## Usage

Use `modelmaite.image_classification.TorchvisionICModel` to wrap torchvision
image-classification models as MAITE-compatible image-classification models.

Use `modelmaite.image_classification.OnnxICModel` to wrap JATIC_ONNX v1
image-classification models as MAITE-compatible image-classification models.

Use `modelmaite.object_detection.TorchvisionODModel` to wrap torchvision
object-detection models as MAITE-compatible object-detection models.

Use `modelmaite.object_detection.VisdroneODModel` to wrap Kitware CenterNet
VisDrone models as MAITE-compatible object-detection models.

Use `modelmaite.object_detection.OnnxODModel` to wrap JATIC_ONNX v1
object-detection models as MAITE-compatible object-detection models.
