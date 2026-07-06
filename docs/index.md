# modelmaite

`modelmaite` provides model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

## Installation

```bash
poetry add modelmaite
```

Install ONNX support with Poetry:

```bash
poetry add "modelmaite[onnx]"
```

## Usage

Use `modelmaite.object_detection.OnnxODModel` to wrap JATIC_ONNX v1
object-detection models as MAITE-compatible object-detection models.
