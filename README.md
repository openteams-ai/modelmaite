# modelmaite - MAITE-compliant model wrappers

`modelmaite` is a Python package providing model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

**THIS PACKAGE IS CURRENTLY UNDER CONSTRUCTION**

## Object detection

`modelmaite.object_detection.OnnxODModel` wraps JATIC_ONNX v1 object-detection
models as MAITE-compatible object-detection models.

Install the optional ONNX Runtime dependencies with Poetry:

```bash
poetry add "modelmaite[onnx]"
```
