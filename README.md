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

`modelmaite.multiobject_tracking.ByteTrackMOTModel` combines any MAITE-compatible
object-detection model with ByteTrack to produce MAITE-compatible multi-object tracks.

Install the optional torchvision dependencies with uv:

```bash
uv add "modelmaite[torchvision]"
```

Install the optional multi-object-tracking dependencies with uv:

```bash
uv add "modelmaite[mot]"
```

On Python 3.10–3.12, the `mot` and `visdrone` extras require incompatible NumPy
versions and must be installed in separate environments. This repository also
marks them as mutually exclusive so uv can produce its universal lockfile.

The detector is supplied separately, so the MOT wrapper works with any MAITE-compatible
object-detection model. For example:

```python
from modelmaite import ByteTrackMOTModel, TorchvisionODModel

detector = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
tracker = ByteTrackMOTModel(detector=detector)
```

Install the optional VisDrone dependencies with uv:

```bash
uv add "modelmaite[visdrone]"
```

Install the optional ONNX Runtime dependencies with uv:

```bash
uv add "modelmaite[onnx]"
```
