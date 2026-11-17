# Usage

Install the extra that matches the wrapper you want, then load and call it.
See [Setup](setup.md) for extras, system requirements, and which combinations
can share an environment.

Every image wrapper takes a batch of CHW RGB images: `uint8`, or floating-point
already scaled to `[0, 1]`. Images in a batch must have the same CHW shape,
except `TorchvisionODModel`, which accepts per-image sizes.
Image-classification models return one score array per image. Object-detection
models return a `DetectionTarget` per image (`boxes` in `xyxy`, `labels`,
`scores`). Multi-object tracking takes a batch of video streams and returns a
`MOTTarget` per stream.

`device` is optional on the image wrappers. Torchvision and VisDrone pick the
best available torch device; ONNX picks the best ONNX Runtime provider from the
installed extra (`cpu`, `cuda`, or `mps` / CoreML). An unsupported device
raises `ValueError`. ONNX cannot pick a GPU by index: use `cuda` and set
`CUDA_VISIBLE_DEVICES`. `ByteTrackMOTModel` has no `device` argument; it runs
on whatever device the wrapped detector uses.

All wrappers share a base, so they behave the same way in a few more places:

- An empty batch returns an empty list.
- `batch_size` on the image wrappers is the most images sent to the model at
  once; a larger batch is split. By default torchvision runs the whole batch
  at once, VisDrone uses 3, and ONNX uses the metadata `batchSize` if it is
  set. An ONNX graph with a fixed batch dimension runs exactly that many
  images at a time, so a batch must be a multiple of it, and `batch_size`
  must be at least that many. `batch_size` is read-only once the model is
  built (so are ByteTrack's `detector_batch_size` and `tracker_kwargs`): to
  change it, build a new model, which gets its own id.
- Every image is checked before the model runs (CHW order, channel count,
  dtype, nonzero height and width, finite values; VisDrone also needs values
  in `[0, 255]`), and outputs are checked against the MAITE protocol as soon
  as each chunk (or tracker frame) is done, so a bad output stops the call
  before the rest runs. Problems raise `ValueError`, naming the image or frame.

To wrap a model of your own on the same base, see
[Write a wrapper for your own model](write_a_wrapper.md).

## Image classification

### Torchvision

Supported `model_name` values: `alexnet`, `resnext50_32x4d`. Default weights
and class labels come from torchvision. User-supplied weights need both
`weights_path` (a torch state-dict) and `config_path` (JSON with
`index2label`).

```python
import numpy as np
from modelmaite import TorchvisionICModel

model = TorchvisionICModel(model_name="alexnet")
scores = model([np.zeros((3, 224, 224), dtype=np.uint8)])
```

### ONNX

Both files are yours: a `.onnx` model (`weights_path`) and its JATIC_ONNX v1
metadata JSON (`config_path`). Optional `validate_onnx=True` runs
`onnx.checker.check_model` before the session is created.

```python
import numpy as np
from modelmaite import OnnxICModel

model = OnnxICModel(weights_path="model.onnx", config_path="model-metadata.json")
scores = model([np.zeros((3, 224, 224), dtype=np.uint8)])
```

### `load_models`

`modelmaite.image_classification.load_models` dispatches on `model_type`. Extra
keyword arguments are forwarded to every loaded wrapper.

Mixing types in one call needs every matching extra installed together;
`torchvision` plus `onnx` is unverified rather than supported, see
[Setup](setup.md#combining-extras).

```python
from modelmaite.image_classification import load_models

models = load_models(
    {
        "tv": {"model_type": "alexnet"},
        "onnx": {
            "model_type": "jatic_onnx",
            "model_weights_path": "model.onnx",
            "model_config_path": "model-metadata.json",
        },
    }
)
```

## Object detection

### Torchvision

Supported `model_name` values:

- `fasterrcnn_resnet50_fpn`, `fasterrcnn_resnet50_fpn_v2`
- `fasterrcnn_mobilenet_v3_large_fpn`, `fasterrcnn_mobilenet_v3_large_320_fpn`
- `maskrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn_v2`
- `retinanet_resnet50_fpn`, `retinanet_resnet50_fpn_v2`
- `fcos_resnet50_fpn`
- `keypointrcnn_resnet50_fpn`
- `ssd300_vgg16`
- `ssdlite320_mobilenet_v3_large`

User-supplied weights again need both `weights_path` and `config_path`.

```python
import numpy as np
from modelmaite import TorchvisionODModel

model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
predictions = model([np.zeros((3, 320, 320), dtype=np.uint8)])
boxes, labels, scores = predictions[0].boxes, predictions[0].labels, predictions[0].scores
```

### VisDrone

Choose a backbone with `arch`: `res2net50`, `resnet50`, or `resnet18`. Missing
Kitware weights download into `model_pickle_dir` (see [Setup](setup.md)).

```python
import numpy as np
from modelmaite import VisdroneODModel

model = VisdroneODModel(arch="resnet18")
predictions = model([np.zeros((3, 512, 512), dtype=np.uint8)])
```

### ONNX

Same required files as image classification. This wrapper returns every
JATIC_ONNX output slot after coordinate scaling; it does not apply NMS,
confidence thresholding, or background filtering.

```python
import numpy as np
from modelmaite import OnnxODModel

model = OnnxODModel(weights_path="model.onnx", config_path="model-metadata.json")
predictions = model([np.zeros((3, 512, 512), dtype=np.uint8)])
```

### `load_models`

`modelmaite.object_detection.load_models` uses the same `model_type` keys as the
constructors (`ssdlite320_mobilenet_v3_large`, `resnet18`, `jatic_onnx`, …).
VisDrone specs may include `model_pickle_dir`. ONNX specs require
`model_weights_path` and `model_config_path`. Extra keyword arguments are
forwarded to every loaded wrapper, so they must be valid for each selected
type. The VisDrone `arch` is derived from `model_type`, not forwarded:
`num_workers` and `max_dets` are VisDrone-only forwarded arguments, while
`index2label_key` works with the torchvision and ONNX wrappers but not
VisDrone.

Do not mix VisDrone with `mot` or treat `torchvision` plus `visdrone` as a
supported pair; see [Setup](setup.md#combining-extras).

```python
from modelmaite.object_detection import load_models

models = load_models(
    {
        "tv": {"model_type": "ssdlite320_mobilenet_v3_large"},
        "onnx": {
            "model_type": "jatic_onnx",
            "model_weights_path": "model.onnx",
            "model_config_path": "model-metadata.json",
        },
    }
)
```

## Multi-object tracking

`ByteTrackMOTModel` wraps any MAITE-compatible object-detection model. There is
no `load_models` helper; compose the detector yourself. Each `__call__` takes a
batch of video streams. A stream is an iterable of frames with `pixels`,
`time_s`, `pts`, and `frame_index`. A fresh tracker is used per stream, so
track state does not leak between batch elements.

```python
from dataclasses import dataclass

import numpy as np
from modelmaite import ByteTrackMOTModel, TorchvisionODModel

@dataclass
class VideoFrame:
    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int

detector = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
tracker = ByteTrackMOTModel(detector=detector)
frames = [
    VideoFrame(
        pixels=np.zeros((3, 320, 320), dtype=np.uint8),
        time_s=index / 30,
        pts=index,
        frame_index=index,
    )
    for index in range(3)
]
tracks = tracker([frames])
frame_tracks = tracks[0].frame_tracks
```
