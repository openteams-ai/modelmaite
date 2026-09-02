# modelmaite

`modelmaite` provides model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

## Installation

```bash
uv add modelmaite
```

Install torchvision support with uv:

```bash
uv add "modelmaite[torchvision]"
```

Install VisDrone support with uv:

```bash
uv add "modelmaite[visdrone]"
```

Install ONNX support with uv:

```bash
uv add "modelmaite[onnx]"
```

Install multi-object-tracking support with uv:

```bash
uv add "modelmaite[mot]"
```

On Python 3.10–3.12, the `mot` and `visdrone` extras require incompatible NumPy versions
and must be installed in separate environments. This repository also marks them as
mutually exclusive so uv can produce its universal lockfile.

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

Use `modelmaite.multiobject_tracking.ByteTrackMOTModel` to combine any
MAITE-compatible object-detection model with ByteTrack, producing
MAITE-compatible multi-object tracks. The detector is supplied separately, so
the wrapper works with any of the object-detection wrappers above:

```python
from modelmaite import ByteTrackMOTModel, TorchvisionODModel

detector = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
tracker = ByteTrackMOTModel(detector=detector)
```

## Multi-object tracking with datamaite datasets

[`datamaite`](https://pypi.org/project/datamaite/) reads MOTChallenge, TAO, and
VisDrone-video annotations and presents each video as a MAITE multi-object-tracking
dataset, so `ByteTrackMOTModel` consumes them with no reader of its own.

Install the three packages plus the detector's own extra. `datamaite` 0.5.0 is the first
release whose MAITE view streams image-folder sequences; on 0.4.1 and earlier these
datasets load as zero items:

```bash
uv add "modelmaite[mot,torchvision]" "datamaite>=0.5.0" maite
```

The first run of the example below downloads torchvision's SSDLite weights.

```python
from datamaite import load_mot
from maite.tasks import predict

from modelmaite import ByteTrackMOTModel, TorchvisionODModel

dataset = load_mot("/data/MOT17", dataset_format="motchallenge")
detector = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
predictions, _ = predict(model=ByteTrackMOTModel(detector=detector), dataset=dataset)
```

Use `dataset_format="tao"` or `dataset_format="visdrone_video"` for the other two
readers. `"visdrone"` is the still-image object-detection reader and will not load as a
tracking dataset.

All three formats store a video as a folder of images rather than a single video file,
so they need `datamaite>=0.5.0`, which added the image-sequence MAITE stream.
Decoding frame folders needs only OpenCV, which `modelmaite[mot]` already installs; PyAV
is required for video-file datasets, not these.

### Options worth setting explicitly

**Frame coverage.** `empty_frame_policy` defaults to `"annotated"`, which streams only
the frames carrying annotations. ByteTrack then sees temporal gaps and track quality
drops, so prefer `"all"` where the dataset has an exact frame count:

```python
dataset = load_mot("/data/MOT17", dataset_format="motchallenge").with_mot_options(
    empty_frame_policy="all"
)
```

An exact count comes from `seqinfo.ini`'s `seqLength` (or a frame-file count) for
MOTChallenge, a frame-file count for VisDrone — taken even with the default
`probe_images=False` — and a gap-free frame table for TAO. `probe_images=True` only
fills in frame width and height; it is *not* what enables `"all"`. Without an exact
count, `"all"` logs a warning and falls back to annotated frames.

**Frame timing.** VisDrone annotations carry no frame rate, so the loader defaults to
`fps=0.0` and every frame reports `time_s == 0.0`. Pass `fps=` for real timestamps.
ByteTrack associates detections by IoU, so tracking itself works either way.

**Label spaces do not line up.** A dataset's `index2label` comes from its own categories
and is often sparse; a model's `index2label` is copied from the wrapped detector (the
91-entry COCO map for torchvision). Build the mapping explicitly before scoring
predictions against ground truth — `modelmaite` does not remap labels for you.

**Pair VisDrone data with the torchvision or ONNX detectors, not `VisdroneODModel`.**
The `mot` and `visdrone` extras are mutually exclusive, so ByteTrack and the VisDrone
detector cannot share an environment.
