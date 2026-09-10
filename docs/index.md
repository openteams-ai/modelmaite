# modelmaite

`modelmaite` provides model utilities built on the
[maite](https://mit-ll-ai-technology.github.io/maite/) protocols.

## Installation

```bash
uv add modelmaite
```

The core install contains no model runtime; each wrapper needs an optional
extra. See [Setup](setup.md) for the `torchvision`, `visdrone`, `onnx`,
`onnx-cuda`, and `mot` extras, their system requirements, and which
combinations can share an environment.

## Usage

See [Usage](usage.md) to load and call the torchvision, VisDrone, ONNX, and
ByteTrack wrappers, including `load_models` for image classification and
object detection.

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
