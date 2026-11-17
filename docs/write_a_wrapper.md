# Write a wrapper for your own model

Every modelmaite wrapper is built on the base classes in `modelmaite.base`.
You can use the same classes to wrap your own model. Your wrapper then gets
the same input checks, batching, device handling, metadata and output checks
as the built-in ones, and it is a MAITE model.

!!! note
    The base classes are new. Their hook names may still change before
    modelmaite 1.0.

## Pick a base class

| Task | Base class | Each `__call__` returns |
| --- | --- | --- |
| Image classification | `ImageClassifierBase` | one 1-D score array per image; needs `index2label` |
| Object detection | `ObjectDetectorBase` | one detection target per image, such as `DetectionTarget`; needs `index2label` |
| Multi-object tracking | `TrackerBase` | one `MOTTarget` per video stream |

## What the base does for you

When your model is called on a batch of images, the base:

1. Returns `[]` for an empty batch, without running the model.
2. Checks every image: CHW order, a channel count listed in `input_channels`,
   an integer or floating-point dtype, a nonzero height and width, finite
   values (for float images), values within `input_range` if it is set, and,
   if `same_shape_batches` is `True`, the same shape for the images in each
   `forward` call. Every image is checked before the first `forward` call, and
   a bad image raises `ValueError`, naming it, before your code runs.
3. Splits the batch into chunks of `batch_size` images. `batch_size=None`
   sends the whole batch at once. If `_fixed_batch_size()` returns a number,
   every chunk has exactly that many images, and a batch that is not a
   multiple of it raises `ValueError` before your code runs. `_plan_chunks()`
   can group the images differently; outputs always come back in input order.
4. For each chunk, calls your `preprocess`, then your `forward` inside
   `_inference_context()`, then your `postprocess`.
5. Checks that there is one output per input in each chunk, and checks each output.
   Classifier scores must be finite, in `[0, 1]`, with one score per
   `index2label` entry; score `i` is the score of label `i`, so a
   classifier's `index2label` keys must be exactly `0..n-1` (checked by
   `_init_metadata`). Detections must follow MAITE's rules: `(N, 4)` boxes,
   `N` labels, `N` scores or `(N, C)` per-class scores, every score at most 1,
   and each per-class row summing to at most 1. Boxes and scores must be
   finite. With `index2label`, per-class scores need at least one column per
   label; more are allowed, such as a background class. Extra columns have
   no declared meaning, and `ByteTrackMOTModel` reads each label as a column
   index into the per-class scores.

It also normalises the `device` argument (`None`, an integer index, or a
string such as `"cuda:1"`; a `torch.device` works too) and builds
`metadata`: your `model_id`, or a default id from `name` and a hash of
`_id_fields()`. Without `_id_fields()`, `model_id` is required, so two
differently configured models never share an id by accident.

## What you write

- `__init__`: keyword-only arguments. Call `super().__init__(device=...,
  batch_size=...)` first, load your model, then call
  `self._init_metadata(model_id=..., index2label=...)` last.
- `name`: a short, readable name.
- `preprocess(images)`: turn a list of checked images into model input. Scale
  pixels here: modelmaite passes images through unchanged.
- `forward(prepared)`: run the model, and nothing else. Keeping this to the
  model call lets later tools reuse it, for example to compute gradients.
- `postprocess(raw, prepared)`: turn the raw output into one output per
  image. `prepared` is what `preprocess` returned, so you can carry original
  image sizes through it.

The base classes are generic. `ImageClassifierBase[PreparedT, RawT]` and
`ObjectDetectorBase[TargetT, PreparedT, RawT]` take the types `preprocess`
and `forward` return; a detector also names its output type `TargetT`, which
must have `boxes`, `labels` and `scores` (the `ObjectDetectionTarget`
protocol). Use `DetectionTarget`, or your own type with more fields, such
as a frozen dataclass that also holds masks. Type checkers then see your
type in the outputs, and the wrapper is still a MAITE detector.

Optional class attributes and hooks:

| Name | Default | Override it to |
| --- | --- | --- |
| `input_channels` | `(1, 2, 3, 4)` | accept fewer channel counts |
| `same_shape_batches` | `False` | require one image shape per `forward` call |
| `input_range` | `None` | reject pixel values outside `(low, high)`, such as `(0, 255)` for a model that divides by 255 |
| `positional_index2label` | `True` for classifiers | require `index2label` keys `0..n-1`, for outputs read by position |
| `requires_index2label` | `True` for classifiers and detectors | allow a model without `index2label` (not recommended: CheckMAITE expects it) |
| `_resolve_device(requested)` | keep the request | pick the device; the built-in torch wrappers pick CUDA, then MPS, then CPU |
| `_inference_context()` | no context | return, for example, `torch.no_grad()` |
| `_id_fields()` | `{}` (then `model_id` is required) | return every setting that changes the outputs |
| `_default_model_id()` | `f"{name}_{hash of _id_fields()}"` | use another id scheme |
| `_chunk_size()` | `batch_size` | cap chunks, for example at a memory limit |
| `_fixed_batch_size()` | `None` | run exactly that many images per `forward` call; other batch sizes are rejected before the model runs |
| `_plan_chunks(images, chunk_size)` | consecutive chunks | choose which images share a `forward` call, for example grouping by aspect ratio for a model that pads each batch; return lists of input indices |

Batch composition can change a model's outputs: torchvision's detectors pad
every image in a batch to the batch's largest size. If yours can, put what
decides it (`batch_size`, and any grouping setting) in `_id_fields()`, so
CheckMAITE never reuses results computed with different batches.
A model id is not a record of a run, though: it describes the model, not
how a caller splits its data into batches.

MAITE predictions are not all the information a model produces. Raw logits,
features and similar outputs need their own interface; `forward` is a
natural place to start one.

A few rules keep wrappers safe to share:

- Don't store per-call state on `self`. A model may be called from several
  threads at once.
- Constructor arguments can be any objects the wrapper needs, such as a
  detector, a runtime or a processor. Whatever changes the outputs belongs in
  `_id_fields()`. Values may be JSON values, paths, classes (hashed by name),
  or objects whose attributes are such values (hashed by class and
  attributes, so two equal settings objects give the same id). Anything else
  raises `TypeError` when the default id is built; pass `model_id` instead,
  or put the object's settings or its own id in `_id_fields()`, as trackers
  do with the detector's id.
- Settings in `_id_fields()` must not change after construction, or the id
  would no longer describe the model. Store them read-only (`batch_size` and
  `detector_batch_size` already are); to change one, build a new model.
- Put the framework model on `self.model`.
- `metadata` holds MAITE's `id` and the `index2label` label map, both set by
  `_init_metadata`. To add other keys, pass them as
  `_init_metadata(..., extra={"framework": "onnx"})`; `extra` cannot set `id`,
  `index2label` or any other key `ModelMetadata` defines. Type checkers still
  see `metadata` as `ModelMetadata`, so read extra keys with
  `metadata.get("framework")`.

## Example

This classifier scores images as "dark" or "bright" from their mean pixel
value. It runs in the modelmaite test suite.

```python
--8<-- "docs/examples/brightness_classifier.py"
```

Use it like any other wrapper:

```python
import numpy as np

model = BrightnessClassifier(batch_size=8)
scores = model([np.zeros((3, 32, 32), dtype=np.uint8)])
model.metadata  # {"id": "brightness_<hash>", "index2label": {0: "dark", 1: "bright"}}
```

To check it against the MAITE protocol, install `maite[experimental]` and
wrap it with `spotcheck_component`:

```python
from maite.protocols import image_classification as ic
from maite.spotcheck import spotcheck_component

checked = spotcheck_component(ic.Model)(model)
checked([np.zeros((3, 32, 32), dtype=np.uint8)])
```

## Trackers

`TrackerBase` wraps an object detector instead of a model of its own. Pass
`detector`, `detector_batch_size` and `model_id` to `super().__init__`, set a
`tracker_name` class attribute, and write two hooks:

- `_new_state()`: fresh tracker state, created once per video stream.
- `_update(state, detections, frame)`: update the state with one frame's
  detections and return that frame's `MOTFrameTarget`.

The base sends frames to the detector `detector_batch_size` at a time, calls
`_update` once per frame in order, and builds `metadata` from the detector:
the id defaults to `f"{tracker_name}_{detector id}_{hash of _id_fields()}"`
and `index2label` is copied. The base builds the id in `__init__`, so set
what `_id_fields()` reads before calling `super().__init__`.
`ByteTrackMOTModel` is built this way and hashes its `tracker_kwargs` and `detector_batch_size`. Each frame's
tracks are checked as soon as `_update` returns them, like a detection: `(N, 4)`
boxes, `N` labels, scores and track ids, and finite boxes and scores.
