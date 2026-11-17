# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `modelmaite.base` holds the base classes every wrapper is now built on: `ModelBase`, `ImageModelBase`, `ImageClassifierBase`, `ObjectDetectorBase` and `TrackerBase`. They share input checks, batching, device handling, metadata and output checks across wrappers. You can use them to wrap your own model. A classifier or detector on the base needs `index2label`, as CheckMAITE's conventions require of models. A wrapper on the base needs `model_id`, or an `_id_fields()` method listing the settings that change its outputs, which are hashed into the default id; a tracker's default id also includes its detector's id. The base classes are new and their hook names may still change before 1.0.
- `ObjectDetectorBase` takes its output type as its first type parameter, `ObjectDetectorBase[TargetT, PreparedT, RawT]`. `TargetT` is any type with `boxes`, `labels` and `scores`, so a detector can return its own type with extra fields. The built-in detectors return `DetectionTarget` as before.
- `_init_metadata(..., extra=...)` adds other keys to a wrapper's `metadata`. It cannot set `id`, `index2label` or another key `ModelMetadata` defines.
- `batch_size` on `TorchvisionICModel` and `TorchvisionODModel`. It defaults to `None`, which runs the whole batch at once, as before.

### Changed

- ModelMAITE is now tested against MAITE 0.10.3. The `maite` test dependency is `maite[experimental]>=0.10.3,<0.11`, matching CheckMAITE; the `experimental` extra brings `beartype` for `maite.spotcheck`. The test group also requires `typing-extensions>=4.13`, which maite 0.10 needs but does not declare. `maite` is still test-only and not in published metadata.
- The `typing-extensions` runtime floor is raised from 4.5 to 4.9, the first release with `ReadOnly`.
- Every wrapper's `metadata` is now typed as `ModelMetadata`, a TypedDict with the same fields as MAITE's (`id`, and optional `index2label`). Type checkers now accept each wrapper as a MAITE `Model`; before, none passed because `metadata` was typed as a plain `dict`. At runtime `metadata` is the same dict as before.
- `ByteTrackMOTModel` typing follows MAITE 0.10: video frames may have `time_s` and `pts` set to `None`, and `detector` accepts any MAITE object-detection model. Only the type hints changed.
- The `DetectionTarget` docstring states MAITE 0.10's score rule: each score is at most 1, and each row of `(n_boxes, n_classes)` scores sums to at most 1.
- The wrappers are now built on `modelmaite.base`. Their outputs are unchanged; every default `metadata["id"]` changes (see below). The entries below are what users can notice.
- **Breaking:** `ByteTrackMOTModel`'s default `metadata["id"]` is now `bytetrack_{detector id}_{hash}`, where the hash covers `tracker_kwargs` and `detector_batch_size`, so trackers with different settings no longer share an id. Cached CheckMAITE results for ByteTrack models are not reused.
- **Breaking:** `TorchvisionODModel`'s default `metadata["id"]` now covers `batch_size`. torchvision pads every image in a batch to the batch's largest size, so batch composition changes the detections, and CheckMAITE must not reuse results computed with another batch size.
- **Breaking:** the default ids of `OnnxICModel` and `OnnxODModel` cover `image_height` and `image_width` when they are given, and those of `TorchvisionICModel` and `TorchvisionODModel` cover the keyword arguments forwarded to the torchvision constructor, since both change the outputs. Models built without them keep their ids. A constructor argument with no canonical form, such as a function, raises `TypeError` asking for `model_id`.
- **Breaking:** an empty or whitespace-only `model_id` raises `ValueError`, since callers such as CheckMAITE key cached results by id and every model with an empty id would share them. `ByteTrackMOTModel` no longer falls back to the detector's id for `model_id=""`.
- **Breaking:** default ids are built from a canonical form of the settings, and their hash is 16 hex characters instead of 8, so every hashed default id changes. An object setting, such as ByteTrack's `iou`, is hashed by its class and attributes rather than by `str()`, which could include a memory address: equal settings now give the same id. NumPy scalars are hashed as the Python values they hold. A setting with no canonical form raises `TypeError` asking for `model_id`.
- **Breaking:** `batch_size` on the image wrappers, and `detector_batch_size` and `tracker_kwargs` on `ByteTrackMOTModel`, are read-only once the model is built, because they are part of the default id. Assigning them raises `AttributeError`; `tracker_kwargs` is a read-only copy. Build a new model to change them.
- **Breaking:** a classifier's `index2label` keys must be exactly `0..n-1`, since score `i` is the score of label `i`. Other keys, such as `{1: "cat", 2: "dog"}` in an ONNX or torchvision config, raise `ValueError` when the model is loaded. Detectors are unchanged.
- **An empty batch now returns an empty list.** The image wrappers used to raise `ValueError` ("Input batch must contain at least one image."). `ByteTrackMOTModel` already returned an empty list.
- `batch_size` means the same thing on every image wrapper: the most images sent to the model at once. A larger batch is split into several calls and the outputs are joined in order. On `OnnxICModel` and `OnnxODModel`, a batch larger than `batch_size`, or than a fixed metadata `batchSize`, used to raise `ValueError`; it is now split. If the ONNX graph's `image` input has a fixed batch dimension, every run uses exactly that many images: a batch that is not a multiple of it raises `ValueError` before the model runs, a `batch_size` smaller than it, or a metadata `batchSize` that differs from it, raises `ValueError` when the model is loaded. A larger `batch_size` is accepted, since it is a maximum. `VisdroneODModel` keeps its default of 3.
- Images are checked before the model runs, with the same messages in every wrapper: CHW order, the channel counts the wrapper supports, and an integer or floating-point dtype. `TorchvisionICModel` and `VisdroneODModel` now reject images without exactly 3 channels up front; before, they failed later inside torch, cv2 or smqtk. The ONNX wrappers reject 2-channel images up front. Images of different sizes only need to match within one model call, so `batch_size=1` (or an ONNX graph fixed at one image) accepts mixed sizes; before, the whole batch had to match.
- Every image is also checked, before any of the batch runs, for a nonzero height and width and, for float images, finite values. A wrapper can set `input_range`; `VisdroneODModel` sets `(0, 255)`, since smqtk divides by 255 itself. The error names the image.
- Outputs are checked against the MAITE protocol before they are returned: one output per input, classifier scores finite and in `[0, 1]` with one score per label, detections with MAITE's box, label and score rules (with `index2label`, per-class scores need at least one column per label), and each frame of `ByteTrackMOTModel`'s tracks with MAITE's tracking rules. Boxes and scores must be finite. A wrapper that breaks them raises `ValueError`, naming the image or frame. Each chunk of images, and each tracker frame, is checked before the next one runs.
- `device` is handled the same way in every wrapper. An unsupported device now raises `ValueError` in the torch-based wrappers too (it was torch's `RuntimeError`). The ONNX wrappers now raise `ValueError` for a device index other than 0, such as `cuda:1`; before, the index was silently dropped and GPU 0 was used. Device names are no longer case-sensitive.
- **Breaking:** `TorchvisionICModel.preprocess` and `TorchvisionODModel.preprocess` are now the base-class preprocessing step, which takes a list of images. The torchvision transforms they used to hold moved to a new `transforms` attribute; call `model.transforms(tensor)` instead of `model.preprocess(tensor)`.
- `TorchvisionICModel` applies its transforms to each image before stacking, so a batch can mix image sizes at any `batch_size`, and large images are resized before a full-resolution batch is built (about half the peak memory for 4K images). The preprocessed batch is unchanged.
- `batch_size` is no longer forwarded to the torchvision model constructor by `TorchvisionICModel` and `TorchvisionODModel`. Torchvision models do not accept it.

### Documentation

- A new "Write a wrapper for your own model" page explains the base classes, with an example classifier that runs in the test suite and is type-checked in CI.
- The usage page describes the behaviour all wrappers now share.
- The wrapper guide says that a model id is not a record of a run, and that MAITE predictions are not all a model produces.

### Added

- Every model wrapper is advertised as a MAITE entry point under the protocol it implements, for example `maite.protocols.object_detection.Model`, so MAITE tooling can discover it.

### Tests

- Every wrapper is checked with `maite.spotcheck`, including a rejected 2-D image. `spotcheck` is used in tests only.
- The `pyright-internal` job runs MAITE's static entry point verifier, which type-checks every advertised wrapper against its MAITE protocol. A test checks that every exported model is advertised. Pyright also checks `tests/typing/`, which checks that `ByteTrackMOTModel` accepts MAITE detectors.
- Golden-output tests pin the outputs, `metadata["id"]`, `index2label` and `name` of every wrapper on fixed inputs. A new `test-golden` job runs them with ONNX, MOT and CPU torchvision. The VisDrone golden needs downloaded weights and runs only with `MODELMAITE_RUN_VISDRONE_INTEGRATION=1`.

## [0.2.0] - 2026-10-07

### Added

- CDAO funding acknowledgment (#18, DR-2.2-H-2), laid out as in CheckMAITE: an "Authors and acknowledgment" README section holds the program's standard text with the contract number, and the docs home page gains an "About the JATIC program and CDAO" section that includes it from the README through `pymdownx.snippets`.
- A ByteTrack multi-object-tracking wrapper that composes with any MAITE-compatible object-detection model, available through the optional `mot` extra. On Python 3.10–3.12, the `mot` and `visdrone` extras require incompatible NumPy versions and must be installed separately.
- `ByteTrackMOTModel` runs on `datamaite`'s MOTChallenge, TAO, and VisDrone-video readers through `maite.tasks.predict`, with no reader of its own; documented in the README and docs, and pinned by tests. Those datasets store each video as a folder of images, so this needs `datamaite>=0.5.0`, the first release with the image-sequence MAITE stream.

### Changed

- Python 3.13 and 3.14 are now supported: `requires-python` widened to `>=3.10,<3.15`. NumPy is uncapped with per-Python floors published in wheel metadata (`>=1.24` below 3.13, `>=2.1` on 3.13, `>=2.3.2` on 3.14+ — the first releases with cp313/cp314 wheels); the SMQTK packages behind the `visdrone` extra now declare NumPy-2 compatibility on Python 3.13+, resolving the reason for the previous `<2` pin. `onnxruntime`/`onnxruntime-gpu` widened to `>=1.19,<1.29` with published `python_version` splits at 3.11 and 3.14 (1.19 is the first NumPy-2-ABI build; 1.24 drops CPython 3.10 wheels while still declaring `requires-python >=3.10`; 1.25 is the first with Python 3.14 wheels), so metadata-driven resolvers such as Poetry cannot select an uninstallable version on 3.10 or 3.14. `onnx` 1.19.0 is excluded (unpinned `ml_dtypes` next to a `float4_e2m1fn` usage that needs ml-dtypes 0.5), as is 1.22.0 on CPython 3.13 only (no cp313 wheel). The `torch` floor is raised to 2.4 (first NumPy-2 build) with `torchvision>=0.19` to match.
- The `maite` test dependency is now declared exactly once, as `>=0.9.5,<0.10`. The upper bound matches the range the rest of the JATIC family declares, so co-installed environments resolve without conflict; the floor is 0.9.5 rather than the family's 0.9.2 because `maite.protocols.multiobject_tracking` first ships in 0.9.4 and the MOT tests import it, and 0.9.5 is the version CheckMAITE pins. Everything else derives from that single declaration: `uv.lock` resolves the exact version, and the `test-wheel` job installs the `test` group with `uv pip install --group test` rather than restating the specifier, which is what previously let the locked jobs and the wheel job drift onto different maite versions. maite 0.10 exports `ObjectDetectionTarget` as `Annotated[..., Is[...]]`, which `isinstance` rejects, so the test helper unwraps the alias only when there is something to unwrap and stays correct on either side of the bound. `maite` remains test-only and absent from published metadata.
- Project tooling migrated from Poetry to uv: hatchling build backend, PEP 735 dependency groups, and `uv.lock`; CI, publish, pre-commit, and documented install commands updated accordingly. Published dependency constraints are unchanged.

### Documentation

- An end-user setup page documents every optional extra (`onnx`, `onnx-cuda`, `torchvision`, `visdrone`, `mot`), including the wrappers each one enables, VisDrone weight downloads, GPU and Linux system requirements, and which extras can share an environment. `onnx-cuda` and `mot` were previously undocumented in the published docs.
- A usage page documents how to load and call each wrapper (torchvision, VisDrone, ONNX, ByteTrack), including `load_models` for image classification and object detection. README and the docs home page point at Setup for extras and Usage for load/call.

## [0.1.0] - 2026-07-07

### Added

- Torchvision image-classification wrapper (`modelmaite.image_classification.TorchvisionICModel`) that exposes torchvision models as MAITE-compatible image-classification models.
- Torchvision object-detection wrapper (`modelmaite.object_detection.TorchvisionODModel`) that exposes torchvision models as MAITE-compatible object-detection models.
- VisDrone object-detection wrapper (`modelmaite.object_detection.VisdroneODModel`) that exposes Kitware CenterNet VisDrone models as MAITE-compatible object-detection models.
- JATIC_ONNX v1 image-classification wrapper (`modelmaite.image_classification.OnnxICModel`) and object-detection wrapper (`modelmaite.object_detection.OnnxODModel`) that expose ONNX Runtime models as MAITE-compatible models.
- ONNX metadata/model-output validation, provider selection, image normalization/channel conversion/resizing utilities, and a MAITE-compatible `DetectionTarget` type.
- Image-classification and object-detection `load_models` dispatch for JATIC_ONNX, torchvision, and VisDrone model specifications.
- Optional `onnx`, `onnx-cuda`, `torchvision`, and `visdrone` extras, plus deterministic ONNX/fake-torchvision/fake-VisDrone coverage for wrapper inference.

### Changed

- Project metadata now describes model wrappers, uses the OpenTeams author, removes dataset-specific keywords, declares `numpy` and `typing-extensions` as runtime dependencies, and keeps `maite` in test dependencies only.
- NumPy is constrained to `<2` while VisDrone support depends on SMQTK packages that do not yet allow NumPy 2.x.
- Ruff configuration drops stale rule references that are no longer used by the configured Ruff version.

### Documentation

- README and docs now document torchvision, VisDrone, and ONNX wrapper usage for image classification and object detection, and use Poetry for install commands.
