# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- ModelMAITE is now tested against MAITE 0.10.3. The `maite` test dependency is `maite[experimental]>=0.10.3,<0.11`, matching CheckMAITE; the `experimental` extra brings `beartype` for `maite.spotcheck`. The test group also requires `typing-extensions>=4.13`, which maite 0.10 needs but does not declare. `maite` is still test-only and not in published metadata.
- The `typing-extensions` runtime floor is raised from 4.5 to 4.9, the first release with `ReadOnly`.
- Every wrapper's `metadata` is now typed as `ModelMetadata`, a TypedDict with the same fields as MAITE's (`id`, and optional `index2label`). Type checkers now accept each wrapper as a MAITE `Model`; before, none passed because `metadata` was typed as a plain `dict`. At runtime `metadata` is the same dict as before.
- `ByteTrackMOTModel` typing follows MAITE 0.10: video frames may have `time_s` and `pts` set to `None`, and `detector` accepts any MAITE object-detection model. Only the type hints changed.
- The `DetectionTarget` docstring states MAITE 0.10's score rule: each score is at most 1, and each row of `(n_boxes, n_classes)` scores sums to at most 1.

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
