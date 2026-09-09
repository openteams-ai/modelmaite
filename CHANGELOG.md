# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A ByteTrack multi-object-tracking wrapper that composes with any MAITE-compatible object-detection model, available through the optional `mot` extra. On Python 3.10–3.12, the `mot` and `visdrone` extras require incompatible NumPy versions and must be installed separately.
- `ByteTrackMOTModel` runs on `datamaite`'s MOTChallenge, TAO, and VisDrone-video readers through `maite.tasks.predict`, with no reader of its own; documented in the README and docs, and pinned by tests. Those datasets store each video as a folder of images, so this needs `datamaite>=0.5.0`, the first release with the image-sequence MAITE stream.

### Changed

- Python 3.13 and 3.14 are now supported: `requires-python` widened to `>=3.10,<3.15`. NumPy is uncapped with per-Python floors published in wheel metadata (`>=1.24` below 3.13, `>=2.1` on 3.13, `>=2.3.2` on 3.14+ — the first releases with cp313/cp314 wheels); the SMQTK packages behind the `visdrone` extra now declare NumPy-2 compatibility on Python 3.13+, resolving the reason for the previous `<2` pin. `onnxruntime`/`onnxruntime-gpu` widened to `>=1.19,<1.29` with published `python_version` splits at 3.11 and 3.14 (1.19 is the first NumPy-2-ABI build; 1.24 drops CPython 3.10 wheels while still declaring `requires-python >=3.10`; 1.25 is the first with Python 3.14 wheels), so metadata-driven resolvers such as Poetry cannot select an uninstallable version on 3.10 or 3.14. `onnx` 1.19.0 is excluded (unpinned `ml_dtypes` next to a `float4_e2m1fn` usage that needs ml-dtypes 0.5), as is 1.22.0 on CPython 3.13 only (no cp313 wheel). The `torch` floor is raised to 2.4 (first NumPy-2 build) with `torchvision>=0.19` to match.
- The `maite` test dependency is now declared exactly once, as `>=0.9.5,<0.10`. The upper bound matches the range the rest of the JATIC family declares, so co-installed environments resolve without conflict; the floor is 0.9.5 rather than the family's 0.9.2 because `maite.protocols.multiobject_tracking` first ships in 0.9.4 and the MOT tests import it, and 0.9.5 is the version checkmaite pins. Everything else derives from that single declaration: `uv.lock` resolves the exact version, and the `test-wheel` job installs the `test` group with `uv pip install --group test` rather than restating the specifier, which is what previously let the locked jobs and the wheel job drift onto different maite versions. maite 0.10 exports `ObjectDetectionTarget` as `Annotated[..., Is[...]]`, which `isinstance` rejects, so the test helper unwraps the alias only when there is something to unwrap and stays correct on either side of the bound. `maite` remains test-only and absent from published metadata.
- Project tooling migrated from Poetry to uv: hatchling build backend, PEP 735 dependency groups, and `uv.lock`; CI, publish, pre-commit, and documented install commands updated accordingly. Published dependency constraints are unchanged.

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
