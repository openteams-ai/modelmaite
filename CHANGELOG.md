# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
