# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Torchvision image-classification wrapper (`modelmaite.image_classification.TorchvisionICModel`) that exposes torchvision models as MAITE-compatible image-classification models.
- JATIC_ONNX v1 image-classification wrapper (`modelmaite.image_classification.OnnxICModel`) and object-detection wrapper (`modelmaite.object_detection.OnnxODModel`) that expose ONNX Runtime models as MAITE-compatible models.
- ONNX metadata/model-output validation, provider selection, image normalization/channel conversion/resizing utilities, and a MAITE-compatible `DetectionTarget` type.
- Image-classification and object-detection `load_models` dispatch for JATIC_ONNX and torchvision model specifications.
- Optional `onnx`, `onnx-cuda`, and `torchvision` extras, plus deterministic ONNX/fake-torchvision coverage for wrapper inference.

### Changed

- Project metadata now describes model wrappers, uses the OpenTeams author, removes dataset-specific keywords, declares `numpy` and `typing-extensions` as runtime dependencies, and keeps `maite` in test dependencies only.
- Ruff configuration drops stale rule references that are no longer used by the configured Ruff version.

### Documentation

- README and docs now document torchvision and image-classification/object-detection ONNX wrapper usage and use Poetry for install commands.
