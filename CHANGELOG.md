# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- JATIC_ONNX v1 object-detection wrapper (`modelmaite.object_detection.OnnxODModel`) that exposes ONNX Runtime models as MAITE-compatible object-detection models.
- ONNX metadata/model-output validation, provider selection, image normalization/channel conversion/resizing utilities, and a MAITE-compatible `DetectionTarget` type.
- Object-detection `load_models` dispatch for JATIC_ONNX model specifications.
- Optional `onnx` and `onnx-cuda` extras, plus deterministic ONNX fixture coverage for wrapper inference.

### Changed

- Project metadata now describes model wrappers, uses the OpenTeams author, removes dataset-specific keywords, declares `numpy` as the runtime dependency, and keeps `maite` in test dependencies only.
- Ruff configuration drops stale rule references that are no longer used by the configured Ruff version.

### Documentation

- README and docs now document ONNX wrapper usage and use Poetry for install commands.
