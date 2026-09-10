# Setup

`modelmaite` installs a small core and pulls each model wrapper's heavy
dependencies through an optional extra. Pick the extra that matches the wrapper
you intend to use.

## Requirements

- Python `>=3.10,<3.15`.
- The core install brings only `numpy` and `typing-extensions`. It contains no
  model runtime, so every wrapper needs an extra.
- `maite` is not a runtime dependency. Install it yourself if you want the
  MAITE protocol types in your own code.

```bash
uv add modelmaite
```

```bash
pip install modelmaite
```

!!! note "Published release vs. this branch"

    This page documents the current source tree. Version 0.1.0 on PyPI supports
    Python 3.10–3.12, requires NumPy `<2`, and offers only the `onnx`,
    `onnx-cuda`, `torchvision`, and `visdrone` extras. The source tree also
    includes the `mot` extra, Python 3.13 and 3.14 support, and uncapped NumPy
    floors.

## Choose extras

### `torchvision`

Wrappers: `TorchvisionICModel` (image classification) and `TorchvisionODModel`
(object detection). Installs `torch` and `torchvision`.

```bash
uv add "modelmaite[torchvision]"
```

```bash
pip install "modelmaite[torchvision]"
```

### `visdrone`

Wrapper: `VisdroneODModel`. Installs `httpx`, `smqtk-detection`, and `torch`.

```bash
uv add "modelmaite[visdrone]"
```

```bash
pip install "modelmaite[visdrone]"
```

The Kitware CenterNet weights are not bundled. The first run downloads them
(60–113 MB depending on the backbone) into `model_pickle_dir`, which defaults to
`~/.cache/modelmaite/visdrone`, and verifies the download against an expected
byte size and SHA-512 digest. To work offline, pre-place the weights file as
`{model_name}.pth` in that directory — `model_name` defaults to
`centernet-{arch}` — and no download is attempted.

### `onnx`

Wrappers: `OnnxICModel` and `OnnxODModel`, running on CPU `onnxruntime`.

```bash
uv add "modelmaite[onnx]"
```

```bash
pip install "modelmaite[onnx]"
```

No models are shipped. You supply both files: a `.onnx` model
(`weights_path`) and its JATIC_ONNX v1 metadata JSON (`config_path`).
Optional `validate_onnx` runs `onnx.checker.check_model` before the session
is created and is `False` by default.

### `onnx-cuda`

The same two wrappers against `onnxruntime-gpu` instead of `onnxruntime`.

```bash
uv add "modelmaite[onnx-cuda]"
```

```bash
pip install "modelmaite[onnx-cuda]"
```

Prefer one of `onnx` or `onnx-cuda` — they install competing ONNX Runtime
builds. This is guidance, not a packaging conflict: nothing stops both from
being installed. The host also needs a CUDA stack (NVIDIA driver plus
CUDA/cuDNN runtime) matching the resolved `onnxruntime-gpu` build.

### `mot`

Wrapper: `ByteTrackMOTModel`. Install the detector's extra alongside it, since
the tracker supplies no detector of its own.

```bash
uv add "modelmaite[mot,torchvision]"
```

```bash
pip install "modelmaite[mot,torchvision]"
```

On Linux, the OpenCV dependency needs system OpenGL — install `libgl1` on
Debian/Ubuntu, or the equivalent package for your distribution.

## Combining extras

`torchvision` with `onnx` or `onnx-cuda`, and `torchvision` with `visdrone`,
have no declared conflict. Treat them as unverified rather than supported:
continuous integration exercises the core install, `onnx`, `mot` with `onnx`,
and a built wheel installed as `[onnx,mot]`.

`mot` and `visdrone` need separate environments. On Python 3.10–3.12 they
resolve incompatible NumPy versions (1.26.4 for `visdrone`, 2.x for `mot`). On
3.13+ `visdrone` resolves NumPy 2, but the combination remains untested and
unsupported.

## After install

See [Usage](usage.md) for how to load each wrapper and run inference.

## Version notes

`onnx` 1.19.0 is excluded, as is 1.22.0 on CPython 3.13. ONNX Runtime floors and
caps are published per Python version so that metadata-driven resolvers select a
release that actually has a wheel for your interpreter; you do not need to pin
them yourself.
