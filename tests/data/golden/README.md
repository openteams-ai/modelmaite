# Golden outputs

These JSON files are snapshots of what each ModelMAITE wrapper returns today.
`tests/test_golden_outputs.py` runs every wrapper on fixed inputs and checks
that its outputs still match them.

## What they check

Each file records, for one wrapper:

- its predictions on a few fixed synthetic images, both `uint8` and float
  images in `[0, 1]`
- `metadata["id"]`, `index2label` and `name`

The models are stand-ins, not trained models: tiny ONNX graphs written at test
time, torchvision models with seeded random weights, and simple numpy
detectors. VisDrone is the exception: it uses the real Kitware weights, so its
test only runs when `MODELMAITE_RUN_VISDRONE_INTEGRATION=1` is set.

So the snapshots test the wrapper code: preprocessing, how the model is
called, and how its output is converted to MAITE types. They do not measure
model accuracy.

| File | Wrapper |
| --- | --- |
| `onnx_ic.json` | `OnnxICModel` |
| `onnx_od.json` | `OnnxODModel` |
| `torchvision_ic.json` | `TorchvisionICModel` |
| `torchvision_od.json` | `TorchvisionODModel` |
| `bytetrack_mot.json` | `ByteTrackMOTModel` |
| `visdrone_od.json` | `VisdroneODModel` |
| `default_ids.json` | The default `metadata["id"]` of each wrapper |

## Why they exist

- **Refactors.** A change that should not alter behaviour, such as moving the
  wrappers onto a shared base class, must pass these tests unchanged.
- **Dependency upgrades.** A new torch, torchvision, onnxruntime or `trackers`
  release can change preprocessing or numerics without any other test
  failing.
- **Model IDs.** CheckMAITE uses `metadata["id"]` in its cache keys. If a
  default ID changes, cached results stop being reused, so `default_ids.json`
  runs in every CI job.

## Running them

The `test-golden` CI job runs them with the ONNX, MOT and CPU torchvision
dependencies. Locally:

```bash
uv sync --group test --extra onnx --extra mot --extra torchvision
uv run pytest tests/test_golden_outputs.py
```

Tests whose optional dependencies are missing are skipped. Torch outputs are
compared with `rtol=1e-4, atol=1e-5`, and numpy and ONNX outputs with
`rtol=1e-5, atol=1e-6`.

## When a golden test fails

A failure means a wrapper now returns something different. Find out why
before changing anything here.

- **Unintended change** (a bug, or a dependency upgrade that changed the
  results): fix the code, or pin the dependency.
- **Intended change** (a deliberate fix or new behaviour): re-record the
  snapshots and explain the change in the `CHANGELOG.md`.

Never re-record only to make a failing test pass.

## Updating them

```bash
MODELMAITE_UPDATE_GOLDENS=1 uv run pytest tests/test_golden_outputs.py
```

Only the sections of tests that run are rewritten; skipped tests keep their
recorded values. So install the extras the changed wrappers need, and check
that their tests ran rather than skipped. VisDrone needs a separate
environment, because the `visdrone` and `mot` extras conflict: install
`--extra visdrone` and also set `MODELMAITE_RUN_VISDRONE_INTEGRATION=1`.

Review the JSON diff before committing: it shows reviewers exactly which
outputs changed.

## Adding a new model wrapper

1. In `tests/test_golden_outputs.py`, add a `test_<wrapper>_matches_golden_outputs`
   test, following the existing ones:
   - Build the model without downloads: seeded random weights, a small graph
     written at test time, or a deterministic stand-in for the network.
   - The stand-in's output must depend on where pixels are, not only on
     their values, so that a flipped or transposed input changes the result.
     It must also give the same result on every platform; random-weight
     detectors often produce near-tied scores, which break this.
   - Check `_model_facts(model)` under the `model` section, and the outputs
     on `_uint8_batch(...)` and `_float_batch(...)` inputs under
     `outputs.<case>` sections, with a new `<wrapper>.json` file name.
   - Mark it with the skip decorator for its optional dependencies.
2. Add the wrapper's default `metadata["id"]` to
   `test_default_ids_match_goldens`.
3. Record the new snapshots with `MODELMAITE_UPDATE_GOLDENS=1` (see above),
   check that the new test ran, and commit the new JSON file.
4. Add the file to the table above. If the wrapper needs a new extra, add it
   to the `test-golden` CI job.

Outside these snapshots, a new wrapper also needs a MAITE entry point in
`pyproject.toml` (`tests/test_maite_entry_points.py` fails without one) and
`isinstance` and `maite.spotcheck` tests in its own test file.
