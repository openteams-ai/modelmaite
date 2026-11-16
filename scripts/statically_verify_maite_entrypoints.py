"""Statically verify every MAITE component modelmaite advertises.

Each ``maite.protocols.*`` entry point in ``pyproject.toml`` names a wrapper and
the MAITE protocol it implements. MAITE's verifier type-checks each wrapper
against that protocol with pyright.

Run it inside the project environment (``uv run python scripts/statically_verify_maite_entrypoints.py``).
MAITE runs pyright on a temporary file without ``--pythonpath``, so pyright finds
modelmaite through the active virtual environment.
"""

from __future__ import annotations

import sys

try:
    from maite._internals.testing.project import statically_verify_exposed_component_entrypoints
    from maite._internals.testing.pyright import PYRIGHT_PATH
except ImportError as exc:
    print(f"cannot import MAITE's static entry point verifier (install the test group): {exc}", file=sys.stderr)
    raise SystemExit(1) from exc


def main() -> int:
    if PYRIGHT_PATH is None:
        print("pyright not on PATH; install the dev group", file=sys.stderr)
        return 1
    results = statically_verify_exposed_component_entrypoints(package_name="modelmaite")
    if not results:
        print("no maite.protocols entry points found for modelmaite", file=sys.stderr)
        return 1
    failed = sorted(name for name, ok in results.items() if not ok)
    if failed:
        print("statically_verify failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"verified {len(results)} maite.protocols entry points: " + ", ".join(sorted(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
