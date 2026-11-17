"""Model identity helpers shared by every wrapper, independent of any framework."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import types
from collections.abc import Mapping

import numpy as np

# Callables whose instance attributes don't describe them: two different functions both have an
# empty ``vars()``, so they would hash the same.
_OPAQUE_TYPES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    functools.partial,
)


def id_hash(**kwargs: object) -> str:
    """Return a 16-character hash of the canonical form of the keyword arguments.

    Raises ``TypeError`` if a value has no canonical form (see ``canonicalize``).
    """
    canonical = {name: canonicalize(value, name) for name, value in kwargs.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


def canonicalize(value: object, field: str) -> object:
    """Return a JSON value that stands for ``value`` in a model id.

    JSON scalars, lists, tuples and mappings with string keys are kept (recursively), a NumPy
    scalar becomes the Python scalar it holds, a path
    becomes its string, a class becomes ``"module.qualname"``, and any other object becomes
    ``{"class": "module.qualname", "attributes": {...}}`` built from ``vars(obj)``, if every
    attribute can itself be canonicalised. Anything else raises ``TypeError`` naming ``field``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        # np.float32(30) and the like come from configs and NumPy code; they have no vars().
        return canonicalize(value.item(), field)
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return _canonical_mapping(value, field)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, type):
        return _qualified_name(value)
    if isinstance(value, _OPAQUE_TYPES):
        raise _not_canonical(field, f"a {type(value).__name__} has no attributes that describe it")
    try:
        attributes = vars(value)
    except TypeError:
        raise _not_canonical(field, f"a {_qualified_name(type(value))} has no instance attributes to hash") from None
    return {
        "class": _qualified_name(type(value)),
        "attributes": {name: canonicalize(item, f"{field}.{name}") for name, item in attributes.items()},
    }


def _canonical_mapping(value: Mapping[object, object], field: str) -> dict[str, object]:
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _not_canonical(field, f"it has a non-string key {key!r}")
        result[key] = canonicalize(item, f"{field}.{key}")
    return result


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _not_canonical(field: str, reason: str) -> TypeError:
    return TypeError(
        f"Cannot build a default model id from setting {field!r}: {reason}. Pass model_id to name the model yourself."
    )
