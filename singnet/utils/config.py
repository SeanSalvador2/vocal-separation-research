"""YAML config loading, ``base:`` inheritance, and stable config hashing.

Every experiment in this project is a YAML file; the run registry keys runs by
a **config hash** (MASTER_PLAN §7.1) so an interrupted sweep resumes exactly and
no "I changed a constant in the notebook" experiments exist. The hash must be
stable across processes and independent of key order — see :func:`hash_config`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# Keys that describe *where a run happened* rather than *what was run*. They are
# excluded from the config hash so that resuming the same experiment on a
# different machine (or writing its outputs elsewhere) does not change identity.
_NON_IDENTITY_KEYS = frozenset(
    {"gpu", "output_dir", "shard_root", "musdb_root", "registry_path", "num_workers"}
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dict (empty file -> empty dict)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins), pure/no-mutate."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(path: str | Path) -> dict[str, Any]:
    """Load a config, resolving a single ``base:`` inheritance level.

    A leaf config may declare ``base: base.yaml`` (path relative to the leaf's
    own directory). The base is loaded first, then the leaf is deep-merged on
    top. ``base`` itself is stripped from the returned dict. Only one level of
    inheritance is supported by design — the sweep configs are intentionally
    shallow.
    """
    path = Path(path)
    raw = load_yaml(path)
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    base_path = (path.parent / base_name).resolve()
    base_cfg = resolve_config(base_path)
    return deep_merge(base_cfg, raw)


def _canonical(obj: Any) -> Any:
    """Return a JSON-canonical view: sorted dict keys, lists preserved."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj) if k not in _NON_IDENTITY_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def hash_config(config: dict[str, Any], *, length: int = 12) -> str:
    """Return a stable short hash of a resolved config.

    Stability guarantees (unit-tested in ``tests/test_registry.py``):

    * independent of key insertion order (keys are sorted),
    * independent of non-identity keys such as ``gpu``/``output_dir``,
    * identical across processes (sha256 over canonical JSON, not ``hash()``).
    """
    canonical = _canonical(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
