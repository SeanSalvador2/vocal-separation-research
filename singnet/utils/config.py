"""YAML config loading, ``base:`` inheritance, canonicalization, and hashing.

Every experiment in this project is a YAML file; the run registry keys runs by
a **config hash** (MASTER_PLAN §7.1) so an interrupted sweep resumes exactly and
no "I changed a constant in the notebook" experiments exist. The hash must be
stable across processes and independent of key order — see :func:`hash_config`.

Direction 02 adds an **augmentation switchboard** and a **track allowlist** to
the schema. Both are optional with defaults, and :func:`canonicalize_config`
default-fills them **before** hashing so that three spellings of the *same*
experiment collapse to one hash (Direction 02 MASTER_PLAN §1, §8 gate G0):

* a legacy Direction-01 config (``augment: true`` + top-level ``remix: true``),
* a config with **no** augmentation block at all (relies on defaults), and
* a Direction-02 config with an explicit ``augment: {remix: true, gain: true,
  flip: true}`` block and ``data: {track_allowlist_csv: null}``

all canonicalize to the identical augmentation block and therefore hash equal.
This is what makes Direction 01's ``l1mag`` sweep cell **provably** the same run
as Direction 02's ``full``/``n86`` cell, so it is reused rather than retrained.
"""

from __future__ import annotations

import copy
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

#: The three factorized augmentation transforms (MASTER_PLAN §5), canonical order.
AUGMENT_KEYS: tuple[str, ...] = ("remix", "gain", "flip")
#: Default switchboard — the full standard recipe is on unless a config says otherwise.
AUGMENT_DEFAULTS: dict[str, bool] = {"remix": True, "gain": True, "flip": True}


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
    """Load a config, resolving ``base:`` inheritance (leaf overrides win).

    A leaf config may declare ``base: base.yaml`` (path relative to the leaf's
    own directory). The base is loaded first, then the leaf is deep-merged on
    top. ``base`` itself is stripped from the returned dict. Inheritance is
    resolved recursively, but the project's configs are intentionally shallow
    (leaf -> base).
    """
    path = Path(path)
    raw = load_yaml(path)
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    base_path = (path.parent / base_name).resolve()
    base_cfg = resolve_config(base_path)
    return deep_merge(base_cfg, raw)


def augment_switches(config: dict[str, Any]) -> dict[str, bool]:
    """Return the normalized ``{remix, gain, flip}`` switchboard for a config.

    Accepts every spelling the project uses and folds them to one canonical
    block (MASTER_PLAN §5; Direction 02 §1):

    * ``augment`` **absent** -> all defaults (the full recipe).
    * ``augment: true`` / ``augment: false`` (Direction-01 legacy bool, which
      toggled *gain + flip* together) -> ``gain = flip = <bool>``; ``remix``
      comes from the separate legacy top-level ``remix`` key (default ``True``).
    * ``augment: {remix?, gain?, flip?}`` (Direction-02 dict) -> those keys,
      each defaulting to ``True``; a legacy top-level ``remix`` fills in only if
      the dict omits ``remix``.
    """
    aug = config.get("augment", None)
    legacy_remix = config.get("remix", None)

    if isinstance(aug, dict):
        block = {k: bool(aug.get(k, AUGMENT_DEFAULTS[k])) for k in AUGMENT_KEYS}
        if "remix" not in aug and legacy_remix is not None:
            block["remix"] = bool(legacy_remix)
        return block

    if isinstance(aug, bool):
        return {
            "remix": bool(legacy_remix) if legacy_remix is not None else AUGMENT_DEFAULTS["remix"],
            "gain": bool(aug),
            "flip": bool(aug),
        }

    # augment absent entirely -> defaults, honouring a stray legacy top-level remix.
    return {
        "remix": bool(legacy_remix) if legacy_remix is not None else AUGMENT_DEFAULTS["remix"],
        "gain": AUGMENT_DEFAULTS["gain"],
        "flip": AUGMENT_DEFAULTS["flip"],
    }


def track_allowlist_path(config: dict[str, Any]) -> str | None:
    """Return the subset allowlist CSV path, or ``None`` for the full split.

    Reads ``data.track_allowlist_csv`` (Direction-02 nested form) and also a flat
    top-level ``track_allowlist_csv`` (the canonical form produced by
    :func:`canonicalize_config`). ``None``/absent both mean "train on the full
    86-song split" (MASTER_PLAN §3.2).
    """
    data = config.get("data")
    if isinstance(data, dict) and data.get("track_allowlist_csv") is not None:
        return str(data["track_allowlist_csv"])
    flat = config.get("track_allowlist_csv")
    return str(flat) if flat is not None else None


def corruption_epsilon(config: dict[str, Any]) -> float:
    """Return the ε-bleed level from an optional ``corrupt:`` block (Direction 06 §3.1).

    Reads ``corrupt.epsilon`` (a float in ``[0, 1]``); a missing block, an empty
    block, or ``epsilon: 0`` all mean "no corruption" (ε = 0). ε is a
    **config-hash identity field when present** (see :func:`canonicalize_config`):
    ε = 0 canonicalizes to *no corruption* so the clean cell stays hash-identical
    to Directions 01–05, while ε > 0 mints a distinct run.
    """
    block = config.get("corrupt")
    if isinstance(block, dict):
        return float(block.get("epsilon", 0.0) or 0.0)
    return 0.0


def trim_q(config: dict[str, Any]) -> float | None:
    """Return the trimmed-loss fraction ``q`` from an optional ``trim:`` block (§3.2).

    Reads ``trim.q``; ``None`` (block absent) means "no trimming" (the plain base
    loss). ``q`` is a passthrough identity field — configs that differ only in
    ``trim.q`` (e.g. ``trim30`` vs ``trim30_q10``) already hash distinctly, so no
    special canonicalization is needed.
    """
    block = config.get("trim")
    if isinstance(block, dict) and block.get("q") is not None:
        return float(block["q"])
    return None


#: The four chunk-sampling policies (Direction 08 §4.1) and their default constants.
SAMPLING_POLICIES: tuple[str, ...] = ("uniform", "energy", "drop", "curriculum")
SAMPLING_DEFAULTS: dict[str, Any] = {"policy": "uniform", "theta_db": -60.0, "floor_lambda": 0.1}


def sampling_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized ``{policy, theta_db, floor_lambda}`` sampling spec (§4.1).

    An absent ``sampling:`` block (every Direction 01–06 config) is the ``uniform``
    baseline; a block selects the policy and, optionally, its constants (``theta_db``
    for ``drop``, ``floor_lambda`` for ``energy``/``curriculum``). The returned dict
    always carries all three keys with defaults filled, so :class:`ChunkSampler`
    construction and hashing both see one canonical shape.
    """
    block = config.get("sampling")
    if not isinstance(block, dict):
        return dict(SAMPLING_DEFAULTS)
    policy = str(block.get("policy", SAMPLING_DEFAULTS["policy"]))
    if policy not in SAMPLING_POLICIES:
        raise ValueError(f"unknown sampling policy {policy!r}; expected {SAMPLING_POLICIES}")
    return {
        "policy": policy,
        "theta_db": float(block.get("theta_db", SAMPLING_DEFAULTS["theta_db"])),
        "floor_lambda": float(block.get("floor_lambda", SAMPLING_DEFAULTS["floor_lambda"])),
    }


def canonicalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Default-fill the augmentation + allowlist + corruption schema to canonical form.

    Returns a **copy** in which

    * ``augment`` is always the normalized ``{remix, gain, flip}`` block,
    * the legacy top-level ``remix`` key is removed (folded into ``augment``),
    * the subset allowlist lives in a single flat ``track_allowlist_csv`` slot
      (``None`` for the full split) and any nested ``data.track_allowlist_csv``
      is removed,
    * the Direction-06 ``corrupt`` block is normalized so ε = 0 (or an absent
      block) leaves **no** corruption key — hash-identical to an uncorrupted
      Directions 01–05 config — while ε > 0 becomes the canonical
      ``{"epsilon": <float>}`` identity field,

    and every other key is left untouched (so the ``trim`` block and any Direction
    05 keys pass straight through). This runs **before** hashing so that a
    Direction-01 ``l1mag`` config and the Direction-06 clean cell hash equal
    (MASTER_PLAN §8 G0). It is idempotent.
    """
    cfg = copy.deepcopy(config)
    cfg["augment"] = augment_switches(cfg)
    cfg.pop("remix", None)

    allowlist = track_allowlist_path(cfg)
    data = cfg.get("data")
    if isinstance(data, dict):
        data.pop("track_allowlist_csv", None)
        if not data:
            cfg.pop("data", None)
    cfg.pop("track_allowlist_csv", None)
    cfg["track_allowlist_csv"] = allowlist

    # Direction-06 corruption: ε = 0 / absent -> no key (clean cell ≡ D01); ε > 0
    # -> the canonical identity block. Existing configs carry no `corrupt` key, so
    # this is a no-op for them and every prior hash is unchanged.
    eps = corruption_epsilon(cfg)
    cfg.pop("corrupt", None)
    if eps != 0.0:
        cfg["corrupt"] = {"epsilon": eps}

    # Direction-08 sampling: the `uniform` policy (or an absent block) canonicalizes to
    # NO sampling key — hash-identical to the shared baseline cell, so the D08 uniform
    # arm reuses D01 `l1mag` rather than retraining (§4.2). A non-uniform policy becomes
    # a canonical identity block carrying only the fields that actually steer it (θ for
    # `drop`, λ for `energy`/`curriculum`), so every existing hash is unchanged.
    spec = sampling_policy(cfg)
    cfg.pop("sampling", None)
    if spec["policy"] != "uniform":
        block: dict[str, Any] = {"policy": spec["policy"]}
        if spec["policy"] == "drop":
            block["theta_db"] = spec["theta_db"]
        else:  # energy / curriculum
            block["floor_lambda"] = spec["floor_lambda"]
        cfg["sampling"] = block
    return cfg


def _canonical(obj: Any) -> Any:
    """Return a JSON-canonical view: sorted dict keys, lists preserved."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj) if k not in _NON_IDENTITY_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def hash_config(config: dict[str, Any], *, length: int = 12) -> str:
    """Return a stable short hash of a config (canonicalized first).

    Stability guarantees (unit-tested in ``tests/test_registry.py`` and
    ``tests/test_config_schema.py``):

    * the augmentation/allowlist schema is **default-filled** before hashing, so
      an old config with no ``augment`` block and a new explicit-defaults config
      hash identically (MASTER_PLAN §8 G0);
    * independent of key insertion order (keys are sorted),
    * independent of non-identity keys such as ``gpu``/``output_dir``,
    * identical across processes (sha256 over canonical JSON, not ``hash()``).
    """
    canonical = _canonical(canonicalize_config(config))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
