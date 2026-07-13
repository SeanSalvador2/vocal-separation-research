#!/usr/bin/env python3
r"""Deterministic parameter-matching for the band-split variants (Direction 03 §3.1).

The whole experiment stands or falls on **honest parameter matching**: a SI-SDR
gap between arms of different size is a capacity artifact, not a band-split
effect. This script derives — in closed form and then cross-checks against the
actual :class:`~singnet.models.bandsplit_unet.BandSplitUNet` — the shared base
width ``c`` (and a single bottleneck ``+Delta`` bump) that puts the 3-tower
variant within **2 %** of the 9,835,745-param SingNet-C1 baseline, and writes the
committed per-module table (``03-mini-band-split/results/param_match_table.md``).

Search (deterministic; THEORY §4):

1. A ``Conv2d``/``ConvTranspose2d`` ``(c_in, c_out, 5x5)`` has ``25*c_in*c_out +
   c_out`` params; its ``BatchNorm2d(c_out)`` adds ``2*c_out``. The closed form
   ``P_variant(c) = 3*E(c) + D(c) + head = 18100 c^2 + 403 c + 1`` (three encoder
   towers + one baseline-width decoder + the 1x1 head).
2. ``c* = `` the **largest** ``c`` whose *pure* ``P_variant(c, 16c)`` does not
   exceed the baseline (so the base structure never overshoots budget) — ``c=23``
   (``9,584,170``, ``-2.56 %``; ``c=24`` is ``+6.10 %``).
3. Close the residual gap with the **smallest single bottleneck bump** ``Delta``
   (level-5 width ``b5 = 16c + Delta``, ``Delta >= 0``) that **minimizes**
   ``|P - P_0|`` while staying within 2 % — ``Delta = 14`` -> ``b5 = 382`` ->
   ``9,841,896`` (``+0.0625 %``). Both variants (mel and uniform) have this exact
   count: the count is independent of the band edges (conv params are
   spatial-size-independent), which differ only in compute (THEORY §4 FLOPs).

    python scripts/match_params.py --report     # CPU, seconds; re-runnable

Every number printed here is asserted in ``tests/test_match_params.py`` (closed
form == the built ``nn.Module``) and mirrored in THEORY §4.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.models.bandsplit_unet import (  # noqa: E402
    BASELINE_PARAM_COUNT,
    BandSplitUNet,
    mel_edges,
    uniform_edges,
)

DEFAULT_TARGET = BASELINE_PARAM_COUNT  # 9,835,745
DEFAULT_TOL = 0.02  # +-2 % (MASTER_PLAN §3.1)
REPORT_PATH = "03-mini-band-split/results/param_match_table.md"


# --- closed-form parameter accounting (bias + BatchNorm included) -----------

def conv_bn_params(c_in: int, c_out: int) -> int:
    """A ``Conv2d/ConvTranspose2d(c_in, c_out, 5x5)`` (+bias) followed by ``BN(c_out)``."""
    return 25 * c_in * c_out + c_out + 2 * c_out  # 25*in*out (weight) + out (bias) + 2*out (BN)


def encoder_tower_params(c: int, b5: int) -> int:
    """One 5-level tower ``1 -> c -> 2c -> 4c -> 8c -> b5``."""
    return (
        conv_bn_params(1, c)
        + conv_bn_params(c, 2 * c)
        + conv_bn_params(2 * c, 4 * c)
        + conv_bn_params(4 * c, 8 * c)
        + conv_bn_params(8 * c, b5)
    )


def decoder_params(c: int, b5: int) -> int:
    """Baseline-design decoder at width ``c`` (dec1 consumes the ``b5`` bottleneck)."""
    return (
        conv_bn_params(b5, 8 * c)      # dec1
        + conv_bn_params(16 * c, 4 * c)  # dec2 (8c d + 8c skip)
        + conv_bn_params(8 * c, 2 * c)   # dec3
        + conv_bn_params(4 * c, c)       # dec4
        + conv_bn_params(2 * c, c)       # dec5
    )


def head_params(c: int) -> int:
    """The ``1x1`` conv ``c -> 1`` (+bias)."""
    return c + 1


def baseline_params(c: int = 32) -> int:
    """SingNet-C1: one encoder + decoder + head at width ``c`` (``b5 = 16c``)."""
    return encoder_tower_params(c, 16 * c) + decoder_params(c, 16 * c) + head_params(c)


def variant_params(c: int, b5: int | None = None, n_towers: int = 3) -> int:
    """Band-split variant: ``n_towers`` encoder towers + one decoder + head."""
    if b5 is None:
        b5 = 16 * c
    return n_towers * encoder_tower_params(c, b5) + decoder_params(c, b5) + head_params(c)


# --- the deterministic width search -----------------------------------------

@dataclass(frozen=True)
class MatchResult:
    """Outcome of the width search."""

    base_width: int          # c
    bottleneck_width: int    # b5 = 16c + delta
    delta: int               # the single bottleneck bump
    params: int              # exact matched-variant param count
    target: int              # baseline param count matched to
    rel_error: float         # (params - target) / target
    within_tol: bool

    @property
    def pure_params(self) -> int:
        return variant_params(self.base_width, 16 * self.base_width)


def match_width(
    target: int = DEFAULT_TARGET,
    tol: float = DEFAULT_TOL,
    n_towers: int = 3,
    *,
    max_width: int = 128,
    max_delta: int = 256,
) -> MatchResult:
    """Largest base width under budget, then the tightest bottleneck bump (<= tol).

    Deterministic and re-runnable. Returns the pinned ``(c, b5)`` used by the
    variant configs and asserted in the unit tests.
    """
    # Step 1: largest c whose pure variant does not exceed the target.
    c_star = None
    for c in range(1, max_width + 1):
        if variant_params(c, 16 * c, n_towers) <= target:
            c_star = c
        else:
            break
    if c_star is None:
        raise ValueError(f"no width <= {max_width} keeps the pure variant under {target}")

    # Step 2: the delta >= 0 minimizing |P - target| (bottleneck bump only).
    best_delta, best_params, best_err = 0, variant_params(c_star, 16 * c_star, n_towers), None
    for delta in range(0, max_delta + 1):
        params = variant_params(c_star, 16 * c_star + delta, n_towers)
        err = abs(params - target)
        if best_err is None or err < best_err:
            best_delta, best_params, best_err = delta, params, err
        elif params - target > best_err:
            break  # params increase monotonically in delta; no closer point ahead

    rel = (best_params - target) / target
    return MatchResult(
        base_width=c_star,
        bottleneck_width=16 * c_star + best_delta,
        delta=best_delta,
        params=best_params,
        target=target,
        rel_error=rel,
        within_tol=abs(rel) <= tol,
    )


# --- cross-check: closed form == the built nn.Module ------------------------

def verify_against_model(match: MatchResult) -> dict[str, int]:
    """Build both variants and assert their real param counts equal the closed form.

    The credibility crux — a reader (and the test suite) sees the match is exact,
    not hand-waved. Returns the actual counts keyed by arm.
    """
    counts = {}
    for arm, model in (
        ("split_mel", BandSplitUNet.from_mel_bands(base_width=match.base_width,
                                                   bottleneck_width=match.bottleneck_width)),
        ("split_uniform", BandSplitUNet.from_uniform_bands(base_width=match.base_width,
                                                           bottleneck_width=match.bottleneck_width)),
    ):
        actual = model.num_parameters
        if actual != match.params:
            raise AssertionError(
                f"{arm}: built model has {actual} params, closed form says {match.params}"
            )
        counts[arm] = actual
    return counts


# --- per-module table (committed markdown; mirrored in THEORY §4) ------------

def per_module_rows(match: MatchResult, n_towers: int = 3) -> list[tuple[str, int, int]]:
    """Rows ``(module, baseline_params, variant_params)`` for the committed table.

    Encoder rows show the **total across towers** (baseline: 1 encoder at c=32;
    variant: ``n_towers`` towers at c=23) so a reader sees exactly where the 3x
    encoder capacity is spent and how the decoder/head absorb the rest.
    """
    cb, c, b5 = 32, match.base_width, match.bottleneck_width
    base_b5 = 16 * cb

    def enc_rows(width: int, bott: int, towers: int) -> list[tuple[str, int]]:
        chans = [(1, width), (width, 2 * width), (2 * width, 4 * width),
                 (4 * width, 8 * width), (8 * width, bott)]
        return [(f"enc{i+1}", towers * conv_bn_params(a, b)) for i, (a, b) in enumerate(chans)]

    def dec_rows(width: int, bott: int) -> list[tuple[str, int]]:
        chans = [(bott, 8 * width), (16 * width, 4 * width), (8 * width, 2 * width),
                 (4 * width, width), (2 * width, width)]
        return [(f"dec{i+1}", conv_bn_params(a, b)) for i, (a, b) in enumerate(chans)]

    base = dict(enc_rows(cb, base_b5, 1) + dec_rows(cb, base_b5) + [("head", head_params(cb))])
    var = dict(enc_rows(c, b5, n_towers) + dec_rows(c, b5) + [("head", head_params(c))])
    order = [f"enc{i}" for i in range(1, 6)] + [f"dec{i}" for i in range(1, 6)] + ["head"]
    return [(m, base[m], var[m]) for m in order]


def render_markdown(match: MatchResult, actual: dict[str, int], n_towers: int = 3) -> str:
    """The committed ``param_match_table.md`` (summary + per-module breakdown)."""
    c, b5 = match.base_width, match.bottleneck_width
    rows = per_module_rows(match, n_towers)
    enc_base = sum(r[1] for r in rows if r[0].startswith("enc"))
    enc_var = sum(r[2] for r in rows if r[0].startswith("enc"))
    dec_base = sum(r[1] for r in rows if r[0].startswith("dec"))
    dec_var = sum(r[2] for r in rows if r[0].startswith("dec"))

    lines = [
        "# Parameter-match table — Direction 03 (mini band-split)",
        "",
        "*Generated by `scripts/match_params.py` (deterministic, CPU-seconds, "
        "re-runnable). Every number here is asserted in `tests/test_match_params.py` "
        "against the built `nn.Module` and mirrored in `THEORY.md` §4.*",
        "",
        "## Chosen width",
        "",
        f"- Baseline (`SingNetC1`, `c = 32`): **{BASELINE_PARAM_COUNT:,}** params.",
        f"- Search: largest base width with the *pure* 3-tower variant under budget "
        f"is **c = {c}** (pure `{match.pure_params:,}`, "
        f"`{100 * (match.pure_params - match.target) / match.target:+.3f} %` — outside 2 %).",
        f"- Single bottleneck bump **Δ = +{match.delta}** → level-5 width "
        f"**b5 = {b5}** closes the gap.",
        f"- Matched variant: **{match.params:,}** params, "
        f"**{100 * match.rel_error:+.4f} %** vs baseline "
        f"(within 2 %: **{str(match.within_tol).lower()}**).",
        "",
        "## Three arms",
        "",
        "| Arm | Front-end | c | b5 | Params | Δ vs baseline | ≤ 2 % |",
        "|---|---|---|---|---|---|---|",
        f"| `baseline` | single full-spectrum encoder | 32 | 512 | "
        f"{BASELINE_PARAM_COUNT:,} | 0.000 % | ✓ |",
        f"| `split_uniform` | 3 equal-**bin** towers | {c} | {b5} | "
        f"{actual['split_uniform']:,} | {100 * match.rel_error:+.4f} % | "
        f"{'✓' if match.within_tol else '✗'} |",
        f"| `split_mel` | 3 equal-**mel** towers | {c} | {b5} | "
        f"{actual['split_mel']:,} | {100 * match.rel_error:+.4f} % | "
        f"{'✓' if match.within_tol else '✗'} |",
        "",
        "Both variants have the **identical** count by construction — conv "
        "parameters are independent of spatial extent, so the band edges (which "
        "differ between the arms) change only the compute, not the parameters "
        "(THEORY §4).",
        "",
        "## Per-module breakdown",
        "",
        "Encoder rows are the **total across towers** (baseline: 1 encoder at "
        f"c = 32; variants: 3 towers at c = {c}).",
        "",
        "| Module | baseline (c=32) | split_mel / split_uniform (c=%d, b5=%d) |" % (c, b5),
        "|---|---|---|",
    ]
    for name, base_p, var_p in rows:
        lines.append(f"| {name} | {base_p:,} | {var_p:,} |")
    lines += [
        f"| **encoder Σ** | {enc_base:,} | {enc_var:,} |",
        f"| **decoder Σ** | {dec_base:,} | {dec_var:,} |",
        f"| **total** | {BASELINE_PARAM_COUNT:,} | {match.params:,} |",
        "",
        f"Mel band edges (bins): `{mel_edges()}` — interior ≈ 1.53 kHz, 6.43 kHz.  ",
        f"Uniform band edges (bins): `{uniform_edges()}` — interior ≈ 7.35 kHz, 14.70 kHz.",
        "",
    ]
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="baseline param count")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="fractional tolerance (0.02)")
    parser.add_argument("--report", action="store_true",
                        help=f"write {REPORT_PATH} (else just print the summary)")
    parser.add_argument("--out", default=REPORT_PATH, help="report path")
    args = parser.parse_args(argv)

    match = match_width(args.target, args.tol)
    actual = verify_against_model(match)
    markdown = render_markdown(match, actual)

    print(f"baseline target : {args.target:,}")
    print(f"chosen c        : {match.base_width}  (pure {match.pure_params:,}, "
          f"{100 * (match.pure_params - match.target) / match.target:+.3f} %)")
    print(f"bottleneck bump : Δ = +{match.delta}  ->  b5 = {match.bottleneck_width}")
    print(f"matched params  : {match.params:,}  ({100 * match.rel_error:+.4f} %, "
          f"within {args.tol:.0%}: {match.within_tol})")
    print(f"built models    : split_mel={actual['split_mel']:,}  "
          f"split_uniform={actual['split_uniform']:,}  (== closed form)")

    if args.report:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
