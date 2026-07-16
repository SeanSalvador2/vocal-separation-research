"""museval wrapper is import-guarded (skipped unless museval is installed)."""

from __future__ import annotations

import numpy as np
import pytest


def test_museval_wrapper_runs_if_available() -> None:
    from singnet.utils.npcompat import install_numpy2_aliases

    install_numpy2_aliases()  # stempeg (via musdb/museval) predates NumPy 2
    try:
        pytest.importorskip("museval")
    except RuntimeError as err:  # stempeg raises at import if ffmpeg is absent
        pytest.skip(f"museval import needs ffmpeg on PATH: {err}")
    from singnet.metrics.museval_wrap import bss_eval_sdr, museval_version

    assert isinstance(museval_version(), str)
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(44100)
    est = ref + 0.1 * rng.standard_normal(44100)
    result = bss_eval_sdr(ref, est, sample_rate=44100)
    assert "sdr_median" in result and "museval_version" in result


def test_importing_metrics_package_does_not_require_museval() -> None:
    """`import singnet.metrics` must not pull the optional heavy deps."""
    import singnet.metrics as m

    assert hasattr(m, "si_sdr")
    # museval_wrap is intentionally NOT imported by the package __init__
    assert not hasattr(m, "bss_eval_sdr")
