"""NumPy-2 alias shim for the legacy data-time stack (stempeg/musdb/museval)."""

from __future__ import annotations

import numpy as np

from singnet.utils.npcompat import install_numpy2_aliases


def test_expired_aliases_resolve_after_install() -> None:
    install_numpy2_aliases()
    # the exact attribute stempeg 0.2.3 dereferences at import time:
    assert np.float_ is np.float64
    assert np.complex_ is np.complex128
    assert np.unicode_ is np.str_
    assert np.string_ is np.bytes_
    assert np.Inf == np.inf and np.NaN != np.NaN  # NaN semantics preserved


def test_install_is_idempotent_and_reports_nothing_second_time() -> None:
    install_numpy2_aliases()
    assert install_numpy2_aliases() == []  # everything already present


def test_existing_numpy_attributes_are_never_clobbered() -> None:
    # sanity: real NumPy 2 scalars/values are untouched by the shim
    before = (np.float64, np.inf)
    install_numpy2_aliases()
    assert (np.float64, np.inf) == before
