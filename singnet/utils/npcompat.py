"""NumPy 2.x compatibility aliases for legacy data-time dependencies.

The pinned data-time extras (``stempeg==0.2.3``, and transitively ``musdb`` /
``museval``) predate NumPy 2.0 and reference expired aliases at import time —
``stempeg/read.py`` uses ``np.float_`` in a function default, so merely importing
``musdb`` crashes under the project's pinned ``numpy==2.4.6``
(``AttributeError: np.float_ was removed in the NumPy 2.0 release``).

:func:`install_numpy2_aliases` restores the removed scalar aliases as plain module
attributes **before** those libraries are imported. Assignment sticks for the whole
process (NumPy's ``__getattr__`` guard only fires for *missing* attributes), it is
idempotent, and it is a no-op on NumPy 1.x where the names still exist.

Call sites: every RUN-LATER path that imports the legacy stack
(``scripts/prepare_data.py``, ``scripts/teacher_label.py``,
``singnet/metrics/museval_wrap.py``) and the museval test's import guard.
Discovered on the first Windows/NumPy-2.4 execution of the data-prep path.
"""

from __future__ import annotations

import numpy as np

#: expired-in-2.0 alias -> its NumPy 2.x replacement (values, not dtypes, for Inf/NaN)
_ALIASES = {
    "float_": np.float64,
    "complex_": np.complex128,
    "unicode_": np.str_,
    "string_": np.bytes_,
    "Inf": np.inf,
    "Infinity": np.inf,
    "NaN": np.nan,
    "NAN": np.nan,
}


def install_numpy2_aliases() -> list[str]:
    """Restore expired NumPy 1.x aliases; return the names actually installed."""
    installed: list[str] = []
    for name, value in _ALIASES.items():
        # getattr would raise AttributeError on NumPy 2 (expired attribute), so
        # probe the module dict directly; skip anything that genuinely exists.
        if name not in np.__dict__:
            setattr(np, name, value)
            installed.append(name)
    return installed
