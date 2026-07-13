"""SingNet: a compact magnitude-mask U-Net for music source separation.

This package holds all load-bearing logic for the SingNet project (data, STFT
front-end, model, losses, metrics, training loop, evaluation). Notebooks and
scripts orchestrate and visualise; they import from here so that every
experiment is reproducible and unit-tested.

Direction 01 ("Train on what you test?", the loss-function study) is the first
consumer of this package; directions 02-10 reuse the same components.

Nothing in this package downloads data, runs a GPU, or trains a model on import.
The heavy lifting (``singnet.train.run``, ``singnet.eval.evaluate``) is invoked
explicitly and is marked "RUN LATER" throughout the notebooks.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
