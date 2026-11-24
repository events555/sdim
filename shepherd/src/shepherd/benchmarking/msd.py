"""
msd.py -- the "msd" experiment (magic state distillation). Placeholder: registered so it can be
looked up, created and listed, but `build` raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np
from sdim import Circuit

from shepherd.experiment import Experiment
from shepherd.qecc import QECC


class MagicStateDistillation(Experiment, name="msd"):
    def __init__(self, dimension: int, **params):
        self.dimension = dimension
        self.params = params

    def build(self, code: QECC) -> Circuit:
        raise NotImplementedError("the msd (magic state distillation) experiment is not "
                                  "implemented yet")

    def expected_observables(self, code: QECC) -> np.ndarray:
        raise NotImplementedError("the msd (magic state distillation) experiment is not "
                                  "implemented yet")
