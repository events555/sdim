from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..circuit import Circuit


@dataclass
class CompiledCircuit:
    """Immutable bundle produced by the compiler.

    Holds everything the samplers need so they never reach back into
    the original ``Circuit`` for IR, reference sample, or metadata.
    """

    circuit: "Circuit"
    ir_array: np.ndarray
    args_pool: np.ndarray
    reference_sample: np.ndarray
    num_qudits: int
    dimension: int
    num_measurements: int
    measurement_records: list = field(default_factory=list)
