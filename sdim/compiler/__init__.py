"""Circuit compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .compiled_circuit import CompiledCircuit
from .lowering import lower_to_ir, reference_sample

if TYPE_CHECKING:
    from ..circuit import Circuit

__all__ = ["compile", "CompiledCircuit", "lower_to_ir", "reference_sample"]


def compile(
    circuit: "Circuit",
    *,
    skip_reference_sample: bool = False,
    ref_sample: Optional[np.ndarray] = None,
    ir_array: Optional[np.ndarray] = None,
) -> CompiledCircuit:
    """Lower a Circuit AST into a CompiledCircuit ready for sampling."""
    ir = ir_array if ir_array is not None else lower_to_ir(circuit)

    if ref_sample is not None:
        ref = ref_sample
    elif skip_reference_sample:
        if circuit.num_measurements > 0:
            raise ValueError(
                "Cannot skip reference sample when circuit has measurements."
            )
        ref = np.array([], dtype=np.int64)
    elif circuit.num_measurements > 0:
        ref = reference_sample(ir, circuit.num_qudits, circuit.dimension)
    else:
        ref = np.array([], dtype=np.int64)

    return CompiledCircuit(
        circuit=circuit,
        ir_array=ir,
        reference_sample=ref,
        num_qudits=circuit.num_qudits,
        dimension=circuit.dimension,
        num_measurements=circuit.num_measurements,
    )
