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

    records: list = []
    if skip_reference_sample:
        if circuit.num_measurements > 0:
            raise ValueError(
                "Cannot skip reference sample when circuit has measurements."
            )
        ref = np.array([], dtype=np.int64)
    elif circuit.num_measurements > 0:
        # The destabilizer records are trajectory-specific, so the reference
        # sample and records must come from one pass. A caller-supplied
        # ref_sample cannot be paired with matching records and is ignored
        # here (the records' collapse choices would not align with it).
        ref = reference_sample(
            ir, circuit.num_qudits, circuit.dimension, records=records
        )
    elif ref_sample is not None:
        ref = ref_sample
    else:
        ref = np.array([], dtype=np.int64)

    return CompiledCircuit(
        circuit=circuit,
        ir_array=ir,
        reference_sample=ref,
        num_qudits=circuit.num_qudits,
        dimension=circuit.dimension,
        num_measurements=circuit.num_measurements,
        measurement_records=records,
    )
