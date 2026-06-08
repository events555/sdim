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
    args_pool: Optional[np.ndarray] = None,
) -> CompiledCircuit:
    """Lower a Circuit AST into a CompiledCircuit ready for sampling."""
    if ir_array is not None:
        if args_pool is None:
            raise ValueError(
                "args_pool must accompany an externally supplied ir_array."
            )
        ir = ir_array
    else:
        ir, args_pool = lower_to_ir(circuit)

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
            ir,
            circuit.num_qudits,
            circuit.dimension,
            args_pool,
            records=records,
        )
    elif ref_sample is not None:
        ref = ref_sample
    else:
        ref = np.array([], dtype=np.int64)

    return CompiledCircuit(
        circuit=circuit,
        ir_array=ir,
        args_pool=args_pool,
        reference_sample=ref,
        num_qudits=circuit.num_qudits,
        dimension=circuit.dimension,
        num_measurements=circuit.num_measurements,
        measurement_records=records,
    )
