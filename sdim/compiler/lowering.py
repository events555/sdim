"""Circuit lowering and reference sample generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gates.registry import (
    gate_id_to_name,
    is_gate_collapsing,
    is_gate_records,
    is_gate_two_qubit,
)

if TYPE_CHECKING:
    from ..circuit import Circuit


def lower_to_ir(circuit: "Circuit") -> np.ndarray:
    """Flatten circuit operations into a structured numpy IR array."""
    ir_list: list[tuple[int, int, int, float]] = []

    for instruction in circuit.operations:
        gate_id = instruction.gate_type
        arg0 = float(instruction.args[0]) if instruction.args else np.nan
        if is_gate_two_qubit(gate_id):
            for i in range(0, len(instruction.targets), 2):
                if i + 1 < len(instruction.targets):
                    control = instruction.targets[i]
                    target = instruction.targets[i + 1]
                    ir_list.append((gate_id, control.value, target.value, arg0))
        else:
            for target in instruction.targets:
                ir_list.append(
                    (gate_id, target.value, np.iinfo(np.int64).max, arg0)
                )

    ir_dtype = np.dtype(
        [
            ("gate_id", np.int64),
            ("qudit_index", np.int64),
            ("target_index", np.int64),
            ("arg0", np.float64),
        ]
    )
    return np.array(ir_list, dtype=ir_dtype)


def reference_sample(
    ir: np.ndarray,
    num_qudits: int,
    dimension: int,
) -> np.ndarray:
    """Run a noiseless tableau simulation over the IR to produce reference measurements.

    Attempts to use the Rust backend (_sdim_rs) for performance.
    Falls back to the Python TableauSimulator if the extension is unavailable.
    """
    try:
        from .._sdim_rs import run_ir as _rust_run_ir

        arg0_raw = ir['arg0']
        arg0_int = np.where(np.isnan(arg0_raw), -1, arg0_raw).astype(np.int64)
        plain = np.column_stack([
            ir['gate_id'].astype(np.int64),
            ir['qudit_index'].astype(np.int64),
            ir['target_index'].astype(np.int64),
            arg0_int,
        ])
        return _rust_run_ir(plain, num_qudits, dimension)
    except ImportError:
        pass

    return _reference_sample_python(ir, num_qudits, dimension)


def _reference_sample_python(
    ir: np.ndarray,
    num_qudits: int,
    dimension: int,
) -> np.ndarray:
    """Pure-Python fallback for reference_sample."""
    from ..simulators.tableau_simulator import TableauSimulator

    tableau = TableauSimulator(num_qudits, dimension)
    measurements: list[int] = []
    gate_count = 0

    for inst in ir:
        gate_id = inst["gate_id"]
        qudit_index = inst["qudit_index"]
        target_index = inst["target_index"]
        arg0 = inst["arg0"]
        gate_name = gate_id_to_name(gate_id)

        if gate_name == "HERALDED_ERASURE":
            measurements.append(0)
        elif is_gate_collapsing(gate_id):
            if gate_name in ("M_X", "MR_X"):
                tableau.hadamard(qudit_index, dagger=True)

            measurement = tableau.measure(qudit_index)

            if gate_name in ("MR", "MR_X", "RESET"):
                correction = (-measurement) % dimension
                tableau.pauli_x(qudit_index, correction)
                if gate_name == "MR_X":
                    tableau.hadamard(qudit_index)

            if is_gate_records(gate_id):
                measurements.append(measurement)
        else:
            if qudit_index < 0:
                if gate_name == "CNOT":
                    tableau.pauli_x(target_index, measurements[qudit_index])
                elif gate_name == "CZ":
                    tableau.pauli_z(target_index, measurements[qudit_index])
            elif target_index < 0:
                if gate_name == "CNOT":
                    raise ValueError(
                        "CNOT gate cannot be applied to measurement record target."
                    )
                elif gate_name == "CZ":
                    tableau.pauli_z(qudit_index, measurements[target_index])
            else:
                tableau.apply_gate(gate_id, qudit_index, target_index, arg0)
        gate_count += 1
        if gate_count % 128 == 0:
            tableau.modulo()

    return np.array(measurements)
