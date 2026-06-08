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


def lower_to_ir(circuit: "Circuit") -> tuple[np.ndarray, np.ndarray]:
    """Flatten circuit operations into a structured IR array + args pool.

    Each gate's float args are concatenated into a flat ``args_pool``; every IR
    row carries an ``(arg_start, arg_len)`` span into it. All IR rows
    expanded from one multi-target instruction share that instruction's span.
    """
    ir_list: list[tuple[int, int, int, int, int]] = []
    args_pool: list[float] = []

    for instruction in circuit.operations:
        gate_id = instruction.gate_type
        if instruction.args:
            arg_start = len(args_pool)
            arg_len = len(instruction.args)
            args_pool.extend(float(a) for a in instruction.args)
        else:
            arg_start = 0
            arg_len = 0
        if is_gate_two_qubit(gate_id):
            for i in range(0, len(instruction.targets), 2):
                if i + 1 < len(instruction.targets):
                    control = instruction.targets[i]
                    target = instruction.targets[i + 1]
                    ir_list.append(
                        (
                            gate_id,
                            control.value,
                            target.value,
                            arg_start,
                            arg_len,
                        )
                    )
        else:
            no_target = np.iinfo(np.int64).max
            for target in instruction.targets:
                ir_list.append(
                    (gate_id, target.value, no_target, arg_start, arg_len)
                )

    ir_dtype = np.dtype(
        [
            ("gate_id", np.int64),
            ("qudit_index", np.int64),
            ("target_index", np.int64),
            ("arg_start", np.int64),
            ("arg_len", np.int64),
        ]
    )
    return (
        np.array(ir_list, dtype=ir_dtype),
        np.array(args_pool, dtype=np.float64),
    )


def reference_sample(
    ir: np.ndarray,
    num_qudits: int,
    dimension: int,
    args_pool: np.ndarray,
    *,
    records: list | None = None,
) -> np.ndarray:
    """Run a noiseless tableau simulation over the IR to produce reference measurements.

    If ``records`` is given, one entry is appended per recorded measurement:
    ``None`` for a deterministic outcome, or ``(eta, s, S0_z, S0_x)`` for a
    random one (the destabilizer the frame sampler folds in).
    """
    from ..simulators.tableau_simulator import TableauSimulator

    tableau = TableauSimulator(num_qudits, dimension)
    measurements: list[int] = []
    gate_count = 0

    for inst in ir:
        gate_id = inst["gate_id"]
        qudit_index = inst["qudit_index"]
        target_index = inst["target_index"]
        arg0 = args_pool[inst["arg_start"]] if inst["arg_len"] else np.nan
        gate_name = gate_id_to_name(gate_id)

        if gate_name == "HERALDED_ERASURE":
            measurements.append(0)
            if records is not None:
                records.append(None)
        elif is_gate_collapsing(gate_id):
            if gate_name in ("M_X", "MR_X"):
                tableau.hadamard(qudit_index, dagger=True)

            measurement = tableau.measure(qudit_index)

            if is_gate_records(gate_id):
                measurements.append(measurement)
                if records is not None:
                    records.append(tableau.last_destabilizer)

            if gate_name in ("MR", "MR_X", "RESET"):
                correction = (-measurement) % dimension
                tableau.pauli_x(qudit_index, correction)
                if gate_name == "MR_X":
                    tableau.hadamard(qudit_index)
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
