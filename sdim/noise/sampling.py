"""Noise sampling functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..gates.registry import (
    gate_id_to_name,
    is_gate_noisy,
    is_gate_two_qubit,
)

if TYPE_CHECKING:
    from ..circuit import Circuit


def sample_x_error(d: int, shots: int, error_prob: float) -> np.ndarray:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    noise = np.zeros((shots, 2), dtype=np.int64)
    rnd = np.random.rand(shots)
    mask = rnd < error_prob
    if d == 2:
        noise[mask, 0] = 1
    else:
        noise[mask, 0] = np.random.randint(1, d, size=int(np.sum(mask)))
    return noise


def sample_z_error(d: int, shots: int, error_prob: float) -> np.ndarray:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    noise = np.zeros((shots, 2), dtype=np.int64)
    rnd = np.random.rand(shots)
    mask = rnd < error_prob
    if d == 2:
        noise[mask, 1] = 1
    else:
        noise[mask, 1] = np.random.randint(1, d, size=int(np.sum(mask)))
    return noise


def sample_y_error(d: int, shots: int, error_prob: float) -> np.ndarray:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    noise = np.zeros((shots, 2), dtype=np.int64)
    rnd = np.random.rand(shots)
    mask = rnd < error_prob
    if d == 2:
        noise[mask, 0] = 1
        noise[mask, 1] = 1
    else:
        a = np.random.randint(1, d, size=int(np.sum(mask)))
        noise[mask, 0] = a
        noise[mask, 1] = a
    return noise


def sample_depolarize1(d: int, shots: int, error_prob: float) -> np.ndarray:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    noise = np.zeros((shots, 2), dtype=np.int64)
    rnd = np.random.rand(shots)
    mask = rnd < error_prob
    num_errors = int(np.sum(mask))
    if num_errors > 0:
        errors = np.array(
            [(x, z) for x in range(d) for z in range(d) if not (x == 0 and z == 0)],
            dtype=np.int64,
        )
        indices = np.random.randint(0, len(errors), size=num_errors)
        noise[mask, :] = errors[indices]
    return noise


def sample_depolarize2(d: int, shots: int, error_prob: float) -> np.ndarray:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    noise = np.zeros((shots, 4), dtype=np.int64)
    rnd = np.random.rand(shots)
    mask = rnd < error_prob
    num_errors = int(np.sum(mask))
    if num_errors > 0:
        errors = np.array(
            [(x, z) for x in range(d) for z in range(d) if not (x == 0 and z == 0)],
            dtype=np.int64,
        )
        indices1 = np.random.randint(0, len(errors), size=num_errors)
        indices2 = np.random.randint(0, len(errors), size=num_errors)
        noise[mask, :2] = errors[indices1]
        noise[mask, 2:] = errors[indices2]
    return noise


def sample_pauli_channel1(d: int, shots: int, pmf: ArrayLike) -> np.ndarray:
    arr = np.array(pmf, dtype=float)
    if arr.size == d**2 - 1:
        identity_prob = 1 - np.sum(arr)
        arr = np.concatenate(([identity_prob], arr))
    elif arr.size != d**2:
        raise ValueError(
            f"PMF for PAULI_CHANNEL_1 must have length {d**2} or {d**2 - 1} "
            f"for dimension {d}, got {arr.size}."
        )
    arr = arr / np.sum(arr)
    cum_probs = np.cumsum(arr)
    r = np.random.rand(shots)
    indices = np.searchsorted(cum_probs, r)
    errors = np.zeros((shots, 2), dtype=np.int64)
    errors[:, 0] = indices // d
    errors[:, 1] = indices % d
    return errors


def sample_pauli_channel2(d: int, shots: int, pmf: ArrayLike) -> np.ndarray:
    arr = np.array(pmf, dtype=float)
    if arr.size == d**4 - 1:
        identity_prob = 1 - np.sum(arr)
        arr = np.concatenate(([identity_prob], arr))
    elif arr.size != d**4:
        raise ValueError(
            f"PMF for PAULI_CHANNEL_2 must have length {d**4} or {d**4 - 1} "
            f"for dimension {d}, got {arr.size}."
        )
    arr = arr / np.sum(arr)
    cum_probs = np.cumsum(arr)
    r = np.random.rand(shots)
    indices = np.searchsorted(cum_probs, r)
    errors = np.zeros((shots, 4), dtype=np.int64)
    i1 = indices // (d**2)
    errors[:, 0] = i1 // d
    errors[:, 1] = i1 % d
    i2 = indices % (d**2)
    errors[:, 2] = i2 // d
    errors[:, 3] = i2 % d
    return errors


def sample_heralded_erasure(
    d: int, shots: int, error_prob: float
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= error_prob <= 1.0):
        raise ValueError("Probability p must be in [0,1].")
    pauli = np.zeros((shots, 2), dtype=np.int64)
    erased = np.zeros(shots, dtype=np.int8)
    rnd = np.random.rand(shots)
    erase_mask = rnd >= (1.0 - error_prob)
    n_erase = int(np.sum(erase_mask))
    if n_erase:
        pauli_erase = np.column_stack(
            (
                np.random.randint(0, d, size=n_erase),
                np.random.randint(0, d, size=n_erase),
            )
        )
        pauli[erase_mask] = pauli_erase
        erased[erase_mask] = 1
    return pauli, erased


def build_noise_banks(
    circuit: "Circuit", shots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build pre-sampled noise arrays from a circuit's noise instructions."""
    d = circuit.dimension
    noise1_list: list[np.ndarray] = []
    noise2_list: list[np.ndarray] = []
    erasure_list: list[np.ndarray] = []
    measurement_list: list[np.ndarray] = []

    for instruction in circuit.operations:
        gate_id = instruction.gate_type
        gate_name = gate_id_to_name(gate_id)
        if not is_gate_noisy(gate_id):
            continue
        if is_gate_two_qubit(gate_id):
            for i in range(0, len(instruction.targets), 2):
                if i + 1 < len(instruction.targets):
                    if gate_name == "DEPOLARIZE2" and instruction.args:
                        noise2_list.append(
                            sample_depolarize2(d, shots, instruction.args[0])
                        )
                    elif (
                        gate_name == "PAULI_CHANNEL_2"
                        and len(instruction.args) >= 15
                    ):
                        noise2_list.append(
                            sample_pauli_channel2(d, shots, instruction.args)
                        )
        else:
            for _ in instruction.targets:
                if gate_name == "X_ERROR" and instruction.args:
                    noise1_list.append(
                        sample_x_error(d, shots, instruction.args[0])
                    )
                elif gate_name == "Z_ERROR" and instruction.args:
                    noise1_list.append(
                        sample_z_error(d, shots, instruction.args[0])
                    )
                elif gate_name == "Y_ERROR" and instruction.args:
                    noise1_list.append(
                        sample_y_error(d, shots, instruction.args[0])
                    )
                elif gate_name == "DEPOLARIZE1" and instruction.args:
                    noise1_list.append(
                        sample_depolarize1(d, shots, instruction.args[0])
                    )
                elif (
                    gate_name == "PAULI_CHANNEL_1"
                    and len(instruction.args) >= 3
                ):
                    noise1_list.append(
                        sample_pauli_channel1(d, shots, instruction.args[:3])
                    )
                elif gate_name == "HERALDED_ERASURE" and instruction.args:
                    pauli, locations = sample_heralded_erasure(
                        d, shots, instruction.args[0]
                    )
                    noise1_list.append(pauli)
                    erasure_list.append(locations)
                elif gate_name in ("M", "M_X", "MR", "MR_X"):
                    probability = (
                        instruction.args[0] if instruction.args else 0.0
                    )
                    measurement_list.append(
                        sample_x_error(d, shots, probability)
                    )

    noise1 = (
        np.array(noise1_list, dtype=np.int64)
        if noise1_list
        else np.empty((0, shots, 2), dtype=np.int64)
    )
    noise2 = (
        np.array(noise2_list, dtype=np.int64)
        if noise2_list
        else np.empty((0, shots, 4), dtype=np.int64)
    )
    erasure = np.array(erasure_list, dtype=np.int8) if erasure_list else None
    measurement = (
        np.array(measurement_list, dtype=np.int64) if measurement_list else None
    )

    return noise1, noise2, erasure, measurement
