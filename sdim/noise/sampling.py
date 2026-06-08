"""Noise sampling functions.

Each sampler is written once against an array module ``xp`` and a Generator
``rng`` whose API (``rng.random``, ``rng.integers``) is shared by
``numpy.random.Generator`` and ``cupy.random.Generator``. They default to
numpy + a fresh Generator, so callers can use ``sample_x_error(d, shots, p)``;
passing ``xp=cupy`` plus a device Generator builds every array on-device with
no host round-trip.

All channels draw ``shots`` values and select with ``xp.where`` rather than
counting the error mask first. Counting (``int(mask.sum())``) would force a
device->host sync on cupy; the extra host draws are negligible next to the
simulation.
"""

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


def _check_prob(p: float) -> None:
    if not (0.0 <= p <= 1.0):
        raise ValueError("Probability p must be in [0,1].")


def _resolve_rng(xp, rng):
    """Default to a fresh numpy Generator; require one for other backends."""
    if rng is not None:
        return rng
    if xp is np:
        return np.random.default_rng()
    raise ValueError("A Generator must be provided for non-numpy backends.")


def _errors_table(d: int, xp):
    """Non-identity Weyl labels ``(x, z)`` on the active backend."""
    errs = [
        (x, z) for x in range(d) for z in range(d) if not (x == 0 and z == 0)
    ]
    return xp.asarray(np.array(errs, dtype=np.int64))


def _pmf_cumsum(d: int, pmf, size: int, xp):
    """Normalize a Pauli-channel PMF (host) and return its CDF on device.

    The PMF prep (``d^2`` or ``d^4`` entries) is tiny and stays on host;
    only the per-shot inverse-CDF sampling runs on device.
    """
    arr = np.array(pmf, dtype=float)
    if arr.size == size - 1:  # identity probability implied
        arr = np.concatenate(([1.0 - arr.sum()], arr))
    elif arr.size != size:
        raise ValueError(
            f"PMF must have length {size} or {size - 1} for dimension {d}, "
            f"got {arr.size}."
        )
    arr = arr / arr.sum()
    return xp.asarray(np.cumsum(arr))


def _sample_single_pauli(d, shots, p, *, xp, rng, want_x, want_z, tie):
    """X/Z/Y single-qudit error: nonzero value(s) with probability ``p``.

    Shared by :func:`sample_x_error`, :func:`sample_z_error`,
    :func:`sample_y_error` (which differ only in which axes get the value and
    whether they are tied) and the measurement-flip channel.
    """
    _check_prob(p)
    noise = xp.zeros((shots, 2), dtype=xp.int64)
    if p <= 0.0:
        return noise
    mask = rng.random(size=shots) < p
    vals = rng.integers(1, d, size=shots)
    if want_x:
        noise[:, 0] = xp.where(mask, vals, 0)
    if want_z:
        # tie=True (Y error) reuses the same value so X and Z match.
        zvals = vals if tie else rng.integers(1, d, size=shots)
        noise[:, 1] = xp.where(mask, zvals, 0)
    return noise


def sample_x_error(d, shots, error_prob, *, xp=np, rng=None):
    return _sample_single_pauli(
        d,
        shots,
        error_prob,
        xp=xp,
        rng=_resolve_rng(xp, rng),
        want_x=True,
        want_z=False,
        tie=False,
    )


def sample_z_error(d, shots, error_prob, *, xp=np, rng=None):
    return _sample_single_pauli(
        d,
        shots,
        error_prob,
        xp=xp,
        rng=_resolve_rng(xp, rng),
        want_x=False,
        want_z=True,
        tie=False,
    )


def sample_y_error(d, shots, error_prob, *, xp=np, rng=None):
    return _sample_single_pauli(
        d,
        shots,
        error_prob,
        xp=xp,
        rng=_resolve_rng(xp, rng),
        want_x=True,
        want_z=True,
        tie=True,
    )


def sample_depolarize1(d, shots, error_prob, *, xp=np, rng=None):
    """Depolarizing: uniform non-identity single-qudit Pauli with prob ``p``."""
    _check_prob(error_prob)
    rng = _resolve_rng(xp, rng)
    noise = xp.zeros((shots, 2), dtype=xp.int64)
    if error_prob <= 0.0:
        return noise
    errors = _errors_table(d, xp)
    mask = rng.random(size=shots) < error_prob
    picked = errors[rng.integers(0, int(errors.shape[0]), size=shots)]
    noise[:, 0] = xp.where(mask, picked[:, 0], 0)
    noise[:, 1] = xp.where(mask, picked[:, 1], 0)
    return noise


def sample_depolarize2(d, shots, error_prob, *, xp=np, rng=None):
    """Standard two-qudit depolarizing: uniform over the d**4 - 1 non-identity
    two-qudit Paulis (includes the weight-1 terms P (x) I and I (x) P)."""
    _check_prob(error_prob)
    rng = _resolve_rng(xp, rng)
    noise = xp.zeros((shots, 4), dtype=xp.int64)
    if error_prob <= 0.0:
        return noise
    mask = rng.random(size=shots) < error_prob
    idx = rng.integers(0, d**4 - 1, size=shots) + 1
    noise[:, 0] = xp.where(mask, idx // d**3, 0)
    noise[:, 1] = xp.where(mask, (idx // d**2) % d, 0)
    noise[:, 2] = xp.where(mask, (idx // d) % d, 0)
    noise[:, 3] = xp.where(mask, idx % d, 0)
    return noise


def sample_pauli_channel1(d, shots, pmf: ArrayLike, *, xp=np, rng=None):
    """Inverse-CDF sample one Pauli per shot from an arbitrary PMF."""
    rng = _resolve_rng(xp, rng)
    cum = _pmf_cumsum(d, pmf, d**2, xp)
    idx = xp.searchsorted(cum, rng.random(size=shots))
    idx = xp.minimum(idx, d**2 - 1)  # guard the fp boundary at CDF=1
    out = xp.zeros((shots, 2), dtype=xp.int64)
    out[:, 0] = idx // d
    out[:, 1] = idx % d
    return out


def sample_pauli_channel2(d, shots, pmf: ArrayLike, *, xp=np, rng=None):
    """Inverse-CDF sample a two-qudit Pauli per shot from a PMF."""
    rng = _resolve_rng(xp, rng)
    cum = _pmf_cumsum(d, pmf, d**4, xp)
    idx = xp.searchsorted(cum, rng.random(size=shots))
    idx = xp.minimum(idx, d**4 - 1)
    out = xp.zeros((shots, 4), dtype=xp.int64)
    i1 = idx // (d**2)
    i2 = idx % (d**2)
    out[:, 0] = i1 // d
    out[:, 1] = i1 % d
    out[:, 2] = i2 // d
    out[:, 3] = i2 % d
    return out


def sample_heralded_erasure(d, shots, error_prob, *, xp=np, rng=None):
    """With probability ``p``, flag erasure and apply a uniform Pauli.

    The erasure Pauli is drawn from all ``d^2`` Weyl labels (identity
    included).
    """
    _check_prob(error_prob)
    rng = _resolve_rng(xp, rng)
    pauli = xp.zeros((shots, 2), dtype=xp.int64)
    if error_prob <= 0.0:
        return pauli, xp.zeros(shots, dtype=xp.int8)
    mask = rng.random(size=shots) >= (1.0 - error_prob)
    px = rng.integers(0, d, size=shots)
    pz = rng.integers(0, d, size=shots)
    pauli[:, 0] = xp.where(mask, px, 0)
    pauli[:, 1] = xp.where(mask, pz, 0)
    return pauli, mask.astype(xp.int8)


_MEASUREMENT_GATES = ("M", "M_X", "MR", "MR_X")


def sample_channel(gate_name, d, shots, args, *, xp=np, rng=None):
    """Draw one gate's inline noise. The single per-gate channel dispatch,
    shared by the streaming frame simulator and :func:`build_noise_banks`.

    Returns ``(noise, erased)``:
      ``noise``  -- ``(shots, 2)`` single-qudit or ``(shots, 4)`` two-qudit
                    Pauli to add to the frame, or ``None`` if the gate carries
                    no usable args.
      ``erased`` -- ``(shots,)`` int8 erasure flags for ``HERALDED_ERASURE``,
                    else ``None``.
    Measurement gates always return their ``(shots, 2)`` flip noise (zeros when
    p=0), so the caller uses the X component.
    """
    rng = _resolve_rng(xp, rng)
    n = len(args)
    if gate_name == "DEPOLARIZE2":
        if not n:
            return None, None
        return sample_depolarize2(d, shots, args[0], xp=xp, rng=rng), None
    if gate_name == "PAULI_CHANNEL_2":
        if n < 15:
            return None, None
        return sample_pauli_channel2(d, shots, args, xp=xp, rng=rng), None
    if gate_name == "X_ERROR":
        if not n:
            return None, None
        return sample_x_error(d, shots, args[0], xp=xp, rng=rng), None
    if gate_name == "Z_ERROR":
        if not n:
            return None, None
        return sample_z_error(d, shots, args[0], xp=xp, rng=rng), None
    if gate_name == "Y_ERROR":
        if not n:
            return None, None
        return sample_y_error(d, shots, args[0], xp=xp, rng=rng), None
    if gate_name == "DEPOLARIZE1":
        if not n:
            return None, None
        return sample_depolarize1(d, shots, args[0], xp=xp, rng=rng), None
    if gate_name == "PAULI_CHANNEL_1":
        if n < 3:
            return None, None
        return sample_pauli_channel1(d, shots, args, xp=xp, rng=rng), None
    if gate_name == "HERALDED_ERASURE":
        if not n:
            return None, None
        return sample_heralded_erasure(d, shots, args[0], xp=xp, rng=rng)
    if gate_name in _MEASUREMENT_GATES:
        p = args[0] if n else 0.0
        return sample_x_error(d, shots, p, xp=xp, rng=rng), None
    return None, None


def build_noise_banks(circuit: "Circuit", shots: int, *, xp=np, rng=None):
    """Build pre-sampled noise arrays from a circuit's noise instructions.

    Banks are returned as ``xp`` arrays, already resident on that backend.
    ``rng`` defaults to a fresh numpy Generator; for cupy pass the device
    Generator so entropy is drawn on-device with no host->device copy.

    The streaming frame simulator draws the same noise inline via
    :func:`sample_channel`; this banked builder remains for direct callers.
    """
    rng = _resolve_rng(xp, rng)
    d = circuit.dimension
    noise1_list: list = []
    noise2_list: list = []
    erasure_list: list = []
    measurement_list: list = []

    for instruction in circuit.operations:
        gate_id = instruction.gate_type
        gate_name = gate_id_to_name(gate_id)
        if not is_gate_noisy(gate_id):
            continue
        args = instruction.args or []
        two_qubit = is_gate_two_qubit(gate_id)
        is_meas = gate_name in _MEASUREMENT_GATES
        if two_qubit:
            spots = sum(
                1
                for i in range(0, len(instruction.targets), 2)
                if i + 1 < len(instruction.targets)
            )
        else:
            spots = len(instruction.targets)
        for _ in range(spots):
            noise, erased = sample_channel(
                gate_name, d, shots, args, xp=xp, rng=rng
            )
            if is_meas:
                measurement_list.append(noise)
            elif noise is None:
                continue
            elif two_qubit:
                noise2_list.append(noise)
            else:
                noise1_list.append(noise)
                if erased is not None:
                    erasure_list.append(erased)

    noise1 = (
        xp.stack(noise1_list)
        if noise1_list
        else xp.empty((0, shots, 2), dtype=xp.int64)
    )
    noise2 = (
        xp.stack(noise2_list)
        if noise2_list
        else xp.empty((0, shots, 4), dtype=xp.int64)
    )
    erasure = xp.stack(erasure_list) if erasure_list else None
    measurement = xp.stack(measurement_list) if measurement_list else None

    return noise1, noise2, erasure, measurement
