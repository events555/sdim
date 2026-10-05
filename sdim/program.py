from __future__ import annotations

from typing import Tuple, Optional, Callable
from .circuit import CircuitInstruction, Circuit
from .tableau.tableau_composite import WeylTableau
from .tableau.tableau_prime import ExtendedTableau
from .tableau.tableau_gates import *
from .tableau.tableau_gates import _apply_pauli_powers
from ._jit import _kernel
from itertools import product, repeat
from sympy import isprime
from numba import prange
from numba.core import types
from numba.typed import Dict
import numpy as np
import ast
import contextlib
import copy
import gc
import math
import operator
import re 

# Gate function dictionary
GATE_FUNCTIONS: dict[int, Callable] = {
    0: apply_I,      # I gate
    1: apply_X,      # X gate
    2: apply_X_inv,   # X inverse gate
    3: apply_Z,      # Z gate
    4: apply_Z_inv,  # Z inverse gate
    5: apply_H,      # H gate
    6: apply_H_inv,  # H inverse gate
    7: apply_P,      # P gate
    8: apply_P_inv,  # P inverse gate
    9: apply_CNOT,   # CNOT gate
    10: apply_CNOT_inv,  # CNOT inverse gate
    11: apply_CZ,  # CZ gate
    12: apply_CZ_inv,  # CZ inverse gate
    13: apply_SWAP,  # SWAP gate
    14: apply_measure, # Measure gate in computational basis
    15: apply_measure_x, # Measure gate in X basis
    16: apply_reset, # Reset gate
    17: apply_single_qudit_noise, # Single qudit Pauli noise gate, skipped in the noiseless reference tableau shot of the frame sampler
    18: apply_two_qudit_noise, # 2 qudit Pauli noise gate, uniform non-identity with probability 'prob', or an explicit 'prob_dist' over the d**4 Paulis; skipped in the noiseless reference shot
    19: apply_I, # Generic detectors
    20: apply_I, # Logical operator detectors
    21: apply_I, # TICK, do nothing.
    22: apply_multiplication, # Multiplication gate
}

MEASUREMENT_DTYPE = np.dtype([
    ('qudit_index', np.int64),
    ('meas_round', np.int64),
    ('shot', np.int64),
    ('deterministic', np.bool_),
    ('measurement_value', np.int64)
])

noise_gate_indices = {17, 18}

# TODO: Clean up typing to expose the now hidden structure of detector_data
@dataclass
class DetectorData:
    detector_data : np.ndarray | None = None # Format is (unique_detector_function_index, label, arguments, is_logical)
    detector_functions : list = None
    total_measurements : int = 1
    num_detector_events : int = 0
    num_logical_operators : int = 0

@dataclass
class DetectorResults:
    detection_events : np.ndarray = None
    logical_operator_shifts : np.ndarray = None


@contextlib.contextmanager
def _gc_paused():
    """
    Pauses the cyclic garbage collector while millions of result objects are built.

    The objects hold no reference cycles, but each allocation counts toward the collector's
    thresholds, and the repeated full collections over a growing heap dominate the build time.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@dataclass
class NoiseModel:
    """
    Per-noise-gate parameters for sampling noise inside the frame simulation.

    `Program._build_ir(..., sample_noise=False)` returns one of these instead of a pre-sampled
    noise array, and `simulate_frame` draws each gate's Paulis from it while it runs, so noise
    takes O(shots) memory per gate instead of O(noise gates x shots) in total.

    Entry j describes the j-th noise gate (N1 or N2) of the IR, in circuit order:
        kind[j]: 0 = N1 'd', 1 = N1 'f', 2 = N1 'p', 3 = N2 with prob, 4 = N2 with prob_dist.
        mode[j]: 0 = never fires, 1 = fires on every shot, 2 = fires with probability p
            (sampled by geometric skipping with log_q[j] = log(1 - p)), 3 = prob_dist draw per shot.
        log_q[j]: log(1 - p) when mode[j] == 2.
        cdf_offset[j]: start of this gate's cumulative distribution in cdf_data (kind 4 only, else -1).
    cdf_data holds the normalized cumulative distributions over the d**4 two-qudit Paulis, each
    computed exactly as np.random.choice computes it.
    """
    kind: np.ndarray
    mode: np.ndarray
    log_q: np.ndarray
    cdf_offset: np.ndarray
    cdf_data: np.ndarray


# Noise kinds in NoiseModel.kind
_NOISE_N1_D = 0
_NOISE_N1_F = 1
_NOISE_N1_P = 2
_NOISE_N2_UNIFORM = 3
_NOISE_N2_DIST = 4

# Noise modes in NoiseModel.mode
_NOISE_NEVER = 0
_NOISE_ALWAYS = 1
_NOISE_GEOMETRIC = 2
_NOISE_PER_SHOT = 3

# Gate ids that change the Pauli frame (everything else is a no-op for the frame)
_FRAME_GATES = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22], dtype=np.int64)

# Throwaway generator used only to run numpy's own validation of prob_dist (size=0 draws nothing)
_PROB_DIST_VALIDATOR = np.random.RandomState(0)


def _event_probability_mode(probability: float) -> tuple[int, float]:
    """
    Turns a noise probability into (mode, log(1 - p)) for the frame kernel.

    The pre-sampled noise of Program._build_ir fires N1 when U >= 1 - p and N2 when U < p, for U
    uniform in [0, 1): an error with probability p.  Both fire on every shot for p >= 1 or
    p = NaN and never for p <= 0, and so does the sampled-noise kernel.
    """
    p = float(probability)
    if np.isnan(p) or p >= 1.0:
        return _NOISE_ALWAYS, 0.0
    if p <= 0.0:
        return _NOISE_NEVER, 0.0
    return _NOISE_GEOMETRIC, float(np.log1p(-p))


def _prob_dist_cdf(distribution, dimension: int) -> np.ndarray:
    """
    Validates an N2 prob_dist exactly as the sampled path does and returns its cumulative distribution.

    The checks run in the same order as before: the length, then the d**4 size limit, then
    np.random.choice's own checks (1-D, non-negative, no NaN, sums to 1).  The returned array is
    the normalized cumulative sum np.random.choice draws from, so searching it with a uniform
    draw gives the same distribution.
    """
    num_events = dimension ** 4
    if len(distribution) != num_events:
        raise ValueError(
            f"N2 prob_dist has length {len(distribution)} instead of the required {num_events}."
        )
    if num_events > 10 ** 7:
        raise ValueError(
            f"Listing all {dimension}**4 two-qudit Pauli operators would take too much memory. "
            "Use N2 with prob= and sdim.dem.DetectorErrorModel.from_circuit instead."
        )
    _PROB_DIST_VALIDATOR.choice(a=num_events, size=0, p=distribution)
    cdf = np.asarray(distribution, dtype=np.float64).cumsum()
    cdf /= cdf[-1]
    return cdf


# --------------------------------------------------------------------------
# Random numbers for the frame kernel
#
# The sampled-noise kernel draws from its own xoshiro256** generator.  _run_frame seeds it with
# one draw from NumPy's global generator, so np.random.seed makes the frame shifts and the detectors
# reproducible.  The reference shot's outcomes, like N1 noise on the tableau, come from Python's
# random module, so reproducing the measurement values needs random.seed as well.

_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)
_MASK32 = np.uint64(0xFFFFFFFF)
_TWO32 = np.uint64(0x100000000)
_INT63_MAX = np.int64(0x7FFFFFFFFFFFFFFF)


@_kernel
def _rng_seed(state, seed):
    """Fills the xoshiro256** state from a 64-bit seed with splitmix64."""
    x = seed
    for i in range(4):
        x = x + _SPLITMIX_GAMMA
        r = x
        r = (r ^ (r >> np.uint64(30))) * _SPLITMIX_M1
        r = (r ^ (r >> np.uint64(27))) * _SPLITMIX_M2
        state[i] = r ^ (r >> np.uint64(31))


@_kernel
def _rng_next(state):
    """Next 64-bit output of xoshiro256**."""
    s0 = state[0]
    s1 = state[1]
    s2 = state[2]
    s3 = state[3]
    m = s1 * np.uint64(5)
    result = ((m << np.uint64(7)) | (m >> np.uint64(57))) * np.uint64(9)
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = (s3 << np.uint64(45)) | (s3 >> np.uint64(19))
    state[0] = s0
    state[1] = s1
    state[2] = s2
    state[3] = s3
    return result


@_kernel
def _rng_double(state):
    """Uniform float in [0, 1) with 53 random bits."""
    return np.float64(_rng_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@_kernel
def _rng_below(state, n):
    """Exactly uniform integer in [0, n), for 1 <= n < 2**63."""
    if n <= 0xFFFFFFFF:
        # Lemire's multiply-and-reject on 32 random bits.
        nn = np.uint64(n)
        m = (_rng_next(state) >> np.uint64(32)) * nn
        low = m & _MASK32
        if low < nn:
            threshold = (_TWO32 - nn) % nn
            while low < threshold:
                m = (_rng_next(state) >> np.uint64(32)) * nn
                low = m & _MASK32
        return np.int64(m >> np.uint64(32))
    # Reject 63-bit draws at or above the largest multiple of n.
    limit = (_INT63_MAX // n) * n
    while True:
        r = np.int64(_rng_next(state) >> np.uint64(1))
        if r < limit:
            return r % n


@_kernel
def _rng_fill_below(state, n, row, width):
    """
    row[0:width] = independent uniform integers in [0, n), for 1 <= n < 2**32.

    Same method as _rng_below (Lemire's multiply-and-reject on the top 32 bits of each output),
    with the generator state kept in registers for the whole row.
    """
    s0 = state[0]
    s1 = state[1]
    s2 = state[2]
    s3 = state[3]
    nn = np.uint64(n)
    threshold = (_TWO32 - nn) % nn
    for i in range(width):
        while True:
            m = s1 * np.uint64(5)
            out = ((m << np.uint64(7)) | (m >> np.uint64(57))) * np.uint64(9)
            t = s1 << np.uint64(17)
            s2 ^= s0
            s3 ^= s1
            s1 ^= s2
            s0 ^= s3
            s2 ^= t
            s3 = (s3 << np.uint64(45)) | (s3 >> np.uint64(19))
            prod = (out >> np.uint64(32)) * nn
            if (prod & _MASK32) >= threshold:
                break
        row[i] = np.int64(prod >> np.uint64(32))
    state[0] = s0
    state[1] = s1
    state[2] = s2
    state[3] = s3


@_kernel
def _rng_next_hit(state, log_q, s, limit):
    """
    Next shot after s, in [s + 1, limit), on which an event of probability p = 1 - exp(log_q)
    happens, or limit if there is none.  The gap is geometric, which is the same as an
    independent Bernoulli(p) trial on every shot.
    """
    u = 1.0 - _rng_double(state)  # (0, 1]
    gap = np.log(u) / log_q
    if gap >= limit - 1 - s:
        return limit
    return s + 1 + np.int64(gap)


# --------------------------------------------------------------------------
# Frame kernel
#
# Frame values are kept reduced to [0, d) after every gate, so no intermediate value exceeds
# 2 * d (or d**2 for MUL) and nothing can overflow int64 for d < 2**31.
#
# Z-frame liveness: the Z part of the frame only matters where it can still flow into the X part
# (through H, H_INV or M_X) before M or RESET overwrites it with a random value.  M_X records the
# Z part, leaves it unchanged, and replaces the X part with a random value instead, since the
# qudit is left in an X eigenstate.  A backward
# pass over the ops marks which Z updates are needed; the others are skipped.  Every X update and
# every measurement record is always computed.

_Z_A = 1  # the op's Z update of its first qudit is needed (for M / M_X / RESET: its new random Z)
_Z_B = 2  # the op's Z update of its second qudit is needed


@_kernel
def _z_liveness(op, qa, qb, flags, live):
    """
    Fills flags (zeros, one per op) with the _Z_A/_Z_B bits of the Z updates whose results can
    reach the X frame, and live (False, one per qudit) with which qudits' initial random Z can.
    The caller allocates both, so the kernel does not need numba's array constructors.
    """
    for i in range(op.shape[0] - 1, -1, -1):
        g = op[i]
        a = qa[i]
        b = qb[i]
        if g == 9 or g == 10:  # CNOT(_INV): z[a] -/+= z[b]
            if live[a]:
                flags[i] = _Z_A
                live[b] = True
        elif g == 11 or g == 12 or g == 18:  # CZ(_INV), N2: z[a] and z[b] each updated from x or noise
            f = 0
            if live[a]:
                f |= _Z_A
            if live[b]:
                f |= _Z_B
            flags[i] = f
        elif g == 7 or g == 8 or g == 17 or g == 22:  # P(_INV), N1, MUL: z[a] updated in place
            if live[a]:
                flags[i] = _Z_A
        elif g == 5 or g == 6:  # H(_INV): x[a] takes z[a], z[a] takes x[a]
            if live[a]:
                flags[i] = _Z_A
            live[a] = True
        elif g == 13:  # SWAP moves both rows
            t = live[a]
            live[a] = live[b]
            live[b] = t
            flags[i] = _Z_A | _Z_B
        elif g == 14 or g == 16:  # M, RESET: z[a] replaced by a random value
            if live[a]:
                flags[i] = _Z_A
            live[a] = False
        elif g == 15:  # M_X: records z[a] and keeps it; x[a] becomes random
            live[a] = True


@_kernel
def _add_mod(row, values, d, width):
    for s in range(width):
        v = row[s] + values[s]
        if v >= d:
            v -= d
        row[s] = v


@_kernel
def _sub_mod(row, values, d, width):
    for s in range(width):
        v = row[s] - values[s]
        if v < 0:
            v += d
        row[s] = v


@_kernel
def _add_injected_noise(x, z, a, b, f, two_qudit, noise, k, d, width):
    """
    Adds row k of an injected noise array to the frames: columns (x_a, z_a) for N1 and
    (x_a, z_a, x_b, z_b) for N2, in that order, skipping Z updates that are not live.
    """
    for c in range(4 if two_qudit else 2):
        q = a if c < 2 else b
        if c % 2 == 0:
            row = x[q]
        elif f & (_Z_A if c == 1 else _Z_B):
            row = z[q]
        else:
            continue
        for s in range(width):
            w = noise[k, s, c]
            if w < 0 or w >= d:  # the division is slow, and injected noise is normally in range
                w %= d
            v = row[s] + w
            if v >= d:
                v -= d
            row[s] = v


@_kernel
def _add_one(row, s, value, d):
    v = row[s] + value
    if v >= d:
        v -= d
    row[s] = v


@_kernel
def _apply_sampled_pauli(kind, a, b, s, za_live, zb_live, x, z, d, rng):
    """Draws the non-identity Pauli of an N1 / N2-with-prob gate and applies it on shot s."""
    if kind == _NOISE_N1_D:
        r = 1 + _rng_below(rng, d * d - 1)  # uniform non-identity (a, b), r = a + d * b
        _add_one(x[a], s, r % d, d)
        if za_live:
            _add_one(z[a], s, r // d, d)
    elif kind == _NOISE_N1_F:
        _add_one(x[a], s, 1 + _rng_below(rng, d - 1), d)
    elif kind == _NOISE_N1_P:
        v = 1 + _rng_below(rng, d - 1)
        if za_live:
            _add_one(z[a], s, v, d)
    else:
        # N2 with prob: uniform non-identity (x1, z1, x2, z2), redrawing the identity.
        while True:
            x1 = _rng_below(rng, d)
            z1 = _rng_below(rng, d)
            x2 = _rng_below(rng, d)
            z2 = _rng_below(rng, d)
            if x1 != 0 or z1 != 0 or x2 != 0 or z2 != 0:
                break
        _add_one(x[a], s, x1, d)
        if za_live:
            _add_one(z[a], s, z1, d)
        _add_one(x[b], s, x2, d)
        if zb_live:
            _add_one(z[b], s, z2, d)


@_kernel
def _apply_lazy_noise(k, a, b, f, x, z, d, width, col0, shots, kinds, modes, log_q, next_hit,
                      cdf_offset, cdf_data, cdf_len, rng):
    """Samples and applies noise gate k (on qudits a and b) to the shots col0..col0+width-1."""
    kind = kinds[k]
    mode = modes[k]
    za_live = (f & _Z_A) != 0
    zb_live = (f & _Z_B) != 0
    if mode == _NOISE_PER_SHOT:
        # N2 with prob_dist: one categorical draw per shot, like np.random.choice.
        off = cdf_offset[k]
        first = cdf_data[off]
        for s in range(width):
            u = _rng_double(rng)
            if u < first:
                continue  # identity
            lo = 1
            hi = cdf_len - 1
            while lo < hi:  # smallest index with cdf > u
                mid = (lo + hi) // 2
                if cdf_data[off + mid] > u:
                    hi = mid
                else:
                    lo = mid + 1
            idx = lo
            z2 = idx % d
            idx //= d
            x2 = idx % d
            idx //= d
            z1 = idx % d
            x1 = idx // d
            _add_one(x[a], s, x1, d)
            if za_live:
                _add_one(z[a], s, z1, d)
            _add_one(x[b], s, x2, d)
            if zb_live:
                _add_one(z[b], s, z2, d)
    elif mode == _NOISE_GEOMETRIC or mode == _NOISE_ALWAYS:
        # The gate fires on every shot (ALWAYS), or on the shots next_hit[k], then the next hits
        # drawn by geometric skipping (GEOMETRIC); next_hit[k] is over all blocks.  One call site
        # for _apply_sampled_pauli keeps the compiled kernel small.
        geometric = mode == _NOISE_GEOMETRIC
        end = col0 + width
        h = next_hit[k] if geometric else col0
        lq = log_q[k]
        while h < end:
            _apply_sampled_pauli(kind, a, b, h - col0, za_live, zb_live, x, z, d, rng)
            if geometric:
                h = _rng_next_hit(rng, lq, h, shots)
            else:
                h += 1
        if geometric:
            next_hit[k] = h


@_kernel
def _init_sampled_noise(rng, seed, modes, log_q, shots, next_hit):
    """
    Seeds the kernel's generator and draws the first shot on which each geometric noise gate
    fires; the gates then step through the blocks in order, so each gate needs O(1) work per
    block plus O(1) per error.
    """
    _rng_seed(rng, seed)
    # A typed value rather than the literal -1: numba compiles a separate copy of a callee for
    # every literal argument it is called with.
    before_first = np.int64(-1)
    for k in range(modes.shape[0]):
        if modes[k] == _NOISE_GEOMETRIC:
            next_hit[k] = _rng_next_hit(rng, log_q[k], before_first, shots)


@_kernel
def _frame_ops(op, qa, qb, pa, pb, zf, start, end, d, x, z, z_live0, block, records, shots,
               noise, kinds, modes, log_q, next_hit, cdf_offset, cdf_data, cdf_len, rng, lazy):
    """
    Applies frame ops start..end-1 to all the shots.

    pa/pb carry per-op data: the record row for M/M_X/RESET, the noise gate index for N1/N2, and
    (a mod d, a^-1 mod d) for MUL; zf holds the Z-liveness flags.  Frame entries stay in [0, d).
    Measurement-like ops write their outcome shifts into records[row, :].

    With lazy=True (sampled noise), the shots run in blocks of `block` shots, and x and z are
    scratch rows for one block: each block starts from x = 0 and a fresh random Z on the qudits
    in z_live0, noise comes from the NoiseModel arrays (with next_hit and rng set up by
    _init_sampled_noise), and the kernel re-randomizes the frame after each measurement.
    With lazy=False (injected noise), x and z hold all the shots, noise comes from the injected
    array (indexed by shot), and the caller re-randomizes the frame after each measurement, which
    keeps NumPy's random stream identical to the previous sampler.

    Both paths share this one compiled function, so the first simulation compiles it only once.
    """
    col0 = np.int64(0)
    while col0 < shots:
        if lazy:
            width = min(block, shots - col0)
            x[:, :] = 0
            for q in range(z_live0.shape[0]):
                if z_live0[q]:
                    _rng_fill_below(rng, d, z[q], width)
        else:
            width = shots
        for i in range(start, end):
            g = op[i]
            a = qa[i]
            f = zf[i]
            if g == 9:  # CNOT
                b = qb[i]
                _add_mod(x[b], x[a], d, width)
                if f & _Z_A:
                    _sub_mod(z[a], z[b], d, width)
            elif g == 17 or g == 18:  # noise
                b = qb[i]
                k = pa[i]
                if lazy:
                    mode = modes[k]
                    # Fast path: most gates have no error in most blocks.
                    if mode == _NOISE_NEVER or (mode == _NOISE_GEOMETRIC and next_hit[k] >= col0 + width):
                        continue
                    _apply_lazy_noise(k, a, b, f, x, z, d, width, col0, shots, kinds, modes, log_q, next_hit,
                                      cdf_offset, cdf_data, cdf_len, rng)
                else:
                    _add_injected_noise(x, z, a, b, f, g == 18, noise, k, d, width)
            elif g == 15:  # M_X: the outcome is the Z part, which stays; the X part becomes random
                zr = z[a]
                rec = records[pa[i]]
                for s in range(width):
                    rec[col0 + s] = zr[s]
                if lazy:
                    _rng_fill_below(rng, d, x[a], width)
            elif g == 14 or g == 16:  # M, RESET
                xr = x[a]
                rec = records[pa[i]]
                for s in range(width):
                    rec[col0 + s] = xr[s]
                if g == 16:  # RESET corrects the outcome back to |0>
                    for s in range(width):
                        xr[s] = 0
                if lazy and (f & _Z_A):
                    _rng_fill_below(rng, d, z[a], width)
            elif g == 5:  # H
                xr = x[a]
                zr = z[a]
                if f & _Z_A:
                    for s in range(width):
                        t = xr[s]
                        v = zr[s]
                        xr[s] = d - v if v != 0 else 0
                        zr[s] = t
                else:
                    for s in range(width):
                        v = zr[s]
                        xr[s] = d - v if v != 0 else 0
            elif g == 6:  # H inverse
                xr = x[a]
                zr = z[a]
                if f & _Z_A:
                    for s in range(width):
                        t = xr[s]
                        xr[s] = zr[s]
                        zr[s] = d - t if t != 0 else 0
                else:
                    for s in range(width):
                        xr[s] = zr[s]
            elif g == 7:  # P
                if f & _Z_A:
                    _add_mod(z[a], x[a], d, width)
            elif g == 8:  # P inverse
                if f & _Z_A:
                    _sub_mod(z[a], x[a], d, width)
            elif g == 10:  # CNOT inverse
                b = qb[i]
                _sub_mod(x[b], x[a], d, width)
                if f & _Z_A:
                    _add_mod(z[a], z[b], d, width)
            elif g == 11:  # CZ
                b = qb[i]
                if f & _Z_B:
                    _add_mod(z[b], x[a], d, width)
                if f & _Z_A:
                    _add_mod(z[a], x[b], d, width)
            elif g == 12:  # CZ inverse
                b = qb[i]
                if f & _Z_B:
                    _sub_mod(z[b], x[a], d, width)
                if f & _Z_A:
                    _sub_mod(z[a], x[b], d, width)
            elif g == 13:  # SWAP
                b = qb[i]
                if a != b:
                    xa = x[a]
                    xb = x[b]
                    za = z[a]
                    zb = z[b]
                    for s in range(width):
                        t = xa[s]
                        xa[s] = xb[s]
                        xb[s] = t
                        t = za[s]
                        za[s] = zb[s]
                        zb[s] = t
            elif g == 22:  # MUL: X -> X^a, Z -> Z^(a^-1)
                ma = pa[i]
                xr = x[a]
                for s in range(width):
                    xr[s] = xr[s] * ma % d
                if f & _Z_A:
                    mi = pb[i]
                    zr = z[a]
                    for s in range(width):
                        zr[s] = zr[s] * mi % d

        col0 += width


def _frame_block_size(n_qudits: int, z_used: bool) -> int:
    """
    Shots per block in the sampled-noise kernel: keep the frame rows a block works on near 1 MiB
    (about one core's L2 cache).  Without any live Z update only the X frame is touched.
    """
    bytes_per_shot = (16 if z_used else 8) * max(int(n_qudits), 1)
    target = (1 << 20) // bytes_per_shot
    block = 64
    while block * 2 <= target and block < 4096:
        block *= 2
    return block


@dataclass
class _FrameRun:
    """Raw output of the frame simulation, before it is turned into result objects."""
    records: np.ndarray            # (num_records, shots) int32: x mod d at each M / M_X / RESET, in circuit order
    record_qudit: np.ndarray       # qudit of each record
    record_round: np.ndarray       # measurement round of each record on its qudit
    detector_results: DetectorResults


def _run_frame(ir_array: np.ndarray, reference_results: np.ndarray, n_qudits: int, dimension: int,
               extra_shots: int, noise_array: np.ndarray | None, detector_info: DetectorData | None,
               noise_model: NoiseModel | None) -> _FrameRun:
    """Runs the Pauli frame simulation and evaluates the detectors (see simulate_frame)."""
    d = int(dimension)
    n_qudits = int(n_qudits)
    shots = int(extra_shots)
    if d < 1:
        raise ValueError(f"Dimension must be positive, not {d}.")
    if d >= 2 ** 31:
        raise ValueError(f"Dimension must be below 2**31, not {d}.")

    gate_ids = np.asarray(ir_array['gate_id'], dtype=np.int64)
    keep = np.isin(gate_ids, _FRAME_GATES)
    op = np.ascontiguousarray(gate_ids[keep])
    qa = np.ascontiguousarray(np.asarray(ir_array['qudit_index'], dtype=np.int64)[keep])
    qb = np.ascontiguousarray(np.asarray(ir_array['target_index'], dtype=np.int64)[keep])
    scalars = np.asarray(ir_array['scalar'], dtype=np.int64)[keep]
    num_ops = op.shape[0]

    # Qudit indices follow NumPy's indexing rules (negative ones count from the end), and must be
    # checked here because the kernel does not bounds-check.
    two_qudit = np.isin(op, (9, 10, 11, 12, 13, 18))
    for name, idx, used in (("qudit", qa, np.ones(num_ops, dtype=bool)), ("target", qb, two_qudit)):
        bad = used & ((idx < -n_qudits) | (idx >= n_qudits))
        if bad.any():
            raise IndexError(f"Gate {int(op[bad][0])} has {name} index {int(idx[bad][0])} "
                             f"outside a frame of {n_qudits} qudits.")
        idx[used & (idx < 0)] += n_qudits
    qb[~two_qudit] = qa[~two_qudit]

    pa = np.zeros(num_ops, dtype=np.int64)
    pb = np.zeros(num_ops, dtype=np.int64)

    # Measurement records: one row per M / M_X / RESET in circuit order.
    is_record = (op == 14) | (op == 15) | (op == 16)
    record_ops = np.flatnonzero(is_record)
    num_records = record_ops.shape[0]
    pa[record_ops] = np.arange(num_records, dtype=np.int64)
    record_qudit = qa[record_ops]
    record_round = np.zeros(num_records, dtype=np.int64)
    counts = np.zeros(n_qudits, dtype=np.int64)
    for r, q in enumerate(record_qudit.tolist()):
        record_round[r] = counts[q]
        counts[q] += 1
    max_rounds = reference_results.shape[1] if reference_results.ndim > 1 else 0
    if num_records and int(record_round.max()) >= max_rounds:
        r = int(np.argmax(record_round >= max_rounds))
        raise IndexError(f"Qudit {int(record_qudit[r])} is measured {int(counts[record_qudit[r]])} times, "
                         f"but the reference results hold only {max_rounds} rounds.")
    # Shift rows seen by detectors: the M and M_X records, in order.
    measurement_rows = pa[record_ops[op[record_ops] != 16]]

    # MUL multiplies X by a and Z by a^-1 mod d.
    for i in np.flatnonzero(op == 22).tolist():
        scalar = int(scalars[i]) % d
        pa[i] = scalar
        pb[i] = pow(scalar, -1, d)

    # Noise gates, numbered in circuit order.
    noise_ops = np.flatnonzero((op == 17) | (op == 18))
    num_noise = noise_ops.shape[0]
    pa[noise_ops] = np.arange(num_noise, dtype=np.int64)

    zf = np.zeros(num_ops, dtype=np.int64)
    z_live0 = np.zeros(n_qudits, dtype=np.bool_)
    _z_liveness(op, qa, qb, zf, z_live0)
    # Records hold values in [0, d) with d < 2**31, so int32 is exact and halves their memory.
    records = np.empty((num_records, shots), dtype=np.int32)

    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    if noise_array is None:
        if num_noise and noise_model is None:
            raise ValueError("The circuit has noise gates, so simulate_frame needs a noise_array or a noise_model.")
        if noise_model is not None and num_noise:
            if len(noise_model.kind) < num_noise:
                raise IndexError(f"The noise model describes {len(noise_model.kind)} noise gates, "
                                 f"but the circuit has {num_noise}.")
            if d < 2:
                raise ValueError("Noise needs a dimension of at least 2.")
            kinds = np.ascontiguousarray(noise_model.kind, dtype=np.int64)
            modes = np.ascontiguousarray(noise_model.mode, dtype=np.int64)
            log_q = np.ascontiguousarray(noise_model.log_q, dtype=np.float64)
            cdf_offset = np.ascontiguousarray(noise_model.cdf_offset, dtype=np.int64)
            cdf_data = np.ascontiguousarray(noise_model.cdf_data, dtype=np.float64)
            cdf_len = d ** 4 if cdf_data.shape[0] else 0
        else:
            kinds, modes, log_q, cdf_offset, cdf_data, cdf_len = empty_i, empty_i, empty_f, empty_i, empty_f, 0
        # One draw from NumPy's global generator seeds the kernel's generator.
        seed = np.uint64(np.random.randint(0, 2 ** 64, dtype=np.uint64))
        if shots:
            block = _frame_block_size(n_qudits, bool(zf.any() or z_live0.any()))
            width = min(block, shots)
            x = np.zeros((n_qudits, width), dtype=np.int64)
            z = np.zeros((n_qudits, width), dtype=np.int64)
            rng = np.zeros(4, dtype=np.uint64)
            next_hit = np.zeros(kinds.shape[0], dtype=np.int64)
            _init_sampled_noise(rng, seed, modes, log_q, shots, next_hit)
            _frame_ops(op, qa, qb, pa, pb, zf, 0, num_ops, d, x, z, z_live0, block, records, shots,
                       np.zeros((0, 0, 4), dtype=np.int64), kinds, modes, log_q, next_hit,
                       cdf_offset, cdf_data, cdf_len, rng, True)
    else:
        # Injected noise: same random draws, in the same order, as the previous sampler.
        if num_noise:
            noise = np.asarray(noise_array)
            if not (np.issubdtype(noise.dtype, np.integer) or noise.dtype == np.bool_):
                raise TypeError(f"noise_array must hold integers, not {noise.dtype}.")
            if noise.ndim != 3:
                raise ValueError(f"noise_array must have shape (num_noise_gates, shots, 4), not {noise.shape}.")
            if noise.shape[0] < num_noise:
                raise IndexError(f"noise_array has {noise.shape[0]} rows, but the circuit has {num_noise} noise gates.")
            needed = 4 if np.any(op[noise_ops] == 18) else 2
            if noise.shape[2] < needed:
                raise IndexError(f"noise_array has {noise.shape[2]} columns, but the noise gates need {needed}.")
            if noise.shape[1] != shots:
                noise = np.broadcast_to(noise, (noise.shape[0], shots, noise.shape[2]))
            noise = np.ascontiguousarray(noise, dtype=np.int64)
        else:
            noise = np.zeros((0, 0, 4), dtype=np.int64)
        rng = np.zeros(4, dtype=np.uint64)
        x = np.zeros((n_qudits, shots), dtype=np.int64)
        z = np.ascontiguousarray(np.random.randint(0, d, size=(n_qudits, shots)), dtype=np.int64)
        start = 0
        for pos in record_ops.tolist():
            _frame_ops(op, qa, qb, pa, pb, zf, start, pos + 1, d, x, z, z_live0, shots, records, shots,
                       noise, empty_i, empty_i, empty_f, empty_i, empty_i, empty_f, 0, rng, False)
            row = np.random.randint(0, d, size=shots)
            if op[pos] == 15:   # M_X is H_INV, M, then H: the random Z row ends up as -row in X
                x[qa[pos]] = (d - row) % d
            else:
                z[qa[pos]] = row
            start = pos + 1
        _frame_ops(op, qa, qb, pa, pb, zf, start, num_ops, d, x, z, z_live0, shots, records, shots,
                   noise, empty_i, empty_i, empty_f, empty_i, empty_i, empty_f, 0, rng, False)

    detector_results = _evaluate_detectors(gate_ids, records, measurement_rows, detector_info, shots)
    return _FrameRun(records, record_qudit, record_round, detector_results)


_FLOAT_EXACT_BOUND = 1 << 52


@_kernel
def _floor_mod_int64(values, d, out):
    """
    out[i] = values[i] % d with Python's sign rule, for d >= 1, without a hardware division.

    For |v| < 2**52, v and v * (1 / d) are within one rounding of exact, so the float quotient is
    off by at most one and the remainder needs at most one correction by d.  Larger values use
    the integer remainder.
    """
    inv = 1.0 / d
    for i in range(values.shape[0]):
        v = values[i]
        if -_FLOAT_EXACT_BOUND < v < _FLOAT_EXACT_BOUND:
            r = v - np.int64(np.floor(np.float64(v) * inv)) * d
            while r < 0:
                r += d
            while r >= d:
                r -= d
            out[i] = r
        else:
            out[i] = v % d


_INT64 = np.dtype(np.int64)


def _detector_mod(value, d: int):
    """
    Returns value % d, the last step of every compiled detector expression.  In the frame sampler,
    ring expressions are evaluated exactly mod d; other expressions are evaluated with NumPy int64
    arithmetic as before, which is exact as long as their intermediate values stay within int64.

    NumPy's int64 remainder costs a hardware division per entry, which dominates detector
    evaluation in the frame sampler, so plain int64 arrays go through _floor_mod_int64 instead.
    Every other input (Python ints, scalars, other dtypes) uses the % operator itself, so the
    result is the same as `value % d` for any input.
    """
    if (type(value) is np.ndarray and value.dtype == _INT64 and value.ndim > 0
            and type(d) is int and 0 < d < 2 ** 62):
        flat = np.ascontiguousarray(value)
        if flat.ndim > 1:
            flat = flat.reshape(-1)
        out = np.empty(flat.shape[0], dtype=_INT64)
        _floor_mod_int64(flat, d, out)
        # Detector values are rows, and reshaping costs as much as the remainder for a few shots.
        return out if value.ndim == 1 else out.reshape(value.shape)
    return value % d


def _pow_mod(base, exponent: int, d: int):
    """base ** exponent mod d for |base| < d and exponent >= 1, reducing every product mod d."""
    result = None
    while True:
        if exponent & 1:
            result = base if result is None else _detector_mod(result * base, d)
        exponent >>= 1
        if not exponent:
            return result
        base = _detector_mod(base * base, d)


# A measurement record reference in a detector or observable expression: rec indexed by an
# integer literal with an optional sign, written rec[-1], rec[0], rec[+1], rec[ - 1], rec [2], ...
# Brackets that do not index rec (a list literal [5, 7][1], ...) are left as they are.
_RECORD_REFERENCE = re.compile(r"\brec\s*\[\s*([+-]?)\s*(\d+)\s*\]")
# Any indexing of rec, to catch references that are not integer literals (rec[i], rec[-1 - 1], ...)
_REC_INDEXING = re.compile(r"\brec\s*\[")


def _resolve_record_references(source: str, num_records: int, name: str) -> tuple[str, list[int]]:
    """
    Resolves the measurement record references of a detector or observable expression.

    Records are the M and M_X outcomes, numbered 0, 1, ... in circuit order over the whole
    program (RESET does not add one).  rec[k] with k >= 0 is record k, and rec[-k] is the k-th
    most recent record before the detector, like Python indexing, so with n records so far
    -n <= k < n is required; anything else raises a ValueError naming the detector.  Only
    indexing of rec is a record reference; other brackets in the expression keep their meaning.

    Returns the expression with every reference rewritten as rec[j], where j is its position in
    the returned list of distinct absolute record indices (in order of first use).  Two
    references to the same record (rec[1] and rec[-1] with 2 records, ...) share one position.
    The compiled detector function is called with the shift rows of those records, in that order.
    """
    for match in _REC_INDEXING.finditer(source):
        if not _RECORD_REFERENCE.match(source, match.start()):
            raise ValueError(f"{name} indexes rec with something other than an integer: {source!r}. "
                             "Write record references as integer literals, like rec[-1] or rec[2].")
    arguments = []
    position = {}

    def resolve(match):
        k = int(match.group(1) + match.group(2))
        if not -num_records <= k < num_records:
            if num_records == 0:
                available = "no measurement (M or M_X) comes before it"
            else:
                available = (f"only {num_records} measurement{'s' if num_records != 1 else ''} (M or M_X) "
                             f"{'come' if num_records != 1 else 'comes'} before it, so the index must be "
                             f"in [{-num_records}, {num_records - 1}]")
            raise ValueError(f"{name} refers to rec[{k}], but {available}.")
        absolute = k + num_records if k < 0 else k
        j = position.get(absolute)
        if j is None:
            j = position[absolute] = len(arguments)
            arguments.append(absolute)
        return f"rec[{j}]"

    return _RECORD_REFERENCE.sub(resolve, source), arguments


def _records_needed(source: str) -> int | None:
    """
    The fewest measurement records before a detector with this expression for all of its record
    references to be in range (rec[k] needs k + 1 for k >= 0, and -k for k < 0), or None when
    it indexes rec with something other than an integer literal.
    """
    needed = 0
    for match in _REC_INDEXING.finditer(source):
        reference = _RECORD_REFERENCE.match(source, match.start())
        if reference is None:
            return None
        k = int(reference.group(1) + reference.group(2))
        needed = max(needed, -k if k < 0 else k + 1)
    return needed


def _detector_name(instruction: CircuitInstruction, index: int) -> str:
    """
    How errors name a DETECTOR or LOGICAL_OBSERVABLE: by its index among the detectors (or among
    the observables), followed by its label unless the label is empty, as in DETECTOR 3 or
    DETECTOR 6 (label 'parity').  A label such as 5 then cannot be mistaken for an index.
    """
    label = instruction.params.get('label', '')
    if isinstance(label, str) and label == '':
        return f"{instruction.name} {index}"
    return f"{instruction.name} {index} (label {label!r})"


def _check_record_references(circuits: list) -> None:
    """
    Checks the record references of every DETECTOR and LOGICAL_OBSERVABLE expression, as
    Program._build_ir does, without compiling anything: a ValueError names the first one that
    refers to a measurement that does not exist (yet) or indexes rec with a non-integer.
    The tableau simulation does not evaluate detectors, so it runs this check instead.
    Each distinct expression is parsed once, so repeated detectors cost a dictionary lookup.
    """
    seen_measurements = 0
    counts = {19: 0, 20: 0}
    needed_by_source = {}
    for circuit in circuits:
        for instruction in circuit.operations:
            gate_id = instruction.gate_id
            if gate_id == 14 or gate_id == 15:
                seen_measurements += 1
            elif gate_id == 19 or gate_id == 20:
                params = instruction.params
                if params and 'expr' in params:
                    source = str(params['expr'])
                    needed = needed_by_source.get(source, -1)
                    if needed == -1:
                        needed = needed_by_source[source] = _records_needed(source)
                    if needed is None or needed > seen_measurements:
                        # Raises the error that names the detector and the bad reference.
                        _resolve_record_references(source, seen_measurements,
                                                   _detector_name(instruction, counts[gate_id]))
                counts[gate_id] += 1


# --------------------------------------------------------------------------
# Ring expressions in the frame sampler
#
# The frame sampler evaluates detector expressions on int64 rows of record values in [0, d).  A sum
# of a few products already leaves int64 for d near 2**31, as do literals of 2**63 or more and
# powers of records at any d.  Ring expressions are built from integer literals, rec[j], +, -,
# unary + and -, *, ** by a non-negative literal and % by a literal that d divides, so they need
# their operands only mod d.  The frame sampler evaluates one that could leave int64 by calling its
# compiled function on _RingRows, which reduce operands mod d where needed (see _detector_values).

_INT64_MAX = 2 ** 63 - 1

# A sum of terms c * rec[j], c * c', rec[j] and c with c, c' < 10**4, with any signs.  Each term is
# below 10**4 * 2**31 < 2**45, and a source shorter than 2**17 characters has fewer than 2**16
# terms, so evaluating it stays inside int64.  The quantifiers are possessive, which is faster: no
# source of this shape needs backtracking.
_LINEAR_TERM = r"(?:[0-9]{1,4}+\s*+\*\s*+)?+(?:rec\[[0-9]++\]|[0-9]{1,4}+)"
_SMALL_LINEAR = re.compile(rf"[-+\s]*+{_LINEAR_TERM}(?:\s*+[-+][-+\s]*+{_LINEAR_TERM})*+\s*+")
# The same for literals of any size, and one of its terms, read as (c, c') for c * c', (c, '') for
# c * rec[j], ('', c) for c and ('', '') for rec[j].
_ANY_LINEAR_TERM = r"(?:[0-9]++\s*+\*\s*+)?+(?:rec\[[0-9]++\]|[0-9]++)"
_LINEAR = re.compile(rf"[-+\s]*+{_ANY_LINEAR_TERM}(?:\s*+[-+][-+\s]*+{_ANY_LINEAR_TERM})*+\s*+")
_LINEAR_TERMS = re.compile(r"(?:([0-9]++)\s*+\*\s*+)?+(?:rec\[[0-9]++\]|([0-9]++))")
# The tokens of ring expressions: rec[j], integer literals, + - * % ( ), white space and line
# continuations.  Besides arithmetic with the rows, they only make calls and empty tuples, such as
# rec[0](1) and (), which fail on _RingRows with _NotRing or a TypeError.  The first pattern is a
# faster one for the usual spelling: rec[j] as _resolve_record_references writes it and decimal
# literals.
_RING_TOKENS_FAST = re.compile(r"[-+*%()\s0-9]*+(?:rec\[[0-9]++\][-+*%()\s0-9]*+)*+")
_RING_TOKENS = re.compile(r"(?:[-+*%()\s]|\\[\r\n]|rec\s*+\[\s*+[-+]?+\s*+[0-9]++\s*+\]"
                          r"|(?:0[xX][0-9a-fA-F_]++|0[oO][0-7_]++|0[bB][01_]++|[0-9][0-9_]*+)(?![\w.]))*+")


class _NotRing(Exception):
    """An operation that _RingRows cannot follow (see _RingEvaluation)."""


class _RingEvaluation:
    """
    One call of a compiled detector function on _RingRows, for a dimension d below 2**31.

    `reduced` says whether an operand was reduced mod d, after which the rows hold values only mod
    d.  `other` says whether a % that is not a ring operation ran, which the rows follow in int64
    as the compiled function would.  A call that needs both raises _NotRing, as does an operand
    that is neither an int nor a _RingRow, or a power by anything but an int k >= 0.
    """
    __slots__ = ('d', 'reduced', 'other')

    def __init__(self, d: int):
        self.d = d
        self.reduced = False
        self.other = False

    def mark_reduced(self):
        if self.other:
            raise _NotRing
        self.reduced = True

    def term(self, x):
        """The (row or int, bound on its magnitude) of an operand."""
        if type(x) is _RingRow:
            return x.row, x.bound
        if type(x) is not int:
            raise _NotRing
        return (x, abs(x)) if abs(x) <= _INT64_MAX else self.reduce((x, abs(x)))

    def reduce(self, term):
        """The term mod d: as it is if its magnitude is below d, an int's residue of least magnitude, or the row mod d."""
        value, bound = term
        if bound < self.d:
            return term
        self.mark_reduced()
        if type(value) is int:
            r = value % self.d
            if r > self.d // 2:
                r -= self.d
            return r, abs(r)
        return _detector_mod(value, self.d), self.d - 1

    def ring(self, op, a, b):
        """op(a, b) for op in operator.add, sub and mul, on _RingRows and ints."""
        a, b = self.term(a), self.term(b)
        combine = operator.mul if op is operator.mul else operator.add
        # The right operand first: long sums nest to the left, so this reduces their terms rather
        # than the running sum.  Two reduced operands never leave int64, as (d - 1)**2 < 2**62.
        if combine(a[1], b[1]) > _INT64_MAX:
            b = self.reduce(b)
            if combine(a[1], b[1]) > _INT64_MAX:
                a = self.reduce(a)
        return _RingRow(op(a[0], b[0]), combine(a[1], b[1]), self)

    def power(self, base, k):
        """base ** k for a _RingRow base and an int k >= 0."""
        if type(k) is not int or k < 0:
            raise _NotRing
        term = self.term(base)
        bound = _power_bound(term[1], k)
        if bound is None:
            term = self.reduce(term)
            bound = _power_bound(term[1], k)
        if bound is None:
            self.mark_reduced()
            return _RingRow(_pow_mod(term[0], k, self.d), self.d - 1, self)
        # NumPy's integer power squares and multiplies, so no intermediate value is larger than
        # the bound on the result.
        return _RingRow(term[0] ** k, bound, self)

    def modulo(self, a, b):
        """a % b, where a or b is a _RingRow.  % by a nonzero int multiple of d is a ring operation."""
        if type(b) is int and b and b % self.d == 0:
            if abs(b) > _INT64_MAX:
                self.mark_reduced()
                return a  # a % b is a mod d
            return _RingRow(_detector_mod(a.row, b), abs(b) - 1, self)
        if self.reduced:
            raise _NotRing
        self.other = True
        a, b = self.term(a), self.term(b)
        # The compiled function gives these values again, with its own warnings (see _detector_values).
        with np.errstate(all='ignore'):
            row = a[0] % b[0]
        return _RingRow(row, max(b[1] - 1, 0), self)  # NumPy gives 0 for % 0


def _power_bound(bound: int, k: int):
    """A bound on |x ** k| for |x| <= bound and k >= 0, or None if it or k could leave int64."""
    if k > _INT64_MAX or (k >= 63 and bound > 1):
        return None
    power = bound ** k
    return power if power <= _INT64_MAX else None


def _ring_operator(op, reflected: bool):
    """The _RingRow method for op in operator.add, sub and mul, or its reflection."""
    combine = operator.mul if op is operator.mul else operator.add

    def method(self, other):
        if type(other) is _RingRow:
            bound, value = combine(self.bound, other.bound), other.row
        elif type(other) is int and -_INT64_MAX <= other <= _INT64_MAX:
            bound, value = combine(self.bound, abs(other)), other
        else:
            bound = None
        if bound is None or bound > _INT64_MAX:
            return self.evaluation.ring(op, other, self) if reflected else self.evaluation.ring(op, self, other)
        return _RingRow(op(value, self.row) if reflected else op(self.row, value), bound, self.evaluation)
    return method


class _RingRow:
    """
    The values of part of a detector expression, one int64 entry per shot, with a bound on their
    magnitude over all record rows in [0, d).  Its operators reduce operands mod d only where a
    value could leave int64, so the row holds the part's int64 values until an operand is reduced,
    and its values mod d after.
    """
    __slots__ = ('row', 'bound', 'evaluation')

    def __init__(self, row, bound, evaluation):
        self.row = row
        self.bound = bound
        self.evaluation = evaluation

    __add__ = _ring_operator(operator.add, False)
    __radd__ = _ring_operator(operator.add, True)
    __sub__ = _ring_operator(operator.sub, False)
    __rsub__ = _ring_operator(operator.sub, True)
    __mul__ = _ring_operator(operator.mul, False)
    __rmul__ = _ring_operator(operator.mul, True)

    def __neg__(self):
        return _RingRow(-self.row, self.bound, self.evaluation)

    def __pos__(self):
        return self

    def __pow__(self, k):
        # Only compiled functions call this, and their ring exponents are literals of at most 2**12
        # (see _large_powers): a larger k was computed, so the source is not a ring expression.
        if type(k) is int and k > 2 ** 12:
            raise _NotRing
        return self.evaluation.power(self, k)

    def __mod__(self, other):
        return self.evaluation.modulo(self, other)

    def __rmod__(self, other):
        return self.evaluation.modulo(other, self)


def _int_literal(node):
    """The value of an integer literal (not a bool) with an optional sign, or None."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return sign * node.value
    return None


# The nodes of a ring expression's syntax tree.
_RING_NODES = frozenset([ast.Expression, ast.Constant, ast.Subscript, ast.Name, ast.Load, ast.UnaryOp, ast.UAdd,
                         ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.Pow])


def _ring_tree(source: str, d: int):
    """
    The syntax tree of a source made of _RING_TOKENS if it is a ring expression, else None: no
    call or tuple, every % by an integer literal that d divides and every ** by a non-negative one.
    _RingRows see only the values of these operands, which may come from arithmetic on literals
    instead.
    """
    try:
        tree = ast.parse('(' + source + ')', mode='eval')  # in parentheses, as compiled (see _compile_detector)
    except (SyntaxError, RecursionError):
        return None
    for node in ast.walk(tree):
        if type(node) not in _RING_NODES:
            return None
        if type(node) is ast.BinOp and type(node.op) in (ast.Mod, ast.Pow):
            k = _int_literal(node.right)
            if k is None or (k == 0 or k % d if isinstance(node.op, ast.Mod) else k < 0):
                return None
    return tree


# The exponent of a ** when it is an integer literal, past white space, line continuations, ( and +.
# In a source that compiles, a backslash there can only start a line continuation.
_EXPONENT = re.compile(r"\*\*[(+\s\\]*+(0[xXoObB][0-9a-fA-F_]++|[0-9][0-9_]*+)")


def _large_powers(source: str) -> bool:
    """
    Whether the literal exponents in `source` are large enough (their product above 2**12, or one
    with more than 18 digits) for a power of a constant to take long to compute, so that the frame
    sampler evaluates it from its syntax tree (see _ring_tree_values).
    """
    product = 1
    for exponent in _EXPONENT.findall(source):
        if len(exponent) > 18:
            return True
        product *= max(int(exponent, 0), 1)
    return product > 2 ** 12


# A decimal number other than 0.
_NONZERO_NUMBER = re.compile(r"[1-9][0-9]*+")


def _could_leave_int64(source: str, d: int) -> bool:
    """
    Whether `source` is made of _RING_TOKENS and its compiled function could leave int64 on records
    in [0, d), as far as cheap bounds tell; _RingRows decide the rest exactly.

    Most detectors are short sums of records with small coefficients (see _SMALL_LINEAR).  Every
    value of another sum of terms c * rec[j], c * c', rec[j] and c is at most the sum over its
    terms of |c| * max(d - 1, 1), |c * c'|, max(d - 1, 1) and |c|.  In any other source of rec[j],
    decimal literals, + - * % and parentheses, as |a % b| < |b|, every value is at most the sum of
    max(d - 1, 1) per record and of its numbers other than 0 (literals, and record indices, which
    only make the bound larger) if it has no *, and else their product times 2 for each + or -.
    Literals are only read from sources of at most 4000 characters, which cannot hold one too long
    for int().
    """
    if len(source) < 2 ** 17 and _SMALL_LINEAR.fullmatch(source):
        return False
    if len(source) <= 4000 and _LINEAR.fullmatch(source):
        record = max(d - 1, 1)
        return sum((int(c) if c else 1) * (int(k) if k else record)
                   for c, k in _LINEAR_TERMS.findall(source)) > _INT64_MAX
    if _RING_TOKENS_FAST.fullmatch(source):
        if '**' in source or len(source) > 4000:
            return True
        numbers = [int(number) for number in _NONZERO_NUMBER.findall(source)]
        if '*' not in source:
            return sum(numbers) + max(d - 1, 1) * source.count('rec') > _INT64_MAX
        bound = math.prod(numbers) * max(d - 1, 1) ** source.count('rec')
        return bound << (source.count('+') + source.count('-')) > _INT64_MAX
    return _RING_TOKENS.fullmatch(source) is not None


# For each (source, dimension) of a function from _compile_detector that the frame sampler has
# evaluated: True for a ring expression that needs _RingRows, the syntax tree of one with large
# literal exponents (see _large_powers), and False for every other expression, whose function
# gives its values itself.
# Its keys are the functions' own compiled_from tuples.  It forgets its oldest half when it holds
# _RING_SOURCES_LIMIT sources, many more than a circuit usually has, which bounds its memory.
_RING_SOURCES = {}
_RING_SOURCES_LIMIT = 2 ** 17


def _ring_values(function, rows, d: int):
    """The function's value on _RingRows of the int64 shift rows, as a row, and the evaluation."""
    evaluation = _RingEvaluation(d)
    value = function([_RingRow(row, d - 1, evaluation) for row in rows])
    return (value.row if type(value) is _RingRow else value), evaluation


_RING_TREE_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Mod: operator.mod}


def _ring_tree_values(tree, rows, d: int):
    """
    The value mod d of a ring expression's syntax tree (see _ring_tree) on _RingRows of the int64
    shift rows, as a row.  Unlike its compiled function, this takes the large powers of constants
    mod d.  The tree is walked with a stack, as sources can nest deeply.
    """
    evaluation = _RingEvaluation(d)
    rec = [_RingRow(row, d - 1, evaluation) for row in rows]
    values = []
    stack = [(tree.body, False)]
    while stack:
        node, ready = stack.pop()
        if isinstance(node, ast.Constant):
            values.append(node.value)
        elif isinstance(node, ast.Subscript):
            values.append(rec[_int_literal(node.slice)])
        elif not ready:
            stack.append((node, True))
            if isinstance(node, ast.BinOp):
                stack.append((node.right, False))
                stack.append((node.left, False))
            else:
                stack.append((node.operand, False))
        elif isinstance(node, ast.UnaryOp):
            value = values.pop()
            values.append(-value if isinstance(node.op, ast.USub) else value)
        else:
            right = values.pop()
            left = values.pop()
            if isinstance(node.op, ast.Pow):
                if type(left) is int and abs(left) > 1 and left.bit_length() * right > 2 ** 16:
                    evaluation.mark_reduced()
                    values.append(pow(left, right, d))
                elif type(left) is _RingRow:
                    values.append(evaluation.power(left, right))  # by any literal, unlike left ** right
                else:
                    values.append(left ** right)
            else:
                values.append(_RING_TREE_OPERATORS[type(node.op)](left, right))
    value = values[0] % d
    return value.row if type(value) is _RingRow else value


def _detector_values(function, compiled_from, rows):
    """
    The values of a function from _compile_detector on int64 shift rows: exact mod d for a ring
    expression, and the function's own int64 values for any other.  The first call for each
    (source, dimension) finds out which, and records it in _RING_SOURCES; a call that fails for
    another reason, such as rows that do not fit the expression, records nothing.
    """
    source, dimension = compiled_from
    mode = _RING_SOURCES.get(compiled_from)
    if mode is True:
        return _ring_values(function, rows, int(dimension))[0]
    if mode is False:
        return function(rows)
    if mode is not None:
        return _ring_tree_values(mode, rows, int(dimension))
    mode = False
    values = None
    d = int(dimension) if isinstance(dimension, (int, np.integer)) else 0
    if 1 <= d < 2 ** 31 and _could_leave_int64(source, d):
        if '**' in source and _large_powers(source):
            tree = _ring_tree(source, d)
            if tree is not None:
                mode = tree
                values = _ring_tree_values(tree, rows, d)
        else:
            try:
                values, evaluation = _ring_values(function, rows, d)
            except (_NotRing, TypeError):
                pass  # not a ring expression, whatever the rows: the function gives its values or raises
            except MemoryError:
                raise  # rather than give int64 values that may be wrong
            except Exception:
                # Not about the expression (rows that do not fit it, ...): record nothing, so that a
                # later call decides, and let the function raise its own error.
                return function(rows)
            else:
                # Values with no operand reduced are the function's own, but a % that is not a ring
                # operation ran without its warnings, so the function gives those values itself.
                mode = evaluation.reduced
                if evaluation.other or (mode and ('%' in source or '**' in source) and _ring_tree(source, d) is None):
                    mode = False
                    values = None
    if len(_RING_SOURCES) >= _RING_SOURCES_LIMIT:
        # Dicts keep insertion order, so this forgets the oldest decisions.
        for key in list(_RING_SOURCES)[:_RING_SOURCES_LIMIT // 2]:
            _RING_SOURCES.pop(key, None)  # another thread may have forgotten it already
    _RING_SOURCES[compiled_from] = mode
    return function(rows) if values is None else values


def _paired_parentheses(tokens: str, depth: int) -> str:
    """A pattern for `tokens` with groups of them in parentheses, nested up to `depth` deep."""
    pattern = tokens
    for _ in range(depth):
        pattern = rf"{tokens}(?:\({pattern}\){tokens})*+"
    return pattern


# A source made of rec[j], decimal literals, + - * %, spaces and tabs, and parentheses that pair
# up (nested up to 16 deep), which is not blank.  It cannot close a parenthesis that it did not
# open, so past its leading white space it is a complete expression on its own exactly when it is
# one in parentheses.
_PLAIN_SOURCE = re.compile(r"(?=[ \t]*+[^ \t])" + _paired_parentheses(r"(?:[-+*% \t0-9]++|rec\[[0-9]++\])*+", 16))


def _closes_only_its_own_parentheses(source: str) -> bool:
    """
    Whether a source that compiles in parentheses closes no parenthesis, bracket or brace that it
    did not open, so that the parentheses of `(source)` pair with each other.  The first one that it
    closed would close both the ( of `(source)` and the [ of `[source]`, and it cannot match both,
    so `[source]` compiles only if it closes none.  A few sources that close none do not compile in
    a list either, such as `yield rec[0]`, and give False.
    """
    try:
        compile('[' + source + ']', '<detector>', 'eval')
    except SyntaxError:
        return False
    return True


def _compile_detector(source: str, dimension: int):
    """
    Compiles a detector expression into `lambda rec : (source) % dimension`.

    When `source` is a complete expression on its own (past any leading white space), or in
    parentheses when it spans lines and closes no parenthesis that it did not open,
    `(source) % dimension` is that expression modulo dimension, and the function computes the
    modulo with _detector_mod, which gives the same value faster on int64 arrays.  Anything else,
    such as "rec[0]) + (rec[1]", is compiled exactly as written.  A plain source (see _PLAIN_SOURCE)
    is complete when the first lambda compiles, so it is compiled once.

    Only a function of the first kind keeps (source, dimension), as `compiled_from`, for the frame
    sampler: ring expressions are evaluated exactly mod d (see _RingRow); other expressions are
    evaluated with NumPy int64 arithmetic as before, which is exact as long as their intermediate
    values stay within int64.
    """
    try:
        if not _PLAIN_SOURCE.fullmatch(source):
            try:
                # Alone, leading white space is an unexpected indent; in the parentheses it is not.
                compile(source.lstrip(), '<detector>', 'eval')
            except SyntaxError:
                # Nor is a line break, if the parentheses hold all of the source; then the first
                # lambda compiles exactly when the source is complete in them.
                if ('\n' not in source and '\r' not in source) or not _closes_only_its_own_parentheses(source):
                    raise
        function = eval('lambda rec : _detector_mod((' + source + '), ' + str(dimension) + ')')
    except SyntaxError:
        return eval('lambda rec : (' + source + ") % " + str(dimension))
    function.compiled_from = source, dimension
    return function


def _evaluate_detectors(gate_ids: np.ndarray, records: np.ndarray, measurement_rows: np.ndarray,
                        detector_info: DetectorData | None, shots: int) -> DetectorResults:
    """
    Evaluates the detector and observable functions, in circuit order, on the measurement shifts.

    Each function gets the list of shift rows (x mod d at each measurement it reads, one int64
    entry per shot) that the frame loop used to hand it.  Shift rows never change once written,
    so evaluating after the frame loop gives the same values as evaluating in place.

    Ring expressions are evaluated exactly mod d: a function from _compile_detector whose ring
    expression could leave int64 is called on _RingRows (see _detector_values).  Other expressions
    are evaluated with NumPy int64 arithmetic as before, which is exact as long as their
    intermediate values stay within int64.
    """
    detector_gates = gate_ids[(gate_ids == 19) | (gate_ids == 20)]
    if detector_info is None:
        if detector_gates.shape[0]:
            raise ValueError("The circuit has detectors, so simulate_frame needs detector_info.")
        detector_info = DetectorData(detector_data=[], detector_functions=[])
    detector_events = np.zeros((detector_info.num_detector_events, shots), dtype=np.int64)
    logical_operator_events = np.zeros((detector_info.num_logical_operators, shots), dtype=np.int64)
    detector_counter = 0
    lo_counter = 0
    for gate_id in detector_gates.tolist():
        function_index, _, arguments, _ = detector_info.detector_data[detector_counter + lo_counter]
        shift_params = [records[measurement_rows[a]].astype(np.int64) for a in arguments]
        function = detector_info.detector_functions[function_index]
        compiled_from = getattr(function, 'compiled_from', None)
        if compiled_from is None or _RING_SOURCES.get(compiled_from) is False:
            value = function(shift_params)
        else:
            value = _detector_values(function, compiled_from, shift_params)
        if gate_id == 19:
            detector_events[detector_counter] = value
            detector_counter += 1
        else:
            logical_operator_events[lo_counter] = value
            lo_counter += 1
    return DetectorResults(
        detection_events=detector_events,
        logical_operator_shifts=logical_operator_events
    )


def simulate_frame(ir_array: np.ndarray, reference_results: np.ndarray,
                  n_qudits: int, dimension: int, extra_shots: int,
                  noise_array: np.ndarray = None, 
                  detector_info : DetectorData = None,
                  noise_model : NoiseModel = None) -> tuple[np.ndarray, DetectorResults]:
    """
    Simulates quantum circuit using Pauli frame simulation.

    The frame update loop runs in a compiled kernel and keeps every frame entry reduced mod d,
    so it cannot overflow for any dimension below 2**31.  Detector and observable expressions
    are evaluated on the measurement shifts: ring expressions (see _RingRow) are evaluated
    exactly mod d; other expressions are evaluated with NumPy int64 arithmetic as before, which
    is exact as long as their intermediate values stay within int64.
    
    Args:
        ir_array: Array of (gate_id, qudit_index, target_index, scalar) tuples
        reference_results: Reference measurement results
        n_qudits: Number of qudits
        dimension: Qudit dimension
        extra_shots: Number of additional shots to simulate
        noise_array: Pre-computed noise samples of shape (num_noise_gates, extra_shots, 4), as
            returned by Program._build_ir.  When it is None, noise is sampled while the frame
            runs from noise_model instead.
        detector_info: The detector data returned by Program._build_ir.
        noise_model: Per-noise-gate parameters from Program._build_ir(..., sample_noise=False),
            used when noise_array is None.  Needed only if the circuit has noise gates.
        
    Returns:
        frame_results: Array of simulated measurement results with shape (n_qudits, num_rounds, extra_shots)
        detector_results: Detection events and observable shifts, each indexed as (index, shot)
    """
    run = _run_frame(ir_array, reference_results, n_qudits, dimension, extra_shots,
                     noise_array, detector_info, noise_model)
    shots = int(extra_shots)
    frame_results = np.zeros((int(n_qudits), reference_results.shape[1], shots), dtype=MEASUREMENT_DTYPE)
    shot_index = np.arange(shots, dtype=np.int64)
    for r in range(run.records.shape[0]):
        q = int(run.record_qudit[r])
        m = int(run.record_round[r])
        out = frame_results[q, m]
        out['qudit_index'] = q
        out['meas_round'] = m
        out['shot'] = shot_index
        out['deterministic'] = reference_results[q, m]['deterministic']
        out['measurement_value'] = (reference_results[q, m]['measurement_value'] + run.records[r].astype(np.int64)) % dimension
    return frame_results, run.detector_results

@dataclass
class SimulationOptions:
    shots: int = 1
    show_measurement: bool = False
    record_tableau: bool = False
    force_tableau: bool = False
    verbose: bool = False
    show_gate: bool = False
    exact: bool = False
    raw_detector_output : bool = False


class Program:
    """
    Represents a quantum program with a circuit and stabilizer tableau.

    This class handles the initialization and simulation of a quantum program,
    including applying gates and managing measurement results.

    Attributes:
        stabilizer_tableau: The current state of the quantum system.
        circuits: The list of Circuit objects the program runs, in order: the constructor's circuit,
            then each one added with append_circuit.
        initial_tableau: The state every shot starts from: the constructor's tableau, or by default the
            all zero computational basis state, with any qudits that appended circuits add in |0>.
        measurement_results: The MeasurementResult objects of the last simulation, indexed as
            measurement_results[qudit][measurement round][shot].

    Args:
        circuit (Circuit): A Circuit object representing the quantum circuit.
        tableau (Optional[Tableau]): An optional stabilizer tableau. If not provided,
            the default is the all zero computational basis.
    """
    def __init__(self, circuit: Circuit, tableau=None):
        if tableau is None:
            if isprime(circuit.dimension):
                self.stabilizer_tableau = ExtendedTableau(circuit.num_qudits, circuit.dimension)
            else:
                self.stabilizer_tableau = WeylTableau(circuit.num_qudits, circuit.dimension)
        else:
            self.stabilizer_tableau = tableau
            if type(tableau.num_qudits) is not int or type(tableau.dimension) is not int:
                # NumPy sizes break pow(a, -1, d) in MUL (and float ones never worked: TypeError here).
                # A copy keeps the caller's tableau as it is.
                self.stabilizer_tableau = copy.copy(tableau)
                self.stabilizer_tableau.num_qudits = operator.index(tableau.num_qudits)
                self.stabilizer_tableau.dimension = operator.index(tableau.dimension)
        self.circuits = [circuit]
        self.measurement_results = []
        self.initial_tableau = copy.copy(self.stabilizer_tableau)
        self._tableau_noise_enabled = True

    # def enumerate_detector_shifts(self) -> dict[str, dict[str, str | np.ndarray]]:

    #     return

    def simulate(self, shots: int = 1, show_measurement: bool = False, record_tableau: bool = False, force_tableau: bool = False,
                 verbose: bool = False, show_gate: bool = False, exact: bool = False, 
                 building_error_mechanism : bool = False, raw_detector_output : bool = False, options: SimulationOptions = None) -> list[list[list[MeasurementResult]]] | tuple[list[list[list[MeasurementResult]]], dict[str, list(dict[str, str | np.ndarray])]]:
        """
        Runs the list of `Circuit` and applies the gates to the `stabilizer_tableau`.
        
        Note that using multiple shots without `record_tableau=True` or `force_tableau=True` will use the Pauli frame sampler.
        
        `show_gate` and `verbose` then print **only the first shot**, the noiseless reference shot, while it runs (also
        when the shots after it run on the tableau, see below).  `show_measurement` prints every shot, as returned,
        once the simulation is done.

        The Pauli frame sampler needs a program that starts in a computational basis state, such as the default |0...0>.
        If the initial `tableau` is not one, the shots after the reference shot run on the tableau instead, which is as
        slow as `force_tableau=True`, and the results take the frame sampler's form described below.  Detectors are then
        evaluated on each shot's outcome shifts from the reference shot, (value - reference value) mod d, as for frame shots.

        Args:
            shots (int): The number of times to run the simulation.
            show_measurement (bool): Whether to print the measurement results of each shot, numbered from 1,
                with the values that are returned.
            verbose (bool): Whether to print the stabilizer tableau at each time step (of the first shot only
                with the Pauli frame sampler).
            show_gate (bool): Whether to print the gate name at each time step (of the first shot only
                with the Pauli frame sampler).
            record_tableau (bool): Whether to record the tableau after each measurement.
            force_tableau (bool): Whether to force the use of the tableau method.
            exact (bool): Kept for compatibility. Composite-dimension measurements are always
                computed exactly, so it has no effect.
            building_error_mechanism (bool): Flag to generate exhaustive noise sequences to sample detector and logical operator shift data.
                Not for manual use, and only for programs that start in a computational basis state
            raw_detector_output (bool): Flag for returning 2D matrices for detection events indexed as (sequential detector index, shot)
            options (SimulationOptions): An optional SimulationOptions object.

        Returns:
            list, 3D list, or tuple: Depending on how the circuit is simulated:
                - If `shots == 1`, returns a list of `MeasurementResult` instances.
                - If `shots > 1` with `record_tableau=True` or `force_tableau=True`, returns a 3D list of
                `MeasurementResult` objects.  The first axis is the qudit position,
                the second axis is the measurement index (number of times a qudit was measured in a Circuit),
                and the third axis is the shot number.
                - Otherwise (`shots > 1` with the Pauli frame sampler), returns a tuple `(measurements, detectors)`.
                `measurements` is the 3D list described above, with shot 0 being the reference tableau shot.
                `detectors` is a dict with keys 'detectors' and 'logicals', each a list of {'label', 'data'} entries
                in circuit order, or a pair of 2D arrays indexed as (detector index, shot) when `raw_detector_output=True`.
                Detector data covers the `shots - 1` frame shots and does not include the reference shot.
                RESET also adds a measurement round for its qudit, since it is applied as a measurement followed by a correction.
        """
        if options is None:
            options = SimulationOptions(
                shots=shots,
                show_measurement=show_measurement,
                record_tableau=record_tableau,
                force_tableau=force_tableau,
                verbose=verbose,
                show_gate=show_gate,
                exact=exact,
                raw_detector_output=raw_detector_output
            )
        # Every mode checks the detector record references up front, before simulating anything.
        _check_record_references(self.circuits)
        if options.shots > 1 and not options.record_tableau and not options.force_tableau:
            if not self._starts_in_basis_state():
                return self._sample_with_tableau(options, building_error_mechanism)
            tableau_options = copy.copy(options)
            tableau_options.shots = 1
            tableau_options.show_measurement = False  # every shot is printed at the end
            self._tableau_noise_enabled = False
            try:
                self._simulate_tableau(tableau_options)
            finally:
                self._tableau_noise_enabled = True
            
            # Convert flattened reference results to structured array
            ref_array = self._results_to_array(self.measurement_results)
            
            # Build the IR.  Normal sampling draws noise inside the frame simulation, one gate at a time;
            # building error mechanisms enumerates every noise event up front instead.
            num_shots = options.shots - 1 if not building_error_mechanism else options.shots

            num_qudits = self.stabilizer_tableau.num_qudits
            if building_error_mechanism:
                ir_array, noise, detector_info = self._build_ir(self.circuits, num_shots, building_error_mechanism, num_qudits=num_qudits)
                noise_model = None
            else:
                ir_array, noise, detector_info, noise_model = self._build_ir(self.circuits, num_shots, sample_noise=False, num_qudits=num_qudits)
            
            # Run frame simulation
            frame_run = _run_frame(
                ir_array, ref_array, 
                self.stabilizer_tableau.num_qudits,
                self.stabilizer_tableau.dimension,
                num_shots,
                noise, 
                detector_info,
                noise_model
            )
            
            # Combine results
            measurements = self._combine_frame_run(frame_run, ref_array, self.stabilizer_tableau.dimension)
            if options.show_measurement:
                for shot in range(num_shots + 1):
                    self._print_shot(shot)
            return measurements, self._combine_detector_results(detector_info, frame_run.detector_results, options.raw_detector_output)
        else:
            return self._simulate_tableau(options)

    def _simulate_tableau(self, options: SimulationOptions) -> list:
        """
        Simulates the circuit using the stabilizer tableau method.
        
        In single-shot mode (options.shots == 1), the simulation returns a flattened list of 
        MeasurementResult objects—one per measurement round per qudit—by taking the first (and only)
        shot from the internal 3D measurement results structure:
        
            self.measurement_results[qudit_index][measurement_round][shot]
        
        For multiple shots (shots > 1) when not recording the tableau, a single reference shot is computed
        via the full tableau simulation and its measurement outcomes are stored in the internal grouped
        structure (by qudit and measurement round). Later, extra shots are generated using a vectorized
        Pauli frame simulation (via _simulate_frame) and then recombined with the reference shot using
        _combine_results. The final combined results are returned as a 3D list of MeasurementResult objects
        with dimensions:
        
            [qudit_index][measurement_round][shot]
        
        where shot index 0 is the reference simulation result.
        
        Args:
            options: A SimulationOptions object containing simulation parameters (e.g., shots, verbose, etc.)
        
        Returns:
            If options.shots == 1, a flat list of MeasurementResult objects (one per measurement round) is returned.
            If options.shots > 1, a 3D list of MeasurementResult objects is returned, where the axes correspond to 
            qudit index, measurement round, and shot number.
        """
        num_qudits = self.stabilizer_tableau.num_qudits
        # Prepare the measurement results container.
        self.measurement_results = [[] for _ in range(num_qudits)]
        length = sum(len(circuit.operations) for circuit in self.circuits)

        # Set exact mode if necessary.
        if isinstance(self.stabilizer_tableau, WeylTableau) and options.exact:
            self.stabilizer_tableau.exact = True

        # Iterate over each shot.
        for shot in range(options.shots):
            self.stabilizer_tableau = copy.deepcopy(self.initial_tableau)
            measurement_counts = [0] * num_qudits  
            for circuit in self.circuits:
                for time, gate in enumerate(circuit.operations):
                    if time == 0 and options.verbose:
                        print("Initial state")
                        self.stabilizer_tableau.print_tableau()
                        print("\n")
                    if time % 64 == 0:
                        self.stabilizer_tableau.modulo()

                    measurement_result = self.apply_gate(gate)
                    if measurement_result is not None:
                        qudit_index = measurement_result.qudit_index
                        if options.record_tableau:
                            measurement_result.stabilizer_tableau = copy.deepcopy(self.stabilizer_tableau)

                        measurement_number = measurement_counts[qudit_index]
                        measurement_counts[qudit_index] += 1

                        if len(self.measurement_results[qudit_index]) <= measurement_number:
                            self.measurement_results[qudit_index].append([])

                        self.measurement_results[qudit_index][measurement_number].append(measurement_result)

                        # Handle reset gate (gate_id == 16)
                        if gate.gate_id == 16:
                            # Shift the measured value back to |0> with a single power of X.
                            steps_to_zero = (-measurement_result.measurement_value) % self.stabilizer_tableau.dimension
                            _apply_pauli_powers(self.stabilizer_tableau, gate.qudit_index, steps_to_zero, 0)

                    if options.show_gate:
                        gate_info = gate.target_index if gate.target_index is not None else ""
                        if time < length - 1:
                            print("Time step", time, "\t", gate.name, gate.qudit_index, gate_info)
                        else:
                            print("Final step", time, "\t", gate.name, gate.qudit_index, gate_info)

                    if options.verbose:
                        self.stabilizer_tableau.print_tableau()
                        print("\n")
            self.stabilizer_tableau.modulo()
            if options.show_measurement:
                self._print_shot(shot)

        # Return results in the desired format.
        if options.shots == 1:
            flattened_results = []
            for measurements_per_qudit in self.measurement_results:
                # Each qudit may have multiple measurement rounds; we take the first shot.
                for shots_list in measurements_per_qudit:
                    flattened_results.append(shots_list[0])
            return flattened_results
        else:
            return self.measurement_results

    def _print_shot(self, shot: int):
        """Prints the stored results of one shot, numbered from 1, for show_measurement."""
        print(f"Measurement results for shot {shot + 1}:")
        # Only this shot: the stored results also hold the others.
        shot_results = [shots_list[shot] for rounds in self.measurement_results for shots_list in rounds]
        print("\n".join(map(str, shot_results)) if shot_results else "No measurements recorded.")
    
    def _starts_in_basis_state(self) -> bool:
        """
        Whether the initial state is a computational basis state, which the Pauli frame sampler
        assumes: it gives every qudit a uniformly random Z frame at the start, and only basis
        states are unchanged by that.  A stabilizer state is one exactly when no generator has an
        X part mod d.
        """
        x_block = self.initial_tableau.x_block
        # The default tableau has no nonzero entry at all, which is much quicker to see than x mod d.
        return not x_block.any() or not (x_block % self.initial_tableau.dimension != 0).any()

    def _sample_with_tableau(self, options: SimulationOptions, building_error_mechanism: bool = False) -> tuple:
        """
        The frame sampler's simulation for an initial state that is not a computational basis state.

        The noiseless reference shot and the shots after it all run on the tableau, and the results
        take the frame sampler's form: the reference shot comes first, and detectors are evaluated
        on the shift of each M / M_X outcome from the reference shot, (value - reference) mod d.
        """
        if building_error_mechanism:
            raise ValueError("Error mechanisms can only be built for a program that starts in a computational basis state.")
        reference_options = copy.copy(options)
        reference_options.shots = 1
        reference_options.show_measurement = False  # every shot is printed at the end
        self._tableau_noise_enabled = False
        try:
            self._simulate_tableau(reference_options)
        finally:
            self._tableau_noise_enabled = True
        reference_results, reference_tableau = self.measurement_results, self.stabilizer_tableau
        # Compiles the detectors before the slow shots, so a bad expression fails right away.
        ir_array, _, detector_info, _ = self._build_ir(self.circuits, options.shots - 1, sample_noise=False)

        self._simulate_tableau(SimulationOptions(shots=options.shots - 1, exact=options.exact))
        for reference_rounds, rounds in zip(reference_results, self.measurement_results):
            for reference_shots, shots_list in zip(reference_rounds, rounds):
                reference_shots.extend(shots_list)
        # The program's state is the reference shot's, as after the frame sampler.
        self.measurement_results, self.stabilizer_tableau = reference_results, reference_tableau

        # The M and M_X outcomes in circuit order.  RESET adds a measurement round, but no record.
        rounds_seen = [0] * len(self.measurement_results)
        records = []
        for circuit in self.circuits:
            for instruction in circuit.operations:
                if instruction.gate_id in (14, 15, 16):
                    q = instruction.qudit_index
                    if instruction.gate_id != 16:
                        records.append([m.measurement_value for m in self.measurement_results[q][rounds_seen[q]]])
                    rounds_seen[q] += 1
        values = np.array(records, dtype=np.int64).reshape(len(records), options.shots)
        shifts = (values[:, 1:] - values[:, :1]) % self.stabilizer_tableau.dimension
        detector_results = _evaluate_detectors(ir_array['gate_id'], shifts, np.arange(len(records)),
                                               detector_info, options.shots - 1)
        if options.show_measurement:
            for shot in range(options.shots):
                self._print_shot(shot)
        return self.measurement_results, self._combine_detector_results(detector_info, detector_results, options.raw_detector_output)

    def apply_gate(self, instruc: CircuitInstruction) -> MeasurementResult:
        """
        Applies a gate to the stabilizer tableau.

        Negative qudit indices count from the end of the program's qudits, and a measurement
        result holds the qudit's non-negative index.

        Args:
            instruc (CircuitInstruction): A CircuitInstruction object from a Circuit's operation list.

        Returns:
            MeasurementResult: A MeasurementResult object if the gate is a measurement gate, otherwise None.

        Raises:
            ValueError: If an invalid gate value is provided, or a two-qudit gate acts on one qudit twice.
            IndexError: If a qudit index is outside the program's qudits.
        """
        try:
            gate_function = GATE_FUNCTIONS[instruc.gate_id]
        except KeyError:
            raise ValueError("Invalid gate value") from None
        tableau = self.stabilizer_tableau
        qudit_index, target_index = instruc.qudit_index, instruc.target_index
        n = tableau.num_qudits
        # Checked before noise gates are skipped, so the reference shot rejects the same indices
        # as the other shots.
        if not (qudit_index is None or 0 <= qudit_index < n) or not (target_index is None or 0 <= target_index < n):
            qudit_index, target_index = self._qudit_positions(instruc, n)
        if not self._tableau_noise_enabled and instruc.gate_id in (17, 18):
            return None
        return gate_function(tableau, qudit_index, target_index, instruc.params)

    @staticmethod
    def _qudit_positions(instruc: CircuitInstruction, num_qudits: int) -> tuple:
        """
        Returns the qudit and target index of a gate with negative indices counted from the end
        of the program's num_qudits qudits.  The target of a one-qudit gate, and the indices of
        DETECTOR, LOGICAL_OBSERVABLE and TICK, are returned as they are, since nothing uses them.

        Raises:
            IndexError: If an index is not in -num_qudits .. num_qudits - 1.
            ValueError: If both indices of a two-qudit gate are the same qudit.
        """
        positions = [instruc.qudit_index, instruc.target_index]
        if instruc.gate_id in (19, 20, 21):
            return tuple(positions)
        two_qudit = instruc.gate_id in (9, 10, 11, 12, 13, 18)
        for i in range(2 if two_qudit else 1):
            index = positions[i]
            if index is None:
                continue
            if not -num_qudits <= index < num_qudits:
                raise IndexError(f"{instruc.name} acts on qudit {index}, but the program has {num_qudits} qudits.")
            if index < 0:
                positions[i] = index + num_qudits
        if two_qudit and positions[0] == positions[1]:
            raise ValueError(f"{instruc.name} needs two different qudits, but {instruc.qudit_index} and "
                             f"{instruc.target_index} are both qudit {positions[0]} of the program's {num_qudits}.")
        return tuple(positions)

    @staticmethod
    def _results_to_array(measurements: list) -> np.ndarray:
        """
        Converts measurement results to a structured NumPy array.

        Returns:
            A structured array with shape (num_qudits, max_rounds) containing measurement data
        """
        def measurement_to_tuple(m: MeasurementResult, meas_round: int = 0, shot: int = 0):
            return (m.qudit_index, meas_round, shot, m.deterministic, m.measurement_value)
        # Ensure the list is not empty and has the expected nested structure.
        if not measurements:
            raise ValueError("Empty measurement results format")

        # A circuit without measurements: no rounds on any qudit.
        if not any(measurements):
            return np.empty((len(measurements), 0), dtype=MEASUREMENT_DTYPE)

        # find the first non-empty element of the list
        j = 0
        while (not measurements[j]):
            j += 1

        # If it's a 3D list (i.e., each measurement round is a list of shots)
        if isinstance(measurements[j][0], list):
            max_rounds = max(len(m) for m in measurements)
            reference_results = np.empty((len(measurements), max_rounds), dtype=MEASUREMENT_DTYPE)
            for q, measurements_per_qudit in enumerate(measurements):
                for m, shots_list in enumerate(measurements_per_qudit):
                    # Convert the first shot in each round into a tuple.
                    reference_results[q, m] = measurement_to_tuple(shots_list[0], meas_round=m, shot=0)
            return reference_results
        # If it's a 2D list (each sublist contains MeasurementResult objects, one per round)
        elif isinstance(measurements[j][0], MeasurementResult):
            max_rounds = max(len(m) for m in measurements)
            reference_results = np.empty((len(measurements), max_rounds), dtype=MEASUREMENT_DTYPE)
            for q, shots_list in enumerate(measurements):
                for m, measurement in enumerate(shots_list):
                    reference_results[q, m] = measurement_to_tuple(measurement, meas_round=m, shot=0)
            return reference_results

        else:
            raise ValueError("Invalid measurement results format")

    def _combine_results(self, frame_results) -> list:
        """
        Combines the reference simulation (stored in self.measurement_results) with
        the extra shots computed in frame_results.

        The frame_results is expected to be a 3D structured array with shape:
            (n_qudits, num_rounds, extra_shots)
        where each element is a record with fields:
            ('qudit_index', 'meas_round', 'shot', 'deterministic', 'measurement_value').

        For each qudit and each measurement round, we create a list of MeasurementResult
        objects (one per shot) and append them.
        """
        n_qudits = frame_results.shape[0]
        # Pull whole field columns out as Python ints and bools, then build the objects in bulk.
        qudit_column = frame_results['qudit_index']
        deterministic_column = frame_results['deterministic']
        value_column = frame_results['measurement_value']
        with _gc_paused():
            for qudit_index in range(n_qudits):
                # It may be that len(self.measurement_results[qudit_index]) equals the number of rounds.
                num_rounds = len(self.measurement_results[qudit_index])
                for meas_round in range(num_rounds):
                    # Extend the already-existing list for this measurement round
                    self.measurement_results[qudit_index][meas_round].extend(map(
                        MeasurementResult,
                        qudit_column[qudit_index, meas_round].tolist(),
                        deterministic_column[qudit_index, meas_round].tolist(),
                        value_column[qudit_index, meas_round].tolist(),
                    ))
        return self.measurement_results

    def _combine_frame_run(self, frame_run: _FrameRun, reference_results: np.ndarray, dimension: int) -> list:
        """
        Appends the frame shots of `frame_run` to the reference shot in self.measurement_results.

        Gives the same result as `_combine_results` on simulate_frame's structured array: every
        shot of a (qudit, round) slot has the slot's qudit index and the reference shot's
        deterministic flag, and its value is (reference value + frame shift) mod d.
        """
        shots = frame_run.records.shape[1]
        values = np.empty(shots, dtype=np.int64)
        with _gc_paused():
            for r in range(frame_run.records.shape[0]):
                q = int(frame_run.record_qudit[r])
                m = int(frame_run.record_round[r])
                if m >= len(self.measurement_results[q]):
                    continue
                # The int64 loop: NumPy 1.x would add the int32 records to the reference value in int32.
                np.add(frame_run.records[r], reference_results[q, m]['measurement_value'], out=values, dtype=np.int64)
                np.remainder(values, dimension, out=values)
                deterministic = bool(reference_results[q, m]['deterministic'])
                self.measurement_results[q][m].extend(map(
                    MeasurementResult, repeat(q, shots), repeat(deterministic, shots), values.tolist()
                ))
        return self.measurement_results

    def _combine_detector_results(self, info : DetectorData, raw_results : DetectorResults, raw_detector_output : bool = False) -> dict[str, list(dict[str, str | np.ndarray])] | tuple[np.array, np.array]:
        """
        Combines all detector mechanism results into a single dictionary organized by their quantitative (e.g. order in the circuit) and qualitative information (e.g. label).
        The topmost dictionary has keys: 'detectors', 'logicals'.
        The list enumerates the detectors / frame change data in order of occurence in the circuit.  
        Finally, the bottom most dictionary has keys: 'label', 'data'
        (unique-index, label, arguments)
        Alternatively, the user may have the 
        """

        detector_events = raw_results.detection_events
        logical_events = raw_results.logical_operator_shifts

        if raw_detector_output:
            return detector_events, logical_events

        else:
            d_ind = 0
            l_ind = 0
            results = {'detectors' : list(), 'logicals' : list()}

            for d_data in info.detector_data:
                label = d_data[1]
                is_logical = d_data[3]

                entry_type, data, index = ('logicals', logical_events, l_ind) if is_logical else ('detectors', detector_events, d_ind)

                event_info = {
                    'label' : label, 
                    'data' : data[index]
                }

                results[entry_type].append(event_info)

                if is_logical:
                    l_ind += 1
                else:
                    d_ind += 1

            return results
        
    @staticmethod
    def _build_ir(circuits: list[Circuit], extra_shots: int, 
    building_error_mechanism : bool = False, sample_noise : bool = True, num_qudits : int | None = None) -> tuple[np.ndarray, np.ndarray, DetectorData] | tuple[np.ndarray, None, DetectorData, NoiseModel]:
        """
        Builds an intermediate representation (IR) for the given circuits and also precomputes
        an array of sampled Pauli noise outcomes for noise gates (if applicable)
        
        Args:
            circuits (list[Circuit]): A list of Circuit objects.
            extra_shots (int): The number of extra shots for which noise outcomes
                            will be sampled.
            building_error_mechanism (bool): Flag for sampling errors in the circuit to build error mechanisms in a detector-error model.
            sample_noise (bool): When False (and not building error mechanisms), no noise is sampled here.
                The noise output is None and a fourth element, a NoiseModel holding each noise gate's
                parameters, is returned so that simulate_frame can sample the noise as it runs.
            num_qudits (int, optional): The program's number of qudits, which negative qudit indices
                count back from.  Defaults to the number of qudits of the widest circuit.
        
        Returns:
            tuple:
                - A NumPy array of IR instructions with each element as a tuple
                (gate_id, qudit_index, target_index, scalar).  Negative qudit indices from -num_qudits
                on are replaced by the qudit they count back to, and -1 stands for no qudit.
                - A NumPy array of shape (num_noise_gates, extra_shots, [x_block, z_block]) containing
                pre-sampled noise outcomes for each noise gate encountered.
                If no noise gate is present, an empty array is returned.
                With sample_noise=False this is None.
                - A DetectorData object containing runtime information for the detection events in the circuit. 
                If no detectors are present, a default object is returned.
                - With sample_noise=False only: a NoiseModel with the parameters of each noise gate.
        """
        lazy_noise = not sample_noise and not building_error_mechanism
        ir_list  = []
        detector_list = []
        detector_data = []
        dimension = circuits[0].dimension
        if num_qudits is None:
            num_qudits = max(circuit.num_qudits for circuit in circuits)

        # Detector related counters
        seen_measurements = 0
        num_detector_events = 0
        num_logical_operators = 0

        # Offset to organize error mechanism sampler
        error_skip_offset = 0
        
        # Noise output: the sampled (or enumerated) noise array is allocated once and filled
        # gate by gate; with lazy noise, only each gate's parameters are kept.
        num_noise_gates = sum(
            1 for circuit in circuits for instruction in circuit.operations if instruction.gate_id in (17, 18)
        )
        noise_array = None
        noise_counter = 0
        if not lazy_noise and num_noise_gates:
            noise_array = np.zeros((num_noise_gates, extra_shots, 4), dtype=np.int64)
        noise_kind = []
        noise_mode = []
        noise_log_q = []
        noise_cdf_offset = []
        cdf_parts = []
        cdf_offsets_by_distribution = {}
        cdf_size = 0

        # Common arrays used and re-used during computation.  The list of all d**4
        # two-qudit Pauli powers is only built when something needs it: building
        # error mechanisms for the legacy DEM.
        two_qudit_event_pauli_powers = None
        two_qudit_number_of_noise_events = (dimension ** 4)
        two_qudit_trivial_event = np.zeros(4, dtype=np.int64)

        def _check_two_qudit_powers():
            # Refuse to work with the d**4 two-qudit Paulis when listing them would be huge.
            if dimension ** 4 > 10 ** 7:
                raise ValueError(
                    f"Listing all {dimension}**4 two-qudit Pauli operators would take too much memory. "
                    "Use N2 with prob= and sdim.dem.DetectorErrorModel.from_circuit instead."
                )

        def _two_qudit_powers():
            # Build the d**4 list on first use, and refuse when it would be huge.
            nonlocal two_qudit_event_pauli_powers
            if two_qudit_event_pauli_powers is None:
                _check_two_qudit_powers()
                two_qudit_event_pauli_powers = list(np.ndindex((dimension, ) * 4))
            return two_qudit_event_pauli_powers
        

        # # If we're building a DEM out of this circuit, then we add 1 shot as our reference shot simulates trivial noise
        # extra_shots = extra_shots if not building_error_mechanism else extra_shots + 1

        for circuit in circuits:

            #TODO: Repeater blocks for detectors that link detectors to earlier detector expressions
            for instruction in circuit.operations:
                if instruction.gate_id == 0:
                    continue
                
                control_index = instruction.qudit_index
                target_index = instruction.target_index
                # -1 means no qudit, so negative indices become the qudits they count back to;
                # out-of-range indices are left as they are.  The sign is tested first because
                # this runs for every instruction.
                if control_index is None:
                    control_index = -1
                elif control_index < 0 and control_index >= -num_qudits:
                    control_index += num_qudits
                if target_index is None:
                    target_index = -1
                elif target_index < 0 and target_index >= -num_qudits:
                    target_index += num_qudits
                scalar = -1
                if instruction.gate_id == 22:  # MUL carries its multiplier in the IR scalar field
                    if instruction.params is None:
                        raise ValueError("Multiplication gate requires an 'a' parameter.")
                    scalar = instruction.params.get('a', instruction.params.get('scalar'))
                    if scalar is None:
                        raise ValueError("Multiplication gate requires an 'a' parameter.")
                    # Only a mod d matters, and reducing it here keeps any integer a (even one
                    # beyond int64) in the int64 IR field.  Same check as the tableau's multiply.
                    scalar = int(scalar) % dimension
                    if math.gcd(scalar, dimension) != 1:
                        raise ValueError(f"Scalar {scalar} is not coprime with the dimension {dimension}.")

                ir_list.append((instruction.gate_id, control_index, target_index, scalar))

                # Count measurements here in order to properly track measurements in detector expressions
                if (instruction.gate_id == 14 or instruction.gate_id == 15):
                    seen_measurements += 1

                if instruction.gate_id == 17:
                    # Always add a noise sample, but only actually sample non-identity with some probability.
                    channel = instruction.params.get('noise_channel', instruction.params.get('channel', 'd'))
                    if channel not in ('d', 'f', 'p'):
                        raise ValueError(f"N1 noise_channel must be 'd', 'f' or 'p', not {channel!r}.")
                    if lazy_noise:
                        # The sampled path applies the error when U >= 1 - prob.
                        probability = float(instruction.params['prob'])
                        mode, log_q = _event_probability_mode(probability)
                        noise_kind.append({'d': _NOISE_N1_D, 'f': _NOISE_N1_F, 'p': _NOISE_N1_P}[channel])
                        noise_mode.append(mode)
                        noise_log_q.append(log_q)
                        noise_cdf_offset.append(-1)

                    elif not building_error_mechanism:
                        if channel == 'd':
                            # Sample integer r from 1 to dimension**2 - 1 for each extra shot.
                            r = np.random.randint(1, dimension**2, size=extra_shots)
                            a = r % dimension
                            b = r // dimension
                        elif channel == 'f':
                            a = np.random.randint(1, dimension, size=extra_shots)
                            b = np.zeros(extra_shots, dtype=np.int64)
                        elif channel == 'p':
                            a = np.zeros(extra_shots, dtype=np.int64)
                            b = np.random.randint(1, dimension, size=extra_shots)

                        # Roll to see if the channel applies on this each shot
                        shot_dice_rolls = np.random.uniform(0.0, 1.0, size=extra_shots)
                        # Mask that checks for failure to clear threshold, aka applying I = X^0 Z^0
                        probability = float(instruction.params['prob'])
                        mask = shot_dice_rolls < 1.0 - probability

                        # Apply mask to both Pauli exponents
                        a[mask] = 0
                        b[mask] = 0

                        noise_array[noise_counter, :, 0] = a
                        noise_array[noise_counter, :, 1] = b
                    
                    else:

                        num_noise_events = 0
                        short_a = []
                        short_b = []

                        if channel == 'd':
                            num_noise_events = dimension**2 - 1    
                            r = np.array( list( product(range(dimension), repeat=2) ) )
                            r = r[1:]
                            short_a = r[:, 0]
                            short_b = r[:, 1]

                        
                        elif channel in ('f', 'p'):
                            num_noise_events = dimension - 1
                            r = np.array( range(1, dimension) )
                            short_a, short_b = (r, np.zeros(dimension - 1, dtype=np.int64)) if channel == 'f' else (np.zeros(dimension - 1, dtype=np.int64), r)
                            

                        front_zero_pad =  np.zeros(error_skip_offset, dtype=np.int64)
                        back_zero_pad = np.zeros(extra_shots - (error_skip_offset + num_noise_events), dtype=np.int64) 
                        a = np.concatenate((front_zero_pad, short_a, back_zero_pad))
                        b = np.concatenate((front_zero_pad, short_b, back_zero_pad))

                        noise_array[noise_counter, :, 0] = a
                        noise_array[noise_counter, :, 1] = b

                        error_skip_offset += num_noise_events  

                    noise_counter += 1


                if instruction.gate_id == 18:
                    distribution = instruction.params.get('prob_dist', None)

                    if lazy_noise:
                        if distribution is None:
                            # The sampled path applies the error when U < prob.
                            probability = float(instruction.params.get('prob', 0.0))
                            mode, log_q = _event_probability_mode(probability)
                            noise_kind.append(_NOISE_N2_UNIFORM)
                            noise_mode.append(mode)
                            noise_log_q.append(log_q)
                            noise_cdf_offset.append(-1)
                        else:
                            offset = cdf_offsets_by_distribution.get(id(distribution))
                            if offset is None:
                                cdf = _prob_dist_cdf(distribution, dimension)
                                offset = cdf_size
                                cdf_parts.append(cdf)
                                cdf_size += cdf.shape[0]
                                cdf_offsets_by_distribution[id(distribution)] = offset
                            noise_kind.append(_NOISE_N2_DIST)
                            noise_mode.append(_NOISE_PER_SHOT)
                            noise_log_q.append(0.0)
                            noise_cdf_offset.append(offset)

                    elif not building_error_mechanism:

                        if distribution is None:
                            # Two-qudit depolarizing: with probability prob, apply a uniformly random
                            # non-identity Pauli (a, b, c, d).  Redraw any all-zero rows, then zero out
                            # the shots where no error happens.
                            probability = float(instruction.params.get('prob', 0.0))
                            noise = noise_array[noise_counter]
                            noise[:] = np.random.randint(0, dimension, size=(extra_shots, 4))
                            zero_rows = ~noise.any(axis=1)
                            while zero_rows.any():
                                noise[zero_rows] = np.random.randint(0, dimension, size=(int(zero_rows.sum()), 4))
                                zero_rows = ~noise.any(axis=1)
                            noise[np.random.uniform(0.0, 1.0, size=extra_shots) >= probability] = 0
                        elif len(distribution) == (dimension ** 4):
                            _check_two_qudit_powers()
                            noise_indices = np.random.choice(a=dimension ** 4, size=extra_shots, p=distribution)
                            # Index i is the i-th tuple (x1, z1, x2, z2) of np.ndindex((d,) * 4), z2 fastest.
                            noise = noise_array[noise_counter]
                            for column in (3, 2, 1, 0):
                                noise[:, column] = noise_indices % dimension
                                noise_indices = noise_indices // dimension
                        else:
                            raise ValueError(
                                f"N2 prob_dist has length {len(distribution)} instead of the required {dimension ** 4}."
                            )

                    else:
                        nontrivial_noise_events = np.array( _two_qudit_powers()[1:] )
                        front_zero_pad =  np.tile(two_qudit_trivial_event, (error_skip_offset, 1)) 
                        back_zero_pad = np.tile(two_qudit_trivial_event, (extra_shots - (error_skip_offset + two_qudit_number_of_noise_events - 1), 1)) 
                        noise = np.vstack([front_zero_pad, nontrivial_noise_events, back_zero_pad])
                        noise_array[noise_counter] = noise
                        error_skip_offset += two_qudit_number_of_noise_events - 1

                    noise_counter += 1

                if instruction.gate_id in (19, 20):
                    if 'expr' not in instruction.params:
                        raise ValueError("No detector provided.")
                    is_logical = instruction.gate_id == 20
                    label = instruction.params['label'] if 'label' in instruction.params else ''
                    name = _detector_name(instruction, num_logical_operators if is_logical else num_detector_events)
                    # TODO: Sanitize input; the expression is evaluated as Python code.
                    source, arguments = _resolve_record_references(
                        str(instruction.params['expr']), seen_measurements, name)
                    # Store the compiled function, and the detector data that points to it
                    detector_data.append((len(detector_list), label, arguments, is_logical))
                    detector_list.append(_compile_detector(source, dimension))

                    if instruction.gate_id == 19:
                        num_detector_events += 1
                    else:
                        num_logical_operators +=1
                    
        ir_dtype = np.dtype([
            ('gate_id', np.int64),
            ('qudit_index', np.int64),
            ('target_index', np.int64),
            ('scalar', np.int64)
        ])

        ir_array = np.array(ir_list, dtype=ir_dtype)

        if noise_array is None and not lazy_noise:
            noise_array = np.empty((1, extra_shots, 2), dtype=np.int64)

        detection_info = DetectorData(
            detector_data=detector_data,
            detector_functions=detector_list,
            total_measurements=seen_measurements,
            num_detector_events=num_detector_events,
            num_logical_operators=num_logical_operators
        )

        if lazy_noise:
            noise_model = NoiseModel(
                kind=np.array(noise_kind, dtype=np.int64),
                mode=np.array(noise_mode, dtype=np.int64),
                log_q=np.array(noise_log_q, dtype=np.float64),
                cdf_offset=np.array(noise_cdf_offset, dtype=np.int64),
                cdf_data=np.concatenate(cdf_parts) if cdf_parts else np.zeros(0, dtype=np.float64),
            )
            return ir_array, None, detection_info, noise_model

        #print(noise_array)
        return ir_array, noise_array, detection_info

    def append_circuit(self, circuit: Circuit):
        """
        Appends a circuit to the existing Program.

        A circuit on more qudits than the program adds the extra qudits in |0>: the program then
        starts in the tensor product of its initial state and |0...0>.  The circuits are not modified.
        Negative qudit indices count from the end of the program's qudits, as in `c1 + c2`, so in
        the earlier circuits they move to the new last qudits after a wider circuit is appended.

        Args:
            circuit (Circuit): The Circuit object to append.

        Raises:
            ValueError: If the circuits have different dimensions.
        """
        if self.circuits[-1].dimension != circuit.dimension:
            raise ValueError("Circuits must have the same dimension")
        initial_tableau = self._with_zero_qudits(self.initial_tableau, circuit.num_qudits)
        stabilizer_tableau = self._with_zero_qudits(self.stabilizer_tableau, circuit.num_qudits)
        self.initial_tableau, self.stabilizer_tableau = initial_tableau, stabilizer_tableau
        self.circuits.append(circuit)
        

    @staticmethod
    def _with_zero_qudits(tableau, num_qudits: int):
        """
        Returns the tableau extended to num_qudits qudits by qudits in |0>, which get the stabilizer
        Z (and in an ExtendedTableau the destabilizer X) of their own, or the tableau itself if it
        has enough qudits.  The tableau is not modified.
        """
        n = tableau.num_qudits
        k = num_qudits - n
        if k <= 0:
            return tableau

        def extend(block, new_block):
            # Rows are qudits and columns generators, so the new qudits also add k columns.
            columns = block.shape[1]
            extended = np.zeros((n + k, columns + k), dtype=block.dtype)
            extended[:n, :columns] = block
            extended[n:, columns:] = new_block
            return extended

        def extend_phases(phases):
            return np.concatenate((phases, np.zeros(k, dtype=phases.dtype)))

        identity = np.eye(k, dtype=np.int64)
        extended = copy.copy(tableau)
        extended.num_qudits = num_qudits
        extended.z_block = extend(tableau.z_block, identity)
        extended.x_block = extend(tableau.x_block, 0)
        extended.phase_vector = extend_phases(tableau.phase_vector)
        if isinstance(tableau, ExtendedTableau):
            extended.destab_z_block = extend(tableau.destab_z_block, 0)
            extended.destab_x_block = extend(tableau.destab_x_block, identity)
            extended.destab_phase_vector = extend_phases(tableau.destab_phase_vector)
        return extended

    def print_measurements(self):
        """
        Prints the measurement results.

        This method iterates through the stored measurement results and prints each one.
        """
        shot_count = 0
        for qudit_measurements in self.measurement_results:
            for measurement_group in qudit_measurements:
                shot_count = max(shot_count, len(measurement_group))
                
        if shot_count == 0:
            print("No measurements recorded.")
            return
        if shot_count == 1:
            # The stored shot, in the order simulate() returns it, without simulating again.
            for measurements_per_qudit in self.measurement_results:
                for shots_list in measurements_per_qudit:
                    print(shots_list[0])
        else:
            for shot_index in range(shot_count):
                print(f"Shot {shot_index + 1}:")
                for qudit_index, measurements_per_qudit in enumerate(self.measurement_results):
                    for measurement_number, shots_list in enumerate(measurements_per_qudit):
                        measurement_result = shots_list[shot_index]
                        print(f"{measurement_result} during measurement {measurement_number}")
                print()

    def __str__(self) -> str:
        return str(self.stabilizer_tableau)
