"""
Gate and measurement kernels for the prime-dimension tableau (`ExtendedTableau`).

There are two families of kernels here.

* The plain Python loops (`hadamard_optimized`, ..., `cnot_inv_optimized`) are the exact reference
  path.  They work on any integer arrays, including `dtype=object` arrays of Python integers, and
  leave the touched entries unreduced; `ExtendedTableau` reduces them afterwards.

* The numba kernels (names starting with an underscore) are the fast path.  `ExtendedTableau` uses
  them only for C-contiguous, writeable int64 arrays when the dimension d is odd or 2 and below
  2**31, and only when numba's JIT is on (`JIT_ENABLED`).  The phase order (d for odd d, 4 for
  d = 2) is then also below 2**31.  These kernels read every entry reduced (mod d for the X and Z
  blocks, mod the order for the phases), write back fully reduced values, and compute exactly the
  same results as the reference path.

Overflow: no kernel ever forms a value of 2**63 or more.  Every product is of two reduced values,
so it is below order**2 < 2**62, and it is reduced before it is added to anything.  Sums of two
reduced values are below 2**32.

Reductions use `_modq`, which avoids the slow 64-bit integer division.  It estimates the quotient
in double precision and corrects the estimate exactly with integer arithmetic (see its docstring).
"""

from numba import prange
from numba.core.dispatcher import Dispatcher
import numpy as np

from .._jit import _kernel


# @njit(parallel=True)
def hadamard_optimized(x_block, z_block, phase_vector,
                       destab_x_block, destab_z_block,
                       destab_phase_vector, qudit_index, 
                       num_qudits, phase_order):
    
    # Get array slices for better cache locality
    x_row = x_block[qudit_index]
    z_row = z_block[qudit_index]
    dx_row = destab_x_block[qudit_index]
    dz_row = destab_z_block[qudit_index]

    # Handle stabilizer blocks
    for i in prange(num_qudits):
        new_x = -z_row[i]
        new_z = x_row[i]
        phase_vector[i] += phase_order * new_x * new_z
        x_row[i] = new_x
        z_row[i] = new_z

    # Handle destabilizer blocks
    for i in prange(num_qudits):
        new_x = -dz_row[i]
        new_z = dx_row[i]
        destab_phase_vector[i] += phase_order * new_x * new_z
        dx_row[i] = new_x
        dz_row[i] = new_z

# @njit(parallel=True)
def hadamard_inv_optimized(x_block, z_block, phase_vector,
                          destab_x_block, destab_z_block,
                          destab_phase_vector, qudit_index, 
                          num_qudits, phase_order):
    
    # Get array slices for better cache locality
    x_row = x_block[qudit_index]
    z_row = z_block[qudit_index]
    dx_row = destab_x_block[qudit_index]
    dz_row = destab_z_block[qudit_index]

    # Handle stabilizer blocks
    for i in prange(num_qudits):
        new_x = z_row[i]
        new_z = -x_row[i]
        phase_vector[i] += phase_order * new_x * new_z
        x_row[i] = new_x
        z_row[i] = new_z

    # Handle destabilizer blocks
    for i in prange(num_qudits):
        new_x = dz_row[i]
        new_z = -dx_row[i]
        destab_phase_vector[i] += phase_order * new_x * new_z
        dx_row[i] = new_x
        dz_row[i] = new_z


# @njit(parallel=True)
def phase_optimized(x_block, z_block, phase_vector,
                   destab_x_block, destab_z_block, 
                   destab_phase_vector,
                   qudit_index: int,
                   num_qudits: int,
                   even: int):
    # Parallelize loop over qudits
    for i in prange(num_qudits):
        if even:
            phase_vector[i] += x_block[qudit_index, i] ** 2
            destab_phase_vector[i] += destab_x_block[qudit_index, i] ** 2
        else:
            phase_vector[i] += (x_block[qudit_index, i] * (x_block[qudit_index, i] - 1)) // 2
            destab_phase_vector[i] += (destab_x_block[qudit_index, i] * (destab_x_block[qudit_index, i] - 1)) // 2
        z_block[qudit_index, i] += x_block[qudit_index, i]
        destab_z_block[qudit_index, i] += destab_x_block[qudit_index, i]


# @njit(parallel=True)
def phase_inv_optimized(x_block, z_block, phase_vector,
                       destab_x_block, destab_z_block, 
                       destab_phase_vector,
                       qudit_index: int,
                       num_qudits: int,
                       even: bool):
    for i in prange(num_qudits):
        if even:
            phase_vector[i] -= x_block[qudit_index, i] ** 2
            destab_phase_vector[i] -= destab_x_block[qudit_index, i] ** 2
        else:
            phase_vector[i] -= (x_block[qudit_index, i] * (x_block[qudit_index, i]-1)) // 2
            destab_phase_vector[i] -= (destab_x_block[qudit_index, i] * (destab_x_block[qudit_index, i]-1)) // 2
            
        z_block[qudit_index, i] -= x_block[qudit_index, i]
        destab_z_block[qudit_index, i] -= destab_x_block[qudit_index, i]

# @njit(parallel=True)
def cnot_optimized(x_block, z_block, destab_x_block, destab_z_block,
                   num_qudits: int, dimension: int,
                   control: int, target: int):
    for i in prange(num_qudits):
        x_block[target, i] += x_block[control, i]
        z_block[control, i] += (z_block[target, i] * (dimension - 1))
    for i in prange(num_qudits):
        destab_x_block[target, i] += destab_x_block[control, i]
        destab_z_block[control, i] += (destab_z_block[target, i] * (dimension - 1))

# @njit(parallel=True)
def cnot_inv_optimized(x_block, z_block, destab_x_block, destab_z_block,
                       num_qudits: int, dimension: int,
                       control: int, target: int):
    for i in prange(num_qudits):
        x_block[target, i] -= x_block[control, i]
        z_block[control, i] -= z_block[target, i] * (dimension - 1)
    for i in prange(num_qudits):
        destab_x_block[target, i] -= destab_x_block[control, i]
        destab_z_block[control, i] -= destab_z_block[target, i] * (dimension - 1)


# ---------------------------------------------------------------------------------------------
# Fast path: numba kernels for int64 arrays.
#
# Layout: x[row, col] is the X power of stabilizer `col` on qudit `row` (likewise z), and ph[col]
# is the phase of stabilizer `col`.  d is the dimension, `order` the phase order (d for odd d, 4 for
# d = 2) and `po` the phase_order factor (1 for odd d, 2 for d = 2), so order == po * d.
# ---------------------------------------------------------------------------------------------

@_kernel
def _canon(v, m):
    """v mod m in [0, m).  Skips the division when v is already reduced, which is the usual case."""
    if v >= 0 and v < m:
        return v
    return v % m


@_kernel
def _modq(t, m, minv):
    """
    t mod m without a 64-bit integer division, for 2 <= m < 2**31, 0 <= t < 2**62 and t // m < 2**31.

    `minv` is the double 1.0 / m.  The estimate float(t) * minv goes through three roundings, so its
    relative error is at most about 3 * 2**-53 and it is within 2**31 * 3.4e-16 < 1e-6 of the exact
    quotient t / m.  Truncating it therefore gives floor(t / m) or one more or one less, the
    remainder t - q * m lies in [-m, 2m), and one correction step makes it exact.  q * m is at most
    t + m < 2**63, so nothing overflows.
    """
    q = np.int64(np.float64(t) * minv)
    r = t - q * m
    if r < 0:
        r += m
    elif r >= m:
        r -= m
    return r


@_kernel
def _mulmod(a, b, m, minv):
    """a * b mod m for 0 <= a, b < m < 2**31: the product is below 2**62 and the quotient below m."""
    return _modq(a * b, m, minv)


@_kernel
def _addmod(a, b, m):
    """a + b mod m for 0 <= a, b < m."""
    s = a + b
    if s >= m:
        s -= m
    return s


@_kernel
def _submod(a, b, m):
    """a - b mod m for 0 <= a, b < m."""
    s = a - b
    if s < 0:
        s += m
    return s


@_kernel
def _triangular_mod(n, m, minv):
    """(n * (n - 1) // 2) mod m for 0 <= n < m < 2**31 (n * (n - 1) // 2 is below m**2 / 2)."""
    return _modq(n * (n - 1) // 2, m, minv)


@_kernel
def _scaled_mod(c, po, m):
    """(po * c) mod m for po in {1, 2} and 0 <= c < 4 * m."""
    while c >= m:
        c -= m
    c *= po
    if c >= m:
        c -= m
    return c


@_kernel
def _count_unreduced(v, m):
    """Number of entries of a 1D array outside [0, m).  Branch-free, so it vectorizes."""
    bad = 0
    for i in range(v.shape[0]):
        a = v[i]
        bad += (a < 0) | (a >= m)
    return bad


@_kernel
def _reduce_vector(v, m):
    """
    Reduces a 1D array (or a row of a matrix) mod m in place.

    The entries are almost always reduced already, so a vectorized check runs first and the
    division only runs, entry by entry, when something is out of range.
    """
    if _count_unreduced(v, m):
        for i in range(v.shape[0]):
            a = v[i]
            if a < 0 or a >= m:
                v[i] = a % m


@_kernel
def _reduce_matrix(a, m):
    """Reduces a C-contiguous matrix mod m in place, like _reduce_vector."""
    _reduce_vector(a.reshape(a.size), m)


@_kernel
def _reduce_kernel(x, z, ph, dx, dz, dph, d, order):
    """The full reduction of ExtendedTableau.modulo: blocks mod d, phases mod the order."""
    _reduce_matrix(x, d)
    _reduce_matrix(z, d)
    _reduce_vector(ph, order)
    _reduce_matrix(dx, d)
    _reduce_matrix(dz, d)
    _reduce_vector(dph, order)


# The gate kernels first reduce the rows they touch and the phases, as ExtendedTableau._reduce_rows
# does after the reference kernels.  The arithmetic after that can assume reduced inputs, has no
# divisions and no data-dependent branches, and vectorizes.

@_kernel
def _hadamard_rows(xr, zr, ph, d, order, po, inverse, dinv):
    _reduce_vector(xr, d)
    _reduce_vector(zr, d)
    _reduce_vector(ph, order)
    for i in range(ph.shape[0]):
        xv = xr[i]
        zv = zr[i]
        if inverse:
            # H^-1: (x, z) -> (z, -x)
            nx = zv
            nz = d - xv if xv != 0 else 0
        else:
            # H: (x, z) -> (-z, x)
            nx = d - zv if zv != 0 else 0
            nz = xv
        # The phase gains po * nx * nz, and po * (nx * nz mod d) is that mod po * d, the order.
        p = ph[i] + po * _mulmod(nx, nz, d, dinv)
        if p >= order:
            p -= order
        xr[i] = nx
        zr[i] = nz
        ph[i] = p


@_kernel
def _hadamard_kernel(x, z, ph, dx, dz, dph, q, d, order, po, inverse):
    """
    H (or H^-1 when `inverse`) on qudit q, for the stabilizers and destabilizers.

    Same result as hadamard_optimized / hadamard_inv_optimized followed by
    ExtendedTableau._reduce_rows(q): every phase gains po * x' * z' for the new powers (x', z').
    """
    dinv = 1.0 / d
    _hadamard_rows(x[q], z[q], ph, d, order, po, inverse, dinv)
    _hadamard_rows(dx[q], dz[q], dph, d, order, po, inverse, dinv)


@_kernel
def _phase_rows(xr, zr, ph, d, order, po, inverse, oinv):
    _reduce_vector(xr, d)
    _reduce_vector(zr, d)
    _reduce_vector(ph, order)
    for i in range(ph.shape[0]):
        xv = xr[i]
        if po == 2:
            t = _mulmod(xv, xv, order, oinv)        # even d: the phase gains x**2
        else:
            t = _triangular_mod(xv, order, oinv)    # odd d: the phase gains x * (x - 1) / 2
        if inverse:
            ph[i] = _submod(ph[i], t, order)
            zr[i] = _submod(zr[i], xv, d)
        else:
            ph[i] = _addmod(ph[i], t, order)
            zr[i] = _addmod(zr[i], xv, d)


@_kernel
def _phase_kernel(x, z, ph, dx, dz, dph, q, d, order, po, inverse):
    """
    P (or P^-1 when `inverse`) on qudit q, for the stabilizers and destabilizers.

    Same result as phase_optimized / phase_inv_optimized followed by ExtendedTableau._reduce_rows(q).
    """
    oinv = 1.0 / order
    _phase_rows(x[q], z[q], ph, d, order, po, inverse, oinv)
    _phase_rows(dx[q], dz[q], dph, d, order, po, inverse, oinv)


@_kernel
def _cnot_rows(x, z, c, t, d, inverse):
    _reduce_vector(x[c], d)
    _reduce_vector(x[t], d)
    _reduce_vector(z[c], d)
    _reduce_vector(z[t], d)
    xc = x[c]
    xt = x[t]
    zc = z[c]
    zt = z[t]
    # Each statement reads both operands before it writes, so c == t gives what the reference gives.
    if inverse:
        for i in range(x.shape[1]):
            xt[i] = _submod(xt[i], xc[i], d)    # x_t -= x_c
            zc[i] = _addmod(zc[i], zt[i], d)    # z_c -= (d - 1) z_t
    else:
        for i in range(x.shape[1]):
            xt[i] = _addmod(xt[i], xc[i], d)    # x_t += x_c
            zc[i] = _submod(zc[i], zt[i], d)    # z_c += (d - 1) z_t


@_kernel
def _cnot_kernel(x, z, ph, dx, dz, dph, c, t, d, order, inverse):
    """
    CNOT (or CNOT^-1 when `inverse`) with control c and target t.

    Same result as cnot_optimized / cnot_inv_optimized followed by ExtendedTableau._reduce_rows(c, t).
    """
    _cnot_rows(x, z, c, t, d, inverse)
    _cnot_rows(dx, dz, c, t, d, inverse)
    _reduce_vector(ph, order)
    _reduce_vector(dph, order)


@_kernel
def _multiply_rows(xr, zr, scalar, inverse, d, dinv):
    _reduce_vector(xr, d)
    _reduce_vector(zr, d)
    for i in range(xr.shape[0]):
        zr[i] = _mulmod(zr[i], inverse, d, dinv)
        xr[i] = _mulmod(xr[i], scalar, d, dinv)


@_kernel
def _multiply_kernel(x, z, dx, dz, q, scalar, inverse, d):
    """
    The multiplication gate on qudit q: X powers times `scalar`, Z powers times its inverse mod d
    (both in [1, d)).  Same result as the reference code in ExtendedTableau.multiply.
    """
    dinv = 1.0 / d
    _multiply_rows(x[q], z[q], scalar, inverse, d, dinv)
    _multiply_rows(dx[q], dz[q], scalar, inverse, d, dinv)


@_kernel
def _pauli_rows(xr, zr, ph, a, b, d, order, po, dinv):
    _reduce_vector(xr, d)
    _reduce_vector(zr, d)
    _reduce_vector(ph, order)
    for i in range(ph.shape[0]):
        s = _submod(_mulmod(b, xr[i], d, dinv), _mulmod(a, zr[i], d, dinv), d)
        p = ph[i] + po * s
        if p >= order:
            p -= order
        ph[i] = p


@_kernel
def _pauli_kernel(x, z, ph, dx, dz, dph, q, a, b, d, order, po):
    """
    The Pauli X^a Z^b on qudit q (0 <= a, b < d).

    It leaves the X and Z blocks alone and adds po * (b * x_q - a * z_q) to every phase, where
    (x_q, z_q) is that stabilizer's power on qudit q.  This is exactly what the H/P/multiplication
    gate sequences of apply_X, apply_Z and _apply_pauli_powers in tableau_gates.py do (those leave
    row q reduced and the phases reduced, as this does).
    """
    dinv = 1.0 / d
    _pauli_rows(x[q], z[q], ph, a, b, d, order, po, dinv)
    _pauli_rows(dx[q], dz[q], dph, a, b, d, order, po, dinv)


@_kernel
def _first_x_kernel(x, q, d):
    """Index and reduced value of the first stabilizer with a non-zero X power on qudit q, or (-1, 0)."""
    for i in range(x.shape[1]):
        v = _canon(x[q, i], d)
        if v != 0:
            return i, v
    return -1, 0


@_kernel
def _exponentiate_kernel(x, z, ph, col, e, d, order, po):
    """
    Raises stabilizer `col` to the power e (1 <= e < d), leaving it reduced:
    (X^a Z^b)^e = omega^(ab * e(e-1)/2) X^(ea) Z^(eb).

    Same result as ExtendedTableau.exponentiate followed by the full reduction in measure.
    """
    dinv = 1.0 / d
    oinv = 1.0 / order
    xz = 0
    for r in range(x.shape[0]):
        xv = _canon(x[r, col], d)
        zv = _canon(z[r, col], d)
        if xv != 0 and zv != 0:
            xz = _addmod(xz, _mulmod(xv, zv, order, oinv), order)
        x[r, col] = _mulmod(xv, e, d, dinv)
        z[r, col] = _mulmod(zv, e, d, dinv)
    power_phase = _mulmod(xz, _triangular_mod(e, order, oinv), order, oinv) * po
    p = _mulmod(_canon(ph[col], order), e, order, oinv) + power_phase
    while p >= order:
        p -= order
    ph[col] = p


@_kernel
def _collapse_phase(p, phf, fac, dot, xz, order, po, oinv):
    """
    The phase of a generator after `fac` times stabilizer f is added to it:
    p + phf * fac + po * (fac * dot + po * (xz * fac(fac-1)/2 mod order))  (mod order),
    where phf is stabilizer f's phase, dot = (this generator's Z powers) . (stabilizer f's X powers)
    is the commutation term and xz = (stabilizer f's X powers) . (its Z powers) the exponentiation term.
    """
    commute = (_mulmod(fac, dot, order, oinv)
               + _mulmod(xz, _triangular_mod(fac, order, oinv), order, oinv) * po)
    p = _canon(p, order) + _mulmod(phf, fac, order, oinv)
    p += _scaled_mod(commute, po, order)
    while p >= order:
        p -= order
    return p


@_kernel
def _random_measure_kernel(x, z, ph, dx, dz, dph, q, f, d, order, po):
    """
    The tableau update of a random Z measurement of qudit q, where stabilizer f is the first one with
    a non-zero X power on q and has already been scaled so that power is 1.

    Every other stabilizer, and every destabilizer, with a non-zero X power on q gets the multiple of
    stabilizer f that cancels it.  Then destabilizer f becomes stabilizer f, and stabilizer f becomes
    Z_q; the caller draws the outcome and sets its phase.

    Same result as ExtendedTableau._random_measurement on the reduced tableau.  The reference adds the
    multiples column by column; the columns are independent (column f, the only one read by all of
    them, never changes), so here the work is done row by row, which reads the arrays in memory order.
    Destabilizer f is skipped, since the reference overwrites it afterwards anyway.
    """
    nr = x.shape[0]
    nc = x.shape[1]
    dinv = 1.0 / d
    oinv = 1.0 / order

    # Stabilizer f, reduced, and the dot product of its X and Z powers.
    xf = np.empty(nr, np.int64)
    zf = np.empty(nr, np.int64)
    xz = 0
    for r in range(nr):
        xv = _canon(x[r, f], d)
        zv = _canon(z[r, f], d)
        xf[r] = xv
        zf[r] = zv
        if xv != 0 and zv != 0:
            xz = _addmod(xz, _mulmod(xv, zv, order, oinv), order)
    phf = _canon(ph[f], order)

    # The multiples, from row q before anything changes.
    dcols = np.empty(nc, np.int64)
    dfac = np.empty(nc, np.int64)
    scols = np.empty(nc, np.int64)
    sfac = np.empty(nc, np.int64)
    nd = 0
    ns = 0
    for i in range(nc):
        if i == f:
            continue
        v = _canon(dx[q, i], d)
        if v != 0:
            dcols[nd] = i
            dfac[nd] = d - v
            nd += 1
        v = _canon(x[q, i], d)
        if v != 0:
            scols[ns] = i
            sfac[ns] = d - v
            ns += 1

    # Add the multiples.  dsum / ssum collect each column's (Z powers) . (X powers of stabilizer f)
    # mod the order, from the Z powers before the update.
    dsum = np.zeros(nd, np.int64)
    ssum = np.zeros(ns, np.int64)
    for r in range(nr):
        a = xf[r]
        b = zf[r]
        if a == 0 and b == 0:
            continue
        for k in range(nd):
            i = dcols[k]
            vz = _canon(dz[r, i], d)
            if a != 0:
                dsum[k] = _addmod(dsum[k], _mulmod(vz, a, order, oinv), order)
                dx[r, i] = _addmod(_canon(dx[r, i], d), _mulmod(a, dfac[k], d, dinv), d)
            if b != 0:
                vz = _addmod(vz, _mulmod(b, dfac[k], d, dinv), d)
            dz[r, i] = vz
        for k in range(ns):
            i = scols[k]
            vz = _canon(z[r, i], d)
            if a != 0:
                ssum[k] = _addmod(ssum[k], _mulmod(vz, a, order, oinv), order)
                x[r, i] = _addmod(_canon(x[r, i], d), _mulmod(a, sfac[k], d, dinv), d)
            if b != 0:
                vz = _addmod(vz, _mulmod(b, sfac[k], d, dinv), d)
            z[r, i] = vz

    for k in range(nd):
        i = dcols[k]
        dph[i] = _collapse_phase(dph[i], phf, dfac[k], dsum[k], xz, order, po, oinv)
    for k in range(ns):
        i = scols[k]
        ph[i] = _collapse_phase(ph[i], phf, sfac[k], ssum[k], xz, order, po, oinv)

    # Destabilizer f becomes stabilizer f, and stabilizer f becomes Z_q.
    for r in range(nr):
        dx[r, f] = xf[r]
        dz[r, f] = zf[r]
        x[r, f] = 0
        z[r, f] = 0
    z[q, f] = 1
    dph[f] = phf


@_kernel
def _det_measure_kernel(x, z, ph, dx, q, d, order, po):
    """
    The phase (mod the order) of the product of stabilizers that equals Z_q, for a deterministic
    Z measurement of qudit q.  Destabilizer i's X power on qudit q is the power of stabilizer i in
    that product.  The outcome is (-phase // po) % d.  Nothing is modified.

    Same result as the ancilla loop of ExtendedTableau._det_measurement.  That loop multiplies the
    stabilizers in one at a time, in column order; stabilizer i's commutation phase against the
    running product is fac_i * sum_r (sum_{j < i} fac_j z[r, j]) * x[r, i].  Here it is computed row
    by row with a running prefix sum, which reads the arrays in memory order.
    """
    nr = x.shape[0]
    nc = x.shape[1]
    oinv = 1.0 / order
    cols = np.empty(nc, np.int64)
    facs = np.empty(nc, np.int64)
    k = 0
    for i in range(nc):
        v = _canon(dx[q, i], d)
        if v != 0:
            cols[k] = i
            facs[k] = v
            k += 1

    commute = np.zeros(k, np.int64)   # sum_r prefix_r * x[r, i]
    xz = np.zeros(k, np.int64)        # sum_r x[r, i] * z[r, i]
    for r in range(nr):
        prefix = 0                    # sum_{j < i} fac_j * z[r, j], mod the order
        for j in range(k):
            i = cols[j]
            xv = _canon(x[r, i], d)
            zv = _canon(z[r, i], d)
            if xv != 0:
                if prefix != 0:
                    commute[j] = _addmod(commute[j], _mulmod(prefix, xv, order, oinv), order)
                if zv != 0:
                    xz[j] = _addmod(xz[j], _mulmod(xv, zv, order, oinv), order)
            if zv != 0:
                prefix = _addmod(prefix, _mulmod(zv, facs[j], order, oinv), order)

    phase = 0
    for j in range(k):
        fac = facs[j]
        c = (_mulmod(fac, commute[j], order, oinv)
             + _mulmod(xz[j], _triangular_mod(fac, order, oinv), order, oinv) * po)
        phase += _mulmod(fac, _canon(ph[cols[j]], order), order, oinv)
        phase += _scaled_mod(c, po, order)
        while phase >= order:
            phase -= order
    return phase


# False when numba's JIT is switched off (NUMBA_DISABLE_JIT=1).  The kernels above are then plain
# Python functions, much slower than the reference code, so ExtendedTableau doesn't use them.
JIT_ENABLED = isinstance(_cnot_kernel, Dispatcher)
