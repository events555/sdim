"""
gf_q.py -- linear algebra over the prime field F_q, on integer NumPy arrays.

Every function takes integer arrays (any values; they are reduced mod q) and returns int64 arrays
with entries in [0, q). Vectors are rows: a "basis" is a 2-D array whose rows are the basis
vectors, and an empty basis has shape (0, n).

Entries stay below q, so products stay below q**2 and fit int64 for any q below 2**31.
"""

from __future__ import annotations

import numpy as np


def _as_matrix(a, q: int) -> np.ndarray:
    m = np.asarray(a, dtype=np.int64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {m.shape}")
    return m % q


def rref(a, q: int) -> tuple[np.ndarray, list[int]]:
    """
    Reduced row echelon form mod q. Returns (R, pivots): the non-zero rows of the RREF and the
    pivot column of each. R[i, pivots[i]] == 1, and every other entry of a pivot column is 0.
    """
    r_mat = _as_matrix(a, q).copy()
    rows, cols = r_mat.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(r_mat[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            r_mat[[r, p]] = r_mat[[p, r]]
        r_mat[r] = r_mat[r] * pow(int(r_mat[r, c]), -1, q) % q
        others = np.nonzero(r_mat[:, c])[0]
        others = others[others != r]
        if others.size:
            r_mat[others] = (r_mat[others] - np.outer(r_mat[others, c], r_mat[r])) % q
        pivots.append(c)
        r += 1
    return r_mat[:r], pivots


def rank(a, q: int) -> int:
    return len(rref(a, q)[1])


def nullspace(a, q: int) -> np.ndarray:
    """Basis (rows) of {x : a @ x == 0 mod q}."""
    m = _as_matrix(a, q)
    n = m.shape[1]
    r_mat, pivots = rref(m, q)
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    basis = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for i, pc in enumerate(pivots):
            basis[k, pc] = (-r_mat[i, f]) % q
    return basis


def row_space_contains(basis, v, q: int) -> bool:
    """Whether v lies in the row space of `basis`."""
    b = _as_matrix(basis, q)
    if b.shape[0] == 0:
        return not np.any(np.asarray(v) % q)
    return rank(np.vstack([b, _as_matrix(v, q)]), q) == rank(b, q)


def quotient_basis(v, w, q: int) -> np.ndarray:
    """
    Representatives (rows) of a basis of span(v) / span(w): vectors of span(v), independent of
    each other and of span(w). Used for logical operators: ker(H_X) / rowspace(H_Z).

    Each row of v is first reduced against the RREF of w, which zeroes it on w's pivot columns
    without changing its class mod span(w); the RREF of what is left is the basis.
    """
    vm = _as_matrix(v, q)
    n = vm.shape[1]
    wm = _as_matrix(w, q) if np.size(w) else np.zeros((0, n), dtype=np.int64)
    if wm.shape[1] != n:
        raise ValueError(f"v has {n} columns but w has {wm.shape[1]}")
    rw, pw = rref(wm, q)
    reduced = (vm - vm[:, pw] @ rw) % q if pw else vm
    basis, _ = rref(reduced, q)
    return basis


def solve(a, b, q: int) -> np.ndarray | None:
    """
    One solution x of a @ x == b mod q (free variables set to 0), or None if there is none.
    `a` is m x n (n may be 0) and `b` has length m.
    """
    m = np.asarray(a, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.int64).reshape(-1)
    if m.ndim != 2:
        m = m.reshape(rhs.shape[0], -1)
    n = m.shape[1]
    if m.shape[0] != rhs.shape[0]:
        raise ValueError(f"a has {m.shape[0]} rows but b has {rhs.shape[0]} entries")
    r_mat, pivots = rref(np.hstack([m % q, (rhs % q).reshape(-1, 1)]), q)
    if pivots and pivots[-1] == n:
        return None                      # a pivot in the right-hand side: inconsistent
    x = np.zeros(n, dtype=np.int64)
    for i, pc in enumerate(pivots):
        x[pc] = r_mat[i, n]
    return x


def solve_with_kernel(a, b, q: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Every solution of a @ x == b mod q, from one row reduction: (x0, K), where x0 is the solution
    `solve` returns and the rows of K are a basis of ker(a), so the solutions are exactly
    x0 + c @ K (mod q) for c in Z_q^len(K). None if there is no solution.
    """
    m = np.asarray(a, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.int64).reshape(-1)
    if m.ndim != 2:
        m = m.reshape(rhs.shape[0], -1)
    n = m.shape[1]
    if m.shape[0] != rhs.shape[0]:
        raise ValueError(f"a has {m.shape[0]} rows but b has {rhs.shape[0]} entries")
    r_mat, pivots = rref(np.hstack([m % q, (rhs % q).reshape(-1, 1)]), q)
    if pivots and pivots[-1] == n:
        return None
    x0 = np.zeros(n, dtype=np.int64)
    for i, pc in enumerate(pivots):
        x0[pc] = r_mat[i, n]
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    kernel = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        kernel[k, f] = 1
        for i, pc in enumerate(pivots):
            kernel[k, pc] = (-r_mat[i, f]) % q
    return x0, kernel


def inverse(a, q: int) -> np.ndarray:
    """Inverse of a square matrix mod q; raises ValueError if it is singular."""
    m = _as_matrix(a, q)
    k = m.shape[0]
    if m.shape != (k, k):
        raise ValueError(f"expected a square matrix, got shape {m.shape}")
    r_mat, pivots = rref(np.hstack([m, np.eye(k, dtype=np.int64)]), q)
    if pivots[:k] != list(range(k)) or len(pivots) < k:
        raise ValueError("matrix is singular mod q")
    return r_mat[:k, k:]
