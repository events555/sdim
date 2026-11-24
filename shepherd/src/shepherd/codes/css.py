"""
css.py -- qudit CSS codes from their check matrices, and the hypergraph product.

A CSS code over F_q (q prime) is given by H_X and H_Z with H_X @ H_Z.T == 0 mod q. Row r of H_X is
the stabilizer X^{H_X[r, 0]} (x) X^{H_X[r, 1]} (x) ..., and likewise for H_Z with Z. The logical
operators are computed here: Z-logicals span ker(H_X) / rowspace(H_Z) and X-logicals span
ker(H_Z) / rowspace(H_X), paired so that L_X @ L_Z.T == I mod q.

Constructors:
    repetition_code(L, q)              [[L, 1, L]], Z checks Z_i Z_{i+1}^-1 only
    surface_code(L, q)                 hgp(rep_L, rep_L): planar code [[L^2 + (L-1)^2, 1, L]]
    lacross_code(n, ell, q, periodic)  hgp(H, H) with H the checks of 1 + x + x^ell, length n
    hgp(H1, H2, q)                     any hypergraph product
"""

from __future__ import annotations

import math

import numpy as np

from shepherd import gf_q
from shepherd.qecc import QECC, PauliString


class CSSCode(QECC, name="css"):
    """
    Args:
        dimension: Prime q.
        H_X, H_Z: Check matrices, entries mod q, each with n columns. Either may have no rows.
        distance: The code distance if known (None if not). Only recorded, never computed here.
        label: Name used in task metadata, e.g. "surface_L5".
        family: Code family for grouping results, e.g. "surface".
        check: Verify H_X @ H_Z.T == 0 mod q.
    """

    def __init__(self, dimension: int, H_X, H_Z, *, distance: int | None = None,
                 label: str = "css", family: str = "css", check: bool = True):
        super().__init__(dimension)
        q = dimension
        hx = np.asarray(H_X, dtype=np.int64)
        hz = np.asarray(H_Z, dtype=np.int64)
        n = hx.shape[1] if hx.ndim == 2 and hx.size else hz.shape[1]
        self.H_X = (hx.reshape(-1, n) if hx.size else np.zeros((0, n), dtype=np.int64)) % q
        self.H_Z = (hz.reshape(-1, n) if hz.size else np.zeros((0, n), dtype=np.int64)) % q
        if self.H_X.shape[1] != self.H_Z.shape[1]:
            raise ValueError(f"H_X has {self.H_X.shape[1]} columns but H_Z has {self.H_Z.shape[1]}")
        if check and np.any((self.H_X @ self.H_Z.T) % q):
            raise ValueError("H_X @ H_Z.T is not 0 mod q: the X and Z checks do not commute")
        self.distance = distance
        self.label = label
        self.family = family
        self._logicals: tuple[np.ndarray, np.ndarray] | None = None

    # -- parameters -----------------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.H_X.shape[1]

    @property
    def k(self) -> int:
        q = self.dimension
        return self.n - gf_q.rank(self.H_X, q) - gf_q.rank(self.H_Z, q)

    def metadata(self) -> dict:
        """What results rows need to be grouped and normalized: JSON values only."""
        return {"code": self.label, "family": self.family, "n": self.n, "k": self.k,
                "d": self.distance, "q": self.dimension}

    # -- logical operators ----------------------------------------------------------------------

    def logical_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """(L_X, L_Z), each k x n mod q, with L_X @ L_Z.T == I and both commuting with the checks."""
        if self._logicals is None:
            q = self.dimension
            lz = gf_q.quotient_basis(gf_q.nullspace(self.H_X, q), self.H_Z, q)
            lx0 = gf_q.quotient_basis(gf_q.nullspace(self.H_Z, q), self.H_X, q)
            if lz.shape[0] != lx0.shape[0]:
                raise AssertionError(f"found {lx0.shape[0]} X-logicals but {lz.shape[0]} Z-logicals")
            # Change basis on the X side only, so L_Z keeps its (reduced, sparse-ish) form.
            lx = (gf_q.inverse(lx0 @ lz.T % q, q) @ lx0) % q if lz.shape[0] else lx0
            self._logicals = (lx, lz)
        return self._logicals

    def stabilizers(self) -> list[PauliString]:
        q, n = self.dimension, self.n
        zeros = [0] * n
        return ([PauliString(q, 0, row.tolist(), zeros) for row in self.H_X]
                + [PauliString(q, 0, zeros, row.tolist()) for row in self.H_Z])

    def logical_operators(self) -> tuple[list[PauliString], list[PauliString]]:
        q, n = self.dimension, self.n
        lx, lz = self.logical_matrices()
        zeros = [0] * n
        return ([PauliString(q, 0, row.tolist(), zeros) for row in lx],
                [PauliString(q, 0, zeros, row.tolist()) for row in lz])

    def __repr__(self) -> str:
        return (f"CSSCode({self.label!r}, q={self.dimension}, n={self.n}, k={self.k}, "
                f"d={self.distance})")


# -- classical seeds -----------------------------------------------------------------------------

def repetition_checks(L: int, q: int) -> np.ndarray:
    """(L-1) x L checks e_i - e_{i+1} mod q."""
    h = np.zeros((L - 1, L), dtype=np.int64)
    for i in range(L - 1):
        h[i, i], h[i, i + 1] = 1, q - 1
    return h


def lacross_checks(n: int, ell: int, periodic: bool = False) -> np.ndarray:
    """
    Checks of the polynomial 1 + x + x^ell on n bits: row i has ones at i, i+1, i+ell.
    Open boundary: n - ell rows (full rank over every field, so the classical code has
    dimension ell for every q). Periodic: n rows, indices mod n (the dimension then depends on q).
    """
    if not 1 < ell < n:
        raise ValueError(f"need 1 < ell < n, got ell={ell}, n={n}")
    rows = n if periodic else n - ell
    h = np.zeros((rows, n), dtype=np.int64)
    for i in range(rows):
        for s in (0, 1, ell):
            h[i, (i + s) % n] = 1
    return h


def classical_distance(h, q: int, max_codewords: int = 1 << 22) -> float:
    """
    Minimum Hamming weight of a non-zero x with h @ x == 0 mod q, by enumerating the code
    (math.inf for the zero code). Refuses codes with more than `max_codewords` codewords.
    """
    basis = gf_q.nullspace(h, q)
    kc, n = basis.shape
    if kc == 0:
        return math.inf
    total = q ** kc
    if total > max_codewords:
        raise ValueError(f"the code has {total} codewords; raise max_codewords to enumerate it")
    best = n
    powers = q ** np.arange(kc, dtype=np.int64)
    chunk = 1 << 16
    for start in range(1, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        coeffs = (idx[:, None] // powers) % q
        words = coeffs @ basis % q
        best = min(best, int(np.count_nonzero(words, axis=1).min()))
    return best


# -- hypergraph product --------------------------------------------------------------------------

def hgp(H1, H2, q: int, *, label: str = "hgp", family: str = "hgp",
        distance: int | None | str = "compute") -> CSSCode:
    """
    Hypergraph product over F_q, on n1*n2 + m1*m2 qudits:
        H_X = [ H1 (x) I_n2 | I_m1 (x) H2^T ]
        H_Z = [ I_n1 (x) H2 | -H1^T (x) I_m2 ]
    The minus sign makes H_X @ H_Z.T == 0 for every q (it does nothing at q = 2).

    distance="compute" uses d = min(d(H1), d(H2), d(H1^T), d(H2^T)) over the codes that are not
    zero (Tillich-Zemor), with each classical distance found by enumeration.
    """
    h1 = np.asarray(H1, dtype=np.int64) % q
    h2 = np.asarray(H2, dtype=np.int64) % q
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    hx = np.hstack([np.kron(h1, np.eye(n2, dtype=np.int64)),
                    np.kron(np.eye(m1, dtype=np.int64), h2.T)]) % q
    hz = np.hstack([np.kron(np.eye(n1, dtype=np.int64), h2),
                    -np.kron(h1.T, np.eye(m2, dtype=np.int64))]) % q
    if distance == "compute":
        ds = [classical_distance(m, q) for m in (h1, h2, h1.T, h2.T)]
        finite = [x for x in ds if x != math.inf]
        distance = int(min(finite)) if finite else None
    return CSSCode(q, hx, hz, distance=distance, label=label, family=family)


# -- named families ------------------------------------------------------------------------------

def repetition_code(L: int, q: int) -> CSSCode:
    return CSSCode(q, np.zeros((0, L), dtype=np.int64), repetition_checks(L, q), distance=L,
                   label=f"repetition_L{L}", family="repetition")


def surface_code(L: int, q: int) -> CSSCode:
    h = repetition_checks(L, q)
    return hgp(h, h, q, label=f"surface_L{L}", family="surface", distance=L)


def lacross_code(n: int, ell: int, q: int, periodic: bool = False) -> CSSCode:
    h = lacross_checks(n, ell, periodic)
    bc = "periodic" if periodic else "open"
    return hgp(h, h, q, label=f"lacross_n{n}_l{ell}_{bc}", family=f"lacross_{bc}")
