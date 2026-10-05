r"""
Stabilizer tableau for qudits of any dimension, in the linearized formalism of de Beaudrap (2013).

Conventions (checked against an exact statevector simulator):

- A generator stored as (p, z, x) is the operator $\omega^{-p} W(z, x)$ with
  $W(z, x) = \tau^{x \cdot z} X^x Z^z$, $\omega = e^{2\pi i/d}$ and $\tau = e^{i\pi(d^2+1)/d}$,
  so $\tau^2 = \omega$.  The state is the +1 eigenstate of every generator, so $W(z, x)$ has
  eigenvalue $\omega^p$ on it, and a generator $(m, e_q, 0)$ says that qudit q measures as m.
- $\tau$ has order D = 2d for even d and D = d for odd d (the `order`), and $W(z, x)$ depends on
  z and x mod D only.  For even d, $W(z, x + d e_k) = (-1)^{z_k} W(z, x)$, so the z and x blocks
  are kept mod D and never reduced mod d on their own.  The phase is a power of $\omega$ and is
  kept mod d.
- $W(v) W(w) = \tau^{[v, w]} W(v + w)$ with $[v, w] = z_v \cdot x_w - x_v \cdot z_w$, and
  $W(v)^k = W(k v)$.  Elements of a stabilizer group commute, so $[v, w] = 0 \bmod d$; for even
  d, $[v, w]/2$ is then an integer and
  $(\omega^{-p} W(v))^a (\omega^{-q} W(w))^b = \omega^{-(a p + b q - a b [v, w]/2)} W(a v + b w)$.
  For odd d the correction vanishes, since $\tau^{[v, w]} = 1$.
- $W(v)$ with $v = 0 \bmod d$ is the identity (for even d, $\tau^{d^2} = 1$).  So whether an
  element of the group is a power of $Z_q$, or the identity, only depends on its vector mod d.

A stabilizer group of n qudits can need up to 2n generators when d is composite, so the number
of columns of the blocks varies between n and 2n.
"""
import random
from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Tuple

import numpy as np

from sdim.tableau.dataclasses import MeasurementResult, Tableau

# A generator while measuring: (phase as a Python int mod d, vector [z..., x...] mod the order).
_Generator = Tuple[int, np.ndarray]


def _mulmod(values: np.ndarray, k: int, modulus: int) -> np.ndarray:
    """
    Returns (values * k) % modulus without int64 overflow, for a modulus below 2**32.

    The blocks of a composite tableau are kept mod 2d, which is close to 2**32 for d near 2**31,
    so a plain product of two entries can reach 2**64.  Splitting k into 16-bit halves keeps every
    intermediate value below 2**49.  Arrays of Python integers (dtype=object) are exact already.
    """
    k = int(k) % modulus
    if values.dtype != np.int64:
        return (values * k) % modulus
    values = values % modulus
    if modulus <= 3037000499:        # (modulus - 1)**2 < 2**63
        return (values * k) % modulus
    high, low = divmod(k, 1 << 16)
    return ((((values * high) % modulus) << 16) + values * low) % modulus


def _nonzero(values: np.ndarray, modulus: int) -> np.ndarray:
    """
    Boolean mask of the entries of `values` that are nonzero mod `modulus`, for any dtype.

    The comparison makes the mask boolean before any reduction: on numpy 1.x, `np.any` of an
    array of Python integers (dtype=object) returns the integers themselves, not booleans, so
    `~np.any(values % modulus, axis=0)` would be a bitwise NOT of integers and not a mask.
    """
    return (values % modulus) != 0


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Returns (g, x, y) with a x + b y = g = gcd(a, b) for integers a, b >= 0, not both zero."""
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


@dataclass
class WeylTableau(Tableau):
    """
    Stabilizer tableau for any qudit dimension (used for composite d).

    See the module docstring for the conventions.  The generators are the columns of `z_block`
    and `x_block` (mod the order), with their phases in `phase_vector` (mod d).

    Attributes:
        exact (bool): Kept for compatibility with `Program.simulate(exact=True)`.  Measurements
            are always exact now, so it has no effect.
    """
    exact: bool = False

    def modulo(self):
        """
        Reduces the blocks mod the order and the phases mod the dimension.

        Reducing the blocks mod d, as `Tableau.modulo` does for prime dimensions, would flip the
        sign of generators for even d (W(z, x + d e_k) = (-1)^(z_k) W(z, x)).
        """
        self.z_block %= self.order
        self.x_block %= self.order
        self.phase_vector %= self.dimension

    @staticmethod
    def _generate_measurement_outcome(kappa: int, eta: int, dimension: int) -> int:
        """
        Given distribution parameters generate a random measurement result.

        The outcome is uniform over kappa + eta * k mod dimension, where eta divides the
        dimension.  eta = 0 or eta = dimension gives kappa.

        Args:
            kappa (int): The kappa distribution parameter.
            eta (int): The eta distribution parameter.
            dimension (int): The dimension of the distribution.

        Returns:
            int: A random measurement result.

        Examples:
            >>> _generate_measurement_outcome(2, 3, 6)   # one of
            {2, 5}
            >>> _generate_measurement_outcome(0, 2, 4)   # one of
            {0, 2}
            >>> _generate_measurement_outcome(1, 2, 4)   # one of
            {1, 3}
        """
        if eta == 0 or eta == dimension:
            return kappa % dimension
        else:
            distribution = (eta * random.randint(0, dimension - 1))
            return (kappa + distribution) % dimension
        
    @staticmethod
    def _symplectic_product(row1: np.ndarray, row2: np.ndarray, num_qudits: int) -> int:
        """
        Compute the symplectic product of two rows.

        Args:
            row1 (np.ndarray): First row.
            row2 (np.ndarray): Second row.
            num_qudits (int): Number of qudits.

        Returns:
            int: The symplectic product.
        """
        return np.dot(row1[:num_qudits], row2[num_qudits:]) - np.dot(row1[num_qudits:], row2[:num_qudits])

    def symplectic_product(self, index1: int, index2: int) -> int:
        """
        Compute the symplectic product of two generators, as an exact integer.

        Args:
            index1 (int): Index of the first generator.
            index2 (int): Index of the second generator.

        Returns:
            int: The symplectic product.
        """
        z = self.z_block.astype(object)
        x = self.x_block.astype(object)
        return int(np.dot(z[:, index1], x[:, index2]) - np.dot(z[:, index2], x[:, index1]))

    def append(self, pauli_vector: np.ndarray) -> None:
        """
        Append a Pauli vector to the tableau.

        Args:
            pauli_vector (np.ndarray): The Pauli vector to append.

        Raises:
            ValueError: If the Pauli vector dimensions do not match.
        """
        if pauli_vector.size != self.pauli_size:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.shape[0]}")
        new_phase = pauli_vector[0]
        new_z = np.c_[pauli_vector[1:self.num_qudits+1]]
        new_x = np.c_[pauli_vector[self.num_qudits+1:]]
        self.phase_vector = np.hstack((self.phase_vector, new_phase))
        self.z_block = np.hstack((self.z_block, new_z))
        self.x_block = np.hstack((self.x_block, new_x))

    def update(self, pauli_vector: np.ndarray, index: int) -> None:
        """
        Update a generator in the tableau.

        Args:
            pauli_vector (np.ndarray): The new Pauli vector.
            index (int): The index of the generator to update.

        Raises:
            ValueError: If the Pauli vector dimensions do not match or if the index is invalid.
        """
        if pauli_vector.size != self.pauli_size:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.shape[0]}")
        
        if index < 0 or index >= self.z_block.shape[1]:
            raise ValueError(f"Invalid generator index. Must be between 0 and {self.z_block.shape[1] - 1}")
        self.phase_vector[index] = pauli_vector[0]
        self.z_block[:, index] = pauli_vector[1:self.num_qudits+1]
        self.x_block[:, index] = pauli_vector[self.num_qudits+1:]

    def add_generators(self, index1: int, index2: int, scalar: int = 1):
        """
        Replace the generator at column index1 by its product with the generator at index2 to
        the power scalar.

        Args:
            index1 (int): Index of the generator that is replaced.
            index2 (int): Index of the other generator.
            scalar (int, optional): Power of the generator at index2. Defaults to 1.
        """
        # g**order is the identity, so the power can be reduced, which keeps the products small.
        columns = self._generator_columns()
        p, v = self._combine(columns[index1], 1, columns[index2], int(scalar) % self.order)
        self._store_column(index1, p, v)

    def multiply_generator(self, index: int, scalar: int, allow_non_coprime: bool = False):
        """
        Raise the generator at column index to the power scalar.

        Args:
            index (int): Index of the generator to multiply.
            scalar (int): Scalar to multiply by.
            allow_non_coprime (bool, optional): Allow non-coprime scalars. Defaults to False.

        Raises:
            ValueError: If the scalar is not coprime with the order and allow_non_coprime is False.
        """
        if gcd(int(scalar), self.order) != 1 and not allow_non_coprime:
            raise ValueError(f"Scalar {scalar} is not coprime with the order {self.order}.")
        self.z_block[:, index] = _mulmod(self.z_block[:, index], scalar, self.order)
        self.x_block[:, index] = _mulmod(self.x_block[:, index], scalar, self.order)
        self.phase_vector[index] = (int(self.phase_vector[index]) * int(scalar)) % self.dimension

    def swap_generators(self, index1: int, index2: int):
        """
        Swap generators at index1 to index2.

        Args:
            index1 (int): Index of the first generator.
            index2 (int): Index of the second generator.

        Raises:
            ValueError: If the indices are invalid.
        """
        if index1 < 0 or index2 < 0 or index1 >= self.z_block.shape[1] or index2 >= self.z_block.shape[1]:
            raise ValueError(f"Invalid indices. Must be between 0 and {self.z_block.shape[1] - 1}")
        self.z_block[:, [index1, index2]] = self.z_block[:, [index2, index1]]
        self.x_block[:, [index1, index2]] = self.x_block[:, [index2, index1]]
        self.phase_vector[[index1, index2]] = self.phase_vector[[index2, index1]]

    # Exact group arithmetic used by the measurement.

    def _work_dtype(self):
        """
        int64 when no intermediate value of the measurement arithmetic can overflow, else object.

        Coefficients stay below 2d in absolute value and entries below 2d, so a combination of two
        vectors stays below 8 d**2 and a symplectic product below 8 n d**2.
        """
        d = self.dimension
        if self.z_block.dtype == np.int64 and self.x_block.dtype == np.int64 \
                and 8 * max(self.num_qudits, 1) * d * d < 2**62:
            return np.int64
        return object

    def _generator_columns(self) -> List[_Generator]:
        """The generators as (phase mod d, vector [z..., x...] mod the order)."""
        dtype = self._work_dtype()
        vectors = np.vstack((self.z_block, self.x_block)).astype(dtype) % self.order
        return [(int(self.phase_vector[j]) % self.dimension, vectors[:, j].copy())
                for j in range(vectors.shape[1])]

    def _store_column(self, index: int, phase: int, vector: np.ndarray):
        n = self.num_qudits
        self.phase_vector[index] = phase
        self.z_block[:, index] = vector[:n]
        self.x_block[:, index] = vector[n:]

    def _set_generators(self, columns: List[_Generator]):
        """Replaces all generators, keeping the dtype of the tableau arrays."""
        n = self.num_qudits
        vectors = np.stack([v for _, v in columns], axis=1)
        self.z_block = vectors[:n].astype(self.z_block.dtype)
        self.x_block = vectors[n:].astype(self.x_block.dtype)
        self.phase_vector = np.array([p for p, _ in columns], dtype=self.phase_vector.dtype)

    def _combine(self, a: _Generator, alpha: int, b: _Generator, beta: int) -> _Generator:
        """
        Returns the group element a**alpha * b**beta.

        Raises:
            RuntimeError: If a and b do not commute, which no two elements of a stabilizer group do.
        """
        d, n = self.dimension, self.num_qudits
        pa, va = a
        pb, vb = b
        sp = int(np.dot(va[:n], vb[n:])) - int(np.dot(va[n:], vb[:n]))
        if sp % d:
            raise RuntimeError("Stabilizer generators do not commute; the tableau is inconsistent.")
        p = alpha * pa + beta * pb
        if self.even:
            p -= alpha * beta * (sp // 2)
        return p % d, (alpha * va + beta * vb) % self.order

    def _drop_identities(self, vectors: np.ndarray, phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes the columns that are the identity: W(v) = I when v = 0 mod d.  Such a column with a
        nonzero phase would put a nontrivial multiple of I in the stabilizer group.
        """
        keep = np.any(_nonzero(vectors, self.dimension), axis=0)
        if keep.all():
            return vectors, phases
        if np.any(_nonzero(phases[~keep], self.dimension)):
            raise RuntimeError("The stabilizer group contains a nontrivial multiple of the identity.")
        return vectors[:, keep], phases[keep]

    def _eliminate_row(self, vectors: np.ndarray, phases: np.ndarray, row: int):
        """
        Takes out a pivot for `row`: afterwards every remaining column is 0 mod d in `row`.

        The columns are generators of a group (vectors mod the order, phases mod d).  The pivot is
        a column whose entry has the smallest gcd with d of the row; if there is none, pairs of
        columns are first replaced by (a**x b**y, a**(-b_row/g) b**(a_row/g)) with
        x a_row + y b_row = g, which has determinant 1, so the pair generates the same group.
        Every other column c then becomes c * pivot**f, with f chosen to clear its entry.

        The group generated by the remaining columns and pivot**s, where s = d / gcd(pivot entry, d),
        is the subgroup of elements that are 0 mod d in `row`; pivot**s is appended for that.

        Returns:
            (vectors, phases, pivot) with pivot = (phase, vector), or None if the row is already 0.
        """
        d, n, order = self.dimension, self.num_qudits, self.order
        entries = vectors[row] % d
        nonzero = np.flatnonzero(entries)
        if nonzero.size == 0:
            return vectors, phases, None
        values = [int(v) for v in entries[nonzero]]
        row_gcd = gcd(d, *values)
        candidates = [j for j, v in zip(nonzero, values) if gcd(v, d) == row_gcd]
        merged = not candidates
        if candidates:
            p_index = int(candidates[0])
        else:
            p_index = int(nonzero[0])
            for j in nonzero[1:]:
                a = int(vectors[row, p_index]) % d
                b = int(vectors[row, j]) % d
                g, x, y = _extended_gcd(a, b)
                first = (int(phases[p_index]), vectors[:, p_index])
                second = (int(phases[j]), vectors[:, j])
                new_first = self._combine(first, x, second, y)
                new_second = self._combine(first, -(b // g), second, a // g)
                phases[p_index], vectors[:, p_index] = new_first
                phases[j], vectors[:, j] = new_second
                if gcd(g, d) == row_gcd:
                    break
            entries = vectors[row] % d
            nonzero = np.flatnonzero(entries)

        pivot_vector = vectors[:, p_index].copy()
        pivot_phase = int(phases[p_index])
        g = gcd(int(entries[p_index]), d)
        others = nonzero[nonzero != p_index]
        drop = [p_index]
        if others.size:
            inverse = pow(int(entries[p_index]) // g, -1, d // g)
            factors = ((-(entries[others] // g)) * inverse) % (d // g)
            block = vectors[:, others]
            products = block[:n].T.dot(pivot_vector[n:]) - block[n:].T.dot(pivot_vector[:n])
            if np.any(_nonzero(products, d)):
                raise RuntimeError("Stabilizer generators do not commute; the tableau is inconsistent.")
            block = (block + np.outer(pivot_vector, factors)) % order
            vectors[:, others] = block
            correction = factors * pivot_phase
            if self.even:
                correction = correction - factors * ((products // 2) % d)
            phases[others] = (phases[others] + correction) % d
            # Only the columns changed here can have become the identity.
            identities = others[~np.any(_nonzero(block, d), axis=0)]
            if identities.size:
                if np.any(_nonzero(phases[identities], d)):
                    raise RuntimeError("The stabilizer group contains a nontrivial multiple of the identity.")
                drop.extend(int(j) for j in identities)

        vectors = np.delete(vectors, drop, axis=1)
        phases = np.delete(phases, drop)
        if merged:
            vectors, phases = self._drop_identities(vectors, phases)
        s = d // g
        annihilator = (s * pivot_vector) % order
        if np.any(_nonzero(annihilator, d)):
            vectors = np.concatenate((vectors, annihilator[:, None]), axis=1)
            phases = np.append(phases, np.array([(s * pivot_phase) % d], dtype=phases.dtype))
        elif (s * pivot_phase) % d:
            raise RuntimeError("The stabilizer group contains a nontrivial multiple of the identity.")
        return vectors, phases, (pivot_phase, pivot_vector)

    def _reduce_row(self, target: _Generator, pivot: _Generator, row: int) -> _Generator:
        """
        Multiplies target by the power of pivot that makes its entry in `row` 0 mod d.

        Raises:
            RuntimeError: If no power does, which means target is not in the group.
        """
        d = self.dimension
        t = int(target[1][row]) % d
        if t == 0:
            return target
        a = int(pivot[1][row]) % d
        g = gcd(a, d)
        if t % g:
            raise RuntimeError("Measured operator is not in the stabilizer group; the tableau is inconsistent.")
        k = (-(t // g) * pow(a // g, -1, d // g)) % (d // g)
        return self._combine(target, 1, pivot, k)

    def measure_z(self, qudit_index: int) -> Optional[MeasurementResult]:
        r"""
        Perform a Z measurement on a qudit.

        With S the stabilizer group, let s be the smallest power with $Z_q^s \in S$ up to a phase.
        Eliminating the X row of the qudit first leaves one pivot P that is nonzero there, with
        s = d / gcd(P_x, d), and the subgroup of S that commutes with Z_q is generated by P**s and
        the other columns.  An echelon form of that subgroup mod d reduces $Z_q^{-s}$ row by row to
        an element $\omega^{-c} I$, so $Z_q^s$ has eigenvalue $\omega^c$, and the outcome m is
        uniform over the solutions of s m = c mod d.  Rows are taken where $Z_q^{-s}$ is still
        nonzero, so a deterministic measurement stops as soon as it is reduced.  If s > 1, the
        generators become the pivots of the commuting subgroup plus $\omega^{-m} Z_q$ (at most 2n).

        All arithmetic is exact (int64 where it provably fits, Python integers otherwise), and the
        result does not depend on which representatives mod 2d the tableau holds.

        Args:
            qudit_index (int): Index of the qudit to measure.

        Returns:
            MeasurementResult: The result of the measurement.
        """
        d, n, order = self.dimension, self.num_qudits, self.order
        dtype = self._work_dtype()
        vectors = np.vstack((self.z_block, self.x_block)).astype(dtype) % order
        phases = self.phase_vector.astype(dtype) % d
        rows_left = list(range(2 * n))

        x_row = n + qudit_index
        vectors, phases, pivot = self._eliminate_row(vectors, phases, x_row)
        rows_left.remove(x_row)
        s = 1 if pivot is None else d // gcd(int(pivot[1][x_row]) % d, d)

        # Reduce Z_q^-s with the commuting subgroup, one row where it is nonzero at a time.
        target_vector = np.zeros(2 * n, dtype=dtype)
        target_vector[qudit_index] = (-s) % order
        target = (0, target_vector)
        pivots = []
        while True:
            row = next((r for r in rows_left if target[1][r] % d), None)
            if row is None:
                break
            vectors, phases, pivot = self._eliminate_row(vectors, phases, row)
            rows_left.remove(row)
            if pivot is None:
                raise RuntimeError("Measured operator is not in the stabilizer group; the tableau is inconsistent.")
            pivots.append(pivot)
            target = self._reduce_row(target, pivot, row)
        c = target[0]
        if c % s:
            raise RuntimeError("Inconsistent measurement phase; the tableau is inconsistent.")
        kappa, eta = c // s, d // s

        if s == 1:
            return MeasurementResult(qudit_index=qudit_index, deterministic=True, measurement_value=int(kappa % d))

        # Finish the echelon form, so the commuting subgroup has at most one pivot per row.
        for row in rows_left:
            vectors, phases, pivot = self._eliminate_row(vectors, phases, row)
            if pivot is not None:
                pivots.append(pivot)
        if self._drop_identities(vectors, phases)[0].shape[1]:
            raise RuntimeError("Echelon form left a non-identity column; the tableau is inconsistent.")

        measurement_value = int(self._generate_measurement_outcome(kappa, eta, d))
        z_q = np.zeros(2 * n, dtype=dtype)
        z_q[qudit_index] = 1
        self._set_generators(pivots + [(measurement_value, z_q)])
        return MeasurementResult(qudit_index=qudit_index, deterministic=False, measurement_value=measurement_value)

    def multiply(self, qudit_index: int, scalar: int):
        """
        Apply multiplication gate to qudit at index.

        M_a |j> = |a j mod d> only depends on a mod d, so the scalar is reduced mod d first.  The
        conjugation M_a W(z, x) M_a^-1 = W(a^-1 z, a x) needs a * a^-1 = 1 mod the order (2d for
        even d), not just mod d: the factor tau^(x.z) of W changes sign otherwise.  A scalar
        coprime to an even d is odd, so it is invertible mod 2d as well.

        Args:
            qudit_index (int): Index of the qudit.
            scalar (int): Scalar value to multiply by.

        Raises:
            ValueError: If the scalar is not coprime with the dimension.
        """
        a = int(scalar) % self.dimension
        if gcd(a, self.dimension) != 1:
            raise ValueError(f"Scalar {scalar} is not coprime with the dimension {self.dimension}.")
        self.z_block[qudit_index, :] = _mulmod(self.z_block[qudit_index, :], pow(a, -1, self.order), self.order)
        self.x_block[qudit_index, :] = _mulmod(self.x_block[qudit_index, :], a, self.order)

    def hadamard(self, qudit_index: int):
        """
        Apply generalized Hadamard gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit.
        """
        self.z_block[qudit_index, :], self.x_block[qudit_index, :] = self.x_block[qudit_index, :], -self.z_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order
        self.x_block[qudit_index, :] %= self.order

    def hadamard_inv(self, qudit_index: int):
        """
        Apply inverse generalized Hadamard gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit.
        """
        # Swap and negate the values
        new_z_block = -self.x_block[qudit_index, :].copy()
        new_x_block = self.z_block[qudit_index, :].copy()

        # Apply the modulus operation
        self.z_block[qudit_index, :] = new_z_block % self.order
        self.x_block[qudit_index, :] = new_x_block % self.order

    def phase(self, qudit_index: int):
        """
        Apply phase gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the phase gate.
        """
        self.z_block[qudit_index, :] += self.x_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order

    def phase_inv(self, qudit_index: int):
        """
        Apply inverse phase gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the inverse phase gate.
        """
        self.z_block[qudit_index, :] -= self.x_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order

    def cnot(self, control_index: int, target_index: int):
        """
        Apply CNOT gate to control and target qudits.

        Args:
            control_index (int): Index of the control qudit.
            target_index (int): Index of the target qudit.
        """
        self.z_block[control_index, :] -= self.z_block[target_index, :]
        self.x_block[target_index, :] += self.x_block[control_index, :]
        self.z_block[control_index, :] %= self.order
        self.x_block[target_index, :] %= self.order

    def cnot_inv(self, control_index: int, target_index: int):
        """
        Apply inverse CNOT gate to control and target qudits.

        Args:
            control_index (int): Index of the control qudit.
            target_index (int): Index of the target qudit.
        """
        self.z_block[control_index, :] += self.z_block[target_index, :]
        self.x_block[target_index, :] -= self.x_block[control_index, :]
        self.z_block[control_index, :] %= self.order
        self.x_block[target_index, :] %= self.order

    def x(self, qudit_index: int):
        """
        Apply Pauli X gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the Pauli X gate.
        """
        factor = self.z_block[qudit_index, :]
        self.phase_vector += factor
        self.phase_vector %= self.dimension

    def x_inv(self, qudit_index: int):
        """
        Apply Pauli X inverse gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the inverse Pauli X gate.
        """
        factor = self.z_block[qudit_index, :]
        self.phase_vector -= factor
        self.phase_vector %= self.dimension
    
    def z(self, qudit_index: int):
        """
        Apply Pauli Z gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the Pauli Z gate.
        """
        factor = self.x_block[qudit_index, :]
        self.phase_vector -= factor
        self.phase_vector %= self.dimension
    
    def z_inv(self, qudit_index: int):
        """
        Apply Pauli Z inverse gate to qudit at index.

        Args:
            qudit_index (int): Index of the qudit to apply the inverse Pauli Z gate.
        """
        factor = self.x_block[qudit_index, :]
        self.phase_vector += factor
        self.phase_vector %= self.dimension
    


