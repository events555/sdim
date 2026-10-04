import math
import operator
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional
from sdim.tableau.dataclasses import MeasurementResult, Tableau
from sdim.tableau.tableau_optimized import hadamard_optimized, phase_optimized, hadamard_inv_optimized, phase_inv_optimized, cnot_optimized, cnot_inv_optimized
from sdim.tableau.tableau_optimized import (
    JIT_ENABLED, _cnot_kernel, _det_measure_kernel, _exponentiate_kernel, _first_x_kernel,
    _hadamard_kernel, _multiply_kernel, _pauli_kernel, _phase_kernel, _random_measure_kernel,
    _reduce_kernel,
)
from numba.core.errors import TypingError

_INT64 = np.dtype(np.int64)

@dataclass
class ExtendedTableau(Tableau):
    """
    Represents an extended stabilizer tableau for quantum circuit simulation.

    This class extends the Tableau class by including destabilizer information,
    which allows for more efficient simulation of certain quantum operations.

    This follows as a generalization to prime dimensions from
    "Improved Simulation of Stabilizer Circuits" by Aaronson and Gottesman.

    Attributes:
        destab_phase_vector (np.ndarray): The phase vector for destabilizers.
        destab_z_block (np.ndarray): The Z block for destabilizers.
        destab_x_block (np.ndarray): The X block for destabilizers.
    """
    destab_phase_vector: Optional[np.ndarray] = None
    destab_z_block: Optional[np.ndarray] = None
    destab_x_block: Optional[np.ndarray] = None

    # Cache for _fast_params: the six arrays, the dimension, the number of qudits and the answer.
    # A plain class attribute without an annotation, so it is not a dataclass field.
    _fast_key = None

    def __getstate__(self):
        # Copies and pickles leave the cache behind and check their own arrays on first use, so a
        # pickle loaded elsewhere (say, with numba's JIT off) never trusts an answer from here.
        state = self.__dict__.copy()
        state.pop("_fast_key", None)
        return state

    def _arrays(self):
        return (self.x_block, self.z_block, self.phase_vector,
                self.destab_x_block, self.destab_z_block, self.destab_phase_vector)

    def _fast_params(self):
        """
        Returns (d, order, phase_order) when the numba kernels in tableau_optimized apply, else None.

        The kernels need six distinct, C-contiguous, writeable int64 arrays of the tableau's shapes,
        and an odd dimension or d = 2, below 2**31, so the order is below 2**31 as well.  Everything
        else, such as dtype=object arrays of Python integers, goes through the exact reference code.

        The answer is cached on the identity of the arrays, the dimension and the number of qudits.
        An array that is made read-only after that keeps its identity; the gate methods notice it
        when numba refuses to compile the kernel for it (see _drop_fast), and measure checks it.
        """
        key = self._fast_key
        if (key is not None and key[0] is self.x_block and key[1] is self.z_block
                and key[2] is self.phase_vector and key[3] is self.destab_x_block
                and key[4] is self.destab_z_block and key[5] is self.destab_phase_vector
                and key[6] == self.dimension and key[7] == self.num_qudits):
            return key[8]
        arrays = self._arrays()
        params = self._check_fast_params(arrays)
        self._fast_key = arrays + (self.dimension, self.num_qudits, params)
        return params

    def _drop_fast(self):
        """
        Sends this tableau's current arrays to the reference code from now on.

        Called when numba raises TypingError for a kernel call: the kernels are compiled for
        writeable int64 arrays, and numba refuses to compile a variant that writes to a read-only
        array.  That happens while numba picks the compiled code, before anything runs, so nothing
        has changed yet, and the reference code then behaves exactly as it always did (for a
        read-only array, it raises ValueError: assignment destination is read-only).
        """
        self._fast_key = self._arrays() + (self.dimension, self.num_qudits, None)

    def _writeable(self) -> bool:
        """Whether all six arrays are still writeable (they were when _fast_params cached its answer)."""
        return (self.x_block.flags.writeable and self.z_block.flags.writeable
                and self.phase_vector.flags.writeable and self.destab_x_block.flags.writeable
                and self.destab_z_block.flags.writeable and self.destab_phase_vector.flags.writeable)

    def _check_fast_params(self, arrays):
        if not JIT_ENABLED:
            return None
        d = self.dimension
        n = self.num_qudits
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)):
            return None
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            return None
        d = int(d)
        n = int(n)
        if not 2 <= d < 2**31 or (d % 2 == 0 and d != 2):
            return None
        shapes = ((n, n), (n, n), (n,), (n, n), (n, n), (n,))
        for array, shape in zip(arrays, shapes):
            if type(array) is not np.ndarray or array.dtype != _INT64 or array.shape != shape:
                return None
            flags = array.flags
            if not (flags.c_contiguous and flags.writeable and flags.aligned):
                return None
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                if np.may_share_memory(arrays[i], arrays[j]):
                    return None
        if d % 2:
            return d, d, 1
        return d, 2 * d, 2

    def _fast(self, *indices):
        """
        Returns (d, order, phase_order, *rows) for the numba kernels, or None for the reference code.

        The kernels do no bounds checking, so the qudit indices must be integers in range; negative
        ones count from the end as in NumPy.  Anything else takes the reference code, which raises
        the same error as before.
        """
        params = self._fast_params()
        if params is None:
            return None
        n = self.x_block.shape[0]
        rows = []
        for index in indices:
            if type(index) is not int:
                # NumPy treats a bool index as a mask, not as 0 or 1.
                if isinstance(index, (bool, np.bool_)):
                    return None
                try:
                    index = operator.index(index)
                except TypeError:
                    return None
            if index < -n or index >= n:
                return None
            rows.append(index + n if index < 0 else index)
        return params + tuple(rows)

    def _fast_pauli(self, qudit_index: int, x_exp: int, z_exp: int) -> bool:
        """
        Applies X^x_exp Z^z_exp to a qudit with a single kernel if the fast path applies.

        Conjugating by X^a Z^b leaves the X and Z blocks alone and adds phase_order * (b * x - a * z)
        to every phase, where (x, z) are the powers on the qudit.  That is exactly the effect of the
        H/P/multiplication sequences in tableau_gates.  Returns False, having done nothing, when the
        reference sequence has to run instead (including when it would raise, for an exponent that
        is not invertible mod a composite d).
        """
        fast = self._fast(qudit_index)
        if fast is None:
            return False
        d, order, po, q = fast
        a = int(x_exp) % d
        b = int(z_exp) % d
        if (a > 1 and math.gcd(a, d) != 1) or (b > 1 and math.gcd(b, d) != 1):
            return False
        try:
            _pauli_kernel(self.x_block, self.z_block, self.phase_vector,
                          self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                          q, a, b, d, order, po)
        except TypingError:
            self._drop_fast()
            return False
        return True

    def print_destab_phase_vector(self):
        """
        Prints the phase vector of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer Phase Vector", self.destab_phase_vector)

    def print_destab_z_block(self):
        """
        Prints the Z block of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer Z Block", self.destab_z_block)

    def print_destab_x_block(self):
        """
        Prints the X block of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer X Block", self.destab_x_block)

    def print_tableau(self):
        """
        Prints the full tableau, including phase vector, Z block, X block,
        and the destabilizer components.
        """
        super().print_tableau()
        self.print_destab_phase_vector()
        self.print_destab_z_block()
        self.print_destab_x_block()

    @property
    def destab_tableau(self) -> np.ndarray:
        """
        Returns the destabilizer tableau as a vertically stacked matrix.

        Returns:
            np.ndarray: The destabilizer tableau.
        """
        return np.vstack((self.destab_phase_vector, self.destab_z_block, self.destab_x_block))
    
    @property
    def tableau(self) -> np.ndarray:
        """
        Returns the full tableau with stabilizers and destabilizers.

        Returns:
            np.ndarray: The full tableau.
        """
        return np.hstack((self.stab_tableau, self.destab_tableau))

    def __post_init__(self):
        """
        Initializes the extended tableau with default values if not provided.
        """
        super().__post_init__()
        if self.destab_z_block is None:
            self.destab_z_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)
        if self.destab_x_block is None:
            self.destab_x_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.destab_phase_vector is None:
            self.destab_phase_vector = np.zeros(self.num_qudits, dtype=np.int64)

    def modulo(self):
        """
        Reduces the tableau modulo the qudit dimension.
        """
        fast = self._fast()
        if fast is not None:
            try:
                _reduce_kernel(self.x_block, self.z_block, self.phase_vector,
                               self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                               fast[0], fast[1])
                return
            except TypingError:
                self._drop_fast()
        super().modulo()
        self.destab_x_block %= self.dimension
        self.destab_z_block %= self.dimension
        self.destab_phase_vector %= self.order
    
    def _reduce_rows(self, *rows: int):
        """
        Reduces the given qudit rows mod the dimension and both phase vectors mod the order.

        The gate kernels don't reduce, and the full `modulo()` only runs every 64 gates.  Without this,
        a few chained CNOTs at large d overflow int64, since CNOT adds z * (d - 1).  Reducing after
        every gate keeps all intermediate values below about d**2.
        """
        for row in rows:
            self.x_block[row] %= self.dimension
            self.z_block[row] %= self.dimension
            self.destab_x_block[row] %= self.dimension
            self.destab_z_block[row] %= self.dimension
        self.phase_vector %= self.order
        self.destab_phase_vector %= self.order

    def hadamard(self, qudit_index: int):
        """
        Applies the Hadamard gate to the qudit at the specified index.

        The Hadamard gate performs the following transformations:

        | Input | Output   |
        |-------|----------|
        | $X$   | $Z$      |
        | $Z$   | $X^{-1}$ |

        The phase transformation is given by:
        
        $$ H\cdot X\cdot Z\ket{\psi} = Z\cdot X^{-1}\ket{\psi} = \omega^{d-1} XZ\ket{\psi} $$

        where $\omega = e^{2\pi i / d}$ and $d$ is the qudit dimension.

        Args:
            qudit_index (int): The index of the qudit to apply the Hadamard gate to.
        """
        fast = self._fast(qudit_index)
        if fast is not None:
            d, order, po, q = fast
            try:
                _hadamard_kernel(self.x_block, self.z_block, self.phase_vector,
                                 self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                                 q, d, order, po, False)
                return
            except TypingError:
                self._drop_fast()
        hadamard_optimized(
            self.x_block, self.z_block, self.phase_vector,
            self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
            qudit_index, self.num_qudits, self.phase_order
        )
        self._reduce_rows(qudit_index)

    def hadamard_inv(self, qudit_index: int):
        """
        Applies the inverse Hadamard gate to the qudit at the specified index.

        The inverse Hadamard gate performs the following transformations:

        | Input | Output   |
        |-------|----------|
        | $X$   | $Z^{-1}$ |
        | $Z$   | $X$      |

        Args:
            qudit_index (int): The index of the qudit to apply the inverse Hadamard gate to.
        """
        fast = self._fast(qudit_index)
        if fast is not None:
            d, order, po, q = fast
            try:
                _hadamard_kernel(self.x_block, self.z_block, self.phase_vector,
                                 self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                                 q, d, order, po, True)
                return
            except TypingError:
                self._drop_fast()
        hadamard_inv_optimized(
            self.x_block, self.z_block, self.phase_vector,
            self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
            qudit_index, self.num_qudits, self.phase_order
        )
        self._reduce_rows(qudit_index)

    def phase(self, qudit_index: int):
        """
        Applies the Phase gate to the qudit at the specified index.

        The Phase gate transformations depend on whether the qudit dimension is odd or even:

        For odd dimensions:

        | Input | Output |
        |-------|--------|
        | $X$   | $XZ$   |
        | $Z$   | $Z$    |

        For even dimensions:

        | Input           | Output               |
        |-----------------|----------------------|
        | $X$             | $\omega^{1/2} XZ$    |
        | $Z$             | $Z$                  |

        Where $\omega = e^{2\pi i / d}$ and $d$ is the qudit dimension.

        The phase accumulation for even dimensions is given by:

        $$\\text{phase} += x^2 $$

        where $x$ is the X-power in the Pauli string.

        Args:
            qudit_index (int): The index of the qudit to apply the Phase gate to.
        """
        fast = self._fast(qudit_index)
        if fast is not None:
            d, order, po, q = fast
            try:
                _phase_kernel(self.x_block, self.z_block, self.phase_vector,
                              self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                              q, d, order, po, False)
                return
            except TypingError:
                self._drop_fast()
        phase_optimized(self.x_block, self.z_block, self.phase_vector,
                       self.destab_x_block, self.destab_z_block,
                       self.destab_phase_vector, 
                       qudit_index,
                       self.num_qudits,
                       self.even)
        self._reduce_rows(qudit_index)

    def phase_inv(self, qudit_index: int):
        """
        Applies the inverse Phase gate to the qudit at the specified index.

        The inverse Phase gate transformations depend on whether the qudit dimension is odd or even:

        For odd dimensions:

        | Input | Output    |
        |-------|-----------|
        | $X$   | $XZ^{-1}$ |
        | $Z$   | $Z$       |

        For even dimensions:

        | Input        | Output                  |
        |--------------|-------------------------|
        | $X$          | $\omega^{-1/2} XZ^{-1}$ |
        | $Z$          | $Z$                     |

        Args:
            qudit_index (int): The index of the qudit to apply the inverse Phase gate to.
        """
        fast = self._fast(qudit_index)
        if fast is not None:
            d, order, po, q = fast
            try:
                _phase_kernel(self.x_block, self.z_block, self.phase_vector,
                              self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                              q, d, order, po, True)
                return
            except TypingError:
                self._drop_fast()
        phase_inv_optimized(self.x_block, self.z_block, self.phase_vector,
                       self.destab_x_block, self.destab_z_block,
                       self.destab_phase_vector, 
                       qudit_index,
                       self.num_qudits,
                       self.even)
        self._reduce_rows(qudit_index)
        
    def cnot(self, control: int, target: int):
        """
        Applies the CNOT gate with the specified control and target qudits.

        The CNOT gate performs the following transformations:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes I$ | $X \otimes X$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $I \otimes Z$ | $Z^{-1}Z$     |


        Args:
            control (int): The index of the control qudit.
            target (int): The index of the target qudit.
        """
        fast = self._fast(control, target)
        if fast is not None:
            d, order, po, c, t = fast
            try:
                _cnot_kernel(self.x_block, self.z_block, self.phase_vector,
                             self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                             c, t, d, order, False)
                return
            except TypingError:
                self._drop_fast()
        cnot_optimized(self.x_block, self.z_block, self.destab_x_block, self.destab_z_block,
                       self.num_qudits, self.dimension,
                       control, target)
        self._reduce_rows(control, target)
    
    def cnot_inv(self, control: int, target: int):
        """
        Applies the inverse CNOT gate with the specified control and target qudits.

        The inverse CNOT gate transformations depend on whether the qudit dimension is odd or even:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes X$ | $X \otimes I$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $Z^{-1}Z$     | $I \otimes Z$ |

        For even dimensions:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes X$ | $X \otimes I$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $Z^{-1}Z$     | $I \otimes Z$ |
        Args:
            control (int): The index of the control qudit.
            target (int): The index of the target qudit.
        """
        fast = self._fast(control, target)
        if fast is not None:
            d, order, po, c, t = fast
            try:
                _cnot_kernel(self.x_block, self.z_block, self.phase_vector,
                             self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
                             c, t, d, order, True)
                return
            except TypingError:
                self._drop_fast()
        cnot_inv_optimized(self.x_block, self.z_block, self.destab_x_block, self.destab_z_block,
                       self.num_qudits, self.dimension,
                       control, target)
        self._reduce_rows(control, target)

    def multiply(self, qudit_index: int, scalar: int):
        """
        Apply a multiplicative Clifford gate to the specified qudit.

        Args:
            qudit_index (int): Index of the qudit.
            scalar (int): Multiplicative factor modulo the qudit dimension.
        """
        scalar = int(scalar) % self.dimension
        if math.gcd(scalar, self.dimension) != 1:
            raise ValueError(f"Scalar {scalar} is not coprime with the dimension {self.dimension}.")

        inverse = pow(scalar, -1, self.dimension)
        fast = self._fast(qudit_index)
        if fast is not None:
            try:
                _multiply_kernel(self.x_block, self.z_block, self.destab_x_block, self.destab_z_block,
                                 fast[3], scalar, inverse, fast[0])
                return
            except TypingError:
                self._drop_fast()
        self.z_block[qudit_index, :] = (self.z_block[qudit_index, :] * inverse) % self.dimension
        self.x_block[qudit_index, :] = (self.x_block[qudit_index, :] * scalar) % self.dimension
        self.destab_z_block[qudit_index, :] = (self.destab_z_block[qudit_index, :] * inverse) % self.dimension
        self.destab_x_block[qudit_index, :] = (self.destab_x_block[qudit_index, :] * scalar) % self.dimension
            
    def measure(self, qudit_index: int) -> MeasurementResult:
        """
        Measures the qudit at the specified index in the Z basis.

        The reference code reduces the whole tableau first.  The int64 kernels don't need to, since
        they reduce every entry they read and write back reduced values; they give the same outcome
        and the same tableau mod the dimension (mod the order for the phases), and the very same
        arrays when the tableau starts reduced, as it always does inside Program.

        Args:
            qudit_index (int): The index of the qudit to measure.

        Returns:
            MeasurementResult: The result of the measurement, including whether it was
                               deterministic and the measured value.
        """
        fast = self._fast(qudit_index)
        if fast is not None and not self._writeable():
            # Made read-only since _fast_params cached its answer.  The kernels below that only
            # read would still run, so check here, before any of them does.
            self._drop_fast()
            fast = None
        if fast is None:
            return self._measure_exact(qudit_index)
        d, order, po, q = fast
        x, z, phases = self.x_block, self.z_block, self.phase_vector
        first_xpow, xpow = _first_x_kernel(x, q, d)
        if first_xpow < 0:
            phase = _det_measure_kernel(x, z, phases, self.destab_x_block, q, d, order, po)
            return MeasurementResult(qudit_index, True, (-phase // po) % d)
        if xpow != 1:
            _exponentiate_kernel(x, z, phases, first_xpow, pow(int(xpow), -1, d), d, order, po)
        _random_measure_kernel(x, z, phases, self.destab_x_block, self.destab_z_block,
                               self.destab_phase_vector, q, first_xpow, d, order, po)
        measurement_outcome = random.choice(range(d))
        phases[first_xpow] = (-measurement_outcome * po) % order
        return MeasurementResult(qudit_index, False, measurement_outcome)

    def _measure_exact(self, qudit_index: int) -> MeasurementResult:
        """
        The reference implementation of `measure`, for arrays the numba kernels don't take
        (such as dtype=object arrays of Python integers).  It reduces the whole tableau.
        """
        first_xpow = None
        # Find the first non-zero X in the tableau zlogical
        for i in range(self.num_qudits):
            xpow = self.x_block[qudit_index, i] % self.dimension
            if xpow > 0:
                first_xpow = i
                if xpow != 1:
                    # Calculate multiplicative inverse
                    inverse = pow(int(xpow), -1, self.dimension) 
                    self.exponentiate(first_xpow, inverse)   
                break
        self.x_block %= self.dimension
        self.z_block %= self.dimension
        self.phase_vector %= self.order
        self.destab_x_block %= self.dimension
        self.destab_z_block %= self.dimension
        self.destab_phase_vector %= self.order
        if first_xpow is not None:
            return self._random_measurement(qudit_index, first_xpow)
        return self._det_measurement(qudit_index)
    
    def _mod_dot(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Dot product of two integer vectors mod the order, without int64 overflow.

        Phases only matter mod the order, and reducing every term first keeps each product below
        order**2.  A plain np.dot can overflow at large d, where the entries are up to about d**2.
        """
        return int(np.sum((a % self.order) * (b % self.order) % self.order)) % self.order

    def _power_phase(self, x_col: np.ndarray, z_col: np.ndarray, exponent: int) -> int:
        """
        Phase term ab * n(n-1)/2 * phase_order (mod order) from raising X^a Z^b to the power n.

        Computed with exact Python integers so it cannot overflow at large d.
        """
        n = int(exponent)
        return (self._mod_dot(x_col, z_col) * ((n * (n - 1) // 2) % self.order) % self.order) * self.phase_order

    def _random_measurement(self, qudit_index: int, first_xpow: int) -> MeasurementResult:
        """
        Make Tableau commute with Z measurement operator at qudit_index using the generator at first_xpow
        
        This method is called when the measurement outcome is not deterministic.

        Args:
            qudit_index (int): The index of the qudit to measure.
            first_xpow (int): The index of the first stabilizer with a non-zero X power.

        Returns:
            MeasurementResult: The result of the random measurement.
        """
        # First make Tableau commute with Z measurement operator
        for i in range(self.num_qudits):
            if self.destab_x_block[qudit_index, i] != 0:
                destab_factor = -self.destab_x_block[qudit_index, i] % self.dimension
                commute_phase = self._mod_dot(self.destab_z_block[:, i], self.x_block[:, first_xpow]*destab_factor) # phase factor from commuting
                commute_phase += self._power_phase(self.x_block[:, first_xpow], self.z_block[:, first_xpow], destab_factor) # phase factor from exponentiation
                self.destab_x_block[:, i] = (self.destab_x_block[:, i] + self.x_block[:, first_xpow] * destab_factor) % self.dimension
                self.destab_z_block[:, i] = (self.destab_z_block[:, i] + self.z_block[:, first_xpow] * destab_factor) % self.dimension
                self.destab_phase_vector[i] = (self.destab_phase_vector[i] + self.phase_vector[first_xpow]*destab_factor + self.phase_order * commute_phase) % self.order
            if self.x_block[qudit_index, i] != 0 and i != first_xpow:
                stab_factor = -self.x_block[qudit_index, i] % self.dimension
                commute_phase = self._mod_dot(self.z_block[:, i], self.x_block[:, first_xpow]*stab_factor)
                commute_phase += self._power_phase(self.x_block[:, first_xpow], self.z_block[:, first_xpow], stab_factor)
                self.x_block[:, i] = (self.x_block[:, i] + self.x_block[:, first_xpow] * stab_factor) % self.dimension
                self.z_block[:, i] = (self.z_block[:, i] + self.z_block[:, first_xpow] * stab_factor) % self.dimension
                self.phase_vector[i] = (self.phase_vector[i] + self.phase_vector[first_xpow]*stab_factor + self.phase_order * commute_phase) % self.order
            
        # Set destabilizer equal to first_xpow
        self.destab_x_block[:, first_xpow] = self.x_block[:, first_xpow]
        self.destab_z_block[:, first_xpow] = self.z_block[:, first_xpow]
        self.destab_phase_vector[first_xpow] = self.phase_vector[first_xpow]
        # Generate measurement outcome
        self.z_block[:, first_xpow] = 0
        self.z_block[qudit_index, first_xpow] = 1
        self.x_block[:, first_xpow] = 0
        measurement_outcome = random.choice(range(self.dimension))
        self.phase_vector[first_xpow] = (-measurement_outcome * self.phase_order) % self.order
        return MeasurementResult(qudit_index, False, measurement_outcome)

    def _det_measurement(self, qudit_index: int) -> MeasurementResult:
        """
        Use ancilla to obtain the right phase value for the measurement outcome
        
        This method is called when the measurement outcome is deterministic.

        Args:
            qudit_index (int): The index of the qudit to measure.

        Returns:
            MeasurementResult: The result of the deterministic measurement.
        """
        ancilla_x = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_z = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_phase = 0
        for i in range(self.num_qudits):
            factor = self.destab_x_block[qudit_index, i] % self.dimension
            if factor != 0:
                commute_phase = self._mod_dot(ancilla_z, factor * self.x_block[:, i]) # phase factor from commuting
                commute_phase += self._power_phase(self.x_block[:, i], self.z_block[:, i], factor) # phase factor from exponentiation
                # Reduce as we go (mod the order, which keeps every phase below the same) so the
                # running sums can't overflow at large d.
                ancilla_x = (ancilla_x + self.x_block[:, i] * factor) % self.order
                ancilla_z = (ancilla_z + self.z_block[:, i] * factor) % self.order
                ancilla_phase = (ancilla_phase + int(factor) * int(self.phase_vector[i]) + self.phase_order * commute_phase) % self.order
        ancilla_x %= self.dimension
        ancilla_z %= self.dimension
        ancilla_phase %= self.order
        measurement_outcome = (-ancilla_phase // self.phase_order) % self.dimension
        return MeasurementResult(qudit_index, True, measurement_outcome)

    def exponentiate(self, col: int, exponent: int):
        """
        Exponentiates a Pauli string by the given exponent.

        This operation performs the following transformation:
        $$(X^a Z^b)^n = \omega^{(ab\cdot n(n-1)/2)} X^{(na)} Z^{(nb)}$$

        Args:
            col (int): The column index of the Pauli string to exponentiate.
            exponent (int): The exponent to raise the Pauli string to.
        """
        self.phase_vector[col] = (int(self.phase_vector[col]) * int(exponent)
                                  + self._power_phase(self.x_block[:, col], self.z_block[:, col], exponent)) % self.order
        self.x_block[:, col] *=  exponent 
        self.z_block[:, col] *= exponent
        self.phase_vector[col] %= self.order
