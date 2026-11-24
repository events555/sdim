from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from sdim import Circuit


@dataclass(slots=True)
class PauliString:
    """
    Class for organizing tensor products of single-qudit Pauli operators.

    Represents ω^overall_phase · ⊗_i X^{x_i} Z^{z_i}, with ω = e^{2πi/d}: on each qudit, Z acts
    first and X second. With Z X = ω X Z, every product of these stays in this form.
    """

    dimension: int
    overall_phase: int
    x_powers: list[int]
    z_powers: list[int]

    def __post_init__(self) -> None:
        if len(self.x_powers) != len(self.z_powers):
            raise ValueError("The X and Z exponent lists must have the same length.")
        if int(self.dimension) < 2:
            raise ValueError("Dimension must be at least 2.")

        q = int(self.dimension)
        self.dimension = q
        self.overall_phase = int(self.overall_phase) % q
        self.x_powers = [int(power) % q for power in self.x_powers]
        self.z_powers = [int(power) % q for power in self.z_powers]

    @property
    def num_qudits(self) -> int:
        return len(self.x_powers)

    @classmethod
    def identity(cls, dimension: int, num_qudits: int) -> PauliString:
        return cls(dimension, 0, [0] * num_qudits, [0] * num_qudits)

    def __mul__(self, other: PauliString) -> PauliString:
        """
        The operator product self · other. Moving other's X's left past self's Z's (Z X = ω X Z)
        adds Σ_i z_i(self) · x_i(other) to the phase.
        """
        if not isinstance(other, PauliString):
            return NotImplemented
        if self.dimension != other.dimension or self.num_qudits != other.num_qudits:
            raise ValueError("Pauli strings must act on the same Hilbert space.")
        phase = (self.overall_phase + other.overall_phase
                 + sum(z * x for z, x in zip(self.z_powers, other.x_powers)))
        return PauliString(self.dimension, phase,
                           [a + b for a, b in zip(self.x_powers, other.x_powers)],
                           [a + b for a, b in zip(self.z_powers, other.z_powers)])

    def __pow__(self, exponent: int) -> PauliString:
        """
        self^exponent; a negative exponent gives a power of the inverse. Every Pauli satisfies
        P^(2d) = I (P^d is I for odd d and ±I for even d), so the exponent is taken mod 2d.
        """
        result = PauliString.identity(self.dimension, self.num_qudits)
        for _ in range(int(exponent) % (2 * self.dimension)):
            result = result * self
        return result

    @classmethod
    def symplectic_inner_product(cls, a: "PauliString", b: "PauliString") -> int:
        """
        Given two Pauli operators, return the phase penalty by which they anti-commute.  
        """
        if a.dimension != b.dimension or a.num_qudits != b.num_qudits:
            raise ValueError("Pauli strings must act on the same Hilbert space.")

        return sum(
            (ax * bz) - (az * bx)
            for ax, az, bx, bz in zip(a.x_powers, a.z_powers, b.x_powers, b.z_powers)
        ) % a.dimension

    def __str__(self) -> str:
        factors: list[str] = []
        q = self.dimension

        for index, exponent in enumerate(self.z_powers):
            if exponent == 0:
                continue
            if exponent == 1:
                factors.append(f"Z_{index}")
            elif exponent == q - 1:
                factors.append(f"Z_{index}^{{-1}}")
            else:
                factors.append(f"Z_{index}^{{{exponent}}}")

        for index, exponent in enumerate(self.x_powers):
            if exponent == 0:
                continue
            if exponent == 1:
                factors.append(f"X_{index}")
            elif exponent == q - 1:
                factors.append(f"X_{index}^{{-1}}")
            else:
                factors.append(f"X_{index}^{{{exponent}}}")

        return " ".join(factors) if factors else "I"



class QECC(ABC):
    """
    Interface for organizing constructions of various quantum error correcting codes.
    """
    registry : dict[str, type["QECC"]] = {}

    # name -> module that defines it; imported on first lookup only
    _lazy: ClassVar[dict[str, str]] = {
        "five_qudit":       "shepherd.codes.five_qudit",
        "css":              "shepherd.codes.css",
        "rotated_memory_z": "shepherd.codes.surface.rotated_memory_z",
        "qrm_self_dual":    "shepherd.codes.qrm.self_dual",
    }

    @classmethod
    def lookup(cls, name: str) -> type[QECC]:
        if name not in cls.registry:
            module_path = cls._lazy.get(name)
            if module_path is None:
                raise KeyError(f"Unknown code {name!r}; available: {cls.available()}")
            importlib.import_module(module_path)   # class statement runs -> __init_subclass__ registers it
            if name not in cls.registry:
                raise RuntimeError(f"{module_path} was imported but did not register {name!r}")
        return cls.registry[name]

    @classmethod
    def create(cls, name: str, dimension: int, **params) -> QECC:
        return cls.lookup(name)(dimension, **params)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(set(cls.registry) | set(cls._lazy))

    @classmethod
    def register_lazy(cls, name: str, module_path: str) -> None:
        """Let users add their own codes without editing this file."""
        if name in cls._lazy or name in cls.registry:
            raise ValueError(f"Duplicate key: {name}")
        cls._lazy[name] = module_path

    # By default, have this init, which can be overwritten in the subclass or called by super().__init__(...)
    def __init__(self, dimension: int, **params):
        if dimension < 2:
            raise ValueError("Dimension must be at least 2.")
        self.dimension = dimension
        self.params = params

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is None:
            return
        if name in QECC.registry:
            raise ValueError(f"Duplicate key: {name}")
        QECC.registry[name] = cls
    
    @abstractmethod
    def stabilizers(self) -> list[PauliString]:
        """Stabilizer generators. Subclasses read parameters (distance, ...) from self."""

    def logical_operators(self) -> tuple[list[PauliString], list[PauliString]]:
        """
        (logical X's, logical Z's), paired so that symplectic_inner_product(X̄_i, Z̄_j) is 1 if
        i == j and 0 otherwise, and the X̄'s (and the Z̄'s) commute with each other.
        `logical_paulis` and `logical_cliffords` are built on these.

        Default: computed from `stabilizers()` over F_q (prime q). The candidates are the Paulis
        that commute with every stabilizer, modulo the stabilizers themselves (2k of them); a
        symplectic Gram-Schmidt pass pairs them into X̄_i, Z̄_i. Phases are 0. Override for a
        cheaper or more canonical choice (CSSCode does).
        """
        from shepherd import gf_q
        stabs = self.stabilizers()
        if not stabs:
            raise ValueError(f"{type(self).__name__} has no stabilizers to derive logicals from")
        q, n = self.dimension, stabs[0].num_qudits
        s = np.array([st.x_powers + st.z_powers for st in stabs], dtype=np.int64) % q
        # Every pair of generators must commute: S_x S_z^T - S_z S_x^T == 0 (mod q).
        clash = (s[:, :n] @ s[:, n:].T - s[:, n:] @ s[:, :n].T) % q
        if clash.any():
            i, j = map(int, np.argwhere(clash)[0])
            raise ValueError(f"{type(self).__name__}: stabilizers {i} and {j} do not commute "
                             f"(symplectic product {int(clash[i, j])})")
        # v = [v_x | v_z] commutes with every stabilizer:  S_x v_z - S_z v_x = 0.
        centralizer = gf_q.nullspace(np.hstack([-s[:, n:], s[:, :n]]) % q, q)
        cands = list(gf_q.quotient_basis(centralizer, s, q))

        def omega(a: np.ndarray, b: np.ndarray) -> int:
            return int((a[:n] @ b[n:] - a[n:] @ b[:n]) % q)

        xs: list[PauliString] = []
        zs: list[PauliString] = []
        while cands:
            x = cands.pop(0)
            j = next((j for j, c in enumerate(cands) if omega(x, c)), None)
            if j is None:
                # Cannot happen for commuting stabilizers: the form is non-degenerate on N(S)/S.
                raise RuntimeError(f"{type(self).__name__}: no symplectic partner for a logical "
                                   "operator; this is a bug in logical_operators")
            z = cands.pop(j)
            z = z * pow(omega(x, z), -1, q) % q                      # now omega(x, z) == 1
            cands = [(u - omega(u, z) * x + omega(u, x) * z) % q for u in cands]
            xs.append(PauliString(q, 0, x[:n].tolist(), x[n:].tolist()))
            zs.append(PauliString(q, 0, z[:n].tolist(), z[n:].tolist()))
        return xs, zs

    @property
    def num_logical_qudits(self) -> int:
        return len(self.logical_operators()[0])

    def logical_paulis(self, k: int, j: int, qudit: int = 0) -> PauliString:
        """
        The logical Pauli X̄^k Z̄^j on logical qudit `qudit`, as a PauliString on the physical
        qudits. Negative k or j give powers of the inverses.
        """
        xs, zs = self.logical_operators()
        self._check_logical_qudits((qudit,))
        return xs[qudit] ** k * zs[qudit] ** j

    # Gates logical_cliffords accepts: name -> (sdim's name for it, number of logical qudits).
    LOGICAL_CLIFFORDS: ClassVar[dict[str, tuple[str, int]]] = {
        "H": ("H", 1),             "H_INV": ("H_INV", 1),
        "F": ("H", 1),             "F_INV": ("H_INV", 1),          # F (Fourier) is sdim's H
        "P": ("P", 1),             "P_INV": ("P_INV", 1),
        "CNOT": ("CNOT", 2),       "CNOT_INV": ("CNOT_INV", 2),
        "SUM": ("CNOT", 2),        "SUM_INV": ("CNOT_INV", 2),     # SUM is sdim's CNOT
    }

    def logical_cliffords(self, gate: str, *qudits: int) -> Circuit:
        """
        A circuit on the code's physical qudits that implements the logical `gate` on the given
        logical qudits: one for H, F, P (logical qudit 0 if none is given), control then target
        for CNOT, SUM. Gate names are case-insensitive; see LOGICAL_CLIFFORDS for those accepted.

        Checks the gate and qudits here, then asks `_logical_clifford_circuit`, which codes
        override. Raises NotImplementedError if the code has no implementation of the gate.
        """
        name = gate.upper() if isinstance(gate, str) else gate
        if name not in self.LOGICAL_CLIFFORDS:
            raise ValueError(f"unknown logical Clifford {gate!r}; expected one of "
                             f"{sorted(self.LOGICAL_CLIFFORDS)}")
        sdim_name, arity = self.LOGICAL_CLIFFORDS[name]
        if not qudits and arity == 1:
            qudits = (0,)
        if len(qudits) != arity:
            raise ValueError(f"logical {name} acts on {arity} logical qudit(s), got {len(qudits)}")
        self._check_logical_qudits(qudits)
        circuit = self._logical_clifford_circuit(sdim_name, tuple(qudits))
        if circuit.dimension != self.dimension:
            raise ValueError(f"circuit for logical {name} has dimension {circuit.dimension}, "
                             f"but the code has dimension {self.dimension}")
        return circuit

    def _logical_clifford_circuit(self, gate: str, qudits: tuple[int, ...]) -> Circuit:
        """
        Override in codes that implement logical Cliffords. `gate` is sdim's name (H, H_INV, P,
        P_INV, CNOT or CNOT_INV) and `qudits` the logical qudits, both already checked. The
        circuit acts on physical qudits 0..n-1, plus any ancillas it needs after those. Raise
        NotImplementedError for gates (or qudits) the code cannot do.
        """
        raise NotImplementedError(f"{type(self).__name__} has no circuit for logical {gate}")

    def _check_logical_qudits(self, qudits: tuple[int, ...]) -> None:
        num = self.num_logical_qudits
        for q in qudits:
            if not 0 <= q < num:
                raise IndexError(f"logical qudit {q} out of range; the code has {num}")
        if len(set(qudits)) != len(qudits):
            raise ValueError(f"logical qudits must be distinct, got {qudits}")

    def encoding(self) -> Circuit:
        """Default: project |0...0> onto the code space by measuring every stabilizer.
        Override for codes with a better (fault-tolerant / lower-depth) encoder."""
        return self.encoding_stabilizer_projection(self.stabilizers())

    @staticmethod
    def encoding_stabilizer_projection(stabilizers : list[PauliString]) -> Circuit:
        """
        Simple, universal algorithm for computing an encoding circuit into logical |000...0> for a stabilizer code given its stabilizers.
        Not guaranteed to be fault-tolerant, low depth, or low gate count.
        """
        d = stabilizers[0].dimension
        n = stabilizers[0].num_qudits
        circuit = Circuit(n + len(stabilizers), d)

        for i, s in enumerate(stabilizers):
            anc = n + i
            circuit.add_gate("H", anc)                    # ancilla -> uniform superposition
            for q, (a, b) in enumerate(zip(s.x_powers, s.z_powers)):
                for _ in range(b % d):                    # controlled-Z^b first (X^a Z^b acts Z first)
                    circuit.add_gate("CZ", anc, q)
                for _ in range(a % d):                    # then controlled-X^a
                    circuit.add_gate("CNOT", anc, q)
            circuit.add_gate("H_INV", anc)                # F^dagger, then measure Z
            circuit.add_gate("M", anc)
        return circuit


    @staticmethod
    def encoding_gottesman_algorithm(stabilizers : list[PauliString]) -> Circuit:
        """
        Universal algorithm that derives an encoding of the circuit using Gottesman's symplectic reduction algorithm.
        Not guaranteed to be fault-tolerant, low depth, or low gate count.
        """
        raise NotImplementedError