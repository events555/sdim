# sdim/simulators/frame_simulator.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

_cupy: Any = None
try:  # GPU backend is optional; falls back to numpy when absent.
    import cupy as _cupy
except Exception:
    pass


from ..gates.registry import (
    gate_id_to_name,
    gate_name_to_id,
    is_gate_annotating,
    is_gate_collapsing,
    is_gate_noisy,
    is_gate_pauli,
    is_gate_records,
    is_gate_two_qubit,
)
from ..noise.sampling import sample_channel


@dataclass
class PauliFrameSimulator:
    """
    An internal engine for simulating quantum circuits using Pauli frames.

    This class takes a pre-processed circuit representation (IR), a noiseless
    reference sample, and sampled noise instances. It then simulates the
    circuit for a number of shots by tracking the evolution of Pauli error
    frames and produces raw noisy measurement outcomes.

    This class is intended for internal use by user-facing sampler objects.

    Attributes:
        ir_array (np.ndarray): The intermediate representation of the circuit.
        dimension (int): Dimension of each qudit.
        num_qudits (int): Number of qudits in the circuit.
        num_total_measurements (int): Total number of measurement results per shot.
        id_to_pauli_frame_op_map (Dict[int, Callable]): Maps gate_id to Pauli frame update functions.
    """

    ir_array: np.ndarray
    args_pool: np.ndarray
    dimension: int
    num_qudits: int
    num_total_measurements: int
    measurement_records: Optional[list] = None
    backend: str = "numpy"
    rng_seed: Optional[int] = None

    id_to_pauli_frame_op_map: Dict[int, Callable[..., None]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        """Initialize gate-op map and the array/RNG backend."""
        self._setup_backend()
        self.id_to_pauli_frame_op_map = {
            gate_name_to_id("H"): self._op_H,
            gate_name_to_id("H_INV"): self._op_H_INV,
            gate_name_to_id("P"): self._op_P,
            gate_name_to_id("P_INV"): self._op_P_INV,
            gate_name_to_id("CNOT"): self._op_CNOT,
            gate_name_to_id("CNOT_INV"): self._op_CNOT_INV,
            gate_name_to_id("CZ"): self._op_CZ,
            gate_name_to_id("CZ_INV"): self._op_CZ_INV,
            gate_name_to_id("SWAP"): self._op_SWAP,
            gate_name_to_id("MULTIPLY"): self._op_MUL,
            gate_name_to_id("MULTIPLY_INV"): self._op_MUL,
        }

    def _setup_backend(self) -> None:
        """Select numpy or cupy as the array module + RNG source."""
        self.xp: Any
        self._rng: Any
        if self.backend == "cupy":
            if _cupy is None:
                raise RuntimeError(
                    "backend='cupy' requested but cupy is not installed "
                    "(install the 'bench' dependency group)."
                )
            self.xp = _cupy
            self._fused = True
            self._compile_kernels()
        elif self.backend == "numpy":
            self.xp = np
            self._fused = False
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        self._bind_ops()
        self._seed_base_rng()

    def _bind_ops(self) -> None:
        """Bind add/sub/neg/mul to the backend's reduction strategy.

        cupy reduces to ``[0, d)`` inside each fused kernel; numpy accumulates
        and leaves reduction to the periodic sweep, so its ops are bare
        ufuncs. Either way the gate methods call the same helpers and stay
        branch-free — the lazy-vs-fused split lives only here and at the three
        reduction points in the run loop.
        """
        d = self.dimension
        if self._fused:
            self._addm = lambda a, b, out: self._add_cs(a, b, d, out)
            self._subm = lambda a, b, out: self._sub_cs(a, b, d, out)
            self._negm = lambda a, out: self._neg_cs(a, d, out)
            self._mulm = lambda a, s, out: self._mul_mod(a, s, d, out)
        else:
            self._addm = lambda a, b, out: np.add(a, b, out=out)
            self._subm = lambda a, b, out: np.subtract(a, b, out=out)
            self._negm = lambda a, out: np.negative(a, out=out)
            self._mulm = lambda a, s, out: np.mod(
                np.multiply(a, s, out=out), d, out=out
            )

    def _seed_base_rng(self) -> None:
        """(Re)create the base Generator from ``rng_seed``.

        ``numpy.random.Generator`` and ``cupy.random.Generator`` share the
        ``.random``/``.integers`` API the noise samplers call, so the rest of
        the engine is backend-agnostic. cupy uses counter-based Philox
        (stateless, reproducible, per-shot independent); numpy uses its
        default bit generator. A ``None`` seed means fresh entropy.
        """
        if self.xp is np:
            self._rng = np.random.default_rng(self.rng_seed)
        else:
            seed = 0 if self.rng_seed is None else int(self.rng_seed)
            self._rng = self.xp.random.Generator(
                self.xp.random.Philox4x3210(seed=seed)
            )

    def _compile_kernels(self) -> None:
        """Fused branch-free modular kernels for the cupy gate path.

        Each does one in-register read-modify-write (no temp array, no
        separate reduction pass). Operands must be in ``[0, d)``; a single
        conditional add/subtract restores range after an add/sub of two
        reduced values. This is what lets the gate path drop the periodic
        ``% d`` sweep — the invariant is maintained per op instead.
        """
        cp = self.xp
        self._add_cs = cp.ElementwiseKernel(
            "int64 a, int64 b, int64 d",
            "int64 out",
            "long long t = a + b; if (t >= d) t -= d; out = t;",
            "fs_add_cs",
        )
        self._sub_cs = cp.ElementwiseKernel(
            "int64 a, int64 b, int64 d",
            "int64 out",
            "long long t = a - b; if (t < 0) t += d; out = t;",
            "fs_sub_cs",
        )
        self._neg_cs = cp.ElementwiseKernel(
            "int64 a, int64 d",
            "int64 out",
            "long long t = d - a; if (t >= d) t -= d; out = t;",
            "fs_neg_cs",
        )
        self._mul_mod = cp.ElementwiseKernel(
            "int64 x, int64 a, int64 d",
            "int64 out",
            "out = ((unsigned long long)a * (unsigned long long)x)"
            " % (unsigned long long)d;",
            "fs_mul_mod",
        )

    def _to_device(self, arr: Optional[np.ndarray]):
        """Move a host array onto the active backend (no-op for numpy)."""
        if arr is None or self.xp is np:
            return arr
        return self.xp.asarray(arr)

    def _randint(self, high: int, size: int):
        """Draw ``size`` integers in ``[0, high)`` on the active backend."""
        return self._rng.integers(0, int(high), size=size)

    def _reduce_rows(self, x_frame, z_frame, rows) -> None:
        """Reduce specific frame rows mod d in place (fused-path repair).

        Noise adds leave entries outside ``[0, d)``; the fused gate kernels
        require reduced inputs, so we restore the invariant on just the
        touched rows rather than sweeping the whole frame.
        """
        d = self.dimension
        for r in rows:
            self.xp.mod(x_frame[r], d, out=x_frame[r])
            self.xp.mod(z_frame[r], d, out=z_frame[r])

    # qi is the primary/control qudit index, ti is the target qudit index (or None)
    def _op_H(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        tmp = x_frame[qi].copy()
        self._negm(z_frame[qi], x_frame[qi])  # x = -z
        z_frame[qi] = tmp

    def _op_H_INV(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        tmp = x_frame[qi].copy()
        x_frame[qi] = z_frame[qi]
        self._negm(tmp, z_frame[qi])  # z = -(old x)

    def _op_P(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        self._addm(z_frame[qi], x_frame[qi], z_frame[qi])

    def _op_P_INV(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        self._subm(z_frame[qi], x_frame[qi], z_frame[qi])

    def _op_CNOT(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        if ti is None:
            raise ValueError("CNOT target index (ti) cannot be None")
        self._addm(x_frame[ti], x_frame[qi], x_frame[ti])
        self._subm(z_frame[qi], z_frame[ti], z_frame[qi])

    def _op_CNOT_INV(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        if ti is None:
            raise ValueError("CNOT_INV target index (ti) cannot be None")
        self._subm(x_frame[ti], x_frame[qi], x_frame[ti])
        self._addm(z_frame[qi], z_frame[ti], z_frame[qi])

    def _op_CZ(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        if ti is None:
            raise ValueError("CZ target index (ti) cannot be None")
        self._addm(z_frame[ti], x_frame[qi], z_frame[ti])
        self._addm(z_frame[qi], x_frame[ti], z_frame[qi])

    def _op_CZ_INV(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        if ti is None:
            raise ValueError("CZ_INV target index (ti) cannot be None")
        self._subm(z_frame[ti], x_frame[qi], z_frame[ti])
        self._subm(z_frame[qi], x_frame[ti], z_frame[qi])

    def _op_SWAP(
        self,
        x_frame: np.ndarray,
        z_frame: np.ndarray,
        qi: int,
        ti: Optional[int],
    ):
        if ti is None:
            raise ValueError("SWAP target index (ti) cannot be None")
        tmp = x_frame[qi].copy()
        x_frame[qi] = x_frame[ti]
        x_frame[ti] = tmp
        tmp = z_frame[qi].copy()
        z_frame[qi] = z_frame[ti]
        z_frame[ti] = tmp

    def _op_MUL(self, x, z, qi, ti, a, a_inv):
        self._mulm(x[qi], a, x[qi])
        self._mulm(z[qi], a_inv, z[qi])

    def run_simulation_for_raw_measurements(
        self,
        shots: int,
        reference_sample: np.ndarray,
    ) -> np.ndarray:
        """
        Performs the core noisy simulation.

        Noise is streamed: each noisy gate draws its entropy inline from the
        engine's RNG via ``sample_channel`` at the point of application, so no
        pre-materialized noise banks are needed (memory is independent of
        circuit depth).

        Args:
            shots: Number of shots to simulate.
            reference_sample: The noiseless reference measurement outcomes.

        Returns:
            A NumPy array of shape (shots, num_total_measurements) containing
            the raw noisy measurement outcomes.
        """
        if reference_sample.shape[0] != self.num_total_measurements:
            raise ValueError(
                f"Reference sample length ({reference_sample.shape[0]}) "
                f"does not match expected total measurements ({self.num_total_measurements})."
            )

        xp = self.xp
        d = self.dimension
        rng = self._rng

        x_frame = xp.zeros((self.num_qudits, shots), dtype=xp.int64)
        z_frame = xp.zeros((self.num_qudits, shots), dtype=xp.int64)

        frame_results = xp.empty(
            (self.num_total_measurements, shots), dtype=xp.int64
        )

        measurement_counter = 0
        gate_counter = 0
        no_target = np.iinfo(np.int64).max

        for inst in self.ir_array:
            gate_id = inst["gate_id"]
            q_idx = inst["qudit_index"]
            t_idx = inst["target_index"]
            gate_name = gate_id_to_name(gate_id)
            arg_start = inst["arg_start"]
            args = self.args_pool[arg_start : arg_start + inst["arg_len"]]

            if not self._fused and gate_counter % 128 == 0:
                xp.mod(x_frame, self.dimension, out=x_frame)
                xp.mod(z_frame, self.dimension, out=z_frame)

            if is_gate_collapsing(gate_id):
                if gate_name in ("M_X", "MR_X"):
                    self._op_H_INV(x_frame, z_frame, q_idx, None)
                if is_gate_records(gate_id):
                    reference_outcome = reference_sample[measurement_counter]
                    record = (
                        self.measurement_records[measurement_counter]
                        if self.measurement_records is not None
                        else None
                    )
                    if record is not None:
                        # Random outcome: sample randomness and fold
                        # the destabilizer S0^k into the frame so the choice
                        # propagates to correlated later measurements
                        eta, s, S0_z, S0_x = record
                        k = self._randint(s, shots)
                        S0_z = self._to_device(S0_z)
                        S0_x = self._to_device(S0_x)
                        z_frame += S0_z[:, None] * k[None, :]
                        x_frame += S0_x[:, None] * k[None, :]
                        if self._fused:
                            # Injection breaks the [0,d) invariant the fused
                            # gate kernels assume; restore it before any gate.
                            xp.mod(x_frame, self.dimension, out=x_frame)
                            xp.mod(z_frame, self.dimension, out=z_frame)
                    flip, _ = sample_channel(
                        gate_name, d, shots, args, xp=xp, rng=rng
                    )
                    frame_results[measurement_counter, :] = (
                        reference_outcome + x_frame[q_idx] + flip[:, 0]
                    ) % self.dimension
                if gate_name in ("MR", "MR_X", "RESET"):
                    x_frame[q_idx, :] = 0
                    z_frame[q_idx, :] = 0
                if gate_name in ("M_X", "MR_X"):
                    self._op_H(x_frame, z_frame, q_idx, None)
            elif is_gate_noisy(gate_id) and not is_gate_collapsing(gate_id):
                noise, erased = sample_channel(
                    gate_name, d, shots, args, xp=xp, rng=rng
                )
                if is_gate_two_qubit(gate_id):
                    if t_idx == no_target:
                        raise ValueError(
                            f"Two-qubit noisy gate {gate_name} missing target."
                        )
                    if noise is not None:
                        x_frame[q_idx] += noise[:, 0]
                        z_frame[q_idx] += noise[:, 1]
                        x_frame[t_idx] += noise[:, 2]
                        z_frame[t_idx] += noise[:, 3]
                        if self._fused:
                            self._reduce_rows(x_frame, z_frame, (q_idx, t_idx))
                elif noise is not None:
                    x_frame[q_idx] += noise[:, 0]
                    z_frame[q_idx] += noise[:, 1]
                    if self._fused:
                        self._reduce_rows(x_frame, z_frame, (q_idx,))
                if is_gate_records(gate_id):
                    # HERALDED_ERASURE records its per-shot erasure flags.
                    frame_results[measurement_counter, :] = erased

            elif not is_gate_annotating(gate_id) and not is_gate_noisy(
                gate_id
            ):  # Standard gates, Pauli gates, or feedforward
                is_std_quantum_op = False
                if gate_id in self.id_to_pauli_frame_op_map:
                    if q_idx < 0:  # Classical control via q_idx
                        abs_rec_idx = (
                            measurement_counter + q_idx
                        )  # q_idx is negative lookback, e.g., -1
                        if not (0 <= abs_rec_idx < measurement_counter):
                            raise IndexError(
                                f"Gate {gate_name}: Lookback q_idx={q_idx} resolves to {abs_rec_idx}, "
                                f"but current_measurement_idx={measurement_counter}."
                            )

                        noisy_control_val_all_shots = frame_results[
                            abs_rec_idx, :
                        ]  # (shots,)
                        ref_control_val = reference_sample[
                            abs_rec_idx
                        ]  # scalar
                        control_error = (
                            noisy_control_val_all_shots - ref_control_val
                        ) % self.dimension  # (shots,)

                        if gate_name == "CNOT":  # CNOT rec_ctrl, quantum_target
                            x_frame[t_idx, :] = (
                                x_frame[t_idx, :] + control_error
                            ) % self.dimension
                        elif gate_name == "CZ":  # CZ rec_ctrl, quantum_target
                            z_frame[t_idx, :] = (
                                z_frame[t_idx, :] + control_error
                            ) % self.dimension
                        else:  # A standard gate from map, but q_idx is rec. This is an invalid configuration.
                            raise ValueError(
                                f"Gate {gate_name} expects quantum control but got measurement record rec({q_idx})."
                            )

                    elif t_idx < 0:  # Classical control via t_idx
                        abs_rec_idx = measurement_counter + t_idx
                        if not (0 <= abs_rec_idx < measurement_counter):
                            raise IndexError(
                                f"Gate {gate_name}: Lookback t_idx={t_idx} resolves to {abs_rec_idx}, "
                                f"but current_measurement_idx={measurement_counter}."
                            )

                        noisy_control_val_all_shots = frame_results[
                            abs_rec_idx, :
                        ]
                        ref_control_val = reference_sample[abs_rec_idx]
                        control_error = (
                            noisy_control_val_all_shots - ref_control_val
                        ) % self.dimension

                        if gate_name == "CZ":  # CZ quantum_ctrl, rec_target
                            z_frame[q_idx, :] = (
                                z_frame[q_idx, :] + control_error
                            ) % self.dimension
                        elif (
                            gate_name == "CNOT"
                        ):  # CNOT quantum_ctrl, rec_target - usually an error.
                            raise ValueError(
                                f"Gate {gate_name} with quantum control q({q_idx}) cannot target measurement record rec({t_idx})."
                            )
                        else:  # A standard gate from map, but t_idx is rec. Invalid.
                            raise ValueError(
                                f"Gate {gate_name} expects quantum target but got measurement record rec({t_idx})."
                            )
                    elif gate_name == "MULTIPLY":
                        a = int(self.args_pool[inst["arg_start"]])
                        a_inv = pow(a, -1, self.dimension)
                        self._op_MUL(x_frame, z_frame, q_idx, t_idx, a, a_inv)
                        is_std_quantum_op = True
                    elif gate_name == "MULTIPLY_INV":
                        a = int(self.args_pool[inst["arg_start"]])
                        a_inv = pow(a, -1, self.dimension)
                        self._op_MUL(x_frame, z_frame, q_idx, t_idx, a_inv, a)
                        is_std_quantum_op = True
                    else:
                        self.id_to_pauli_frame_op_map[gate_id](
                            x_frame, z_frame, q_idx, t_idx
                        )
                        is_std_quantum_op = True

                if not is_std_quantum_op and not (q_idx < 0 or t_idx < 0):
                    if is_gate_pauli(
                        gate_id
                    ):  # Pauli gates (X, Z, etc.) are errors, they don't change the frame.
                        pass  # Frame is unchanged by ideal Pauli gates; noise is handled by noisy gate type
                    else:
                        raise ValueError(
                            f"Unknown non-Pauli gate_id {gate_id} ({gate_name}) or unhandled operation type in PauliFrameSimulator."
                        )

            if is_gate_records(gate_id):
                measurement_counter += 1

            gate_counter += 1

        xp.mod(x_frame, self.dimension, out=x_frame)
        xp.mod(z_frame, self.dimension, out=z_frame)

        result = frame_results.T
        if xp is not np:
            # Copy back only the (shots, nmeas) measurement bits, not frames.
            result = xp.asnumpy(result)
        return result
