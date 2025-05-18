# sdim/simulators/frame_simulator.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any
import numpy as np


from ..gatedata import (
    is_gate_pauli, is_gate_collapsing, is_gate_noisy,
    is_gate_two_qubit, gate_name_to_id, gate_id_to_name,
    is_gate_records, is_gate_annotating, is_gate_collapsing_and_records
)

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
    dimension: int
    num_qudits: int
    num_total_measurements: int # Total number of measurement results expected per shot

    id_to_pauli_frame_op_map: Dict[int, Callable[[np.ndarray, np.ndarray, int, Optional[int]], None]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize the mapping from gate IDs to their Pauli frame operations."""
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

    # qi is the primary/control qudit index, ti is the target qudit index (or None)
    def _op_H(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        tmp = x_frame[qi].copy()
        x_frame[qi] = -z_frame[qi]
        z_frame[qi] = tmp
    def _op_H_INV(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        tmp = x_frame[qi].copy()
        x_frame[qi] = z_frame[qi]
        z_frame[qi] = -tmp
    def _op_P(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        z_frame[qi] += x_frame[qi]
    def _op_P_INV(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        z_frame[qi] -= x_frame[qi]
    def _op_CNOT(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        if ti is None: raise ValueError("CNOT target index (ti) cannot be None")
        x_frame[ti] += x_frame[qi]
        z_frame[qi] -= z_frame[ti]
    def _op_CNOT_INV(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        if ti is None: raise ValueError("CNOT_INV target index (ti) cannot be None")
        x_frame[ti] -= x_frame[qi] 
        z_frame[qi] += z_frame[ti]
    def _op_CZ(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        if ti is None: 
            raise ValueError("CZ target index (ti) cannot be None")
        z_frame[ti] += x_frame[qi]
        z_frame[qi] += x_frame[ti]
    def _op_CZ_INV(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        if ti is None: 
            raise ValueError("CZ_INV target index (ti) cannot be None")
        z_frame[ti] -= x_frame[qi]
        z_frame[qi] -= x_frame[ti]
    def _op_SWAP(self, x_frame: np.ndarray, z_frame: np.ndarray, qi: int, ti: Optional[int]):
        if ti is None: raise ValueError("SWAP target index (ti) cannot be None")
        tmp = x_frame[qi].copy()
        x_frame[qi] = x_frame[ti]
        x_frame[ti] = tmp
        tmp = z_frame[qi].copy()
        z_frame[qi] = z_frame[ti]; z_frame[ti] = tmp
    def _op_MUL(self, x, z, qi, ti, a, a_inv):
        x[qi] = (a      * x[qi]) % self.dimension
        z[qi] = (a_inv  * z[qi]) % self.dimension

    def run_simulation_for_raw_measurements(
        self,
        shots: int,
        reference_sample: np.ndarray, # Shape: (num_total_measurements,)
        noise1_bank: np.ndarray,      # Shape: (num_single_qubit_noise_ops, shots, 2)
        noise2_bank: np.ndarray,      # Shape: (num_two_qubit_noise_ops, shots, 4)
        erased_bank: np.ndarray,       # Shape: (num_heralded_erasure_ops, shots)
        measurement_bank: np.ndarray
    ) -> np.ndarray:
        """
        Performs the core noisy simulation.

        Args:
            shots: Number of shots to simulate.
            reference_sample: The noiseless reference measurement outcomes.
            noise1_bank: Pre-sampled noise for single-qubit noise channels.
            noise2_bank: Pre-sampled noise for two-qubit noise channels.
            erased_bank: Pre-sampled erasure flags for HERALDED_ERASURE.

        Returns:
            A NumPy array of shape (shots, num_total_measurements) containing
            the raw noisy measurement outcomes.
        """
        if reference_sample.shape[0] != self.num_total_measurements:
            raise ValueError(f"Reference sample length ({reference_sample.shape[0]}) "
                             f"does not match expected total measurements ({self.num_total_measurements}).")

        x_frame = np.zeros((self.num_qudits, shots), dtype=np.int64)
        z_frame = np.random.randint(0, self.dimension, (self.num_qudits, shots), dtype=np.int64)

        frame_results = np.empty((self.num_total_measurements, shots), dtype=np.int64)

        noise1_counter = 0
        noise2_counter = 0
        erasure_events_counter = 0
        measurement_counter = 0
        measurement_noise_counter = 0
        gate_counter = 0

        for inst in self.ir_array:
            gate_id = inst['gate_id']
            q_idx = inst['qudit_index'] 
            t_idx = inst['target_index']
            arg0 = inst['arg0']
            gate_name = gate_id_to_name(gate_id)

            if gate_counter % 128 == 0: # Periodic modulo
                x_frame %= self.dimension
                z_frame %= self.dimension

            if is_gate_noisy(gate_id) and is_gate_collapsing(gate_id):
                reference_outcome = reference_sample[measurement_counter]
                if gate_name in ("M_X", "MR_X"):
                    self._op_H_INV(x_frame, z_frame, q_idx, None)
                if is_gate_records(gate_id):
                    frame_results[measurement_counter, :] = (reference_outcome + x_frame[q_idx] + measurement_bank[measurement_noise_counter, :, 0]) % self.dimension
                    measurement_noise_counter += 1
                if gate_name in ("MR", "MR_X", "RESET"):
                    x_frame[q_idx, :] = 0
                    z_frame[q_idx, :] = 0
                if gate_name in ("M_X", "MR_X"):
                    self._op_H(x_frame, z_frame, q_idx, None)
            elif is_gate_noisy(gate_id) and not is_gate_collapsing(gate_id):
                if is_gate_two_qubit(gate_id):
                    if t_idx is np.iinfo(np.int64).max: 
                        raise ValueError(f"Two-qubit noisy gate {gate_name} missing target.")
                    x_frame[q_idx] += noise2_bank[noise2_counter, :, 0]
                    z_frame[q_idx] += noise2_bank[noise2_counter, :, 1]
                    x_frame[t_idx] += noise2_bank[noise2_counter, :, 2]
                    z_frame[t_idx] += noise2_bank[noise2_counter, :, 3]
                    noise2_counter += 1
                else:
                    x_frame[q_idx] += noise1_bank[noise1_counter, :, 0]
                    z_frame[q_idx] += noise1_bank[noise1_counter, :, 1]
                    noise1_counter += 1
                if is_gate_records(gate_id):
                    frame_results[measurement_counter, :] = erased_bank[erasure_events_counter, :]
                    erasure_events_counter += 1
                
            elif not is_gate_annotating(gate_id) and not is_gate_noisy(gate_id): # Standard gates, Pauli gates, or feedforward
                is_std_quantum_op = False
                if gate_id in self.id_to_pauli_frame_op_map:
                    if q_idx < 0: # Classical control via q_idx
                        abs_rec_idx = measurement_counter + q_idx # q_idx is negative lookback, e.g., -1
                        print(f"Gate {gate_name}: Lookback q_idx={q_idx} resolves to {abs_rec_idx}, ")
                        if not (0 <= abs_rec_idx < measurement_counter):
                            raise IndexError(f"Gate {gate_name}: Lookback q_idx={q_idx} resolves to {abs_rec_idx}, "
                                            f"but current_measurement_idx={measurement_counter}.")

                        noisy_control_val_all_shots = frame_results[abs_rec_idx, :]    # (shots,)
                        ref_control_val = reference_sample[abs_rec_idx]                     # scalar
                        control_error = (noisy_control_val_all_shots - ref_control_val) % self.dimension # (shots,)

                        if gate_name == "CNOT": # CNOT rec_ctrl, quantum_target
                            x_frame[t_idx, :] = (x_frame[t_idx, :] + control_error) % self.dimension
                        elif gate_name == "CZ":   # CZ rec_ctrl, quantum_target
                            z_frame[t_idx, :] = (z_frame[t_idx, :] + control_error) % self.dimension
                        else: # A standard gate from map, but q_idx is rec. This is an invalid configuration.
                            raise ValueError(f"Gate {gate_name} expects quantum control but got measurement record rec({q_idx}).")
                    
                    elif t_idx < 0: # Classical control via t_idx
                        abs_rec_idx = measurement_counter + t_idx
                        if not (0 <= abs_rec_idx < measurement_counter):
                            raise IndexError(f"Gate {gate_name}: Lookback t_idx={t_idx} resolves to {abs_rec_idx}, "
                                            f"but current_measurement_idx={measurement_counter}.")

                        noisy_control_val_all_shots = frame_results[abs_rec_idx, :]
                        ref_control_val = reference_sample[abs_rec_idx]
                        control_error = (noisy_control_val_all_shots - ref_control_val) % self.dimension
                        
                        if gate_name == "CZ": # CZ quantum_ctrl, rec_target
                            z_frame[q_idx, :] = (z_frame[q_idx, :] + control_error) % self.dimension
                        elif gate_name == "CNOT": # CNOT quantum_ctrl, rec_target - usually an error.
                            raise ValueError(f"Gate {gate_name} with quantum control q({q_idx}) cannot target measurement record rec({t_idx}).")
                        else: # A standard gate from map, but t_idx is rec. Invalid.
                            raise ValueError(f"Gate {gate_name} expects quantum target but got measurement record rec({t_idx}).")
                    elif gate_name == "MULTIPLY":
                        a = int(arg0)
                        a_inv = pow(a, -1, self.dimension)
                        self._op_MUL(x_frame, z_frame, q_idx, t_idx, a, a_inv)
                        is_std_quantum_op = True
                    elif gate_name == "MULTIPLY_INV":
                        a = int(arg0)
                        a_inv = pow(a, -1, self.dimension)
                        self._op_MUL(x_frame, z_frame, q_idx, t_idx, a_inv, a)
                        is_std_quantum_op = True
                    else:
                        self.id_to_pauli_frame_op_map[gate_id](x_frame, z_frame, q_idx, t_idx)
                        is_std_quantum_op = True
                
                if not is_std_quantum_op and not (q_idx < 0 or t_idx < 0):
                    if is_gate_pauli(gate_id): # Pauli gates (X, Z, etc.) are errors, they don't change the frame.
                        pass # Frame is unchanged by ideal Pauli gates; noise is handled by noisy gate type
                    else:
                        raise ValueError(f"Unknown non-Pauli gate_id {gate_id} ({gate_name}) or unhandled operation type in PauliFrameSimulator.")
            

            if is_gate_records(gate_id):
                 measurement_counter += 1
            
            gate_counter += 1

        # Final modulo
        x_frame %= self.dimension
        z_frame %= self.dimension
        
        return frame_results.T