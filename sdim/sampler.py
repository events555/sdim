from typing import Optional, Tuple
from sdim.gatedata import is_gate_pauli, is_gate_collapsing, is_gate_noisy, is_gate_two_qubit, gate_name_to_id, gate_id_to_name
import numpy as np

class CompiledMeasurementSampler():
    def __init__(self,
                 circuit: object,
                 *,
                 skip_reference_sample: bool = False,
                 seed: Optional[int] = None,
                 reference_sample: np.ndarray = None,
                 ir_array: Optional[object] = None,
    ) -> None:
        if ir_array is None:
            ir_array = circuit._build_ir()
        if reference_sample is None:
            reference_sample = circuit.reference_sample()

        self.circuit = circuit
        self.skip_reference_sample = skip_reference_sample
        self.seed = seed
        self.reference_sample = reference_sample
        self.ir_array = ir_array

    def sample(
            self,
            shots: int,
    ) -> np.ndarray:
        """
        Samples the measurement results of the circuit.

        Args:
            shots (int): Number of shots.

        Returns:
            np.ndarray: Measurement results.
        """
        # Get circuit parameters
        n_qudits = self.circuit.num_qudits
        dimension = self.circuit.dimension
        # Initialize frame arrays
        x_frame = np.zeros((n_qudits, shots), dtype=np.int64)
        z_frame = np.random.randint(0, dimension, size=(n_qudits, shots))
        
        # Initialize measurement tracking
        measurement_count = 0
        noise_counter1 = 0
        noise_counter2 = 0
        gate_count = 1
        # Initialize results array
        frame_results = np.empty((self.circuit.num_measurements, shots), 
                            dtype=np.int64)
        # Initialize noise arrays
        noise1, noise2 = self.circuit._build_noise(shots)


        def op_H(qi, ti):
            # Hadamard: swap x and -z.
            tmp = x_frame[qi].copy()
            x_frame[qi] = -z_frame[qi]
            z_frame[qi] = tmp

        def op_H_INV(qi, ti):
            # Inverse Hadamard: swap z and -x.
            tmp = x_frame[qi].copy()
            x_frame[qi] = z_frame[qi]
            z_frame[qi] = -tmp

        def op_P(qi, ti):
            # Phase (P) gate: add x to z.
            z_frame[qi] += x_frame[qi]

        def op_P_INV(qi, ti):
            # Inverse Phase: subtract x from z.
            z_frame[qi] -= x_frame[qi]

        def op_CNOT(qi, ti):
            # qi is control, ti is target.
            x_frame[ti] += x_frame[qi]
            z_frame[qi] -= z_frame[ti]

        def op_CNOT_INV(qi, ti):
            x_frame[ti] -= x_frame[qi]
            z_frame[qi] += z_frame[ti]

        def op_CZ(qi, ti):
            # Apply CZ: add x from control to z of target and vice versa.
            z_frame[ti] += x_frame[qi]
            z_frame[qi] += x_frame[ti]

        def op_CZ_INV(qi, ti):
            z_frame[ti] -= x_frame[qi]
            z_frame[qi] -= x_frame[ti]

        def op_SWAP(qi, ti):
            # Swap both x and z.
            tmp = x_frame[qi].copy()
            x_frame[qi] = x_frame[ti]
            x_frame[ti] = tmp
            tmp = z_frame[qi].copy()
            z_frame[qi] = z_frame[ti]
            z_frame[ti] = tmp

        # Create an op_map dictionary.
        id_to_op = {
            gate_name_to_id("H"): op_H,
            gate_name_to_id("H_INV"): op_H_INV,
            gate_name_to_id("P"): op_P,
            gate_name_to_id("P_INV"): op_P_INV,
            gate_name_to_id("CNOT"): op_CNOT,
            gate_name_to_id("CNOT_INV"): op_CNOT_INV,
            gate_name_to_id("CZ"): op_CZ,
            gate_name_to_id("CZ_INV"): op_CZ_INV,
            gate_name_to_id("SWAP"): op_SWAP,
        }

        for inst in self.ir_array:
            gate_id = inst['gate_id']
            qudit_index = inst['qudit_index']
            target_index = inst['target_index']
            if gate_count % 128 == 0:
                x_frame %= dimension
                z_frame %= dimension
            
            if is_gate_collapsing(gate_id):
                gate_name = gate_id_to_name(gate_id)
                ref_value = self.reference_sample[qudit_index]
                if gate_name in ("M_X", "MR_X"):
                    op_H_INV(qudit_index, target_index)

                for shot in range(shots):
                    new_val = (ref_value + x_frame[qudit_index, shot]) % dimension
                    frame_results[measurement_count, shot] = new_val

                if gate_name != "RESET":
                    measurement_count += 1

                if gate_name in ("MR", "MR_X", "RESET"):
                    x_frame[qudit_index] = 0
                    z_frame[qudit_index] = np.random.randint(0, dimension, size=shots)

                if gate_name in ("M_X", "MR_X"):
                    op_H(qudit_index, target_index)

            elif is_gate_noisy(gate_id):
                if not is_gate_two_qubit(gate_id):
                    x_frame[qudit_index] += noise1[noise_counter1, :, 0]
                    z_frame[qudit_index] += noise1[noise_counter1, :, 1]
                    noise_counter1 += 1
                else:
                    x_frame[qudit_index] += noise2[noise_counter2, :, 0]
                    z_frame[qudit_index] += noise2[noise_counter2, :, 1]
                    x_frame[target_index] += noise2[noise_counter2, :, 2]
                    z_frame[target_index] += noise2[noise_counter2, :, 3]
                    noise_counter2 += 1

            else:
                if not is_gate_pauli(gate_id):
                        # Handle the case when one of the indices refers to a measurement record.
                        if qudit_index < 0:
                            gate_name = gate_id_to_name(gate_id)
                            measurement_index = -qudit_index - 1
                            if gate_name == "CNOT":
                                x_frame[target_index] += frame_results[measurement_index, :]
                            elif gate_name == "CZ":
                                z_frame[target_index] += frame_results[measurement_index, :]
                            else:
                                raise ValueError(f"Unsupported gate {gate_name} for negative qudit_index.")
                        elif target_index < 0:
                            gate_name = gate_id_to_name(gate_id)
                            measurement_index = -target_index - 1
                            if gate_name == "CNOT":
                                raise ValueError("CNOT gate cannot be applied to measurement record target.")
                            elif gate_name == "CZ":
                                z_frame[qudit_index] += frame_results[measurement_index, :]
                            else:
                                raise ValueError(f"Unsupported gate {gate_name} for negative target_index.")
                        else:
                            id_to_op[gate_id](qudit_index, target_index)
            gate_count += 1
        return frame_results

    
    def sample_write(
            self,
            shots: int,
            filepath: str,
            format: str = '01',
    ) -> None:
        """
        Samples the measurement results of the circuit and writes them to a file.

        Args:
            shots (int): Number of shots.
            filename (str): File name.
        """
        ...
    
class CompiledDetectorSampler():
    def __init__(self,
                 circuit: object,
                 *,
                 seed: Optional[int] = None,
    ) -> None:
        ...
    def sample(
            self,
            shots: int,
            *,
            dets_out: Optional[np.ndarray] = None,
            obs_out: Optional[np.ndarray] = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        ...
    def sample_write( # TODO: Fix arguments to match Stim
            self,
            shots: int,
            filepath: str,
            format: str = '01',
    ) -> None:
        ...