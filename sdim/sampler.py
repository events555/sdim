# sdim/sampler.py
from typing import Optional, Tuple, Dict, Callable, Any, TYPE_CHECKING
import numpy as np

from .simulators.frame_simulator import PauliFrameSimulator

from .gatedata import (
    is_gate_pauli, is_gate_collapsing, is_gate_noisy,
    is_gate_two_qubit, gate_name_to_id, gate_id_to_name,
    is_gate_records, is_not_a_gate
)

if TYPE_CHECKING:
    from .circuit import Circuit # Only for type hinting

class CompiledMeasurementSampler():
    def __init__(self,
                 circuit_object: "Circuit", # Keep the original Circuit object
                 *,
                 # skip_reference_sample: bool = False, # This is now implicitly handled by requiring ref_sample
                 seed: Optional[int] = None,
                 reference_sample: np.ndarray, # Make these required from the compile step
                 ir_array: np.ndarray,
    ) -> None:
        self.circuit: "Circuit" = circuit_object
        self.reference_sample = reference_sample
        self.ir_array = ir_array
        self.seed: Optional[int] = seed 
        self.engine = PauliFrameSimulator(
            ir_array=ir_array,
            dimension=circuit_object.dimension,
            num_qudits=circuit_object.num_qudits,
            num_total_measurements=circuit_object.num_measurements
        )

    def sample(
            self,
            shots: int,
    ) -> np.ndarray:
        """
        Samples the measurement results of the circuit.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            import random
            random.seed(self.seed)
        noise1_bank, noise2_bank, erased_bank, measurement_bank = self.circuit._build_noise(shots)

        # Get raw noisy measurements from the engine
        raw_noisy_measurements = self.engine.run_simulation_for_raw_measurements(
            shots=shots,
            reference_sample=self.reference_sample,
            noise1_bank=noise1_bank,
            noise2_bank=noise2_bank,
            erased_bank=erased_bank,
            measurement_bank=measurement_bank
        )
        return raw_noisy_measurements

    def sample_write(
            self,
            shots: int,
            filepath: str,
            format: str = '01', # TODO: Implement
    ) -> None:
        samples = self.sample(shots)
        # Basic 01 format for now
        if format == '01':
            with open(filepath, 'w') as f:
                for shot_idx in range(shots):
                    f.write("".join(map(str, samples[shot_idx, :])) + "\n")
        else:
            raise NotImplementedError(f"Format '{format}' not implemented for sample_write.")


class CompiledDetectorSampler():
    def __init__(self,
                 circuit_object: "Circuit",
                 *,
                 seed: Optional[int] = None,
                 reference_sample: np.ndarray,
                 ir_array: np.ndarray,
    ) -> None:
        self.circuit: "Circuit" = circuit_object
        self.reference_sample = reference_sample
        self.ir_array = ir_array
        self.seed: Optional[int] = seed
        self.engine = PauliFrameSimulator(
            ir_array=ir_array,
            dimension=circuit_object.dimension,
            num_qudits=circuit_object.num_qudits,
            num_total_measurements=circuit_object.num_measurements
        )
        self._parse_annotations()
        self._calculate_reference_annotations()

    def _parse_annotations(self):
        self.detectors_meas_indices: list[list[int]] = []
        self.observables_meas_indices: list[list[int]] = [] # Assuming observables are indexed 0 to N-1
        self.num_total_physical_measurements = 0 # Count of M, MR, MX, MRX, HERALDED_ERASURE outputs

        measurement_op_indices_in_ir = []
        abs_meas_idx_counter = 0
        for op_ir_idx, op_inst_ir in enumerate(self.engine.ir_array): # Use engine's IR
            gate_id = op_inst_ir['gate_id']
            if is_gate_records(gate_id):
                pass

        # --- Simpler parsing based on original Circuit.operations for annotations ---
        self.detectors_meas_indices = []
        observables_map_temp: Dict[int, list[int]] = {}

        abs_meas_output_idx_so_far = 0
        
        op_idx_to_abs_meas_count_before_it = []

        for op_circuit in self.circuit.operations:
            op_idx_to_abs_meas_count_before_it.append(abs_meas_output_idx_so_far)
            if is_gate_records(op_circuit.gate_type):
                abs_meas_output_idx_so_far += len(op_circuit.targets)

        self.num_total_measurements_from_ops = abs_meas_output_idx_so_far
        if self.num_total_measurements_from_ops != self.engine.num_total_measurements:
             raise ValueError("Mismatch in total measurement count between engine and annotation parsing.")


        for op_idx, op_circuit in enumerate(self.circuit.operations):
            gate_name = gate_id_to_name(op_circuit.gate_type)
            num_meas_before_this_op = op_idx_to_abs_meas_count_before_it[op_idx]

            if gate_name == "DETECTOR":
                target_abs_indices = []
                for target in op_circuit.targets: # These are GateTarget objects
                    if target.is_measurement_record_target:
                        abs_idx = num_meas_before_this_op + target.value
                        if not (0 <= abs_idx < self.engine.num_total_measurements):
                            raise ValueError(f"DETECTOR target rec[{target.value}] resolves to "
                                             f"invalid absolute index {abs_idx} at op index {op_idx}.")
                        target_abs_indices.append(abs_idx)
                    # TODO: Handle other target types for DETECTOR if necessary
                self.detectors_meas_indices.append(sorted(list(set(target_abs_indices))))

            elif gate_name == "OBSERVABLE_INCLUDE":
                if not op_circuit.args:
                    raise ValueError("OBSERVABLE_INCLUDE requires a logical index argument.")
                logical_idx = int(op_circuit.args[0])
                
                target_abs_indices = []
                for target in op_circuit.targets:
                    if target.is_measurement_record_target:
                        abs_idx = num_meas_before_this_op + target.value
                        if not (0 <= abs_idx < self.engine.num_total_measurements):
                            raise ValueError(f"OBSERVABLE_INCLUDE target rec[{target.value}] resolves to "
                                             f"invalid absolute index {abs_idx} at op index {op_idx}.")
                        target_abs_indices.append(abs_idx)
                
                if logical_idx not in observables_map_temp:
                    observables_map_temp[logical_idx] = []
                observables_map_temp[logical_idx].extend(target_abs_indices)

        self.num_detectors = len(self.detectors_meas_indices)
        max_obs_idx = -1
        if observables_map_temp:
            max_obs_idx = max(observables_map_temp.keys())
        self.num_observables = max_obs_idx + 1
        
        self.observables_meas_indices = [[] for _ in range(self.num_observables)]
        for idx, targets_for_obs in observables_map_temp.items():
            self.observables_meas_indices[idx] = sorted(list(set(targets_for_obs)))
        # --- End of _parse_annotations sketch ---


    def _calculate_reference_annotations(self):
        # (Implementation from previous responses, using self.reference_sample,
        #  self.detectors_meas_indices, self.observables_meas_indices)
        self.ref_detector_values = np.zeros(self.num_detectors, dtype=np.int64)
        for i, indices in enumerate(self.detectors_meas_indices):
            if not indices: continue # Skip empty detectors
            val = sum(self.reference_sample[m_idx] for m_idx in indices) % self.circuit.dimension
            self.ref_detector_values[i] = val

        self.ref_observable_values = np.zeros(self.num_observables, dtype=np.int64)
        for i, indices in enumerate(self.observables_meas_indices):
            if not indices: continue # Skip empty observables
            val = sum(self.reference_sample[m_idx] for m_idx in indices) % self.circuit.dimension
            self.ref_observable_values[i] = val


    def sample(
            self,
            shots: int,
            *,
            dets_out: Optional[np.ndarray] = None,
            obs_out: Optional[np.ndarray] = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        if self.seed is not None:
            np.random.seed(self.seed)
            import random
            random.seed(self.seed)
        # Generate noise
        noise1_bank, noise2_bank, erased_bank, measurement_bank = self.circuit._build_noise(shots)

        # Get raw noisy measurements from the engine
        raw_noisy_measurements_all_shots = self.engine.run_simulation_for_raw_measurements(
            shots=shots,
            reference_sample=self.reference_sample,
            noise1_bank=noise1_bank,
            noise2_bank=noise2_bank,
            erased_bank=erased_bank,
            measurement_bank=measurement_bank
        ) # Shape: (shots, num_total_measurements)

        # --- Post-process raw measurements into detection events ---
        if dets_out is None and self.num_detectors > 0 :
            dets_out = np.empty((shots, self.num_detectors), dtype=np.uint8)
        elif self.num_detectors == 0:
             dets_out = np.empty((shots, 0), dtype=np.uint8) # Handle case with no detectors


        process_obs = self.num_observables > 0
        if process_obs and obs_out is None:
            obs_out = np.empty((shots, self.num_observables), dtype=np.uint8)
        elif not process_obs: # Ensure obs_out is None or empty if no observables
            if obs_out is not None and obs_out.shape[1] > 0:
                raise ValueError("obs_out provided but num_observables is 0")
            obs_out = np.empty((shots,0), dtype=np.uint8)


        for s_idx in range(shots):
            current_shot_raw_measurements = raw_noisy_measurements_all_shots[s_idx, :]
            
            if self.num_detectors > 0:
                for d_idx in range(self.num_detectors):
                    if not self.detectors_meas_indices[d_idx]: # Empty detector definition
                        dets_out[s_idx, d_idx] = 0 # Or based on convention if args define value
                        continue
                    noisy_det_val = sum(current_shot_raw_measurements[m_idx] 
                                        for m_idx in self.detectors_meas_indices[d_idx]
                                       ) % self.engine.dimension # Use engine's dimension
                    ref_det_val = self.ref_detector_values[d_idx]
                    dets_out[s_idx, d_idx] = 1 if noisy_det_val != ref_det_val else 0
            
            if process_obs and obs_out is not None:
                for o_idx in range(self.num_observables):
                    if not self.observables_meas_indices[o_idx]:
                        obs_out[s_idx, o_idx] = 0
                        continue
                    noisy_obs_val = sum(current_shot_raw_measurements[m_idx] 
                                        for m_idx in self.observables_meas_indices[o_idx]
                                       ) % self.engine.dimension
                    ref_obs_val = self.ref_observable_values[o_idx]
                    obs_out[s_idx, o_idx] = 1 if noisy_obs_val != ref_obs_val else 0
        
        if process_obs: # obs_out will be defined or empty here
            return dets_out, obs_out # type: ignore
        return dets_out # type: ignore


    def sample_write(
            self,
            shots: int,
            filepath: str,
            format: str = '01',
            *,
            obs_out_filepath: Optional[str] = None,
            obs_out_format: str = '01'
    ) -> None:
        # (Implementation from previous responses using self.sample())
        if self.num_observables > 0:
            dets, obs = self.sample(shots) # type: ignore
            if obs_out_filepath:
                 with open(obs_out_filepath, 'w') as f_obs: # Use different file var
                    for s_idx in range(shots):
                        f_obs.write("".join(map(str, obs[s_idx, :])) + "\n")
        else:
            dets = self.sample(shots) # type: ignore

        with open(filepath, 'w') as f_det: # Use different file var
            for s_idx in range(shots):
                f_det.write("".join(map(str, dets[s_idx, :])) + "\n")