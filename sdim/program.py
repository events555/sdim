from typing import Tuple, Optional, Callable
from .circuit import CircuitInstruction, Circuit
from .tableau.tableau_composite import WeylTableau
from .tableau.tableau_prime import ExtendedTableau
from .tableau.tableau_gates import *
from sympy import isprime
from numba import njit, prange
from numba.core import types
from numba.typed import Dict
import numpy as np
import copy
import re 

# Gate function dictionary
GATE_FUNCTIONS: dict[int, Callable] = {
    0: apply_I,      # I gate
    1: apply_X,      # X gate
    2: apply_X_inv,   # X inverse gate
    3: apply_Z,      # Z gate
    4: apply_Z_inv,  # Z inverse gate
    5: apply_H,      # H gate
    6: apply_H_inv,  # H inverse gate
    7: apply_P,      # P gate
    8: apply_P_inv,  # P inverse gate
    9: apply_CNOT,   # CNOT gate
    10: apply_CNOT_inv,  # CNOT inverse gate
    11: apply_CZ,  # CZ gate
    12: apply_CZ_inv,  # CZ inverse gate
    13: apply_SWAP,  # SWAP gate
    14: apply_measure, # Measure gate in computational basis
    15: apply_measure_x, # Measure gate in X basis
    16: apply_reset, # Reset gate
    17: apply_I, # Single qudit Pauli noise gate, implemented in Pauli frame, applied as I in noiseless reference tableau
    18: apply_I, # 2 qudit Pauli noise gate, implemented in Pauli frame, applied as I in noiseless ref.  Input is a distribution on Pauli operators, shape is (d, d, d, d)
    19: apply_I, # Generic detectors
    20: apply_I, # Logical operator detectors
    21: apply_I # TICK, do nothing.
}

MEASUREMENT_DTYPE = np.dtype([
    ('qudit_index', np.int64),
    ('meas_round', np.int64),
    ('shot', np.int64),
    ('deterministic', np.bool_),
    ('measurement_value', np.int64)
])

noise_gate_indices = {17, 18}

# TODO: Clean up typing to expose the now hidden structure of detector_data
@dataclass
class DetectorData:
    detector_data : np.ndarray | None = None # Format is (unique_detector_function_index, label, arguments, is_logical)
    detector_functions : list = None
    total_measurements : int = 1
    num_detector_events : int = 0
    num_logical_operators : int = 0

@dataclass
class DetectorResults:
    detection_events : np.ndarray = None
    logical_operator_shifts : np.ndarray = None


# @njit
def simulate_frame(ir_array: np.ndarray, reference_results: np.ndarray,
                  n_qudits: int, dimension: int, extra_shots: int,
                  noise_array: np.ndarray = None, 
                  detector_info : DetectorData = None) -> tuple[np.ndarray, DetectorResults]:
    """
    Simulates quantum circuit using Pauli frame simulation.
    
    Args:
        ir_array: Array of (gate_id, qudit_index, target_index) tuples
        reference_results: Reference measurement results
        n_qudits: Number of qudits
        dimension: Qudit dimension
        extra_shots: Number of additional shots to simulate
        noise_array: Pre-computed noise samples
        
    Returns:
        frame_results: Array of simulated measurement results with shape (n_qudits, num_rounds, extra_shots)
    """
    # Initialize frame arrays
    x_frame = np.zeros((n_qudits, extra_shots), dtype=np.int64)
    z_frame = np.random.randint(0, dimension, size=(n_qudits, extra_shots))
    
    # Initialize measurement tracking
    measurement_counts = np.zeros(n_qudits, dtype=np.int64)
    noise_counter = 0
    gate_count = 1

    # Initialize detector and shift matrix tracking
    detector_counter = 0
    lo_counter = 0
    total_measurement_counter = 0
    shift_data = np.zeros((detector_info.total_measurements, extra_shots), dtype=np.int64)
    detector_events = np.zeros((detector_info.num_detector_events, extra_shots), dtype=np.int64)
    logical_operator_events = np.zeros((detector_info.num_logical_operators, extra_shots), dtype=np.int64)

    time = 0


    # Initialize results array
    frame_results = np.empty((n_qudits, reference_results.shape[1], extra_shots), 
                           dtype=MEASUREMENT_DTYPE)
    
    # Process each instruction
    for inst in ir_array:
        gate_id = inst['gate_id']
        qudit_index = inst['qudit_index']
        target_index = inst['target_index']
        if gate_count % 64 == 0:
            x_frame %= dimension
            z_frame %= dimension
        if gate_id == 1:  # X gate
            pass
        elif gate_id == 2:  # X inverse
            pass
        elif gate_id == 3:  # Z gate
            pass
        elif gate_id == 4:  # Z inverse
            pass
        elif gate_id == 5:  # Hadamard
            tmp = x_frame[qudit_index].copy()
            x_frame[qudit_index] = -z_frame[qudit_index]
            z_frame[qudit_index] = tmp
        elif gate_id == 6:  # Inverse Hadamard
            tmp = x_frame[qudit_index].copy()
            x_frame[qudit_index] = z_frame[qudit_index]
            z_frame[qudit_index] = -tmp
        elif gate_id == 7:  # P gate
            z_frame[qudit_index] += x_frame[qudit_index]
        elif gate_id == 8:  # Inverse P gate
            z_frame[qudit_index] -= x_frame[qudit_index]
        elif gate_id == 9:  # CNOT gate
            x_frame[target_index] += x_frame[qudit_index]
            z_frame[qudit_index] -= z_frame[target_index]
        elif gate_id == 10:  # Inverse CNOT
            x_frame[target_index] -= x_frame[qudit_index]
            z_frame[qudit_index] += z_frame[target_index]
        elif gate_id == 11:  # CZ gate
            z_frame[target_index] += x_frame[qudit_index]
            z_frame[qudit_index] += x_frame[target_index]
        elif gate_id == 12:  # Inverse CZ
            z_frame[target_index] -= x_frame[qudit_index]
            z_frame[qudit_index] -= x_frame[target_index]
        elif gate_id == 13:  # SWAP gate
            tmp = x_frame[qudit_index].copy()
            x_frame[qudit_index] = x_frame[target_index]
            x_frame[target_index] = tmp
            tmp = z_frame[qudit_index].copy()
            z_frame[qudit_index] = z_frame[target_index]
            z_frame[target_index] = tmp
        if gate_id in (14, 15):  # Measurement gates
            m = measurement_counts[qudit_index]
            shifts = shift_data[total_measurement_counter]
            
            ref_val = reference_results[qudit_index, m]['measurement_value']
            deterministic = reference_results[qudit_index, m]['deterministic']
            
            # Handle X-basis measurement
            if gate_id == 15:
                tmp = x_frame[qudit_index].copy()
                x_frame[qudit_index] = z_frame[qudit_index]
                z_frame[qudit_index] = -tmp
                
            # Compute and store measurements for each shot
            for shot in range(extra_shots):
                new_val = (ref_val + x_frame[qudit_index, shot]) % dimension
                shifts[shot] = x_frame[qudit_index, shot] % dimension
                frame_results[qudit_index, m, shot]['qudit_index'] = qudit_index
                frame_results[qudit_index, m, shot]['meas_round'] = m
                frame_results[qudit_index, m, shot]['shot'] = shot
                frame_results[qudit_index, m, shot]['deterministic'] = deterministic
                frame_results[qudit_index, m, shot]['measurement_value'] = new_val

            
            measurement_counts[qudit_index] += 1
            total_measurement_counter += 1
            z_frame[qudit_index] = np.random.randint(0, dimension, size=extra_shots)
            
        elif gate_id == 16:  # Reset
            q = int(qudit_index)
            m = int(measurement_counts[q])
            ref_val = reference_results[q, m]['measurement_value']
            deterministic = reference_results[q, m]['deterministic']

            for shot in range(extra_shots):
                frame_results[q, m, shot] = (q, m, shot, deterministic, ref_val)

            x_frame[q] = 0
            z_frame[q] = np.random.randint(0, dimension, size=extra_shots)
            measurement_counts[q] += 1
            
        elif gate_id == 17:  # 1 qudit noise
            x_frame[qudit_index] += noise_array[noise_counter, :, 0]
            z_frame[qudit_index] += noise_array[noise_counter, :, 1]
            noise_counter += 1

        elif gate_id == 18: # 2 qudit noise
            #print(f" {noise_array[noise_counter, :, 0]} \n {noise_array[noise_counter, :, 1]} \n {noise_array[noise_counter, :, 2]} \n {noise_array[noise_counter, :, 3]} \n\n")
            x_frame[qudit_index] += noise_array[noise_counter, :, 0]
            z_frame[qudit_index] += noise_array[noise_counter, :, 1]
            x_frame[target_index] += noise_array[noise_counter, :, 2]
            z_frame[target_index] += noise_array[noise_counter, :, 3]
            noise_counter += 1

        elif gate_id in (19, 20):
            # Fetch in detector params
            # TODO: Write this using just is_logical since there is a lot of redundant information
            detector_collection_bin = detector_events if gate_id == 19 else logical_operator_events
            position = detector_counter if gate_id == 19 else lo_counter
            function_index, label, arguments, _ = detector_info.detector_data[detector_counter + lo_counter]
            # Compute and store the data appropriately
            shift_params = [shift_data[a] for a in arguments]
            detector = detector_info.detector_functions[function_index]
            detector_collection_bin[position] = detector(shift_params)

            if gate_id == 19:
                detector_counter += 1
            else:
                lo_counter += 1

        # elif gate_id == 21: # TICK, increment internal clock
        #     time += 1

        gate_count += 1
        
    
    detector_results = DetectorResults(
        detection_events=detector_events,
        logical_operator_shifts=logical_operator_events
    )

    return frame_results, detector_results

@dataclass
class SimulationOptions:
    shots: int = 1
    show_measurement: bool = False
    record_tableau: bool = False
    force_tableau: bool = False
    verbose: bool = False
    show_gate: bool = False
    exact: bool = False


class Program:
    """
    Represents a quantum program with a circuit and stabilizer tableau.

    This class handles the initialization and simulation of a quantum program,
    including applying gates and managing measurement results.

    Attributes:
        stabilizer_tableau: The current state of the quantum system.
        circuit: A Circuit object representing the quantum circuit.
        measurement_results: A list of MeasurementResult objects.

    Args:
        circuit (Circuit): A Circuit object representing the quantum circuit.
        tableau (Optional[Tableau]): An optional stabilizer tableau. If not provided,
            the default is the all zero computational basis.
    """
    def __init__(self, circuit: Circuit, tableau=None):
        if tableau is None:
            if isprime(circuit.dimension):
                self.stabilizer_tableau = ExtendedTableau(circuit.num_qudits, circuit.dimension)
            else:
                self.stabilizer_tableau = WeylTableau(circuit.num_qudits, circuit.dimension)
        else:
            self.stabilizer_tableau = tableau
        self.circuits = [circuit]
        self.measurement_results = []
        self.initial_tableau = copy.copy(self.stabilizer_tableau)

    # def enumerate_detector_shifts(self) -> dict[str, dict[str, str | np.ndarray]]:

    #     return

    def simulate(self, shots: int = 1, show_measurement: bool = False, record_tableau: bool = False, force_tableau: bool = False,
                 verbose: bool = False, show_gate: bool = False, exact: bool = False, 
                 building_error_mechanism : bool = False, options: SimulationOptions = None) -> list[list[list[MeasurementResult]]] | tuple[list[list[list[MeasurementResult]]], dict[str, list(dict[str, str | np.ndarray])]]:
        """
        Runs the list of `Circuit` and applies the gates to the `stabilizer_tableau`.
        
        Note that using multiple shots without `record_tableau=True` or `force_tableau=True` will use the Pauli frame sampler.
        
        This means that things like `show_gate` and `verbose` will **not work for any shot after the first**.

        Args:
            shots (int): The number of times to run the simulation.
            show_measurement (bool): Whether to print the measurement results.
            verbose (bool): Whether to print the stabilizer tableau at each time step.
            show_gate (bool): Whether to print the gate name at each time step.
            record_tableau (bool): Whether to record the tableau after each measurement.
            force_tableau (bool): Whether to force the use of the tableau method.
            exact (bool): Whether to use the Diophantine solver instead of column reduction.
                Much slower but fails less often.
            building_error_mechanim (bool): Flag to generate exhaustive noise sequences to sample detector and logical operator shift data.
                Not for manual use
            options (SimulationOptions): An optional SimulationOptions object.

        Returns:
            list or 3D list: Depending on the value of `shots`:
                - If `shots == 1`, returns a list of `MeasurementResult` instances.
                - If `shots > 1`, returns a 3D list of `MeasurementResult` objects.
                The first axis is the qudit position,
                the second axis is the measurement index (number of times a qudit was measured in a Circuit),
                and the third axis is the shot number.
        """
        if options is None:
            options = SimulationOptions(
                shots=shots,
                show_measurement=show_measurement,
                record_tableau=record_tableau,
                force_tableau=force_tableau,
                verbose=verbose,
                show_gate=show_gate,
                exact=exact
            )
        if options.shots > 1 and not options.record_tableau and not options.force_tableau:
            tableau_options = copy.copy(options)
            tableau_options.shots = 1
            self._simulate_tableau(tableau_options)
            
            # Convert flattened reference results to structured array
            ref_array = self._results_to_array(self.measurement_results)
            
            # Build IR and noise arrays
            ir_array, noise, detector_info = self._build_ir(self.circuits, options.shots - 1)
            
            # Run frame simulation
            frame_results, detection_results = simulate_frame(
                ir_array, ref_array, 
                self.stabilizer_tableau.num_qudits,
                self.stabilizer_tableau.dimension,
                options.shots - 1,
                noise, 
                detector_info
            )
            
            # Combine results
            return self._combine_results(frame_results), self._combine_detector_results(detector_info, detection_results)
        else:
            return self._simulate_tableau(options)

    def _simulate_tableau(self, options: SimulationOptions) -> list:
        """
        Simulates the circuit using the stabilizer tableau method.
        
        In single-shot mode (options.shots == 1), the simulation returns a flattened list of 
        MeasurementResult objects—one per measurement round per qudit—by taking the first (and only)
        shot from the internal 3D measurement results structure:
        
            self.measurement_results[qudit_index][measurement_round][shot]
        
        For multiple shots (shots > 1) when not recording the tableau, a single reference shot is computed
        via the full tableau simulation and its measurement outcomes are stored in the internal grouped
        structure (by qudit and measurement round). Later, extra shots are generated using a vectorized
        Pauli frame simulation (via _simulate_frame) and then recombined with the reference shot using
        _combine_results. The final combined results are returned as a 3D list of MeasurementResult objects
        with dimensions:
        
            [qudit_index][measurement_round][shot]
        
        where shot index 0 is the reference simulation result.
        
        Args:
            options: A SimulationOptions object containing simulation parameters (e.g., shots, verbose, etc.)
        
        Returns:
            If options.shots == 1, a flat list of MeasurementResult objects (one per measurement round) is returned.
            If options.shots > 1, a 3D list of MeasurementResult objects is returned, where the axes correspond to 
            qudit index, measurement round, and shot number.
        """
        num_qudits = self.stabilizer_tableau.num_qudits
        # Prepare the measurement results container.
        self.measurement_results = [[] for _ in range(num_qudits)]
        length = sum(len(circuit.operations) for circuit in self.circuits)

        # Set exact mode if necessary.
        if isinstance(self.stabilizer_tableau, WeylTableau) and options.exact:
            self.stabilizer_tableau.exact = True

        # Iterate over each shot.
        for shot in range(options.shots):
            self.stabilizer_tableau = copy.deepcopy(self.initial_tableau)
            measurement_counts = [0] * num_qudits  
            for circuit in self.circuits:
                for time, gate in enumerate(circuit.operations):
                    if time == 0 and options.verbose:
                        print("Initial state")
                        self.stabilizer_tableau.print_tableau()
                        print("\n")
                    if time % 64 == 0:
                        self.stabilizer_tableau.modulo()

                    measurement_result = self.apply_gate(gate)
                    if measurement_result is not None:
                        qudit_index = measurement_result.qudit_index
                        if options.record_tableau:
                            measurement_result.stabilizer_tableau = copy.deepcopy(self.stabilizer_tableau)

                        measurement_number = measurement_counts[qudit_index]
                        measurement_counts[qudit_index] += 1

                        if len(self.measurement_results[qudit_index]) <= measurement_number:
                            self.measurement_results[qudit_index].append([])

                        self.measurement_results[qudit_index][measurement_number].append(measurement_result)

                        # Handle reset gate (gate_id == 16)
                        if gate.gate_id == 16:
                            # Apply X gates until reaching the computational basis state.
                            steps_to_zero = (-measurement_result.measurement_value) % self.stabilizer_tableau.dimension
                            for _ in range(steps_to_zero):
                                apply_X(self.stabilizer_tableau, gate.qudit_index, None)

                    if options.show_gate:
                        gate_info = gate.target_index if gate.target_index is not None else ""
                        if time < length - 1:
                            print("Time step", time, "\t", gate.name, gate.qudit_index, gate_info)
                        else:
                            print("Final step", time, "\t", gate.name, gate.qudit_index, gate_info)

                    if options.verbose:
                        self.stabilizer_tableau.print_tableau()
                        print("\n")
            self.stabilizer_tableau.modulo()
            if options.show_measurement:
                print(f"Measurement results for shot {shot + 1}:")
                self.print_measurements()

        # Return results in the desired format.
        if options.shots == 1:
            flattened_results = []
            for measurements_per_qudit in self.measurement_results:
                # Each qudit may have multiple measurement rounds; we take the first shot.
                for shots_list in measurements_per_qudit:
                    flattened_results.append(shots_list[0])
            return flattened_results
        else:
            return self.measurement_results
    
    def apply_gate(self, instruc: CircuitInstruction) -> MeasurementResult:
        """
        Applies a gate to the stabilizer tableau.

        Args:
            instruc (CircuitInstruction): A CircuitInstruction object from a Circuit's operation list.
            exact (bool): Whether to use exact computation methods.

        Returns:
            MeasurementResult: A MeasurementResult object if the gate is a measurement gate, otherwise None.

        Raises:
            ValueError: If an invalid gate value is provided.
        """
        if instruc.gate_id not in GATE_FUNCTIONS:
            raise ValueError("Invalid gate value")
        gate_function = GATE_FUNCTIONS[instruc.gate_id]
        measurement_result = gate_function(self.stabilizer_tableau, instruc.qudit_index, instruc.target_index, instruc.params)
        return measurement_result

    @staticmethod
    def _results_to_array(measurements: list) -> np.ndarray:
        """
        Converts measurement results to a structured NumPy array.

        Returns:
            A structured array with shape (num_qudits, max_rounds) containing measurement data
        """
        def measurement_to_tuple(m: MeasurementResult, meas_round: int = 0, shot: int = 0):
            return (m.qudit_index, meas_round, shot, m.deterministic, m.measurement_value)
        # TODO: make this check a little less truthy and more explicitly against a list of empty measurement outcomes.
        # TODO: actually, make sure program is okay even if there are no measurements...for some reason.
        # Ensure the list is not empty and has the expected nested structure.
        if not measurements:
            raise ValueError("Empty measurement results format")

        # TODO: make this robust against all elements of measurements being empty, ie None
        # find the first non-empty element of the list
        j = 0
        while (not measurements[j]):
            j += 1

        # If it's a 3D list (i.e., each measurement round is a list of shots)
        if isinstance(measurements[j][0], list):
            max_rounds = max(len(m) for m in measurements)
            reference_results = np.empty((len(measurements), max_rounds), dtype=MEASUREMENT_DTYPE)
            for q, measurements_per_qudit in enumerate(measurements):
                for m, shots_list in enumerate(measurements_per_qudit):
                    # Convert the first shot in each round into a tuple.
                    reference_results[q, m] = measurement_to_tuple(shots_list[0], meas_round=m, shot=0)
            return reference_results
        # If it's a 2D list (each sublist contains MeasurementResult objects, one per round)
        elif isinstance(measurements[j][0], MeasurementResult):
            max_rounds = max(len(m) for m in measurements)
            reference_results = np.empty((len(measurements), max_rounds), dtype=MEASUREMENT_DTYPE)
            for q, shots_list in enumerate(measurements):
                for m, measurement in enumerate(shots_list):
                    reference_results[q, m] = measurement_to_tuple(measurement, meas_round=m, shot=0)
            print(reference_results)
            return reference_results

        else:
            raise ValueError("Invalid measurement results format")

    def _combine_results(self, frame_results) -> list:
        """
        Combines the reference simulation (stored in self.measurement_results) with
        the extra shots computed in frame_results.

        The frame_results is expected to be a 3D structured array with shape:
            (n_qudits, num_rounds, extra_shots)
        where each element is a record with fields:
            ('qudit_index', 'meas_round', 'shot', 'deterministic', 'measurement_value').

        For each qudit and each measurement round, we create a list of MeasurementResult
        objects (one per shot) and append them.
        """
        extra_shots = frame_results.shape[2]
        n_qudits = frame_results.shape[0]
        # Loop over qudits and measurement rounds and build the shot results in one go.
        for qudit_index in range(n_qudits):
            # It may be that len(self.measurement_results[qudit_index]) equals the number of rounds.
            num_rounds = len(self.measurement_results[qudit_index])
            for meas_round in range(num_rounds):
                # Build the list of MeasurementResult objects for this qudit and measurement round
                new_results = [
                    MeasurementResult(
                        qudit_index = int(frame_results[qudit_index, meas_round, shot]['qudit_index']),
                        deterministic = bool(frame_results[qudit_index, meas_round, shot]['deterministic']),
                        measurement_value = int(frame_results[qudit_index, meas_round, shot]['measurement_value'])
                    )
                    for shot in range(extra_shots)
                ]
                # Extend the already-existing list for this measurement round
                self.measurement_results[qudit_index][meas_round].extend(new_results)
        return self.measurement_results

    def _combine_detector_results(self, info : DetectorData, raw_results : DetectorResults) -> dict[str, list(dict[str, str | np.ndarray])]:
        """
        Combines all detector mechanism results into a single dictionary organized by their quantitative (e.g. order in the circuit) and qualitative information (e.g. label).
        The topmost dictionary has keys: 'detectors', 'logicals'.
        The list enumerates the detectors / frame change data in order of occurence in the circuit.  
        Finally, the bottom most dictionary has keys: 'label', 'data'
        (unique-index, label, arguments)
        """

        d_ind = 0
        l_ind = 0
        results = {'detectors' : list(), 'logicals' : list()}
        detector_events = raw_results.detection_events
        logical_events = raw_results.logical_operator_shifts

        for j, d_data in enumerate(info.detector_data):
            label = d_data[1]
            args = d_data[2]
            is_logical = d_data[3]

            entry_type, data, index = ('logicals', logical_events, l_ind) if is_logical else ('detectors', detector_events, d_ind)

            event_info = {
                'label' : label, 
                'data' : data[index]
            }

            results[entry_type].append(event_info)

            if is_logical:
                l_ind += 1
            else:
                d_ind += 1

        return results
        
    def _build_ir(self, circuits: list[Circuit], extra_shots: int, 
    building_error_mechanism : bool = False) -> tuple[np.ndarray, np.ndarray, DetectorData]:
        """
        Builds an intermediate representation (IR) for the given circuits and also precomputes
        an array of sampled Pauli noise outcomes for noise gates (if applicable)
        
        Args:
            circuits (list[Circuit]): A list of Circuit objects.
            extra_shots (int): The number of extra shots for which noise outcomes
                            will be sampled.
            building_error_mechanism (bool): Flag for sampling errors in the circuit to build error mechanisms in a detector-error model.
        
        Returns:
            tuple:
                - A NumPy array of IR instructions with each element as a tuple
                (gate_id, qudit_index, target_index).
                - A NumPy array of shape (num_noise_gates, extra_shots, [x_block, z_block]) containing
                pre-sampled noise outcomes for each noise gate encountered.
                If no noise gate is present, an empty array is returned.
                - A DetectorData object containing runtime information for the detection events in the circuit. 
                If no detectors are present, a default object is returned.
        """
        ir_list  = []
        noise_list = []
        detector_list = []
        detector_data = []
        dimension = self.stabilizer_tableau.dimension

        # Detector related counters
        seen_measurements = 0
        num_detector_events = 0
        num_logical_operators = 0

        # Offset to organize error mechanism sampler
        error_skip_offset = 0

        for circuit in circuits:

            #TODO: Repeater blocks for detectors that link detectors to earlier detector expressions
            unique_detector_index = 0

            for instruction in circuit.operations:
                if instruction.gate_id == 0:
                    continue
                
                control_index = instruction.qudit_index if instruction.qudit_index is not None else -1
                target_index = instruction.target_index if instruction.target_index is not None else -1

                ir_list.append((instruction.gate_id, control_index, target_index))

                # Count measurements here in order to properly track measurements in detector expressions
                if (instruction.gate_id == 14 or instruction.gate_id == 15):
                    seen_measurements += 1

                if instruction.gate_id == 17:
                    # Always add a noise sample, but only actually sample non-identity with some probability.
                    channel = instruction.params['noise_channel']
                    if not building_error_mechanism:
                        if channel == 'd':
                            # Sample integer r from 1 to dimension**2 - 1 for each extra shot.
                            r = np.random.randint(1, dimension**2, size=extra_shots)
                            a = r % dimension
                            b = r // dimension
                        elif channel == 'f':
                            a = np.random.randint(1, dimension, size=extra_shots)
                            b = np.zeros(extra_shots, dtype=np.int64)
                        elif channel == 'p':
                            a = np.zeros(extra_shots, dtype=np.int64)
                            b = np.random.randint(1, dimension, size=extra_shots)

                        # Roll to see if the channel applies on this each shot
                        shot_dice_rolls = np.random.uniform(0.0, 1.0, size=extra_shots)
                        # Mask that checks for failure to clear threshold, aka applying I = X^0 Z^0
                        probability = float(instruction.params['prob'])
                        mask = shot_dice_rolls < 1.0 - probability

                        # Apply mask to both Pauli exponents
                        a[mask] = 0
                        b[mask] = 0

                        zero_vec = [0,] * extra_shots
                        pair = np.stack((a, b, zero_vec, zero_vec), axis=1)
                        noise_list.append(pair)

                    else:
                        print("TODO")
                        # if channel == 'd':
                        #     powers = list(np.ndindex((dimension, ) * 4))

                        # elif channel == 'f':

                        # elif channel == 'p':





                if instruction.gate_id == 18:
                    distribution = instruction.params['prob_dist']
                    
                    powers = list(np.ndindex((dimension, ) * 4))

                    if len(distribution) == (dimension ** 4):
                        powers = list(np.ndindex((dimension, ) * 4))
                        noise_indices = np.random.choice(a=len(powers), size=extra_shots, p=distribution)
                        noise = np.array( [powers[i] for i in noise_indices] )
                        noise_list.append(noise)
                    else: # If the list doesn't have a valid shape, then the channel acts as identity.
                        zero_vec = [0,] * extra_shots
                        noise = np.stack((zero_vec, ) * 4, axis=1)
                        noise_list.append(noise)

                if instruction.gate_id in (19, 20):

                    arguments = []
                    
                    if 'expr' not in instruction.params:
                        raise ValueError("No detector provided.")
                    
                    source = instruction.params['expr']
                    # TODO: Sanitize input.
                    # Extract argument indices and turn detector expression into a general lambda
                    # Pattern match for "(+/-)? [index]"
                    # If the first term is positive, leave the sign field blank to avoid unnecessary copying
                    j = 0
                    matches = list(re.finditer(r"\[(-?\d+)\]", source))
                    seen_args = set()
                    for match in matches:
                        match_string = match.group()
                        arg = int(match_string[1:-1]) % seen_measurements
                        if arg not in seen_args:
                            arguments.append(arg)
                            seen_args.add(arg)
                            source = source.replace(match_string, '[' + str(j) + ']')
                            j += 1
            
                    detector = eval('lambda rec : (' + str(source) + ") % " + str(dimension))
                    # Store lambdas in a map
                    detector_list.append(detector)
                    # Store detector data
                    label = instruction.params['label'] if 'label' in instruction.params else ''
                    is_logical = True if instruction.gate_id == 20 else False
                    detector_data.append((unique_detector_index, label, arguments, is_logical))

                    unique_detector_index += 1
                    if instruction.gate_id == 19:
                        num_detector_events += 1
                    else:
                        num_logical_operators +=1
                    
        ir_dtype = np.dtype([
            ('gate_id', np.int64),
            ('qudit_index', np.int64),
            ('target_index', np.int64)
        ])

        ir_array = np.array(ir_list, dtype=ir_dtype)

        if noise_list:
            noise_array = np.array(noise_list, dtype=np.int64)
        else:
            noise_array = np.empty((1, extra_shots, 2), dtype=np.int64)

        detection_info = DetectorData(
            detector_data=detector_data,
            detector_functions=detector_list,
            total_measurements=seen_measurements,
            num_detector_events=num_detector_events,
            num_logical_operators=num_logical_operators
        )

        #print(noise_array)
        return ir_array, noise_array, detection_info

    def append_circuit(self, circuit: Circuit):
        """
        Appends a circuit to the existing Program.

        Args:
            circuit (Circuit): The Circuit object to append.

        Raises:
            ValueError: If the circuits have different dimensions.
        """
        if self.circuits[-1].num_qudits < circuit.num_qudits:
            self.circuits[-1].num_qudits = circuit.num_qudits
        else:
            circuit.num_qudits = self.circuits[-1].num_qudits
        if self.circuits[-1].dimension != circuit.dimension:
            raise ValueError("Circuits must have the same dimension")
        self.circuits.append(circuit)
        

    def print_measurements(self):
        """
        Prints the measurement results.

        This method iterates through the stored measurement results and prints each one.
        """
        shot_count = 0
        for qudit_measurements in self.measurement_results:
            for measurement_group in qudit_measurements:
                shot_count = max(shot_count, len(measurement_group))
                
        if shot_count == 0:
            print("No measurements recorded.")
            return
        if shot_count == 1:
            for result in self.simulate():
                print(result)
        else:
            for shot_index in range(shot_count):
                print(f"Shot {shot_index + 1}:")
                for qudit_index, measurements_per_qudit in enumerate(self.measurement_results):
                    for measurement_number, shots_list in enumerate(measurements_per_qudit):
                        measurement_result = shots_list[shot_index]
                        print(f"{measurement_result} during measurement {measurement_number}")
                print()

    def __str__(self) -> str:
        return str(self.stabilizer_tableau)