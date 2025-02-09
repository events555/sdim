from typing import Tuple, Optional, Callable
from .circuit import CircuitInstruction, Circuit
from .tableau.tableau_composite import WeylTableau
from .tableau.tableau_prime import ExtendedTableau
from .tableau.tableau_gates import *
from sympy import isprime
from numba import njit, prange
import numpy as np
import copy
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
    17: apply_single_qudit_noise # Random Pauli noise gate
}

MEASUREMENT_DTYPE = np.dtype([
    ('qudit_index', np.int64),
    ('meas_round', np.int64),
    ('shot', np.int64),
    ('deterministic', np.bool_),
    ('measurement_value', np.int64)
])

# Defined to eventually support noise parameters on arbitrary gates
# Also partial work for classifying all gates for more sophisticated behavior in apply_gate
noise_gate_indices = {17}

@dataclass
class SimulationOptions:
    shots: int = 1
    show_measurement: bool = False
    record_tableau: bool = False
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

    def simulate(self, shots: int = 1, show_measurement: bool = False, record_tableau: bool = False,
                 verbose: bool = False, show_gate: bool = False, exact: bool = False, options: SimulationOptions = None) -> list[list[list[MeasurementResult]]]:
        """
        Runs the list of `Circuit` and applies the gates to the `stabilizer_tableau`.
        
        Note that using multiple shots without `record_tableau=True` will use the Pauli frame sampler.
        
        This means that things like `show_gate` and `verbose` will **not work for any shot after the first**.

        Args:
            shots (int): The number of times to run the simulation.
            show_measurement (bool): Whether to print the measurement results.
            verbose (bool): Whether to print the stabilizer tableau at each time step.
            show_gate (bool): Whether to print the gate name at each time step.
            record_tableau (bool): Whether to record the tableau after each measurement.
            exact (bool): Whether to use the Diophantine solver instead of column reduction.
                Much slower but fails less often.
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
                verbose=verbose,
                show_gate=show_gate,
                exact=exact
            )

        # If using the frame simulation strategy, run one full tableau simulation first.
        if options.shots > 1 and not options.record_tableau:
            # Run one shot with the full tableau simulation.
            tableau_options = copy.copy(options)
            tableau_options.shots = 1
            reference_results = self._simulate_tableau(tableau_options)
            
            # Now simulate the additional shots using the Pauli frame method.
            extra_shots = options.shots - 1
            frame_results = self._simulate_frame(reference_results, extra_shots, options)
            
            # Combine the reference shot with the extra shots.
            combined_results = self._combine_results(reference_results, frame_results)
            return combined_results
        else:
            # Otherwise, run the full tableau simulation as usual.
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
                    if gate.gate_id not in GATE_FUNCTIONS:
                        raise ValueError("Invalid gate value")
                    if time == 0 and options.verbose:
                        print("Initial state")
                        self.stabilizer_tableau.print_tableau()
                        print("\n")
                    if time % 200 == 0:
                        self.stabilizer_tableau.modulo()

                    gate_function = GATE_FUNCTIONS[gate.gate_id]
                    measurement_result = gate_function(self.stabilizer_tableau, gate.qudit_index, gate.target_index, gate.params)
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


    @njit(parallel=True)
    def _simulate_frame(self, reference_results: list[list[list[MeasurementResult]]],
                        extra_shots: int,
                        options: SimulationOptions) -> list[list[list[MeasurementResult]]]:
        """
        Augments the reference measurement results using the Pauli frame method.
        This function should simulate the circuit for the additional shots using a
        lightweight, vectorized approach that uses the measurement outcomes from
        the reference shot.
        
        Args:
            reference_results: The measurement results from one full tableau simulation.
            extra_shots: The number of additional shots to simulate.
            options: The SimulationOptions used for the simulation.
        
        Returns:
            A 3D list of MeasurementResult objects with dimensions:
                [qudit_index][measurement_round][shot]
            for the extra shots.
        """

        tableau = self.stabilizer_tableau
        n_qudits = tableau.num_qudits

        measurement_dtype = np.dtype([
            ('qudit_index', np.int64),
            ('meas_round', np.int64),
            ('shot', np.int64),
            ('deterministic', np.bool_),
            ('measurement_value', np.int64)
        ])

        num_rounds = 0
        for q in range(n_qudits):
            if len(self.measurement_results[q]) > num_rounds:
                num_rounds = len(self.measurement_results[q])
            
        # Preallocate the structured array for extra-shot results.
        frame_results = np.empty((n_qudits, num_rounds, extra_shots), dtype=MEASUREMENT_DTYPE)
        
        # Initialize frame arrays.
        x_frame = np.zeros((n_qudits, extra_shots), dtype=np.int8)
        z_frame = np.random.randint(0, tableau.dimension, size=(n_qudits, extra_shots), dtype=np.int8)
        measurement_counts = np.zeros(n_qudits, dtype=np.int64)

        ir_array, noise = self._build_ir(self.circuits, extra_shots)
        noise_counter = 0


        for circuit in self.circuits:
            for time in range(len(circuit.operations)):
                gate = circuit.operations[time]
                # All conjugations are C^dag P C and not C P C^dag.
                if gate.gate_id == 0:  # I gate
                    break
                elif gate.gate_id == 1:  # X gate
                    x_frame[gate.qudit_index] += 1
                elif gate.gate_id == 2:  # X inverse
                    x_frame[gate.qudit_index] -= 1
                elif gate.gate_id == 3:  # Z gate
                    z_frame[gate.qudit_index] += 1
                elif gate.gate_id == 4:  # Z inverse
                    z_frame[gate.qudit_index] -= 1
                elif gate.gate_id == 5:  # Hadamard
                    tmp = x_frame[gate.qudit_index]
                    x_frame[gate.qudit_index] = z_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] = -tmp
                elif gate.gate_id == 6:  # Inverse Hadamard
                    tmp = x_frame[gate.qudit_index]
                    x_frame[gate.qudit_index] = -z_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] = tmp
                elif gate.gate_id == 7:  # P gate
                    z_frame[gate.qudit_index] -= x_frame[gate.qudit_index]
                elif gate.gate_id == 8:  # Inverse P gate
                    z_frame[gate.qudit_index] += x_frame[gate.qudit_index]
                elif gate.gate_id == 9:  # CNOT gate
                    x_frame[gate.target_index] -= x_frame[gate.qudit_index]
                    z_frame[gate.target_index] += z_frame[gate.qudit_index]
                elif gate.gate_id == 10:  # Inverse CNOT
                    x_frame[gate.target_index] += x_frame[gate.qudit_index]
                    z_frame[gate.target_index] -= z_frame[gate.qudit_index]
                elif gate.gate_id == 11:  # CZ gate
                    z_frame[gate.target_index] -= x_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] -= x_frame[gate.target_index]
                elif gate.gate_id == 12:  # Inverse CZ
                    z_frame[gate.target_index] += x_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] += x_frame[gate.target_index]
                elif gate.gate_id == 13:  # SWAP gate
                    tmp = x_frame[gate.qudit_index]
                    x_frame[gate.qudit_index] = x_frame[gate.target_index]
                    x_frame[gate.target_index] = tmp
                    tmp = z_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] = z_frame[gate.target_index]
                    z_frame[gate.target_index] = tmp
                elif gate.gate_id == 14:  # Measurement (computational basis)
                    q = gate.qudit_index
                    meas_round = measurement_counts[q]
                    # Retrieve reference measurement value and deterministic flag.
                    ref_val = self.measurement_results[q][meas_round][0].measurement_value
                    deterministic_flag = self.measurement_results[q][meas_round][0].deterministic
                    # For each extra shot, compute and store the measurement result.
                    for shot in range(extra_shots):
                        new_val = (ref_val + x_frame[q, shot]) % tableau.dimension
                        frame_results[q, meas_round, shot] = (q, meas_round, shot, deterministic_flag, new_val)
                    measurement_counts[q] += 1
                    # Update the z_frame with a fresh random offset after measurement.
                    z_frame[q] += np.random.randint(0, tableau.dimension, size=extra_shots)
                elif gate.gate_id == 15:  # Measurement (X basis)
                    tmp = x_frame[gate.qudit_index]
                    x_frame[gate.qudit_index] = -z_frame[gate.qudit_index]
                    z_frame[gate.qudit_index] = tmp
                    q = gate.qudit_index
                    meas_round = measurement_counts[q]
                    ref_val = self.measurement_results[q][meas_round][0].measurement_value
                    deterministic_flag = self.measurement_results[q][meas_round][0].deterministic
                    for shot in range(extra_shots):
                        new_val = (ref_val + x_frame[q, shot]) % tableau.dimension
                        frame_results[q, meas_round, shot] = (q, meas_round, shot, deterministic_flag, new_val)
                    measurement_counts[q] += 1
                    z_frame[q] += np.random.randint(0, tableau.dimension, size=extra_shots)
                elif gate.gate_id == 16:  # Reset gate
                    x_frame[gate.qudit_index] = 0
                    z_frame[gate.qudit_index] = np.random.randint(0, tableau.dimension, size=extra_shots)
                elif gate.gate_id == 17:  # Single qudit noise
                    x_frame[gate.qudit_index] += noise[noise_counter][:, 0] 
                    z_frame[gate.qudit_index] += noise[noise_counter][:, 1] 
                    noise_counter += 1
                else:
                    pass
                if time % 200 == 0:
                    for i in range(n_qudits):
                        for shot in range(extra_shots):
                            x_frame[i, shot] = x_frame[i, shot] % tableau.dimension
                            z_frame[i, shot] = z_frame[i, shot] % tableau.dimension
        return frame_results
    
    def _combine_results(self, frame_results) -> list:
        """
        Combines the reference shot results with the extra-shot results from the Pauli frame simulation.
        
        This helper appends the extra-shot measurement results directly to self.measurement_results, which is
        a grouped 3D list with dimensions:
            self.measurement_results[qudit_index][measurement_round][shot]
        where shot 0 is the reference shot. For each measurement round, the deterministic flag is taken from the
        reference shot, while the qudit_index and measurement_value are taken from the extra-shot (Pauli frame)
        results contained in frame_results (a NumPy structured array).
        
        Args:
            reference_results: The flattened reference shot results (used only to ensure self.measurement_results is grouped).
            frame_results: A NumPy structured array with shape (num_qudits, num_rounds, extra_shots)
                        containing the extra-shot measurement results.
        
        Returns:
            The updated self.measurement_results—a 3D list of MeasurementResult objects with dimensions:
                [qudit_index][measurement_round][shot],
            where shot 0 is the reference result and subsequent shots are from the frame simulation.
        """
        num_qudits = len(self.measurement_results)
        for q in range(num_qudits):
            num_rounds = len(self.measurement_results[q])
            for m in range(num_rounds):
                deterministic_flag = self.measurement_results[q][m][0].deterministic
                for shot in range(frame_results.shape[2]):
                    rec = frame_results[q, m, shot]
                    new_result = MeasurementResult(
                        qudit_index=int(rec['qudit_index']),
                        deterministic=deterministic_flag,
                        measurement_value=int(rec['measurement_value'])
                    )
                    self.measurement_results[q][m].append(new_result)
        return self.measurement_results
        
    def _build_ir(self, circuits: list[Circuit], extra_shots: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds an intermediate representation (IR) for the given circuits and also precomputes
        an array of sampled Pauli noise outcomes for noise gates (if applicable)
        
        Args:
            circuits (list[Circuit]): A list of Circuit objects.
            extra_shots (int): The number of extra shots for which noise outcomes
                            will be sampled.
        
        Returns:
            tuple:
                - A NumPy array of IR instructions with each element as a tuple
                (gate_id, qudit_index, target_index).
                - A NumPy array of shape (num_noise_gates, extra_shots) containing
                pre-sampled noise outcomes for each noise gate encountered.
                If no noise gate is present, an empty array is returned.
        """
        ir_list  = []
        noise_list = []
        dimension = self.stabilizer_tableau.dimension
        for circuit in circuits:
            for instruction in circuit.operations:
                if instruction.gate_id == 0:
                    continue
                target_index = instruction.target_index if instruction.target_index is not None else -1
                ir_list.append((instruction.gate_id, instruction.qudit_index, target_index))
                
                if instruction.gate_id in noise_gate_indices:
                    # Perform noise sampling only with some probability.
                    if np.random.random() < instruction.params['probability']:
                        channel = instruction.params['noise_channel']
                        if channel == 'd':
                            # Sample integer r from 1 to dimension**2 - 1 for each extra shot.
                            r = np.random.randint(1, dimension**2, size=extra_shots)
                            a = r % dimension
                            b = r // dimension
                            pair = np.stack((a, b), axis=1) 
                            noise_list.append(pair)
                        elif channel == 'f':
                            a = np.random.randint(0, dimension, size=extra_shots)
                            b = np.zeros(extra_shots, dtype=np.int64)
                            pair = np.stack((a, b), axis=1)
                            noise_list.append(pair)
                        elif channel == 'p':
                            a = np.zeros(extra_shots, dtype=np.int64)
                            b = np.random.randint(0, dimension, size=extra_shots)
                            pair = np.stack((a, b), axis=1)
                            noise_list.append(pair)

        ir_dtype = np.dtype([
            ('gate_id', np.int64),
            ('qudit_index', np.int64),
            ('target_index', np.int64)
        ])

        ir_array = np.array(ir_list, dtype=ir_dtype)

        if noise_list:
            noise_array = np.array(noise_list, dtype=np.int64)
        else:
            noise_array = np.empty((0, extra_shots, 2), dtype=np.int64)

        return ir_array, noise_array

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