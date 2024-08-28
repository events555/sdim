from typing import Tuple, Optional, Callable
from .circuit import CircuitInstruction, Circuit
from .tableau.tableau_composite import WeylTableau
from .tableau.tableau_prime import ExtendedTableau
from .tableau.tableau_gates import *
from sympy import isprime
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
    15: apply_measure_x # Measure gate in X basis
}


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

    def simulate(self, show_measurement: bool = False, verbose: bool = False,
                 show_gate: bool = False, exact: bool = False) -> list[MeasurementResult]:
        """
        Runs the circuit and applies the gates to the stabilizer tableau.

        Args:
            show_measurement (bool): Whether to print the measurement results.
            verbose (bool): Whether to print the stabilizer tableau at each time step.
            show_gate (bool): Whether to print the gate name at each time step.
            exact (bool): Whether to use the Diophantine solver instead of column reduction.
                Much slower but fails less often.

        Returns:
            list[MeasurementResult]: A list of MeasurementResult objects representing
            the measurement results. Returns an empty list if no measurements are present.
        """
        length = sum(len(circuit.operations) for circuit in self.circuits)
        if isinstance(self.stabilizer_tableau, WeylTableau) and exact:
            self.stabilizer_tableau.exact = True
        for circuit in self.circuits:
            for time, gate in enumerate(circuit.operations):
                if time == 0 and verbose:
                    print("Initial state")
                    self.stabilizer_tableau.print_tableau()
                    print("\n")
                measurement_result = self.apply_gate(gate)
                if measurement_result is not None:
                    self.measurement_results.append(measurement_result)
                if show_gate:
                    if time < length - 1:
                        print("Time step", time, "\t", gate.name, gate.qudit_index, gate.target_index if gate.target_index is not None else "")
                    else:
                        print("Final step", time, "\t", gate.name, gate.qudit_index, gate.target_index if gate.target_index is not None else "")
                if verbose:
                    self.stabilizer_tableau.print_tableau()
                    print("\n")
            if show_measurement:
                print("Measurement results:")
                self.print_measurements()
        return self.measurement_results

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
        measurement_result = gate_function(self.stabilizer_tableau, instruc.qudit_index, instruc.target_index)
        return measurement_result


    def print_measurements(self):
        """
        Prints the measurement results.

        This method iterates through the stored measurement results and prints each one.
        """
        for result in self.measurement_results:
            print(result)

    def __str__(self) -> str:
        return str(self.stabilizer_tableau)