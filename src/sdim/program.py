from typing import Tuple, Optional, Callable
from .circuit import CircuitInstruction, Circuit
from .tableau import Tableau
from .tableau_gates import *

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
    11: apply_SWAP,  # SWAP gate
    12: apply_measure # Measure gate
}


class Program:
    def __init__(self, circuit: Circuit, tableau=None):
        if tableau is None:
            self.stabilizer_tableau = Tableau(circuit.num_qudits, circuit.dimension)
        else:
            self.stabilizer_tableau = tableau
        self.circuit = circuit
        self.measurement_results = []

    def simulate(self, show_measurement: bool=False, verbose: bool=False, show_gate: bool=False, exact: bool=False) -> list[MeasurementResult]:
        length = len(self.circuit.operations)
        for time, gate in enumerate(self.circuit.operations):
            if time == 0 and verbose:
                print("Initial state")
                self.stabilizer_tableau.print_tableau()
                print("\n")
            measurement_result = self.apply_gate(gate, exact)
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

    def apply_gate(self, instruc: CircuitInstruction, exact: bool) -> Tuple[Tableau, Optional[MeasurementResult]]:
        """
        Apply a gate to the stabilizer tableau
        Args:
            instruc: A CircuitInstruction object from a Circuit's operation list
        Returns:
            The updated stabilizer tableau and a MeasurementResult
        """
        if instruc.gate_id not in GATE_FUNCTIONS:
            raise ValueError("Invalid gate value")
        gate_function = GATE_FUNCTIONS[instruc.gate_id]
        measurement_result = gate_function(self.stabilizer_tableau, instruc.qudit_index, instruc.target_index, exact)
        return measurement_result


    def print_measurements(self):
        for result in self.measurement_results:
            print(result)