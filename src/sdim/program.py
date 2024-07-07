from typing import Tuple, Optional
from .circuit import CircuitInstruction, Circuit
from .tableau import Tableau
from .tableau_composite import apply_H, apply_P, apply_CNOT, measure, MeasurementResult
from .compound_gates import apply_PauliX, apply_PauliZ

GATE_FUNCTIONS = {
    0: lambda tableau, qudit_index, target_index: None,  # I gate
    1: apply_PauliX,  # X gate
    2: apply_PauliZ,  # Z gate
    3: apply_H,  # H gate
    4: apply_P,  # P gate
    5: apply_CNOT,  # CNOT gate
    6: measure  # Measure gate
}


class Program:
    def __init__(self, circuit: Circuit, tableau=None):
        if tableau is None:
            self.stabilizer_tableau = Tableau(circuit.num_qudits, circuit.dimension)
        else:
            self.stabilizer_tableau = tableau
        self.circuit = circuit
        self.measurement_results = []

    def simulate(self, show_measurement: bool=False, verbose: bool=False, show_gate: bool=False) -> list[MeasurementResult]:
        length = len(self.circuit.operations)
        for time, gate in enumerate(self.circuit.operations):
            self.stabilizer_tableau, measurement_result = self.apply_gate(gate)
            if gate.gate_id == 6:
                self.measurement_results.append(measurement_result)
            if show_gate:
                if time < length - 1:
                    print("Time step", time, "\t", gate.name, gate.qudit_index, gate.target_index if gate.target_index is not None else "")
                else:
                    print("Final step", time, "\t", gate.name, gate.qudit_index, gate.target_index if gate.target_index is not None else "")
            if verbose:
                self.print_tableau()
                print("\n")
        if show_measurement:
            print("Measurement results:")
            self.print_measurements()
        return self.measurement_results

    def apply_gate(self, instruc: CircuitInstruction) -> Tuple[Tableau, Optional[MeasurementResult]]:
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
        updated_tableau, measurement_result = gate_function(self.stabilizer_tableau, instruc.qudit_index, instruc.target_index)
        return updated_tableau, measurement_result

    def print_tableau(self):
        self.stabilizer_tableau.print_matrix()
        self.stabilizer_tableau.print_phase_correction()

    def print_measurements(self):
        for result in self.measurement_results:
            print(result)