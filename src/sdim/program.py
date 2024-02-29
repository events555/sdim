from .tableau import Tableau
from .tableau_simulator import apply_H, apply_P, apply_CNOT, measure

GATE_FUNCTIONS = {
    0: lambda tableau, qudit_index, target_index: None,  # I gate
    1: lambda tableau, qudit_index, target_index: None,  # X gate
    2: lambda tableau, qudit_index, target_index: None,  # Z gate
    3: apply_H,  # H gate
    4: apply_P,  # P gate
    5: apply_CNOT,  # CNOT gate
    6: measure  # Measure gate
}


class Program:
    def __init__(self, circuit, tableau=None):
        if tableau is None:
            self.stabilizer_tableau = Tableau(circuit.num_qudits, circuit.dimension)
            self.stabilizer_tableau.identity()
        else:
            self.stabilizer_tableau = tableau
        self.circuit = circuit
        self.measurement_results = []

    def simulate(self, show_measurement=False, verbose=False, show_gate=False):
        for time, gate in enumerate(self.circuit.operations):
            self.stabilizer_tableau, measurement = self.apply_gate(gate)
            if gate.gate_id == 6:
                self.measurement_results.append((gate.qudit_index, measurement[0], measurement[1]))
            if show_gate:
                print("Time step", time, "\t", gate.name)
            if verbose:
                self.print_tableau()
                print("\n")
        if verbose:
            print("Final state:")
            self.print_tableau()
        if show_measurement:
            print("Measurement results:")
            self.print_measurements()

    def apply_gate(self, instruc):
        """
        Apply a gate to the stabilizer tableau
        Args:
            instruc: A Gate object from Circuit
        Returns:
            The updated stabilizer tableau and a flag indicating whether the measurement was deterministic
        """
        if instruc.gate_id not in GATE_FUNCTIONS:
            raise ValueError("Invalid gate value")
        gate_function = GATE_FUNCTIONS[instruc.gate_id]
        updated_tableau, measurement_type = gate_function(self.stabilizer_tableau, instruc.qudit_index, instruc.target_index)
        return updated_tableau, measurement_type

    def print_tableau(self):
        self.stabilizer_tableau.print_tableau_num()

    def print_measurements(self):
        for qudit_index, measurement_type, measurement_value in self.measurement_results:
            measurement_type_str = "deterministic" if measurement_type else "random"
            print(f"Measured qudit ({qudit_index}) as ({measurement_value}) and was {measurement_type_str}.")