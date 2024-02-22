from tableau import Tableau
from tableau_simulator import apply_H, apply_P, apply_CNOT, measure

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

    def simulate(self):
        measurement_results = []
        for time, gate in enumerate(self.circuit.operations):
            print("Time step", time, "\t", gate.name)
            self.stabilizer_tableau.print_tableau_num()
            self.stabilizer_tableau, measurement_type = self.apply_gate(gate)
            if gate.gate_id == 6:
                measurement_value = self.stabilizer_tableau.zlogical[gate.qudit_index].phase
                measurement_results.append((gate.qudit_index, measurement_value, measurement_type))
            print("\n")
        print("Final state")
        self.stabilizer_tableau.print_tableau_num()
        
        print("Measurement results:")
        for qudit_index, measurement_value, measurement_type in measurement_results:
            measurement_type_str = "deterministic" if measurement_type else "random"
            print(f"Measured qudit ({qudit_index}) as ({measurement_value}) and was {measurement_type_str}.")

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