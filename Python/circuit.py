from gatedata import GateData


class CircuitInstruction:
    def __init__(self, gate_name, qudit_index, gate_data, target_index=None):
        self.qudit_index = qudit_index
        self.target_index = target_index
        gate_id = gate_data.gateDataMap[gate_name].gate_id
        if gate_id is None:
            raise ValueError(f"Gate {gate_name} not found")
        self.gate_id = gate_id

    def __str__(self):
        return f"{self.gate_id} {self.qudit_index} {self.target_index}"


class Circuit:
    def __init__(self, num_qudits, dimension=2):
        self.operations = []
        self.num_qudits = num_qudits
        self.dimension = dimension
        self.gate_data = GateData(dimension)

    def add_gate(self, gate_name, qudit_index, target_index=None):
        """
        Add a CircuitInstruction to list operations after finding ID from GateData
        CircuitInstruction contains qubit_index, target_index, and gate_id
        """
        self.operations.append(
            CircuitInstruction(gate_name, qudit_index, self.gate_data, target_index)
        )

    def __str__(self):
        return "\n".join(str(op) for op in self.operations)
