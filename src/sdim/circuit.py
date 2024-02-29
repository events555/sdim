from .gatedata import GateData


class CircuitInstruction:
    def __init__(self, gate_name, qudit_index, gate_data, target_index=None):
        self.qudit_index = qudit_index
        self.target_index = target_index
        gate_id = self.get_gate_id(gate_name, gate_data)
        if gate_id is None:
            raise ValueError(f"Gate {gate_name} not found")
        self.gate_id = gate_id
        self.name = self.get_gate_name(gate_id, gate_data)

    def get_gate_id(self, gate_name, gate_data):
        if gate_name in gate_data.gateDataMap:
            return gate_data.gateDataMap[gate_name].gate_id
        elif gate_name in gate_data.aliasMap:
            return gate_data.gateDataMap[gate_data.aliasMap[gate_name]].gate_id
        else:
            return None
    def get_gate_name(self, gate_id, gate_data):
        for name, gate in gate_data.gateDataMap.items():
            if gate.gate_id == gate_id:
                return name
        raise ValueError(f"Gate ID {gate_id} not found")
    def get_gate_matrix(self, gate_id, gate_data):
        for _, gate in gate_data.gateDataMap.items():
            if gate.gate_id == gate_id:
                return gate.unitary_matrix
        raise ValueError(f"Gate ID {gate_id} not found")

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
