from .gatedata import GateData
from dataclasses import dataclass


@dataclass
class CircuitInstruction:
    gate_data: GateData
    gate_name: str
    qudit_index: int
    target_index: int = None
    gate_id: int = None
    name: str = None

    def __post_init__(self):
        self.gate_id = self.gate_data.get_gate_id(self.gate_name)
        if self.gate_id is None:
            raise ValueError(f"Gate {self.gate_name} not found")
        self.name = self.gate_data.get_gate_name(self.gate_id)

    def __str__(self):
        return f"{self.gate_id} {self.qudit_index} {self.target_index}"


@dataclass
class Circuit:
    num_qudits: int
    dimension: int = 2
    operations: list = None
    gate_data: GateData = None

    def __post_init__(self):
        if self.num_qudits < 1:
            raise ValueError("Number of qudits must be greater than 0")
        if self.dimension < 2:
            raise ValueError("Dimension must be greater than 1")
        self.operations = self.operations or []
        self.gate_data = self.gate_data or GateData(self.dimension)
    
    def add_gate(self, gate_name:str, qudit_index:int, target_index:int=None):
        instruction = CircuitInstruction(self.gate_data, gate_name.upper(), qudit_index, target_index)
        self.operations.append(instruction)

    def __str__(self):
        return "\n".join(str(op) for op in self.operations)
    
    def print_gateData(self):
        print(self.gate_data)
    
    @classmethod
    def from_operation_list(cls, operation_list, num_qudits, dimension):
        circuit = cls(num_qudits, dimension)
        for op in operation_list:
            if isinstance(op, tuple):  # If the operation is a tuple (gate, qudits)
                gate_name = op[0]
                qudits = op[1]
                if len(qudits) == 1:
                    circuit.add_gate(gate_name, qudits[0])
                elif len(qudits) == 2:
                    circuit.add_gate(gate_name, qudits[0], qudits[1])
                else:
                    raise ValueError(f"Unsupported number of qudits for gate {gate_name}")
            elif isinstance(op, CircuitInstruction):  # If the operation is a CircuitInstruction
                circuit.add_gate(op.gate_name, op.qudit_index, op.target_index)
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return circuit