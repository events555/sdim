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
        instruction = CircuitInstruction(self.gate_data, gate_name, qudit_index, target_index)
        self.operations.append(instruction)

    def __str__(self):
        return "\n".join(str(op) for op in self.operations)
