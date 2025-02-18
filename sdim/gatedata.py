from dataclasses import dataclass
import numpy as np
from .tableau.dataclasses import Tableau
from typing import Optional

@dataclass
class Gate:
    """
    Represents a quantum gate.

    Attributes:
        name (str): The name of the gate.
        arg_count (int): The number of arguments (qudits) the gate operates on.
        gate_id (int): A unique identifier for the gate.
    """
    name: str
    arg_count: int
    gate_id: int
    defaults: Optional[dict] = None
    def __str__(self):
        return f"{self.name} {self.gate_id}"

@dataclass
class GateData:
    """
    Manages a collection of quantum gates and their aliases.

    This class handles the creation, storage, and retrieval of quantum gates
    and their associated data for a provided dimension.

    Attributes:
        gateMap (dict): A dictionary mapping gate names to Gate objects.
        aliasMap (dict): A dictionary mapping gate aliases to their primary names.
        num_gates (int): The total number of gates added.
        dimension (int): The dimension of the quantum system (default is 2 for qubits).
    """
    gateMap: dict 
    aliasMap: dict
    num_gates: int = 0
    dimension: int = 2

    def __init__(self, dimension=2):
        self.gateMap = {}
        self.aliasMap = {}
        self.dimension = dimension
        self.append_data_pauli(dimension)
        self.append_hada(dimension)
        self.append_controlled(dimension)
        self.append_collapsing(dimension)
        self.append_noise(dimension)

    def __str__(self):
        return "\n".join(str(gate) for gate in self.gateMap.values())

    def append(self, name, arg_count):
        gate_id = self.num_gates
        gate = Gate(name, arg_count, gate_id)
        self.gateMap[name] = gate
        self.num_gates += 1

    def append_alias(self, name, list_alias):
        for alias in list_alias:
            self.aliasMap[alias] = name

    def append_data_pauli(self, d):
        self.append("I", 1)
        self.append("X", 1)
        self.append("X_INV", 1)
        self.append("Z", 1)
        self.append("Z_INV", 1)

    def append_hada(self, d):
        self.append("H", 1)
        self.append_alias("H", ["R", "DFT"])
        self.append("H_INV", 1)
        self.append_alias("H_INV", ["R_INV", "DFT_INV", "H_DAG", "R_DAG", "DFT_DAG"])
        self.append("P", 1)
        self.append_alias("P", ["PHASE", "S"])
        self.append("P_INV", 1)
        self.append_alias("P_INV", ["PHASE_INV", "S_INV"])

    def append_controlled(self, d):
        self.append("CNOT", 2)
        self.append_alias("CNOT", ["SUM", "CX", "C"])
        self.append("CNOT_INV", 2)
        self.append_alias("CNOT_INV", ["SUM_INV", "CX_INV", "C_INV"])
        self.append("CZ", 2)
        self.append("CZ_INV", 2)
        self.append("SWAP", 2)

    def append_collapsing(self, d):
        self.append("M", 1)
        self.append_alias("M", ["MEASURE", "COLLAPSE", "MZ"])
        self.append("M_X", 1)
        self.append_alias("M_X", ["MEASURE_X", "MX"])
        self.append("RESET", 1)
        self.append_alias("RESET", ["MR", "MEASURE_RESET", "MEASURE_R"])

    def append_noise(self, d):
        self.append("N1", 1)
        self.append_alias("N1", ["NOISE1"])
        self.gateMap["N1"].defaults = {"channel": "d", "prob": 0.01}

    def get_gate_id(self, gate_name):
        if gate_name in self.gateMap:
            return self.gateMap[gate_name].gate_id
        elif gate_name in self.aliasMap:
            return self.gateMap[self.aliasMap[gate_name]].gate_id
        else:
            return None

    def get_gate_name(self, gate_id):
        for name, gate in self.gateMap.items():
            if gate.gate_id == gate_id:
                return name
        raise ValueError(f"Gate ID {gate_id} not found")

    def get_gate_matrix(self, gate_id):
        for _, gate in self.gateMap.items():
            if gate.gate_id == gate_id:
                return gate.unitary_matrix
        raise ValueError(f"Gate ID {gate_id} not found")