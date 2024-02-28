from dataclasses import dataclass
from gate import Gate
from tableau import Tableau
from unitary import generate_cnot_matrix, generate_h_matrix, generate_identity_matrix, generate_p_matrix, generate_x_matrix, generate_z_matrix

@dataclass
class GateData:
    gateDataMap: dict
    aliasMap: dict
    num_gates: int = 0
    dimension: int = 2

    def __init__(self, dimension=2):
        self.gateDataMap = {}
        self.aliasMap = {}
        self.dimension = dimension
        self.add_gate_data_pauli(dimension)
        self.add_gate_hada(dimension)
        self.add_gate_controlled(dimension)
        self.add_gate_collapsing(dimension)

    def __str__(self):
        return "\n".join(str(gate) for gate in self.gateDataMap.values())

    def add_gate(self, name, arg_count, tableau, unitary_matrix):
        gate_id = self.num_gates
        gate = Gate(name, arg_count, gate_id, tableau, unitary_matrix)
        self.gateDataMap[name] = gate
        self.num_gates += 1

    def add_gate_alias(self, name, list_alias):
        for alias in list_alias:
            self.aliasMap[alias] = name

    def add_gate_data_pauli(self, d):
        I_tableau = Tableau(1, d)
        I_tableau.identity()
        self.add_gate("I", 1, I_tableau, generate_identity_matrix(d))
        self.add_gate("X", 1, Tableau(1, d).gate1("Z", "X!"), generate_x_matrix(d))
        self.add_gate("Z", 1, Tableau(1, d).gate1("X", "Z"), generate_z_matrix(d))

    def add_gate_hada(self, d):
        self.add_gate("H", 1, Tableau(1, d).gate1("Z", "X!"), generate_h_matrix(d))
        self.add_gate_alias("H", ["R", "DFT"])
        self.add_gate("P", 1, Tableau(1, d).gate1("XZ", "Z"), generate_p_matrix(d))
        self.add_gate_alias("P", ["PHASE", "S"])

    def add_gate_controlled(self, d):
        self.add_gate(
            "CNOT",
            2,
            Tableau(2, d).gate2(["(X)(X)", "(I)(X)"], ["(Z)(I)", "(Z!)(Z)"]),
            generate_cnot_matrix(d),
        )
        self.add_gate_alias("CNOT", ["SUM", "CX", "C"])

    def add_gate_collapsing(self, d):
        self.add_gate(
            "M",
            1,
            Tableau(1, d).gate1("Z", "Z"),
            generate_identity_matrix(d),
        )
        self.add_gate_alias("M", ["MEASURE", "COLLAPSE", "MZ"])