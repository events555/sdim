from dataclasses import dataclass

import numpy as np
from gate import Gate
from tableau import Tableau


@dataclass
class GateData:
    gateDataMap: dict
    num_gates: int = 0
    dimension: int = 2

    def __init__(self, dimension=2):
        self.gateDataMap = {}
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

    def add_gate_data_pauli(self, d):
        I_tableau = Tableau(1, d)
        I_tableau.identity()
        ## PAULIS ARE WRONG BECAUSE MISSING COMMUTATIVE PHASE FACTOR
        X = np.zeros((d, d), dtype=np.complex128)
        X_tableau = Tableau(1, d).gate1("Z", "X!")
        for i in range(d):
            X[i, (i - 1) % d] = 1
        Z = np.zeros((d, d), dtype=np.complex128)
        Z_tableau = Tableau(1, d).gate1("X", "Z")
        for i in range(d):
            Z[i, i] = np.exp(2 * np.pi * 1j * i / d)
        self.add_gate("I", 1, I_tableau, np.eye(d))
        self.add_gate("X", 1, X_tableau, X)
        self.add_gate("Z", 1, Z_tableau, Z)

    def add_gate_hada(self, d):
        H = np.zeros((d, d), dtype=np.complex128)
        for m in range(d):
            for n in range(d):
                H[m, n] = 1 / np.sqrt(d) * np.exp(2 * np.pi * 1j * m * n / d)
        self.add_gate("H", 1, Tableau(1, d).gate1("Z", "X!"), H)
        P = np.eye(d, dtype=np.complex128)
        if d == 2:
            P[d - 1, d - 1] = np.exp(1j * np.pi / 2)
        else:
            P[d - 1, d - 1] = np.exp(1j * 2 * np.pi / d)
        self.add_gate("P", 1, Tableau(1, d).gate1("XZ", "Z"), P)

    def add_gate_controlled(self, d):
        CNOT = np.zeros((d**d, d**d))
        for i in range(d):
            for j in range(d):
                CNOT[d * i + j, d * i + (i + j) % d] = 1
        CNOT = CNOT.transpose()
        self.add_gate(
            "CNOT",
            2,
            Tableau(2, d).gate2(["(X)(X)", "(I)(X)"], ["(Z)(I)", "(Z!)(Z)"]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        )
    def add_gate_collapsing(self, d):
        # wrong tableau and array
        self.add_gate(
            "M",
            1,
            Tableau(1, d).gate1("Z", "Z"),
            np.array([[1, 0], [0, 1]]),
        )
