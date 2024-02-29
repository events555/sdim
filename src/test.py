import random
import time
from itertools import product

from circuit import Circuit, QuditRegister
from sympy import I, Matrix, N, arg, exp, eye, nsimplify, pi, pprint, re, simplify, sqrt
from sympy.physics.quantum import TensorProduct
from tableau import Program, Tableau


def generate_paulis(d):
    """
    Generate the Pauli matrices for a given dimension.

    Parameters:
    - d (int): The dimension of the matrices.

    Returns:
    - X (Matrix): The shift matrix.
    - Z (Matrix): The clock matrix.
    """
    X = Matrix.zeros(d)
    for i in range(d):
        X[i, (i - 1) % d] = 1

    Z = Matrix.eye(d)
    for i in range(d):
        Z[i, i] = exp(2 * pi * I * i / d)

    return X, Z


def generate_clifford(d):
    """
    Generate the Clifford gates for a given dimension.

    Parameters:
    - d (int): The dimension of the gates.

    Returns:
    - P (Matrix): The P gate.
    - R (Matrix): The DFT matrix.
    - SUM (Matrix): The SUM gate. (no clue why transposed)
    """
    P = Matrix.eye(d)
    if d == 2:
        P[d - 1, d - 1] = exp(I * pi / 2)
    else:
        P[d - 1, d - 1] = exp(I * 2 * pi / d)

    R = Matrix.zeros(d)
    for m in range(d):
        for n in range(d):
            R[m, n] = 1 / sqrt(d) * exp(2 * pi * I * m * n / d)

    SUM = Matrix.zeros(d**2)
    for i, j in product(range(d), repeat=2):
        SUM[d * i + j, d * i + (i + j) % d] = 1
    SUM = SUM.reshape(d**2, d**2)
    SUM = SUM.transpose()
    return P, R, SUM


def discard_global_phase_state(mat):
    """
    Discard the global phase from a matrix.

    Parameters:
    - mat (Matrix): The input matrix.

    Returns:
    - mat_simplify (Matrix): The matrix with the global phase discarded.
    """
    mat = mat.as_mutable()
    global_phase = None
    for i in range(mat.rows):
        if N(mat[i], 15, chop=True) != 0:
            global_phase = arg(mat[i])
            break
    if global_phase is not None:
        for i in range(mat.rows):
            mat[i] = mat[i] * exp(-I * global_phase)
    mat_num = N(mat, 15, chop=True)
    mat_simplify = mat_num.applyfunc(nsimplify)
    return mat_simplify.as_immutable()


def statevec_test(
    d,
    trials=1,
    num_qudits=2,
    num_gates=2,
    seed=None,
    print_circuit=False,
    print_statevec=False,
    print_stabilizer=False,
):
    if seed is None:
        seed = int(time.time())
    """
    Perform a statevector comparison on a random quantum circuit and equivalent tableau.

    Parameters:
    - d (int): The dimension of the qudits.
    - trials (int): The number of trials to run.
    - num_qudits (int): The number of qudits in the circuit.
    - num_gates (int): The number of gates to apply.
    - seed (int): The seed for the random number generator.
    - print_circuit (bool): Whether to print the circuit.
    - print_statevec (bool): Whether to print the statevector.
    - print_stabilizer (bool): Whether to print the Pauli stabilizers.

    Raises:
    - Exception: If the final stabilizer does not stabilize the statevector.
    """
    for trial in range(trials):
        qr = QuditRegister("Trial %d" % trial, d, num_qudits)
        qc = Circuit(qr)
        table = Tableau(d, num_qudits)
        random.seed(seed + trial)
        X, Z = generate_paulis(d)
        P, R, SUM = generate_clifford(d)
        statevec = Matrix([1 if i == 0 else 0 for i in range(d**num_qudits)])
        for _ in range(num_gates):
            gate_name = random.choice(
                ["R", "P", "SUM"] if num_qudits >= 2 else ["R", "P"]
            )
            if gate_name == "SUM":
                qudit_index = random.randrange(num_qudits - 1)
                target_index = qudit_index + 1
                qc.add_gate(gate_name, qudit_index, target_index)
            else:
                qudit_index = random.randrange(num_qudits)
                qc.add_gate(gate_name, qudit_index)

            gate_map = {"R": R, "P": P, "SUM": SUM}
            if gate_name == "SUM":
                matrices = [
                    gate_map[gate_name] if i == qudit_index else eye(d)
                    for i in range(num_qudits)
                    if i != target_index
                ]
            else:
                matrices = [
                    gate_map[gate_name] if i == qudit_index else eye(d)
                    for i in range(num_qudits)
                ]
            tensor_product = matrices[0]
            for matrix in matrices[1:]:
                tensor_product = TensorProduct(tensor_product, matrix)
            statevec = tensor_product * statevec

        prog = Program(table, qc)
        prog.simulate()
        pauli_map = {"I": eye(d), "X": X, "Z": Z}
        for j in range(num_qudits, 2 * num_qudits):
            pauli_matrices = []
            stabilizer = prog.get_stabilizer(j)
            for matrix in stabilizer.pauli_product:
                pauli_stabilizer = eye(d)
                for char in matrix:
                    pauli_stabilizer = pauli_stabilizer * pauli_map[char]
                pauli_matrices.append(pauli_stabilizer)
            pauli_stabilizer = pauli_matrices[0]
            for i in range(1, num_qudits):
                pauli_stabilizer = TensorProduct(pauli_stabilizer, pauli_matrices[i])
            stabilized = discard_global_phase_state(pauli_stabilizer * statevec)
            statevec = discard_global_phase_state(statevec)
            if not all(
                re(abs(simplify(i)).n()) < 0.5
                for i in (Matrix(stabilized) - Matrix(statevec))
            ):
                if print_circuit or print_statevec:
                    print("Trial %d" % trial)
                if print_circuit:
                    print(prog.stabilizer_tableau)
                    print(qc)
                if print_stabilizer:
                    print("\nStabilizer %d" % (j - num_qudits))
                    pprint(pauli_stabilizer)
                if print_statevec:
                    pprint(stabilized)
                raise Exception(
                    f"Trial {trial} was not stabilized at index {i} of statevector. Seed: {seed}"
                )
