import random
import numpy as np
from sdim.circuit import Circuit
from sdim.circuit_io import write_circuit
from sdim.circuit_io import read_circuit
from sdim.program import Program
import math
import pytest


single_qudit_deterministic_gates = ["I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV"]
single_qudit_measurement_gates = ["M"]
single_qudit_noise_gates = ["N1"]
single_qudit_inv = {"H" : "H_INV", "H_INV" : "H", 
		    		"P" : "P_INV", "P_INV" : "P", 
                    "Z" : "Z_INV", "Z_INV" : "Z",
                    "X" : "X_INV", "X_INV" : "X",
                    "I" : "I"}

two_qudit_gates = ["CNOT", "CNOT_INV", "CZ", "CZ_INV"]



# def random_multi_qudit_circuit(depth : int = 10, dimension : int = 3, num_qudits : int = 2, sample_with_noise_gates = True):

#     single_qudit_non_noise_gates = single_qudit_deterministic_gates + single_qudit_measurement_gates
#     single_qudit_gates_with_noise = single_qudit_non_noise_gates + single_qudit_noise_gates

#     if num_qudits < 2:
#         raise ValueError("Can't have less than 2 qudits in a multi qudit test!")

#     c = Circuit(dimension=dimension, num_qudits=num_qudits)

#     for _ in range(depth):

#         apply_single_qudit_gate = random.choice([True, False])

#         if apply_single_qudit_gate:
#             qudit_target = random.randint(0, num_qudits - 1)
#             gate_set = single_qudit_gates_with_noise  if sample_with_noise_gates else single_qudit_non_noise_gates
#             gate = random.choice(gate_set)

#             if gate == "N1":
#                 p = random.uniform(0.0, 1.0)
#                 channel = random.choice(['f', 'p', 'd'])
#                 c.add_gate(gate, qudit_target, prob=p, noise_channel=channel)
#             else:
#                 c.add_gate(gate, qudit_target)
#         else:
#             qudit_control = 0
#             qudit_target = 0
#             gate = random.choice(two_qudit_gates)

#             while (qudit_control == qudit_target):
#                 qudit_control = random.randint(0, num_qudits - 1)
#                 qudit_target = random.randint(0, num_qudits - 1)

#             c.add_gate(gate, qudit_control, qudit_target)

#     return c
def random_multi_qudit_circuit(depth : int = 10, dimension : int = 3, num_qudits : int = 2, sample_with_noise_gates = True):

    single_qudit_non_noise_gates = single_qudit_deterministic_gates
    single_qudit_gates_with_noise = single_qudit_non_noise_gates + single_qudit_noise_gates

    if num_qudits < 2:
        raise ValueError("Can't have less than 2 qudits in a multi qudit test!")

    c = Circuit(dimension=dimension, num_qudits=num_qudits)

    for _ in range(depth):

        apply_single_qudit_gate = random.choice([True, False])

        if apply_single_qudit_gate:
            qudit_target = random.randint(0, num_qudits - 1)
            gate_set = single_qudit_gates_with_noise  if sample_with_noise_gates else single_qudit_non_noise_gates
            gate = random.choice(gate_set)

            if gate == "N1":
                p = random.uniform(0.0, 1.0)
                channel = random.choice(['f', 'p', 'd'])
                c.add_gate(gate, qudit_target, prob=p, noise_channel=channel)
            else:
                c.add_gate(gate, qudit_target)
        else:
            qudit_control = 0
            qudit_target = 0
            gate = random.choice(two_qudit_gates)

            while (qudit_control == qudit_target):
                qudit_control = random.randint(0, num_qudits - 1)
                qudit_target = random.randint(0, num_qudits - 1)

            c.add_gate(gate, qudit_control, qudit_target)

    c.add_gate("M", [i for i in range(num_qudits)])

    return c

def generic_read_write_test(has_noise : bool = True):

    passed_test = True
    depth = random.randint(0, 100000)
    dimension = 5
    num_qudits = random.randint(0, 100)
    c = random_multi_qudit_circuit(depth=depth, dimension=dimension, num_qudits=num_qudits, sample_with_noise_gates=has_noise)

    write_circuit(circuit=c, output_file="test_no_noise_io.chp", comment="To go where no test has ever gone.")
    cc = read_circuit("./circuits/test_no_noise_io.chp")

    # Matching dimension test 
    passed_test = passed_test and (c.dimension == cc.dimension)

    # Testing that every instruction is identically scribed from original circuit
    for index, op in enumerate(c.operations):
        cc_op = cc.operations[index]
        # Matching name test
        passed_test = passed_test and (op.gate_name == cc_op.gate_name)
        # Matching control qudit test
        passed_test = passed_test and (op.qudit_index == cc_op.qudit_index)
        # Matching target qudit test
        passed_test = passed_test and (op.target_index == cc_op.target_index)
        # Matching gate_id test
        passed_test = passed_test and (op.gate_id == cc_op.gate_id)
        # Matching name test
        passed_test = passed_test and (op.name == cc_op.name)
        # Matching optional parameters test
        if op.params is None:
            passed_test = passed_test and (op.params == cc_op.params)
        else:
            for k in op.params.keys():
                passed_test = passed_test and (str(op.params[k]) == str(cc_op.params[k]))

    return passed_test


def generic_single_error_type(testing_X : bool = True):

    channel = 'f' if testing_X else 'p'
    p = random.uniform(0.0, 1.0)
    dimension = random.choice([3, 5, 7, 11, 13, 17])
    shots = 100000
    

    c = Circuit(dimension=dimension, num_qudits=1)

    if not testing_X:
        c.add_gate("H", 0)
    
    c.add_gate("N1", 0, prob=p, noise_channel=channel)

    if not testing_X:
        c.add_gate("H_INV", 0)

    c.add_gate("M", 0)

    result = Program(c).simulate(shots=shots)
    measurement_counts = [0 for _ in range(dimension)]

    for s in range(shots):
        measurement_counts[result[0][0][s].measurement_value] += 1

    empirical_probs = [measurement_counts[i] / shots for i in range(dimension)]
    ideal_probs = [1 - p] + [p / (dimension - 1) for _ in range(dimension - 1)]
    fuzzy_close = [abs(empirical_probs[i] - ideal_probs[i]) < 0.01 for i in range(dimension)]

    return all(fuzzy_close)


def test_io_no_noise():
    assert generic_read_write_test(False)


def test_io():
    assert generic_read_write_test()


def test_single_qudit_depolarizing():

    dimension = random.choice([3, 5, 7, 11, 13, 17])
    maximal_mixing_prob = (dimension * dimension - 1) / (dimension * dimension)
    p = random.uniform(0.0, maximal_mixing_prob)
    shots = 100000

    c = Circuit(dimension=dimension, num_qudits=1)
    c.add_gate("N1", 0, prob=p, noise_channel='d')
    c.add_gate("M", 0)

    result = Program(c).simulate(shots=shots)
    measurement_counts = [0 for _ in range(dimension)]

    for s in range(shots):
        measurement_counts[result[0][0][s].measurement_value] += 1

    empirical_probs = [measurement_counts[i] / shots for i in range(dimension)]
    ideal_probs = [((1 - p) + (dimension - 1) * (p / (dimension * dimension - 1)))] + [dimension * (p / (dimension * dimension - 1)) for _ in range(dimension - 1)]
    fuzzy_close = [abs(empirical_probs[i] - ideal_probs[i]) < 0.01 for i in range(dimension)]

    assert all(fuzzy_close)


def test_flip_error_channel():
    assert generic_single_error_type()


def test_phase_error_channel():
    assert generic_single_error_type(False)


def test_deterministic_gates():
    # Construct noiseless RB circuit
    depth = 200
    dimension = random.choice([3, 5])
    shots = 1000
    gates = []
    inverse_gates = []
    # Sample Clifford sequence that evaluates to identity 
    for i in range(depth):
        gate = random.choice(single_qudit_deterministic_gates)
        inverse_gate = single_qudit_inv[gate]
        gates = gates + [gate]
        inverse_gates = [inverse_gate] + inverse_gates
    # Construct circuit, measuring in Z
    circuit_gates = gates + inverse_gates
    c = Circuit(dimension=dimension, num_qudits=1)
    for g in circuit_gates:
        c.add_gate(g, 0)
    c.add_gate("M", 0)
    # Run circuit
    result = Program(c).simulate(shots=shots)
    measurement_counts = [0 for _ in range(dimension)]

    for s in range(shots):
        measurement_counts[result[0][0][s].measurement_value] += 1

    assert measurement_counts[0] == shots







