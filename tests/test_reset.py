import random
import numpy as np
from sdim.circuit import Circuit
from sdim.circuit_io import write_circuit
from sdim.circuit_io import read_circuit
from sdim.program import Program
import math
import pytest


def test_reset_skip():
    
    dimension = random.choice([3, 5, 7, 11])
    # Maximal mixing probability to uniformly randomly sample qudit Pauli
    p = ((dimension * dimension) - 1) / (dimension * dimension)
    num_X = random.choice([i for i in range(dimension)])
    shots = 1000000
    
    c = Circuit(dimension=dimension, num_qudits=1)
    
    c.add_gate("N1", 0, prob=p, noise_channel='d')
    c.add_gate("RESET", 0)

    for _ in range(num_X):
        c.add_gate("X", 0)

    c.add_gate("M", 0)

    # 3D array as (num_qudit, num_measurement, shot)
    result = Program(c).simulate(shots=shots)

    measurement_counts = [0 for _ in range(dimension)]

    for s in range(shots):
        measurement_counts[result[0][1][s].measurement_value] += 1

    assert(measurement_counts[num_X] == shots)