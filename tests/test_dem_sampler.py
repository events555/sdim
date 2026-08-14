from random import randint
import math
from dataclasses import dataclass
from typing import Union, Optional, List
import random
import numpy as np
import time
from sdim.circuit import Circuit
from sdim.circuit_io import write_circuit, read_circuit
from sdim.circuit_io import circuit_to_cirq_circuit
from sdim.program import Program
import os
import datetime
from matplotlib import pyplot as plt
from typing import Tuple, Optional, Callable
import csv
from sdim.dem import DetectorErrorModel
from pathlib import Path


def chained_CNOT_test(dimension : int = 2, 
                    num_qudits : int = 2,
                    include_single_noise_events : bool = False, 
                    single_noise_event_probablity : float = 0.01, 
                    include_two_qudit_noise_events : bool = True, 
                    two_qudit_noise_channel : list = None
):
    
    c = Circuit(num_qudits=num_qudits, dimension=dimension)

    for j in range(num_qudits - 1):
        c.add_gate("CNOT", j, j + 1)

    if include_single_noise_events:
        for j in range(num_qudits):
            c.add_gate("N1", j, noise_channel='d', prob=single_noise_event_probablity)

    if include_two_qudit_noise_events:
        for j in range(num_qudits - 1):
            c.add_gate("N2", j, j + 1, prob_dist=two_qudit_noise_channel)
    
    c.add_gate("M", [j for j in range(num_qudits)])

    for j in range(1, num_qudits):
        c.add_gate("DETECTOR", expr=f"rec[{j}] - rec[{j - 1}]", label=f"detector {j}")
        # print(f"DETECTOR: rec[{j}] - rec[{j - 1}]")

    return c


def run(dimension : int = 2, 
        num_qudits : int = 2,
        epsilon : float = 0.05, 
        tests : list = [chained_CNOT_test],
        include_single_noise_events : bool = False, 
        single_noise_event_probablity : float = 0.01, 
        include_two_qudit_noise_events : bool = True, 
        two_qudit_noise_channel : list = None):
    
    # If no two qudit channel is passed, by default we use the uniform distribution
    two_qudit_noise_channel = two_qudit_noise_channel if two_qudit_noise_channel is not None else np.array( [ 1 / (dimension ** 4) for _ in range(dimension ** 4) ])

    # Sanitize input
    try:
        assert dimension > 1, f"Dimension must be a positive integer that is at least 2"
        assert num_qudits > 1, f"The number of qudits must be a positive integer that is at least 2"
        assert epsilon > 0, f"The tolerance must be positive"
        assert single_noise_event_probablity > 0, f"The single noise event probability must be positive"
        assert len(two_qudit_noise_channel) == dimension ** 4, f"The number of elements in the provided two qudit noise probability distribution ({len(two_qudit_noise_channel)}) doesn't match the number needed ({dimension ** 4})."
        assert np.isclose(1, sum(two_qudit_noise_channel)), f"The two qudit probabilities do not sum to a valid noise distribution."
        assert tests, f"No tests provided."
    except Exception as e:
        raise e
    
    # Determine number of required samples
    # TVD(A, B) < epsilon with high probability requires O(m / epsilon^2), where m = (dimension ** (num_qudits - 1)) is the number of shift outcomes
    shots = 3 * math.ceil((dimension ** (num_qudits - 1)) / (epsilon ** 2))
    print(f"Need to sample {shots} shots.")
    
    for test in tests:
        test_circuit = test(dimension=dimension, num_qudits=num_qudits, include_single_noise_events=include_single_noise_events, single_noise_event_probablity=single_noise_event_probablity, include_two_qudit_noise_events=include_two_qudit_noise_events, two_qudit_noise_channel=two_qudit_noise_channel)
    
        dem = DetectorErrorModel.from_circuit(test_circuit)
        print(dem)

        # Collect samples via Pauli frames
        p = Program(test_circuit)
        _, pauli_frame_detection_events = p.simulate(shots=shots, raw_detector_output=True)
        detector_events = np.array(pauli_frame_detection_events[0])
        detector_events = detector_events.swapaxes(0, 1)

        # Collect samples via DEM sampler
        dem_samples, _ = dem.sample(shots=shots)

        # Binning samples into counts
        pauli_detector_unique_arrs, pauli_counts = np.unique(detector_events, axis=0, return_counts=True)
        dem_unique_arrs, dem_counts = np.unique(dem_samples, axis=0, return_counts=True)

        #print("Unique arrays:\n", pauli_detector_unique_arrs)
        #print("Counts:\n", pauli_counts)
        #print("Counts:\n", dem_counts)

        pauli_frame_probabilities = pauli_counts / shots
        dem_probabilities = dem_counts / shots
        tvd = np.sum(np.abs(pauli_frame_probabilities - dem_probabilities)) / 2

        print(f"TVD is {tvd}")

    

    return



if __name__ == "__main__":

    run(dimension=5, num_qudits=3, epsilon=0.05, include_single_noise_events=True, single_noise_event_probablity=0.1, include_two_qudit_noise_events=False)