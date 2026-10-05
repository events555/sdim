"""
Regression tests for sdim.program: detectors with coefficients far beyond int64 at a large dimension, from
simulate down, the attributes the Program docstring lists, and what seeding reproduces.
"""

import random
import re

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import Program
from sdim.tableau.tableau_prime import ExtendedTableau

D = 1000003


def _circuit(k, num_qudits=1):
    """Flip errors on qudit 0 up to full mixing, then DETECTOR rec[-1] and DETECTOR k*rec[-1]."""
    c = Circuit(num_qudits, D)
    c.add_gate("N1", 0, noise_channel="f", prob=1 - 1 / D)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr=f"{k}*rec[-1]")
    return c


@pytest.mark.parametrize("k", [10 ** 13 + 7, 2 ** 63 + 5, 10 ** 20 + 1, -(10 ** 19) - 3])
def test_unreduced_coefficients_at_a_large_dimension(k):
    """k * rec[-1] left int64 before the reduction mod d: for k = 10**13 + 7, 163 of 2000 frame shots broke
    k * D0 = D1, and the other coefficients raised OverflowError under NumPy 2."""
    c = _circuit(k)
    (mechanism,) = DetectorErrorModel.from_circuit(c).mechanisms
    (generator,) = mechanism.generators
    assert generator[1] * pow(generator[0], -1, D) % D == k % D

    np.random.seed(13)
    _, (detectors, _) = Program(c).simulate(shots=2001, raw_detector_output=True)
    assert np.count_nonzero(detectors[0]) > 1900
    assert all((k * a - b) % D == 0 for a, b in zip(detectors[0].tolist(), detectors[1].tolist()))

    # A program that does not start in a computational basis state runs its shots on the tableau, and
    # evaluates the same detectors on their shifts.
    tableau = ExtendedTableau(2, D)
    tableau.hadamard(1)
    random.seed(13)
    _, (detectors, _) = Program(_circuit(k, 2), tableau=tableau).simulate(shots=201, raw_detector_output=True)
    assert np.count_nonzero(detectors[0]) > 190
    assert all((k * a - b) % D == 0 for a, b in zip(detectors[0].tolist(), detectors[1].tolist()))


def test_documented_program_attributes_exist():
    """The Program docstring listed a `circuit` attribute, which does not exist."""
    section = Program.__doc__.split("Attributes:")[1].split("Args:")[0]
    names = re.findall(r"^ {8}(\w+):", section, re.MULTILINE)
    assert {"circuits", "initial_tableau"} <= set(names)
    first, second = Circuit(1, 3), Circuit(2, 3)
    program = Program(first)
    program.append_circuit(second)
    assert all(hasattr(program, name) for name in names)
    assert program.circuits == [first, second] and program.initial_tableau.num_qudits == 2


def test_seeds_that_reproduce_a_frame_simulation():
    """np.random.seed reproduces the frame shifts and the detectors; the reference outcome comes from Python's
    random module, so the measurement values need both seeds."""
    c = Circuit(1, D)
    c.add_gate("H", 0)
    c.add_gate("N1", 0, noise_channel="f", prob=0.5)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    runs = []
    for python_seed in (1, 2, 2):
        random.seed(python_seed)
        np.random.seed(3)
        measurements, (detectors, _) = Program(c).simulate(shots=50, raw_detector_output=True)
        values = [r.measurement_value for r in measurements[0][0]]
        runs.append((values, [(v - values[0]) % D for v in values[1:]], detectors[0].tolist()))
    assert runs[0][1:] == runs[1][1:] == runs[2][1:]
    assert runs[0][0] != runs[1][0] and runs[1][0] == runs[2][0]
