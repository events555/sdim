import pytest
import cirq as cirq
from sdim.circuit import Circuit
from sdim.program import Program
from sdim.tableau import MeasurementResult
from sdim.random_circuit import cirq_statevector_from_circuit

def test_phase_kickback():
    circuit = Circuit(2, 2)
    circuit.add_gate("H", 1)
    circuit.add_gate("P", 1)

    circuit.add_gate("H", 0)
    circuit.add_gate("CNOT", 0, 1)
    circuit.add_gate("H", 1)
    circuit.add_gate("CNOT", 0, 1)
    circuit.add_gate("H", 1)

    circuit.add_gate("P", 0)
    circuit.add_gate("H", 0)
    circuit.add_gate("M", 0)
    circuit.add_gate("P", 1)
    circuit.add_gate("H", 1)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 1), MeasurementResult(1, True, 1)]

def test_qubit_flip():
    circuit = Circuit(2, 2)
    circuit.add_gate("H", 0)
    circuit.add_gate("P", 0)
    circuit.add_gate("P", 0)
    circuit.add_gate("H", 0)
    circuit.add_gate("M", 0)
    circuit.add_gate("X", 1)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 1), MeasurementResult(1, True, 1)]

def test_qutrit_flip():
    circuit = Circuit(2, 3)
    circuit.add_gate("H", 0)
    circuit.add_gate("P", 0)
    circuit.add_gate("P", 0)
    circuit.add_gate("H", 0)
    circuit.add_gate("H", 0)
    circuit.add_gate("P", 0)
    circuit.add_gate("H", 0)
    circuit.add_gate("M", 0)

    circuit.add_gate("H", 1)
    circuit.add_gate("P", 1)
    circuit.add_gate("P", 1)
    circuit.add_gate("H", 1)
    circuit.add_gate("H", 1)
    circuit.add_gate("P", 1)
    circuit.add_gate("H", 1)

    circuit.add_gate("X", 1)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 1), MeasurementResult(1, True, 2)]
