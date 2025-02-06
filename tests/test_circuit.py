import pytest
import cirq as cirq
import random
from sdim.circuit import Circuit
from sdim.program import Program
from sdim.tableau.dataclasses import MeasurementResult
from sdim.circuit_io import cirq_statevector_from_circuit

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
    circuit.add_gate("X", 0)
    circuit.add_gate("M", 0)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 1)]

@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_qudit_swap_computational_basis(dimension):
    circuit = Circuit(2, dimension)

    x0 = random.choice([i for i in range(dimension)])
    x1 = random.choice([i for i in range(dimension)])

    for _ in range(x0):
        circuit.add_gate("X", 0)
    for _ in range(x1):
        circuit.add_gate("X", 1)
    
    circuit.add_gate("SWAP", 0, 1)
    circuit.add_gate("M", 0)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, x1), MeasurementResult(1, True, x0)]

@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_qudit_swap_self_inverse(dimension):
    """
    Test that the SWAP gate is self-inverse. Applying two consecutive SWAP gates
    should leave the original state unchanged.
    """
    circuit = Circuit(2, dimension)
    
    # Prepare a random computational basis state.
    x0 = random.choice(range(dimension))
    x1 = random.choice(range(dimension))
    for _ in range(x0):
        circuit.add_gate("X", 0)
    for _ in range(x1):
        circuit.add_gate("X", 1)
    
    # Apply SWAP twice.
    circuit.add_gate("SWAP", 0, 1)
    circuit.add_gate("SWAP", 0, 1)
    
    # Measure both qudits.
    circuit.add_gate("M", 0)
    circuit.add_gate("M", 1)
    
    program = Program(circuit)
    expected = [MeasurementResult(0, True, x0), MeasurementResult(1, True, x1)]
    assert program.simulate() == expected, (
        f"Double SWAP failed for dimension={dimension} with initial states x0={x0}, x1={x1}"
    )

@pytest.mark.parametrize("a", range(3))
@pytest.mark.parametrize("b", range(3))
def test_qutrit_swap_in_x_basis(a, b):
    """
    Test that the SWAP gate correctly swaps states prepared in the X basis.
    
    The preparation for each qudit is as follows:
      1. Start in the computational |0> state.
      2. Apply X^a (or X^b) to shift |0> to |a> (or |b>).
      3. Apply the Fourier gate F so that F|a> is an eigenstate of the X operator
         with eigenvalue ω^a (and similarly for F|b>).
    
    After applying SWAP, the state on qudit 0 should be F|b> and on qudit 1 should be F|a>.
    Measuring directly in the X basis (with "MX") should then return outcomes b and a respectively.
    """
    circuit = Circuit(2, 3)

    for _ in range(a):
        circuit.add_gate("X", 0)
    circuit.add_gate("DFT", 0)

    for _ in range(b):
        circuit.add_gate("X", 1)
    circuit.add_gate("DFT", 1)

    circuit.add_gate("SWAP", 0, 1)

    circuit.add_gate("MX", 0)
    circuit.add_gate("MX", 1)

    program = Program(circuit)
    result = program.simulate()

    expected = [
        MeasurementResult(0, True, b),
        MeasurementResult(1, True, a)
    ]
    assert result == expected, (
        f"SWAP in X basis failed for preparation F|{a}> and F|{b}>: "
        f"expected {expected}, got {result}"
    )

def test_qubit_deutsch():
    # Create circuit
    circuit = Circuit(2, 2)

    # Put qudits into |+> and |-> states
    circuit.add_gate("H", 0)
    circuit.add_gate("X", 1)
    circuit.add_gate("H", 1)

    secret_function = [random.randint(0, 1) for _ in range(2)]

    if secret_function[0]:  # pragma: no cover
        circuit.add_gate("CNOT", 0, 1)
        circuit.add_gate("X", 1)

    if secret_function[1]:  # pragma: no cover
        circuit.add_gate("CNOT", 0, 1)

    circuit.add_gate("H", 0)

    circuit.add_gate("M", 0)

    program = Program(circuit)

    if secret_function[0] == secret_function[1]:
        expected_result = 0
    else:
        expected_result = 1
    
    assert program.simulate() == [MeasurementResult(0, True, expected_result)]

def test_z_stabilizer_extraction():
    """
    Five qutrit circuit that measures two operators initialized to state $|111\rangle$

    $Z_2 \otimes Z_3 \otimes Z_4^\dag$ onto qutrit 0
    
    $Z_2^\dag \otimes Z_3^\dag \otimes Z_4$ onto qutrit 1

    Outcomes are $\omega$ and $\omega^2$ for the operators respectively
    """
    circuit = Circuit(dimension=3, num_qudits=5)
    circuit.add_gate("X", [2, 3, 4])
    circuit.add_gate("H", 0)
    circuit.add_gate("CZ", 0, 2)
    circuit.add_gate("CZ", 0, 3)
    circuit.add_gate("CZ_INV", 0, 4)
    circuit.add_gate("H_INV", 0)
    circuit.add_gate("H", 1)
    circuit.add_gate("CZ_INV", 1, 2)
    circuit.add_gate("CZ_INV", 1, 3)
    circuit.add_gate("CZ", 1, 4)
    circuit.add_gate("H_INV", 1)
    circuit.add_gate("M", 0)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 1), MeasurementResult(1, True, 2)]

def test_x_stabilizer_extraction():
    """
    Five qutrit circuit that measures two operators 
    
    $X_2 \otimes X_3 \otimes X_4^\dag$ onto qutrit 0
    
    $X_2^\dag \otimes X_3^\dag \otimes X_4$ onto qutrit 1
    
    Introduces a phase error on qutrit 2
    """
    circuit = Circuit(dimension=3, num_qudits=5)
    circuit.add_gate("H", [2,3,4])
    circuit.add_gate("Z", 2)
    circuit.add_gate("H", 0)
    circuit.add_gate("CX", 0, 2)
    circuit.add_gate("CX", 0, 3)
    circuit.add_gate("CX_INV", 0, 4)
    circuit.add_gate("H_INV", 0)
    circuit.add_gate("H", 1)
    circuit.add_gate("CX_INV", 1, 2)
    circuit.add_gate("CX_INV", 1, 3)
    circuit.add_gate("CX", 1, 4)
    circuit.add_gate("H_INV", 1)
    circuit.add_gate("M", 0)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    assert program.simulate() == [MeasurementResult(0, True, 2), MeasurementResult(1, True, 1)]

def test_deutsch():
    # input the dimension here
    dimension = 3
    
    # Create circuit
    circuit = Circuit(2, dimension)

    # Put qudits into |+> and |-> states
    circuit.add_gate("H", 0)
    circuit.add_gate("X", 1)
    circuit.add_gate("H", 1)

    #is_constant = random.choice([True, False])
    is_constant = False

    if is_constant:
        # constant function is a random constant between 0 and dimension - 1 inclusive
        function_constant = random.choice([i for i in range(dimension)])
        for i in range(function_constant):
            circuit.add_gate("X", 1)
    else:
        # identity function
        circuit.add_gate("CNOT", 0, 1)

    circuit.add_gate("H_INV", 0)

    circuit.add_gate("M", 0)

    program = Program(circuit)

    if is_constant:
        expected_result = 0
    else:
        expected_result = 2

    result = program.simulate(verbose=False)

    program.stabilizer_tableau.print_tableau()
    
    assert result == [MeasurementResult(0, True, expected_result)]
