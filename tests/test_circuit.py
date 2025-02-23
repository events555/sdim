import pytest
import cirq as cirq
import random
from sdim.circuit import Circuit
from sdim.circuit_io import cirq_statevector_from_circuit
import numpy as np

def test_phase_kickback():
    circuit = Circuit(2, 2)
    circuit.append("H", 1)
    circuit.append("P", 1)

    circuit.append("H", 0)
    circuit.append("CNOT", 0, 1)
    circuit.append("H", 1)
    circuit.append("CNOT", 0, 1)
    circuit.append("H", 1)

    circuit.append("P", 0)
    circuit.append("H", 0)
    circuit.append("M", 0)
    circuit.append("P", 1)
    circuit.append("H", 1)
    circuit.append("M", 1)
    sampler = circuit.compile_sampler()
    assert np.array_equal(sampler.sample(shots=1), np.array([1, 1]))

def test_qubit_flip():
    circuit = Circuit(2, 2)
    circuit.append("H", 0)
    circuit.append("P", 0)
    circuit.append("P", 0)
    circuit.append("H", 0)
    circuit.append("M", 0)
    circuit.append("X", 1)
    circuit.append("M", 1)
    sampler = circuit.compile_sampler()
    assert np.array_equal(sampler.sample(shots=1), np.array([1, 1]))

def test_qutrit_flip():
    circuit = Circuit(2, 3)
    circuit.append("X", 0)
    circuit.append("M", 0)
    sampler = circuit.compile_sampler()
    assert np.array_equal(sampler.sample(shots=1), np.array([1]))

@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_qudit_swap_computational_basis(dimension):
    circuit = Circuit(2, dimension)

    x0 = random.choice([i for i in range(dimension)])
    x1 = random.choice([i for i in range(dimension)])

    for _ in range(x0):
        circuit.append("X", 0)
    for _ in range(x1):
        circuit.append("X", 1)
    
    circuit.append("SWAP", 0, 1)
    circuit.append("M", 0)
    circuit.append("M", 1)
    sampler = circuit.compile_sampler()
    assert np.array_equal(sampler.sample(shots=1), np.array([x1, x0]))

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
        circuit.append("X", 0)
    for _ in range(x1):
        circuit.append("X", 1)
    
    # Apply SWAP twice.
    circuit.append("SWAP", 0, 1)
    circuit.append("SWAP", 0, 1)
    
    # Measure both qudits.
    circuit.append("M", 0)
    circuit.append("M", 1)
    
    sampler = circuit.compile_sampler()
    expected = np.array([x0, x1])
    assert np.array_equal(sampler.sample(shots=1), expected), (
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
        circuit.append("X", 0)
    circuit.append("DFT", 0)

    for _ in range(b):
        circuit.append("X", 1)
    circuit.append("DFT", 1)

    circuit.append("SWAP", 0, 1)

    circuit.append("MX", 0)
    circuit.append("MX", 1)

    sampler = circuit.compile_sampler()
    result = sampler.sample(shots=1)

    expected = np.array([b, a])
    assert np.array_equal(result, expected), (
        f"SWAP in X basis failed for preparation F|{a}> and F|{b}>: "
        f"expected {expected}, got {result}"
    )

def test_qubit_deutsch():
    # Create circuit
    circuit = Circuit(2, 2)

    # Put qudits into |+> and |-> states
    circuit.append("H", 0)
    circuit.append("X", 1)
    circuit.append("H", 1)

    secret_function = [random.randint(0, 1) for _ in range(2)]

    if secret_function[0]:  # pragma: no cover
        circuit.append("CNOT", 0, 1)
        circuit.append("X", 1)

    if secret_function[1]:  # pragma: no cover
        circuit.append("CNOT", 0, 1)

    circuit.append("H", 0)

    circuit.append("M", 0)

    sampler = circuit.compile_sampler()

    if secret_function[0] == secret_function[1]:
        expected_result = 0
    else:
        expected_result = 1
    
    assert np.array_equal(sampler.sample(shots=1), np.array([expected_result]))

def test_z_stabilizer_extraction():
    """
    Five qutrit circuit that measures two operators initialized to state $|111\rangle$

    $Z_2 \otimes Z_3 \otimes Z_4^\dag$ onto qutrit 0
    
    $Z_2^\dag \otimes Z_3^\dag \otimes Z_4$ onto qutrit 1

    Outcomes are $\omega$ and $\omega^2$ for the operators respectively
    """
    circuit = Circuit(dimension=3, num_qudits=5)
    circuit.append("X", [2, 3, 4])
    circuit.append("H", 0)
    circuit.append("CZ", 0, 2)
    circuit.append("CZ", 0, 3)
    circuit.append("CZ_INV", 0, 4)
    circuit.append("H_INV", 0)
    circuit.append("H", 1)
    circuit.append("CZ_INV", 1, 2)
    circuit.append("CZ_INV", 1, 3)
    circuit.append("CZ", 1, 4)
    circuit.append("H_INV", 1)
    circuit.append("M", 0)
    circuit.append("M", 1)
    sampler = circuit.compile_sampler()
    assert np.array_equal(sampler.sample(shots=1), np.array([1, 2]))

def test_x_stabilizer_extraction():
    """
    Five qutrit circuit that measures two operators 
    
    $X_2 \otimes X_3 \otimes X_4^\dag$ onto qutrit 0
    
    $X_2^\dag \otimes X_3^\dag \otimes X_4$ onto qutrit 1
    
    Introduces a phase error on qutrit 2
    """
    circuit = Circuit(dimension=3, num_qudits=5)
    circuit.append("H", [2,3,4])
    circuit.append("Z", 2)
    circuit.append("H", 0)
    circuit.append("CX", 0, 2)
    circuit.append("CX", 0, 3)
    circuit.append("CX_INV", 0, 4)
    circuit.append("H_INV", 0)
    circuit.append("H", 1)
    circuit.append("CX_INV", 1, 2)
    circuit.append("CX_INV", 1, 3)
    circuit.append("CX", 1, 4)
    circuit.append("H_INV", 1)
    circuit.append("M", 0)
    circuit.append("M", 1)
    sampler = circuit.compile_sampler()

    assert np.array_equal(sampler.sample(shots=1), np.array([2, 1]))

def test_deutsch():
    # input the dimension here
    dimension = 3
    
    # Create circuit
    circuit = Circuit(2, dimension)

    # Put qudits into |+> and |-> states
    circuit.append("H", 0)
    circuit.append("X", 1)
    circuit.append("H", 1)

    #is_constant = random.choice([True, False])
    is_constant = False

    if is_constant:
        # constant function is a random constant between 0 and dimension - 1 inclusive
        function_constant = random.choice([i for i in range(dimension)])
        for i in range(function_constant):
            circuit.append("X", 1)
    else:
        # identity function
        circuit.append("CNOT", 0, 1)

    circuit.append("H_INV", 0)

    circuit.append("M", 0)

    sampler = circuit.compile_sampler()

    if is_constant:
        expected_result = 0
    else:
        expected_result = 2

    assert np.array_equal(sampler.sample(shots=1), np.array([expected_result]))
    