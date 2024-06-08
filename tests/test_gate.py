import numpy as np
import pytest
from sdim.unitary import generate_cnot_matrix, generate_h_matrix, generate_identity_matrix, generate_p_matrix, generate_x_matrix, generate_z_matrix

# Expected matrices for dimension 2
I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
P2 = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
CNOT2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)

# Expected matrices for dimension 3
I3 = np.eye(3, dtype=np.complex128)
X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.complex128)
Z3 = np.array([[1, 0, 0], [0, np.exp(2j * np.pi / 3), 0], [0, 0, np.exp(4j * np.pi / 3)]], dtype=np.complex128)
H3 = np.array([[1, 1, 1], [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)], [1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]], dtype=np.complex128) / np.sqrt(3)
P3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.exp(2j * np.pi / 3)]], dtype=np.complex128)
CNOT3 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 1, 0, 0]
], dtype=np.complex128)

@pytest.mark.parametrize("dimension, expected_matrix, generator", [
    (2, I2, generate_identity_matrix),
    (2, X2, generate_x_matrix),
    (2, Z2, generate_z_matrix),
    (2, H2, generate_h_matrix),
    (2, P2, generate_p_matrix),
    (2, CNOT2, generate_cnot_matrix),
    (3, I3, generate_identity_matrix),
    (3, X3, generate_x_matrix),
    (3, Z3, generate_z_matrix),
    (3, H3, generate_h_matrix),
    (3, P3, generate_p_matrix),
    (3, CNOT3, generate_cnot_matrix),
])
def test_matrix_generation(dimension, expected_matrix, generator):
    generated_matrix = generator(dimension)
    np.testing.assert_array_almost_equal(generated_matrix, expected_matrix)

def test_qutrit_gate_products():
    R = generate_h_matrix(3)
    P = generate_p_matrix(3)
    X = generate_x_matrix(3)
    Z = generate_z_matrix(3)
    np.testing.assert_array_almost_equal(X, (R @ P @ R @ R @ P @ P @ R))
    np.testing.assert_array_almost_equal(Z, (R @ R @ P @ R @ R @ P @ P))

def test_qubit_gate_products():
    R = generate_h_matrix(2)
    P = generate_p_matrix(2)
    X = generate_x_matrix(2)
    Z = generate_z_matrix(2)
    np.testing.assert_array_almost_equal(X, (R @ P @ P @ R))
    np.testing.assert_array_almost_equal(Z, (P @ P))
