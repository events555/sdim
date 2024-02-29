import numpy as np
from itertools import product

def generate_identity_matrix(d):
    return np.eye(d)

def generate_x_matrix(d):
    X = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        X[i, (i - 1) % d] = 1
    return X

def generate_z_matrix(d):
    Z = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        Z[i, i] = np.exp(2 * np.pi * 1j * i / d)
    return Z

def generate_h_matrix(d):
    H = np.zeros((d, d), dtype=np.complex128)
    for m in range(d):
        for n in range(d):
            H[m, n] = 1 / np.sqrt(d) * np.exp(2 * np.pi * 1j * m * n / d)
    return H

def generate_p_matrix(d):
    P = np.eye(d, dtype=np.complex128)
    omega = np.exp(2j * np.pi / d)
    for j in range(d):
        if d % 2 == 1:  # For odd d
            P[j, j] = omega ** (j * (j - 1) / 2)
        else:  # For even d
            P[j, j] = omega ** (j ** 2 / 2)
    return P

def generate_cnot_matrix(d):
    CNOT = np.zeros((d**2, d**2))
    for i, j in product(range(d), repeat=2):
        CNOT[d * i + j, d * i + (i + j) % d] = 1
    CNOT = CNOT.reshape(d**2, d**2)
    CNOT = CNOT.transpose()
    return CNOT