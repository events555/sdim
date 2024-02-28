import numpy as np

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
    if d == 2:
        P[d - 1, d - 1] = np.exp(1j * np.pi / 2)
    else:
        P[d - 1, d - 1] = np.exp(1j * 2 * np.pi / d)
    return P

def generate_cnot_matrix(d):
    CNOT = np.zeros((d**d, d**d))
    for i in range(d):
        for j in range(d):
            CNOT[d * i + j, d * i + (i + j) % d] = 1
    CNOT = CNOT.transpose()
    return CNOT
