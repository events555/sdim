"""Generalized qudit unitary matrices."""

from itertools import product

import numpy as np


def generate_tau(d: int) -> complex:
    return np.exp(1j * np.pi * (d**2 + 1) / d)


def generate_identity_matrix(d: int) -> np.ndarray:
    return np.eye(d)


def generate_x_matrix(d: int) -> np.ndarray:
    X = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        X[i, (i - 1) % d] = 1
    return X


def generate_z_matrix(d: int) -> np.ndarray:
    Z = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        Z[i, i] = np.exp(2 * np.pi * 1j * i / d)
    return Z


def generate_h_matrix(d: int) -> np.ndarray:
    omega = np.exp(2j * np.pi / d)
    H = np.zeros((d, d), dtype=np.complex128)
    for m in range(d):
        for n in range(d):
            H[m, n] = omega ** (m * n) / np.sqrt(d)
    return H


def generate_m_matrix(d: int, a: int) -> np.ndarray:
    M = np.zeros((d, d), dtype=np.complex128)
    if np.gcd(a, d) != 1:
        raise ValueError("a and d must be coprime")
    for q in range(d):
        M[a * q % d, q] = 1
    return M


def generate_p_matrix(d: int) -> np.ndarray:
    P = np.eye(d, dtype=np.complex128)
    tau = generate_tau(d)
    omega = tau * tau
    if d % 2 == 0:
        for j in range(d):
            P[j, j] = tau ** (j * j)
    else:
        for j in range(d):
            P[j, j] = omega ** (j * (j - 1) // 2)
    return P


def generate_cnot_matrix(d: int) -> np.ndarray:
    CNOT = np.zeros((d**2, d**2), dtype=np.complex128)
    for i, j in product(range(d), repeat=2):
        CNOT[d * i + j, d * i + (i + j) % d] = 1
    CNOT = CNOT.reshape(d**2, d**2)
    CNOT = CNOT.transpose()
    return CNOT


def generate_multiply_matrix(d: int, a: int) -> np.ndarray:
    if np.gcd(a, d) != 1:
        raise ValueError("a and d must be coprime")
    M = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        M[i, (a * i) % d] = 1
    return M
