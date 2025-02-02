import numpy as np
import cirq
from itertools import product
from sympy import isprime

def generate_tau(d):
    """
    Generates the tau value for dimension d. Given by Beaudrap.
    Args:
        d: The dimension of the tau value
    Returns:
        The tau value for dimension d
    """
    return np.exp(1j * np.pi * (d**2 + 1) / d)

def generate_identity_matrix(d):
    """
    Generates the identity matrix for dimension d.
    Args:
        d: The dimension of the identity matrix
    Returns:
        The identity matrix of dimension d
    """
    return np.eye(d)

def generate_x_matrix(d):
    """
    Generates the generalized Pauli-X matrix for dimension d.
    Args:
        d: The dimension of the X matrix
    Returns:
        The X matrix of dimension d
    """
    X = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        X[i, (i - 1) % d] = 1
    return X

def generate_z_matrix(d):
    """
    Generates the generalized Pauli-Z matrix for dimension d.
    Args:
        d: The dimension of the Z matrix
    Returns:
        The Z matrix of dimension d
    """
    Z = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        Z[i, i] = np.exp(2 * np.pi * 1j * i / d)
    return Z

def generate_h_matrix(d):
    """
    Generates the generalized Hadamard matrix for dimension d.
    Args:
        d: The dimension of the Hadamard matrix
    Returns:
        The Hadamard matrix of dimension d
    """
    H = np.zeros((d, d), dtype=np.complex128)
    if isprime(d):
        for m in range(d):
            for n in range(d):
                H[m, n] = 1 / np.sqrt(d) * np.exp(2 * np.pi * 1j * m * n / d)
    else:
        tau = generate_tau(d)
        for m in range(d):
            for n in range(d):
                H[m, n] = 1 / np.sqrt(d) * tau**(2 * m * n)
    return H

def generate_m_matrix(d, a):
    """
    Generates the multiplicative gate based on some integer a coprime to d and the dimension d.
    Args:
        d: The dimension of the multiplicative gate
        a: The integer coprime to d
    Returns:
        The multiplicative gate of dimension d and integer a
    """
    M = np.zeros((d, d), dtype=np.complex128)
    if np.gcd(a, d) != 1:
        raise ValueError("a and d must be coprime")
    for q in range(d):
        M[a*q % d, q] = 1
    return M

def generate_p_matrix(d):
    """
    Generates the phase shift matrix for dimension d. An important distinction is 
    made between prime and non-prime dimensions. For prime dimensions, the function follows "An Ideal Characterization of the Clifford Operators" (Farinholt).
    For non-prime dimensions, the function follows "A linearized stabilizer formalism for systems of finite dimension" (Beaudrap).
    Args:
        d: The dimension of the phase shift matrix
    Returns:
        The phase shift matrix of dimension d
    """
    P = np.eye(d, dtype=np.complex128)
    if isprime(d):
        omega = np.exp(2j * np.pi / d)
        for j in range(d):
            if d % 2 == 1:  # For odd d
                P[j, j] = omega ** (j * (j - 1) / 2)
            else:  # For even d
                P[j, j] = omega ** (j ** 2 / 2)
    else:
        tau = generate_tau(d)
        for j in range(d):
            P[j, j] = tau ** (j ** 2)
    return P

def generate_cnot_matrix(d):
    """
    Geneates the CNOT (SUM) gate for dimension d. Does not contain the phase correction present in the d even case as present in Farinholt.
    Args:
        d: The dimension of the CNOT gate
    Returns:
        The CNOT gate of dimension d
    """
    CNOT = np.zeros((d**2, d**2),dtype=np.complex128)
    for i, j in product(range(d), repeat=2):
        CNOT[d * i + j, d * i + (i + j) % d] = 1
    CNOT = CNOT.reshape(d**2, d**2)
    CNOT = CNOT.transpose()
    return CNOT


class GeneralizedHadamardGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedHadamardGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return generate_h_matrix(self.d)
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedHadamardGateInverse(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"H_{self.d}"
    
class GeneralizedHadamardGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedHadamardGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return np.conj(generate_h_matrix(self.d)).T
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedHadamardGate(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args): 
        return f"H_{self.d}†"
    

class GeneralizedPhaseShiftGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedPhaseShiftGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return generate_p_matrix(self.d)

    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedPhaseShiftGateInverse(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"P_{self.d}"
    
class GeneralizedPhaseShiftGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedPhaseShiftGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return np.conj(generate_p_matrix(self.d)).T

    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedPhaseShiftGate(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"P_{self.d}†"

class GeneralizedXPauliGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedXPauliGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return generate_x_matrix(self.d)
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedXPauliGateInverse(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"X_{self.d}"
    
class GeneralizedXPauliGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedXPauliGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return np.conj(generate_x_matrix(self.d)).T
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedXPauliGate(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"X_{self.d}†"
    
class GeneralizedZPauliGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedZPauliGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return generate_z_matrix(self.d)
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            inv_gate = GeneralizedZPauliGate(self.d)
            inv_gate._unitary = lambda: np.conj(self._unitary()).T
            return inv_gate
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"Z_{self.d}"
    
class GeneralizedZPauliGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedZPauliGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return np.conj(generate_z_matrix(self.d)).T
    
    def __pow__(self, exponent):
        if exponent == 0:
            return IdentityGate(self.d)
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedZPauliGate(self.d)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"Z_{self.d}†"
    
class GeneralizedCNOTGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedCNOTGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        return generate_cnot_matrix(self.d)
    
    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedCNOTGateInverse(self.d)

    def _circuit_diagram_info_(self, args):
        return (f"CNOT_{self.d}_control", f"CNOT_{self.d}_target")
    
class GeneralizedCNOTGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedCNOTGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        return np.conj(generate_cnot_matrix(self.d)).T
    
    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedCNOTGate(self.d)

    def _circuit_diagram_info_(self, args):
        return (f"CNOT_{self.d}_control†", f"CNOT_{self.d}_target†")
    

class GeneralizedCZGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedCZGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        return (np.kron(generate_identity_matrix(self.d), generate_h_matrix(self.d))) * generate_cnot_matrix(self.d) * (np.kron(generate_identity_matrix(self.d), np.conj(generate_h_matrix(self.d)).T))
    
    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedCZGateInverse(self.d)
        
    def _circuit_diagram_info_(self, args):
        return (f"CZ_{self.d}_control", f"CZ_{self.d}_target")
    

class GeneralizedCZGateInverse(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedCZGateInverse, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        return np.conj((np.kron(generate_identity_matrix(self.d), generate_h_matrix(self.d))) * generate_cnot_matrix(self.d) * (np.kron(generate_identity_matrix(self.d), np.conj(generate_h_matrix(self.d)).T))).T
    
    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return GeneralizedCZGate(self.d)
        
    def _circuit_diagram_info_(self, args):
        return (f"CZ_{self.d}_control†", f"CZ_{self.d}_target†")
    
class IdentityGate(cirq.Gate):
    def __init__(self, d):
        super(IdentityGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return generate_identity_matrix(self.d)
    
    def __pow__(self, _):
        return IdentityGate(self.d)

    def _circuit_diagram_info_(self, args):
        return f"I_{self.d}"