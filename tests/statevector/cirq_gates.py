import cirq
import sdim.unitary as unitary
class GeneralizedHadamardGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedHadamardGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return unitary.generate_h_matrix(self.d)

    def _circuit_diagram_info_(self, args):
        return f"H_{self.d}"


class GeneralizedPhaseShiftGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedPhaseShiftGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return unitary.generate_p_matrix(self.d)

    def _circuit_diagram_info_(self, args):
        return f"P_{self.d}"

class GeneralizedCNOTGate(cirq.Gate):
    def __init__(self, d):
        super(GeneralizedCNOTGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        return unitary.generate_cnot_matrix(self.d)

    def _circuit_diagram_info_(self, args):
        return ("CNOT_{self.d}_control", "CNOT_{self.d}_target")
    
class IdentityGate(cirq.Gate):
    def __init__(self, d):
        super(IdentityGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        return unitary.generate_identity_matrix(self.d)

    def _circuit_diagram_info_(self, args):
        return f"I_{self.d}"
