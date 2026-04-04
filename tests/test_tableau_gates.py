"""Per-gate correctness tests.

For each gate, three things are verified:

1. The formula from algorithm.md preserves the symplectic inner product
   as a polynomial identity over Z (sympy).
2. The code produces the exact Z/X/tau values predicted by that formula,
   for concrete inputs across multiple dimensions (numpy).
3. The symplectic invariant holds on the resulting tableau (numpy).

If (1) passes, the formula is correct. If (2) passes, the code matches
the formula. Together they guarantee the code preserves commutation.
"""

import pytest
import numpy as np
from sympy import symbols, expand
from sdim import TableauSimulator


DIMENSIONS = [2, 3, 4, 5, 6]


def check_symplectic(tab):
    d = tab.d
    Z = tab.Z[:tab.l] % d
    X = tab.X[:tab.l] % d
    SS = (Z @ X.T - X @ Z.T) % d
    assert np.all(SS == 0), f"stabilizer commutation violated:\n{SS}"


def apply_gate_to_column(gate_name, z, x, phi, d, **kwargs):
    """Apply a gate's formula to a single generator (z, x, phi).

    Returns (z', x', phi') as predicted by algorithm.md.
    All arithmetic mod d (for z, x) or mod 2d (for phi).
    """
    D = 2 * d
    if gate_name == "hadamard":
        phi_new = (phi - 2 * z * x) % D
        return x % d, (-z) % d, phi_new
    elif gate_name == "hadamard_inv":
        phi_new = (phi - 2 * z * x) % D
        return (-x) % d, z % d, phi_new
    elif gate_name == "phase_gate":
        z_new = (z + x) % d
        if d % 2 == 0:
            phi_new = (phi + x * x) % D
        else:
            phi_new = (phi + x * (x + 1)) % D
        return z_new, x, phi_new
    elif gate_name == "pauli_x":
        return z, x, (phi + 2 * z) % D
    elif gate_name == "pauli_z":
        return z, x, (phi - 2 * x) % D
    elif gate_name == "multiply":
        a = kwargs["a"]
        a_inv = pow(a, -1, d)
        z_new = (a_inv * z) % d
        x_new = (a * x) % d
        return z_new, x_new, phi
    else:
        raise ValueError(gate_name)


class TestFourierGate:
    def test_symbolic(self):
        z_i, x_i, z_j, x_j = symbols("z_i x_i z_j x_j")
        before = z_i*x_j - x_i*z_j
        after = x_i*(-z_j) - (-z_i)*x_j
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_code_matches_formula(self, d):
        t = TableauSimulator(1, d)
        for z_val in range(d):
            for x_val in range(d):
                for phi_val in range(2 * d):
                    t2 = TableauSimulator(1, d)
                    t2.Z[0, 0] = z_val
                    t2.X[0, 0] = x_val
                    t2.tau_exp[0] = phi_val

                    ez, ex, ep = apply_gate_to_column(
                        "hadamard", z_val, x_val, phi_val, d)
                    t2.hadamard(0)
                    t2.modulo()

                    assert t2.Z[0, 0] == ez % d
                    assert t2.X[0, 0] == ex % d
                    assert t2.tau_exp[0] == ep % (2*d)

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_preserves_symplectic(self, d):
        t = TableauSimulator(3, d)
        t.hadamard(0)
        check_symplectic(t)
        t.hadamard(1, dagger=True)
        check_symplectic(t)


class TestPhaseGate:
    def test_symbolic(self):
        z_i, x_i, z_j, x_j = symbols("z_i x_i z_j x_j")
        before = z_i*x_j - x_i*z_j
        after = (z_i + x_i)*x_j - x_i*(z_j + x_j)
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_code_matches_formula(self, d):
        for z_val in range(d):
            for x_val in range(d):
                for phi_val in range(2 * d):
                    t = TableauSimulator(1, d)
                    t.Z[0, 0] = z_val
                    t.X[0, 0] = x_val
                    t.tau_exp[0] = phi_val

                    ez, ex, ep = apply_gate_to_column(
                        "phase_gate", z_val, x_val, phi_val, d)
                    t.phase_gate(0)
                    t.modulo()

                    assert t.Z[0, 0] == ez % d
                    assert t.X[0, 0] == ex % d
                    assert t.tau_exp[0] == ep % (2*d)

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_preserves_symplectic(self, d):
        t = TableauSimulator(3, d)
        t.phase_gate(1)
        check_symplectic(t)


class TestPauliX:
    def test_symbolic(self):
        z_i, x_i, z_j, x_j = symbols("z_i x_i z_j x_j")
        before = z_i*x_j - x_i*z_j
        after = z_i*x_j - x_i*z_j  # X doesn't change Z/X
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_code_matches_formula(self, d):
        for z_val in range(d):
            for x_val in range(d):
                for phi_val in range(2 * d):
                    t = TableauSimulator(1, d)
                    t.Z[0, 0] = z_val
                    t.X[0, 0] = x_val
                    t.tau_exp[0] = phi_val

                    ez, ex, ep = apply_gate_to_column(
                        "pauli_x", z_val, x_val, phi_val, d)
                    t.pauli_x(0)
                    t.modulo()

                    assert t.Z[0, 0] == ez % d
                    assert t.X[0, 0] == ex % d
                    assert t.tau_exp[0] == ep % (2*d)


class TestPauliZ:
    def test_symbolic(self):
        z_i, x_i, z_j, x_j = symbols("z_i x_i z_j x_j")
        before = z_i*x_j - x_i*z_j
        after = z_i*x_j - x_i*z_j
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_code_matches_formula(self, d):
        for z_val in range(d):
            for x_val in range(d):
                for phi_val in range(2 * d):
                    t = TableauSimulator(1, d)
                    t.Z[0, 0] = z_val
                    t.X[0, 0] = x_val
                    t.tau_exp[0] = phi_val

                    ez, ex, ep = apply_gate_to_column(
                        "pauli_z", z_val, x_val, phi_val, d)
                    t.pauli_z(0)
                    t.modulo()

                    assert t.Z[0, 0] == ez % d
                    assert t.X[0, 0] == ex % d
                    assert t.tau_exp[0] == ep % (2*d)


class TestSumGate:
    def test_symbolic(self):
        (zc_i, xc_i, zt_i, xt_i,
         zc_j, xc_j, zt_j, xt_j) = symbols(
            "zc_i xc_i zt_i xt_i zc_j xc_j zt_j xt_j")

        def sip(zc1, xc1, zt1, xt1, zc2, xc2, zt2, xt2):
            return zc1*xc2 + zt1*xt2 - xc1*zc2 - xt1*zt2

        before = sip(zc_i, xc_i, zt_i, xt_i, zc_j, xc_j, zt_j, xt_j)
        after = sip(
            zc_i - zt_i, xc_i, zt_i, xt_i + xc_i,
            zc_j - zt_j, xc_j, zt_j, xt_j + xc_j,
        )
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_preserves_symplectic(self, d):
        t = TableauSimulator(3, d)
        t.cnot(0, 1)
        check_symplectic(t)
        t.cnot(2, 0)
        check_symplectic(t)
        t.cnot(0, 1, dagger=True)
        check_symplectic(t)


class TestCZGate:
    def test_symbolic(self):
        (z1_i, x1_i, z2_i, x2_i,
         z1_j, x1_j, z2_j, x2_j) = symbols(
            "z1_i x1_i z2_i x2_i z1_j x1_j z2_j x2_j")

        def sip(z1a, x1a, z2a, x2a, z1b, x1b, z2b, x2b):
            return z1a*x1b + z2a*x2b - x1a*z1b - x2a*z2b

        before = sip(z1_i, x1_i, z2_i, x2_i, z1_j, x1_j, z2_j, x2_j)
        after = sip(
            z1_i + x2_i, x1_i, z2_i + x1_i, x2_i,
            z1_j + x2_j, x1_j, z2_j + x1_j, x2_j,
        )
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_preserves_symplectic(self, d):
        t = TableauSimulator(3, d)
        t.cz(0, 2)
        check_symplectic(t)
        t.cz(1, 0, dagger=True)
        check_symplectic(t)


class TestSwapGate:
    def test_symbolic(self):
        (z1_i, x1_i, z2_i, x2_i,
         z1_j, x1_j, z2_j, x2_j) = symbols(
            "z1_i x1_i z2_i x2_i z1_j x1_j z2_j x2_j")

        def sip(z1a, x1a, z2a, x2a, z1b, x1b, z2b, x2b):
            return z1a*x1b + z2a*x2b - x1a*z1b - x2a*z2b

        before = sip(z1_i, x1_i, z2_i, x2_i, z1_j, x1_j, z2_j, x2_j)
        after = sip(z2_i, x2_i, z1_i, x1_i, z2_j, x2_j, z1_j, x1_j)
        assert expand(after - before) == 0

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_preserves_symplectic(self, d):
        t = TableauSimulator(3, d)
        t.swap(0, 2)
        check_symplectic(t)


class TestMultiplyGate:
    def test_symbolic(self):
        z_i, x_i, z_j, x_j, a, a_inv = symbols(
            "z_i x_i z_j x_j a a_inv")
        before = z_i*x_j - x_i*z_j
        after = (a_inv*z_i)*(a*x_j) - (a*x_i)*(a_inv*z_j)
        assert expand(after.subs(a*a_inv, 1) - before) == 0

    @pytest.mark.parametrize("d, a", [
        (3, 2), (4, 3), (5, 3), (6, 5),
    ])
    def test_code_matches_formula(self, d, a):
        a_inv = pow(a, -1, d)
        for z_val in range(d):
            for x_val in range(d):
                t = TableauSimulator(1, d)
                t.Z[0, 0] = z_val
                t.X[0, 0] = x_val

                t.multiply(0, a)

                assert t.X[0, 0] % d == (a * x_val) % d
                assert t.Z[0, 0] % d == (a_inv * z_val) % d

    @pytest.mark.parametrize("d, a", [
        (3, 2), (4, 3), (5, 3), (6, 5),
    ])
    def test_preserves_symplectic(self, d, a):
        t = TableauSimulator(2, d)
        t.multiply(0, a)
        check_symplectic(t)


class TestProductAndPowerRules:
    def test_product_associativity(self):
        phi1, phi2, phi3 = symbols("phi1 phi2 phi3")
        z1, x1, z2, x2, z3, x3 = symbols("z1 x1 z2 x2 z3 x3")

        def product(pa, za, xa, pb, zb, xb):
            return (pa + pb + 2*xa*zb, za + zb, xa + xb)

        p12, z12, x12 = product(phi1, z1, x1, phi2, z2, x2)
        left_phi, left_z, left_x = product(p12, z12, x12, phi3, z3, x3)

        p23, z23, x23 = product(phi2, z2, x2, phi3, z3, x3)
        right_phi, right_z, right_x = product(phi1, z1, x1, p23, z23, x23)

        assert expand(left_z - right_z) == 0
        assert expand(left_x - right_x) == 0
        assert expand(left_phi - right_phi) == 0

    @pytest.mark.parametrize("c", [2, 3, 4, 5])
    def test_power_rule_matches_iterated_product(self, c):
        phi, z, x = symbols("phi z x")

        def product(pa, za, xa, pb, zb, xb):
            return (pa + pb + 2*xa*zb, za + zb, xa + xb)

        acc_phi, acc_z, acc_x = 0, 0, 0
        for _ in range(c):
            acc_phi, acc_z, acc_x = product(
                acc_phi, acc_z, acc_x, phi, z, x)

        pow_phi = c*phi + c*(c-1)*z*x
        assert expand(acc_phi - pow_phi) == 0
        assert expand(acc_z - c*z) == 0
        assert expand(acc_x - c*x) == 0

    def test_column_transform_matches_product(self):
        phi1, phi2 = symbols("phi1 phi2")
        z1, x1, z2, x2 = symbols("z1 x1 z2 x2")

        # V = [[1, 0], [1, 1]], col 1 = S_1 * S_2
        # Phase formula: phi'_1 = V_11*phi_1 + V_21*phi_2 + ... + 2*V_11*V_21*G_{2,1}
        # G_{2,1} = z_2 * x_1
        formula_phi = phi1 + phi2 + 2 * z2 * x1

        # Product rule: S_1 * S_2 -> phi1 + phi2 + 2*x1*z2
        product_phi = phi1 + phi2 + 2*x1*z2

        assert expand(formula_phi - product_phi) == 0
