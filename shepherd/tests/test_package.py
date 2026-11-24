"""Package hygiene: every module imports, every registry entry resolves, placeholders fail loudly."""

from __future__ import annotations

import importlib
import pkgutil

import numpy as np
import pytest

import shepherd
from shepherd.decoder import Decoder
from shepherd.experiment import Experiment
from shepherd.qecc import QECC, PauliString

PLACEHOLDER_DECODERS = {"perfect", "trellis"}
PLACEHOLDER_EXPERIMENTS = {"lrb", "msd"}
PLACEHOLDER_CODES = {"qrm_self_dual"}


def test_every_module_imports():
    for m in pkgutil.walk_packages(shepherd.__path__, "shepherd."):
        if m.name != "shepherd.__main__":
            importlib.import_module(m.name)


@pytest.mark.parametrize("registry", [QECC, Decoder, Experiment])
def test_every_registry_entry_resolves(registry):
    for name in registry.available():
        assert registry.lookup(name) is registry.registry[name]


def test_top_level_api():
    for name in shepherd.__all__:
        assert getattr(shepherd, name) is not None
    with pytest.raises(AttributeError):
        shepherd.no_such_name


def test_placeholders_raise_not_implemented():
    from shepherd.codes.css import surface_code
    for name in PLACEHOLDER_DECODERS:
        with pytest.raises(NotImplementedError):
            Decoder.create(name).compile_decoder_for_dem(None)
    for name in PLACEHOLDER_EXPERIMENTS:
        with pytest.raises(NotImplementedError):
            Experiment.create(name, 3).build(surface_code(3, 3))
    for name in PLACEHOLDER_CODES:
        code = QECC.create(name, 3)
        for method in (code.stabilizers, code.logical_operators):
            with pytest.raises(NotImplementedError):
                method()
    rotated = QECC.create("rotated_memory_z", 3, distance=3)
    assert rotated.encoding().num_qudits == 17          # 9 data + 8 ancillas
    with pytest.raises(NotImplementedError):
        rotated.stabilizers()


@pytest.mark.parametrize("q", (2, 3, 5, 7, 11))
def test_default_logicals_of_the_five_qudit_code(q):
    code = QECC.create("five_qudit", q)
    xs, zs = code.logical_operators()
    _assert_logical_basis(xs, zs, code.stabilizers(), 1, q)
    assert code.num_logical_qudits == 1


@pytest.mark.parametrize("q", (2, 3, 5))
def test_default_logicals_agree_with_css(q):
    from shepherd import gf_q
    from shepherd.codes.css import lacross_code
    code = lacross_code(10, 4, q)
    xs, zs = QECC.logical_operators(code)                 # the generic default
    ip = PauliString.symplectic_inner_product
    assert len(xs) == code.k
    assert all(ip(x, z) == (i == j) for i, x in enumerate(xs) for j, z in enumerate(zs))
    assert all(ip(a, b) == 0 for a in xs for b in xs) and all(ip(a, b) == 0 for a in zs for b in zs)
    # Same logical space as CSSCode's: every default logical is a CSS logical times stabilizers.
    lx, lz = code.logical_matrices()
    stab = np.vstack([np.hstack([code.H_X, np.zeros_like(code.H_X)]),
                      np.hstack([np.zeros_like(code.H_Z), code.H_Z])])
    css = np.vstack([np.hstack([lx, np.zeros_like(lx)]), np.hstack([np.zeros_like(lz), lz])])
    span = np.vstack([stab, css]) % q
    for L in xs + zs:
        assert gf_q.row_space_contains(span, np.array(L.x_powers + L.z_powers), q)


def _assert_logical_basis(xs, zs, stabs, k, q):
    """xs, zs are k symplectic pairs of logicals: commute with the stabilizers, pair as
    <X_i, Z_j> = delta_ij, commute among themselves, and are independent of the stabilizers."""
    from shepherd import gf_q
    ip = PauliString.symplectic_inner_product
    assert len(xs) == len(zs) == k
    assert all(ip(L, s) == 0 for L in xs + zs for s in stabs)
    assert all(ip(x, z) == (i == j) for i, x in enumerate(xs) for j, z in enumerate(zs))
    assert all(ip(a, b) == 0 for a in xs for b in xs)
    assert all(ip(a, b) == 0 for a in zs for b in zs)
    vec = lambda p: p.x_powers + p.z_powers
    s = np.array([vec(p) for p in stabs])
    full = np.array([vec(p) for p in stabs + xs + zs])
    assert gf_q.rank(full, q) == gf_q.rank(s, q) + 2 * k


class _FiveQuditCopies(QECC):
    """`copies` independent five-qudit codes; with `interleave`, their qudits are shuffled by a
    fixed permutation, so no copy sits on contiguous qudits. Non-CSS, k = copies."""

    def __init__(self, dimension, copies, interleave=False):
        super().__init__(dimension, copies=copies, interleave=interleave)
        n = 5 * copies
        self.order = (np.random.default_rng(7).permutation(n) if interleave else np.arange(n))
        self.copies = copies

    def stabilizers(self):
        q, n = self.dimension, 5 * self.copies
        one = QECC.create("five_qudit", q).stabilizers()
        out = []
        for c in range(self.copies):
            where = self.order[5 * c: 5 * c + 5]
            for s in one:
                x, z = [0] * n, [0] * n
                for local, qd in enumerate(where):
                    x[qd], z[qd] = s.x_powers[local], s.z_powers[local]
                out.append(PauliString(q, 0, x, z))
        return out


@pytest.mark.parametrize("q", (2, 3, 5, 7))
@pytest.mark.parametrize("copies", (2, 3))
@pytest.mark.parametrize("interleave", (False, True))
def test_default_logicals_of_five_qudit_copies(q, copies, interleave):
    # Non-CSS with k > 1: the pairing step has to separate the copies' logicals.
    code = _FiveQuditCopies(q, copies, interleave)
    xs, zs = code.logical_operators()
    _assert_logical_basis(xs, zs, code.stabilizers(), copies, q)
    assert code.num_logical_qudits == copies


def test_default_logicals_check_their_input():
    five = QECC.lookup("five_qudit")

    class Redundant(five):              # an extra generator that is a product of two others
        def stabilizers(self):
            s = super().stabilizers()
            return s + [s[0] * s[1]]

    class NonCommuting(five):           # Z_0 fails to commute with the first generator (X on 0)
        def stabilizers(self):
            return super().stabilizers()[:3] + [PauliString(3, 0, [0] * 5, [1, 0, 0, 0, 0])]

    assert Redundant(3).num_logical_qudits == 1
    with pytest.raises(ValueError, match="do not commute"):
        NonCommuting(3).logical_operators()
