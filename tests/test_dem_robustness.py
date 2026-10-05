"""Tests for sdim.dem: NumPy integers in models and circuits, detector expressions that are polynomials in
their records, and the bytecode scan that admits an expression to the symbolic evaluation."""

import dis
import itertools
import os
import random
import re
import subprocess
import sys
import time
import types

import numpy as np
import pytest

import sdim.dem as dem_module
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel, ErrorMechanism
from sdim.program import _compile_detector, _detector_mod
from tests.test_dem_backward import _balanced_sum


def _all_plain_ints(dem):
    """Whether the dimension, the counts and every target and coefficient of the model are Python ints."""
    values = [dem.dimension, dem.num_detectors, dem.num_observables]
    values += [x for m in dem.mechanisms for g in m.generators for x in (*g, *g.values())]
    return all(type(x) is int for x in values)


# --------------------------------------------------------------------------
# NumPy integers


def test_numpy_coefficients_merge_and_split_into_lines():
    """sample() takes NumPy integer targets and coefficients, but merge_lines and to_lines raised TypeError on
    them (NumPy integers have no three-argument pow)."""
    def model(i):
        return DetectorErrorModel(i(3), i(2), i(1), [
            ErrorMechanism(0.1, [{i(0): i(1), i(1): i(2)}], "a"),
            ErrorMechanism(0.2, [{i(0): i(2), i(1): i(1)}], "b"),
            ErrorMechanism(0.3, [{i(0): i(1)}, {i(1): i(1), i(2): i(2)}], "c")])

    for i in (np.int64, np.int32, np.uint8, int):
        dem = model(i)
        assert _all_plain_ints(dem)
        assert dem.to_lines() == model(int).to_lines()
        dem.merge_lines()
        assert _all_plain_ints(dem)
        assert [(m.generators, m.source) for m in dem.mechanisms] == [
            ([{0: 1, 1: 2}], "a+b"), ([{0: 1}, {1: 1, 2: 2}], "c")]


def test_numpy_coefficients_merge_exactly_at_large_dimensions():
    """NumPy coefficients are merged with exact int arithmetic, where int64 products would overflow."""
    d = 2 ** 61 - 1
    dem = DetectorErrorModel(d, 2, 0, [ErrorMechanism(0.1, [{0: np.int64(3), 1: np.uint64(2 ** 63 + 5)}], "a"),
                                       ErrorMechanism(0.2, [{0: 6, 1: (2 ** 64 + 10) % d}], "b")])
    dem.merge_lines()
    assert dem.mechanisms == [ErrorMechanism(dem.mechanisms[0].probability,
                                             [{0: 1, 1: pow(3, -1, d) * (2 ** 63 + 5) % d}], "a+b")]
    assert _all_plain_ints(dem)


@pytest.mark.parametrize("i", [np.int64, np.int32, np.uint8])
def test_numpy_entries_added_after_construction(i):
    """The constructor normalizes the mechanisms it is given, but a mechanism appended later reached merge_lines
    and to_lines as it was: three-argument pow raised TypeError, and narrow NumPy products overflow (16 * 16
    does not fit a uint8)."""
    def model(i):
        dem = DetectorErrorModel(17, 3, 0)
        dem.mechanisms += [ErrorMechanism(0.1, [{i(0): i(16), i(1): i(5)}], "a"),
                           ErrorMechanism(0.2, [{i(0): i(1), i(1): i(12)}], "b"),
                           ErrorMechanism(0.3, [{i(0): i(15), i(2): i(14)}, {i(1): i(13), i(2): i(16)}], "c")]
        return dem

    lines = model(i).to_lines()
    assert lines == model(int).to_lines()
    assert _all_plain_ints(lines)
    dem = model(i)
    dem.merge_lines()
    assert [(m.generators, m.source) for m in dem.mechanisms] == [
        ([{0: 1, 1: 12}], "a+b"), ([{0: 15, 2: 14}, {1: 13, 2: 16}], "c")]
    assert all(type(x) is int for x in (*dem.mechanisms[0].generators[0], *dem.mechanisms[0].generators[0].values()))


def test_numpy_dimension_set_after_construction(monkeypatch):
    """A NumPy dimension set after the constructor overflowed the line count of to_lines (which then started
    listing about d**3 lines) and made merge_lines raise TypeError."""
    def expand(*args):
        raise AssertionError("to_lines expanded the mechanism")

    dem = DetectorErrorModel(3, 2, 0, [ErrorMechanism(0.1, [{0: 1, 1: 2}], "a"),
                                       ErrorMechanism(0.2, [{0: 2, 1: 1}], "b")])
    dem.dimension = np.int64(3)
    dem.merge_lines()
    assert [(m.generators, m.source) for m in dem.mechanisms] == [([{0: 1, 1: 2}], "a+b")]
    monkeypatch.setattr(dem_module, "_lines_from_arrays", expand)
    dem = DetectorErrorModel(3, 2, 0, [ErrorMechanism(0.1, [{0: 1}, {1: 1}, {0: 1, 1: 1}, {1: 2}], "n2")])
    dem.dimension = np.int64(1000003)
    with pytest.raises(ValueError, match="lines for one mechanism"):
        dem.to_lines()


def test_entries_that_are_not_integers_are_left_for_sample_to_reject():
    """Only integers become Python ints: a float is not truncated, and sample() still refuses it."""
    for gen, what in (({0: 1.0}, "coefficient"), ({0.0: 1}, "target"), ({0: "1"}, "coefficient")):
        dem = DetectorErrorModel(3, 1, 0, [ErrorMechanism(1.0, [gen], "x")])
        [kept] = dem.mechanisms[0].generators
        assert [(type(t), type(v)) for t, v in kept.items()] == [(type(t), type(v)) for t, v in gen.items()]
        with pytest.raises(ValueError, match=what):
            dem.sample(4, seed=1)
    dem = DetectorErrorModel(3, 2, 0, [ErrorMechanism(1.0, [{True: True}], "x")])
    assert dem.mechanisms[0].generators == [{1: 1}] and _all_plain_ints(dem)


def test_to_lines_refuses_too_many_lines_for_a_numpy_dimension(monkeypatch):
    """With a NumPy dimension, the line count (d ** k - 1) // (d - 1) overflowed int64 and went negative, so
    to_lines skipped the max_lines_per_mechanism check and started listing about d**3 lines."""
    def expand(*args):
        raise AssertionError("to_lines expanded the mechanism")

    monkeypatch.setattr(dem_module, "_lines_from_arrays", expand)
    dem = DetectorErrorModel(np.int64(1000003), 2, 0,
                             [ErrorMechanism(0.1, [{0: 1}, {1: 1}, {0: 1, 1: 1}, {1: 2}], "n2")])
    assert type(dem.dimension) is int
    with pytest.raises(ValueError, match="lines for one mechanism"):
        dem.to_lines()


def _memory_circuit(d):
    c = Circuit(3, d)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("N1", [0, 1], noise_channel="f", prob=0.01)
    c.add_gate("CNOT", 0, 2)
    c.add_gate("CNOT_INV", 1, 2)
    c.add_gate("N2", 1, 2, prob=0.001)
    c.add_gate("M", [2, 0, 1])
    c.add_gate("DETECTOR", expr="rec[-3]")
    c.add_gate("DETECTOR", expr="rec[-3] - rec[-2] + rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="2*rec[-1] - 3*rec[-2]")
    return c


@pytest.mark.parametrize("dimension", [np.int64(5), np.int32(1000003)])
def test_from_circuit_with_a_numpy_dimension(dimension):
    """The circuit's dimension reached the coefficient arithmetic as it was: as an int32, products of
    coefficients overflowed and a linear detector was reported as not linear."""
    dem = DetectorErrorModel.from_circuit(_memory_circuit(dimension))
    assert _all_plain_ints(dem)
    assert str(dem) == str(DetectorErrorModel.from_circuit(_memory_circuit(int(dimension))))


# --------------------------------------------------------------------------
# Polynomial detector expressions


def _detector_circuit(d, n, expr):
    c = Circuit(n, d)
    c.add_gate("N1", list(range(n)), noise_channel="f", prob=0.5 * (1 - 1 / d))
    c.add_gate("M", list(range(n)))
    c.add_gate("DETECTOR", expr=expr)
    return c


@pytest.mark.parametrize("d, n, expr", [
    (2, 10, "*".join(f"rec[{i}]" for i in range(10))),
    (2, 12, "rec[0] + " + "*".join(f"rec[{i}]" for i in range(12))),
    (3, 10, "*".join(f"rec[{i}]**2" for i in range(10))),
])
def test_products_of_many_records_are_not_linear(d, n, expr):
    """The numeric probes catch a product of a few records, but the AND of 10 records at d = 2 is non-zero on
    only 1 input in 1024: from_circuit returned an empty model, while the frame sampler sees the detector fire."""
    with pytest.raises(ValueError, match=re.escape("detector D0 is not linear in its records")):
        DetectorErrorModel.from_circuit(_detector_circuit(d, n, expr))


@pytest.mark.parametrize("d, expr, same_as", [
    (3, "rec[0]**3", "rec[0]"),
    (5, "2*rec[1]**5 - rec[0]**9", "2*rec[1] - rec[0]"),
    (2, "rec[0]*rec[0] + rec[1]**7", "rec[0] + rec[1]"),
    (2, "(rec[0] + rec[1])**2 - rec[0]**2 - rec[1]**2", "0"),
    (7, "(rec[0] + rec[1])**7 + 7*rec[0]*rec[1]", "rec[0] + rec[1]"),
    (101, "rec[0]**201 - 3*rec[1]*rec[0]**0", "rec[0] - 3*rec[1]"),
    (5, "(rec[0]*rec[1] - rec[1]*rec[0] + rec[1])**1", "rec[1]"),
])
def test_polynomials_that_are_affine_on_z_d(d, expr, same_as):
    """x ** d = x on Z_d, so these are linear detectors."""
    dem = DetectorErrorModel.from_circuit(_detector_circuit(d, 2, expr))
    assert str(dem) == str(DetectorErrorModel.from_circuit(_detector_circuit(d, 2, same_as)))


@pytest.mark.parametrize("d, expr, message", [
    (3, "rec[0]**2", "is not linear"),
    (5, "rec[0]**4 * rec[1]", "is not linear"),
    (3, "(rec[0] + rec[1])**2 - rec[0]**2 - rec[1]**2", "is not linear"),
    (1000003, "rec[0]**1000002", "is not linear"),
    (3, "rec[0]*rec[1] + 1", "has a non-zero constant term"),
    (3, "rec[0]**0", "has a non-zero constant term"),
])
def test_polynomials_that_are_not_affine(d, expr, message):
    """The constant term is checked first, as the numeric path does."""
    with pytest.raises(ValueError, match=re.escape(f"detector D0 {message}")):
        DetectorErrorModel.from_circuit(_detector_circuit(d, 2, expr))


def _random_polynomial(rng, n, depth):
    if depth <= 0 or rng.random() < 0.25:
        return f"rec[{rng.randrange(n)}]" if rng.random() < 0.75 else str(rng.randint(-6, 6))
    a, b = _random_polynomial(rng, n, depth - 1), _random_polynomial(rng, n, depth - 1)
    op = rng.choice(["+", "-", "*", "*", "**", "%", "neg"])
    if op == "neg":
        return f"-({a})"
    if op == "**":
        return f"({a}) ** {rng.choice([0, 1, 2, 3, 4, 5, 7, 8, 13])}"
    if op == "%":
        return f"({a}) % {rng.choice([1, -1, 2]) * rng.choice([2, 3, 5, 7])}"
    return f"({a}) {op} ({b})"


@pytest.mark.parametrize("d", [2, 3, 5])
def test_symbolic_verdicts_match_every_input(d):
    """For prime d the symbolic call decides exactly: its coefficients, or its error, are those of the function
    evaluated on all of Z_d^n."""
    rng = random.Random(d)
    decided = 0
    for _ in range(400):
        n = rng.randint(1, 3)
        src = _random_polynomial(rng, n, rng.randint(0, 5))
        fn = _compile_detector(src, d)
        points = list(itertools.product(range(d), repeat=n))
        values = {p: fn(list(p)) % d for p in points}
        k = values[(0,) * n]
        coeffs = [(values[tuple(int(i == j) for i in range(n))] - k) % d for j in range(n)]
        affine = all(values[p] == (k + sum(c * x for c, x in zip(coeffs, p))) % d for p in points)
        try:
            form = dem_module._symbolic_coefficients(fn, n, d, {}, "D")
        except ValueError as e:
            assert not affine, src
            assert str(e) == ("D has a non-zero constant term" if k else "D is not linear in its records"), src
            decided += 1
            continue
        if form is not None:
            assert affine and form == (k, *coeffs), src
            decided += 1
        else:
            # Outside the symbolic fragment: % by a number that is not a multiple of d.
            assert any(int(m) % d for m in re.findall(r"% (-?\d+)", src)), src
    assert decided > 250


@pytest.mark.parametrize("src", ["(4) - (5) + rec[0]", "(-1) ** 5 * rec[0]", "-((1) * (0)) * rec[1] + rec[0]"])
def test_folded_constants_stay_symbolic(src):
    """Python 3.15 loads a folded -1 with LOAD_COMMON_CONSTANT, which the bytecode scan refused."""
    assert dem_module._is_straight_line_arithmetic(_compile_detector(src, 3))


def test_composite_dimensions_keep_the_numeric_path_for_higher_degrees():
    """2 x**2 = 2 x as functions on Z_4, though not as polynomials: the probes decide, as before."""
    c = _detector_circuit(4, 2, "2*rec[0]*rec[0] + rec[1]")
    dem = DetectorErrorModel.from_circuit(c, check_dimension_prime=False)
    assert str(dem) == str(DetectorErrorModel.from_circuit(_detector_circuit(4, 2, "2*rec[0] + rec[1]"),
                                                           check_dimension_prime=False))
    with pytest.raises(ValueError, match="not linear"):
        DetectorErrorModel.from_circuit(_detector_circuit(4, 2, "rec[0]*rec[1]"), check_dimension_prime=False)


def test_large_products_fall_back_to_the_numeric_path():
    """Expanding a product of long sums, or a high power of one, is bounded; past the bound the probes decide."""
    n = 200
    left = " + ".join(f"rec[{i}]" for i in range(n // 2))
    right = " + ".join(f"rec[{i}]" for i in range(n // 2, n))
    for src in (f"({left}) * ({right})", "(rec[0] + rec[1] + rec[2] + rec[3]) ** 1000"):
        fn = _compile_detector(src, 1000003)
        t = time.perf_counter()
        assert dem_module._symbolic_coefficients(fn, n, 1000003, {}, "D") is None
        assert time.perf_counter() - t < 5
    with pytest.raises(ValueError, match="not linear"):
        DetectorErrorModel.from_circuit(_detector_circuit(1000003, n, f"({left}) * ({right})"))


_HUGE_EXPONENT_SCRIPT = r"""
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
for expr in ("rec[0] ** (10**18 + 1)", "rec[0] ** (10**18)"):
    c = Circuit(1, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.5)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr=expr)
    try:
        print([m.generators for m in DetectorErrorModel.from_circuit(c).mechanisms])
    except ValueError as e:
        print(e)
"""


def test_huge_exponents_reduce_by_fermat():
    """Exponents used to be evaluated on integer probes: 2 ** (10**18 + 1) never finished. (In a subprocess,
    so the old code fails on the timeout.)"""
    import sdim
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(sdim.__file__)), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", _HUGE_EXPONENT_SCRIPT], env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["[[{0: 1}]]", "detector D0 is not linear in its records"]


# --------------------------------------------------------------------------
# Bytecode scan


_ALLOWED = {"RESUME", "NOP", "CACHE", "EXTENDED_ARG", "RETURN_VALUE", "LOAD_FAST", "LOAD_FAST_CHECK",
            "LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "LOAD_CONST",
            "LOAD_SMALL_INT", "LOAD_COMMON_CONSTANT", "BINARY_SUBSCR", "UNARY_NEGATIVE", "BINARY_OP"}
_CALLS = {"LOAD_GLOBAL", "PUSH_NULL", "PRECALL", "CALL"}


def _dis_scan(fn):
    """`_is_straight_line_arithmetic` as it was written with dis.get_instructions, with ** allowed."""
    if type(fn) is not types.FunctionType or fn.__defaults__ or fn.__kwdefaults__ or fn.__closure__:
        return False
    code = fn.__code__
    wraps_mod = code.co_names == ("_detector_mod",) and fn.__globals__.get("_detector_mod") is _detector_mod
    if (code.co_argcount != 1 or code.co_kwonlyargcount or (code.co_names and not wraps_mod) or code.co_freevars
            or code.co_cellvars or code.co_flags & (0x04 | 0x08)):
        return False
    for ins in dis.get_instructions(code):
        name = ins.opname
        if wraps_mod and name in _CALLS:
            if name == "LOAD_GLOBAL" and ins.argval != "_detector_mod":
                return False
            continue
        if name not in _ALLOWED:
            return False
        if name == "BINARY_OP" and ins.argrepr not in {"+", "-", "*", "**", "%", "[]"}:
            return False
        if name in ("LOAD_CONST", "LOAD_SMALL_INT", "LOAD_COMMON_CONSTANT") and type(ins.argval) is not int:
            return False
    return True


_ATOMS = ["rec[{j}]", "rec[{j}]", "{c}", "-{c}", "{big}", "None", "1.5", "'a'", "True", "(1, 2)", "abs(rec[{j}])",
          "x", "rec", "len(rec)", "(1, 2)[0]", "rec[{j}:{j}]", "_detector_mod(rec[{j}], 7)", "rec[{j}].real",
          "(y := rec[{j}])", "2**200"]
_OPERATORS = ["+", "-", "*", "**", "%", "+", "-", "*", "//", "/", "&", "^", "<<", "@", "==", "<", "and", "or"]


def _random_expression(rng, n, depth):
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(_ATOMS).format(j=rng.randrange(n), c=rng.randint(0, 600), big=rng.randint(2 ** 62, 2 ** 70))
    a, b = _random_expression(rng, n, depth - 1), _random_expression(rng, n, depth - 1)
    r = rng.random()
    if r < 0.05:
        return f"-({a})"
    if r < 0.08:
        return f"+({a})"
    if r < 0.1:
        return f"(({a}) if ({b}) else 0)"
    return f"({a}) {rng.choice(_OPERATORS)} ({b})"


def test_bytecode_scan_matches_dis():
    """The scan reads co_code directly and accepts exactly the functions that the instruction-by-instruction
    scan with dis.get_instructions accepts, ** now included."""
    rng = random.Random(5)
    accepted = 0
    for k in range(3000):
        n = rng.randint(1, 5)
        src = _random_expression(rng, n, rng.randint(0, 5))
        if k % 50 == 0:
            # Hundreds of distinct constants, which need EXTENDED_ARG.
            n = rng.randint(200, 800)
            terms = [f"{rng.randint(-3000, 3000)}*rec[{rng.randrange(n)}]" for _ in range(n)]
            if k % 100 == 0:
                terms[rng.randrange(n)] = rng.choice(["1.5", "None", "abs(rec[0])", "rec[0] ** 2", "rec[0] // 2"])
            src = _balanced_sum(terms)
        try:
            fn = _compile_detector(src, rng.choice([2, 3, 1000003]))
        except Exception:
            continue
        expected = _dis_scan(fn)
        assert dem_module._is_straight_line_arithmetic(fn) == expected, src
        accepted += expected
    assert accepted > 500


def test_bytecode_scan_is_a_small_part_of_the_symbolic_call():
    """dis.get_instructions made the scan most of the time of reading a long observable."""
    n = 12000
    fn = _compile_detector(_balanced_sum([f"{j % 5 + 1}*rec[{j}]" for j in range(n)]), 1000003)
    scan, total = [], []
    for _ in range(3):
        for f, times in ((lambda: dem_module._is_straight_line_arithmetic(fn), scan),
                         (lambda: dem_module._symbolic_coefficients(fn, n, 1000003, {}), total)):
            # CPU time: under load a call longer than a scheduler time slice also waits for a core.
            t = time.process_time()
            f()
            times.append(time.process_time() - t)
    # About 0.25 now; it was about 0.9.
    assert min(scan) / min(total) < 0.4, (min(scan), min(total))
