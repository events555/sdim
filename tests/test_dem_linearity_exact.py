"""Tests for the linearity check of detector expressions in sdim.dem: exhaustive evaluation over small Z_d^n,
the error for plain arithmetic too large to check, probes that try more random inputs and do not depend on the
detector's position, and the cache that decides a repeated expression once."""

import random
import re
import types

import pytest

import sdim.dem as dem_module
import sdim.program
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import _compile_detector
from tests.test_dem_robustness import _random_polynomial


def _circuit(d, n, padding, expr):
    """n flipped and measured qudits, `padding` detectors on the last record, then a detector with expr."""
    c = Circuit(n, d)
    c.add_gate("RESET", list(range(n)))
    c.add_gate("N1", list(range(n)), noise_channel="f", prob=0.1)
    c.add_gate("M", list(range(n)))
    for _ in range(padding):
        c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr=expr)
    return c


def _sum(n):
    return " + ".join(f"rec[{i}]" for i in range(n))


# A product of four sums of three records: Q * Q multiplies out 81 * 81 pairs of terms, past _MAX_PRODUCT_TERMS.
_Q = "*".join(f"(rec[{i}] + rec[{i + 1}] + rec[{i + 2}])" for i in range(0, 12, 3))


@pytest.mark.parametrize("k", [4, 17, 24, 35])
def test_product_of_three_records_with_a_call_is_not_linear(k):
    """`//` keeps the expression off the symbolic path, and the 16 probes seeded by the detector's position missed
    the product of three records (non-zero on 1 input in 8) at these positions: from_circuit returned a model
    that never fires the detector, while the frame sampler fires it in about 0.1**3 of the shots. The 8 inputs
    of Z_2^3 are now all evaluated."""
    with pytest.raises(ValueError, match=f"^detector D{k} is not linear in its records$"):
        DetectorErrorModel.from_circuit(_circuit(2, 3, k, "(rec[-3] * rec[-2] * rec[-1]) // 1"))


@pytest.mark.parametrize("k", [9, 11, 12, 22, 18, 27, 37])
def test_arithmetic_past_the_product_bound_is_evaluated_everywhere_when_small(k):
    """Q * Q - Q is 0 mod 2, so this is the product of three records, written as plain arithmetic whose expansion
    stops at the product bound. The probes accepted it at the first four positions; Z_2^12 is evaluated now."""
    with pytest.raises(ValueError, match=f"^detector D{k} is not linear in its records$"):
        DetectorErrorModel.from_circuit(_circuit(2, 12, k, f"({_Q})*({_Q}) - ({_Q}) + rec[0]*rec[1]*rec[2]"))


@pytest.mark.parametrize("k", [18, 27, 37])
def test_arithmetic_past_the_product_bound_over_many_records_is_too_large_to_check(k):
    """S * S - S is 0 mod 2 for the sum S of 65 records, and S * S has 65 * 65 pairs of terms to multiply out.
    The probes missed the product of three records at these positions; at 2**65 inputs it is now rejected
    outright."""
    s = _sum(65)
    message = ("is too large to check for linearity: one of its products multiplies out more than 4096 pairs of "
               "terms. Use fewer records in each product")
    with pytest.raises(ValueError, match=f"^detector D{k} {message}"):
        DetectorErrorModel.from_circuit(_circuit(2, 65, k, f"({s})*({s}) - ({s}) + rec[0]*rec[1]*rec[2]"))


def test_too_large_reports_a_constant_term_first():
    """A constant term is reported before the size, as on every other path."""
    s = _sum(65)
    with pytest.raises(ValueError, match="^detector D0 has a non-zero constant term$"):
        DetectorErrorModel.from_circuit(_circuit(2, 65, 0, f"({s})*({s}) + 1"))


def test_too_large_is_only_for_plain_arithmetic():
    """A negative power is not plain arithmetic, so a product past the bound before it leaves the expression to
    the probes, as for any other operation the symbolic call refuses. S * S - S is 0 mod 2, so this is rec[1]."""
    s = _sum(65)
    expr = f"({s})*({s}) - ({s}) + rec[1] + 0 * (rec[0] + 1) ** -1"
    assert dem_module._symbolic_coefficients(_compile_detector(expr, 2), 65, 2, {}, "D") is None
    dem = DetectorErrorModel.from_circuit(_circuit(2, 65, 0, expr))
    assert str(dem) == str(DetectorErrorModel.from_circuit(_circuit(2, 65, 0, "rec[1]")))


@pytest.mark.parametrize("d, n", [(2, 60), (3, 60), (3, 2), (1000003, 3)])
def test_linear_expressions_with_calls_still_work(d, n):
    """A linear expression that only the numeric path can read is still accepted, probed or evaluated
    everywhere."""
    dem = DetectorErrorModel.from_circuit(_circuit(d, n, 0, f"({_sum(n)}) // 1"))
    assert str(dem) == str(DetectorErrorModel.from_circuit(_circuit(d, n, 0, _sum(n))))


def test_probed_verdicts_do_not_depend_on_the_position():
    """The probes were seeded with the detector's index, so the same expression got different verdicts at
    different places: this degree-3 term over 50 records was accepted at positions 24, 27, 29, 41 and 57."""
    expr = f"(rec[0]*rec[1]*rec[2] + {_sum(50)}) // 1"
    for k in (0, 1, 24, 27, 29, 41, 57):
        with pytest.raises(ValueError, match=f"^detector D{k} is not linear in its records$"):
            DetectorErrorModel.from_circuit(_circuit(2, 50, k, expr))


@pytest.mark.parametrize("degree", [4, 5])
def test_probes_see_a_product_of_a_few_records(degree):
    """The product of 4 or 5 records is non-zero on 1 input in 16 or 32 at d = 2, which 16 random probes can
    miss: they did for both. With 128 they see it."""
    product = "*".join(f"rec[{i}]" for i in range(degree))
    with pytest.raises(ValueError, match="^detector D0 is not linear in its records$"):
        DetectorErrorModel.from_circuit(_circuit(2, 50, 0, f"({product} + {_sum(50)}) // 1"))


@pytest.mark.parametrize("d", [2, 3, 5, 4, 6])
def test_exhaustive_verdicts_match_the_symbolic_ones(d):
    """Where the symbolic call decides, evaluating on all of Z_d^n gives the same coefficients or error."""
    rng = random.Random(100 + d)
    decided = 0
    for _ in range(300):
        n = rng.randint(1, 3)
        src = _random_polynomial(rng, n, rng.randint(0, 5))
        fn = _compile_detector(src, d)
        exhaustive = dem_module._exhaustive_coefficients(fn, n, d)
        try:
            form = dem_module._symbolic_coefficients(fn, n, d, {}, "D")
        except ValueError as e:
            assert str(e) == f"D {exhaustive}", src
            decided += 1
            continue
        if form is not None:
            assert exhaustive == (form if form[0] == 0 else "has a non-zero constant term"), src
            decided += 1
    assert decided > 150


def test_composite_higher_degrees_are_evaluated_everywhere(monkeypatch):
    """2 x**2 = 2 x on Z_4. Over 2 records that is now decided on all of Z_4^2 instead of on probes, and over
    7 records (4**7 inputs) the probes still accept it."""
    def probe(*args):
        raise AssertionError("probed")

    with monkeypatch.context() as m:
        m.setattr(dem_module, "_probed_coefficients", probe)
        dem = DetectorErrorModel.from_circuit(_circuit(4, 2, 0, "2*rec[0]*rec[0] + rec[1]"),
                                              check_dimension_prime=False)
        with pytest.raises(ValueError, match="detector D0 is not linear"):
            DetectorErrorModel.from_circuit(_circuit(4, 2, 0, "rec[0]*rec[1]"), check_dimension_prime=False)
    assert str(dem) == str(DetectorErrorModel.from_circuit(_circuit(4, 2, 0, "2*rec[0] + rec[1]"),
                                                           check_dimension_prime=False))
    dem = DetectorErrorModel.from_circuit(_circuit(4, 7, 0, f"2*rec[0]*rec[0] + {_sum(7)}"),
                                          check_dimension_prime=False)
    assert str(dem) == str(DetectorErrorModel.from_circuit(_circuit(4, 7, 0, f"2*rec[0] + {_sum(7)}"),
                                                           check_dimension_prime=False))


def test_constant_term_is_reported_first():
    """Evaluated everywhere, a constant term is still the error, as on the probes and the symbolic path."""
    with pytest.raises(ValueError, match="^detector D0 has a non-zero constant term$"):
        DetectorErrorModel.from_circuit(_circuit(3, 2, 0, "(rec[0] * rec[1] + 1) // 1"))
    assert dem_module._exhaustive_coefficients(_compile_detector("(rec[0] * rec[1]) // 1", 3), 2, 3) == (
        "is not linear in its records")


@pytest.mark.parametrize("d, decide", [(3, "_exhaustive_coefficients"), (1000003, "_probed_coefficients")])
def test_repeated_expressions_are_decided_once(d, decide, monkeypatch):
    """Every detector is "(rec[0] + rec[1]) // 1" once sdim renumbers its records, so one verdict serves all."""
    calls = []
    original = getattr(dem_module, decide)

    def counted(*args):
        calls.append(args)
        return original(*args)

    def circuit(expr):
        c = _circuit(d, 4, 0, "rec[-1]")
        for _ in range(30):
            c.add_gate("DETECTOR", expr=expr)
        return c

    monkeypatch.setattr(dem_module, decide, counted)
    dem = DetectorErrorModel.from_circuit(circuit("(rec[-1] + rec[-2]) // 1"))
    assert len(calls) == 1
    assert str(dem) == str(DetectorErrorModel.from_circuit(circuit("rec[-1] + rec[-2]")))


def test_verdicts_are_only_cached_for_functions_their_code_defines():
    """A closure, a default or other globals can change the function behind the same code object."""
    cache = {}
    sdim_globals = vars(sdim.program)

    def power(k):
        return lambda rec: (rec[0] ** k) // 1

    linear, square = (types.FunctionType(f.__code__, sdim_globals, None, None, f.__closure__)
                      for f in (power(1), power(2)))
    assert linear.__code__ == square.__code__
    assert dem_module._numeric_coefficients(linear, 1, 3, cache) == (0, 1)
    assert dem_module._numeric_coefficients(square, 1, 3, cache) == "is not linear in its records"

    def power_with_default(rec, k=2):
        return (rec[0] ** k) // 1

    with_default = types.FunctionType(power_with_default.__code__, sdim_globals, None, (2,))
    assert dem_module._numeric_coefficients(with_default, 1, 3, cache) == "is not linear in its records"
    with_default.__defaults__ = (1,)
    assert dem_module._numeric_coefficients(with_default, 1, 3, cache) == (0, 1)

    fn = _compile_detector("(rec[0] ** 2) // 1", 3)
    other = types.FunctionType(fn.__code__, dict(sdim_globals, _detector_mod=lambda value, d: 0))
    assert dem_module._numeric_coefficients(other, 1, 3, cache) == (0, 0)
    assert dem_module._numeric_coefficients(fn, 1, 3, cache) == "is not linear in its records"
    assert list(cache) == [(fn.__code__, 1)]
    assert dem_module._numeric_coefficients(fn, 1, 3, cache) == "is not linear in its records"


def test_too_large_needs_a_name_to_raise():
    """Without a name the symbolic call leaves the verdict to its caller, as for its other errors."""
    s = _sum(65)
    fn = _compile_detector(f"({s})*({s})", 2)
    assert dem_module._symbolic_coefficients(fn, 65, 2, {}) is None
    with pytest.raises(ValueError, match=re.escape("D is too large to check for linearity")):
        dem_module._symbolic_coefficients(fn, 65, 2, {}, "D")
