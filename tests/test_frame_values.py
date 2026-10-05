"""Exact values from the Pauli frame sampler where int64 arithmetic used to overflow.

Ring expressions (integer literals, rec[j], +, -, unary + and -, *, ** by a non-negative literal and %
by a literal that d divides) must give their exact value mod d on the measurement shifts, for dimensions
near 2**31, literals of 2**63 or more, and products and powers of records alike.  Every other expression
must give the same values as before, from NumPy int64 arithmetic.  Measurement outcomes must be
(reference + shift) mod d, under NumPy 2 and under NumPy 1.x's value-based casting alike.
"""

import itertools
import random
import time
import warnings

import numpy as np
import pytest

import sdim.program as program
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import DetectorData, Program, _compile_detector, _evaluate_detectors
from sdim.tableau.dataclasses import MeasurementResult

D = 2 ** 31 - 1
DIMENSIONS = [2, 3, 5, 6, 2147483629, 2147483646, 2147483647]


def _shifts(measurements, d):
    """The shift of every frame shot from the reference shot, one list per qudit's first round."""
    rows = []
    for qudit in measurements:
        values = [r.measurement_value for r in qudit[0]]
        rows.append([(v - values[0]) % d for v in values[1:]])
    return rows


def _exact(expr, rows, d):
    """The expression on Python integers, shot by shot, then mod d."""
    function = eval("lambda rec : " + expr)
    return [function([row[s] for row in rows]) % d for s in range(len(rows[0]))]


def _records(d, n, shots, seed):
    """Random int32 record rows in [0, d), starting with shots of all zeros, all d - 1 and all ones."""
    records = np.random.default_rng(seed).integers(0, d, size=(n, shots)).astype(np.int32)
    records[:, 0] = 0
    records[:, 1] = d - 1
    records[:, 2] = 1 % d
    return records


def _evaluate(records, function):
    """The frame sampler's values of one detector, given its function, on the record rows."""
    n, shots = records.shape
    info = DetectorData(detector_data=[(0, "", list(range(n)), False)], detector_functions=[function],
                        num_detector_events=1)
    return _evaluate_detectors(np.array([19]), records, np.arange(n), info, shots).detection_events[0]


@pytest.mark.parametrize("d", [D, D - 1])
def test_equal_detectors_agree_near_the_largest_dimension(d):
    """(d - 1) * rec[j] summed over 3 records wrapped int64 in about 17% of shots, so two detectors
    that are equal mod d came out different."""
    c = Circuit(3, d)
    c.add_gate("N1", [0, 1, 2], noise_channel="f", prob=1.0)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr=f"{d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2]")
    c.add_gate("DETECTOR", expr="-rec[0] - rec[1] - rec[2]")
    c.add_gate("LOGICAL_OBSERVABLE", expr=f"{d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2]")
    np.random.seed(0)
    measurements, (detectors, observables) = Program(c).simulate(shots=1001, raw_detector_output=True)
    expected = _exact("-rec[0] - rec[1] - rec[2]", _shifts(measurements, d), d)
    np.testing.assert_array_equal(detectors[0], expected)
    np.testing.assert_array_equal(detectors[1], expected)
    np.testing.assert_array_equal(observables[0], expected)


def _linear_form(rng, n, d):
    """A linear form over rec[0 .. n - 1] with coefficients near d, huge and negative ones, and a constant."""
    coefficients = [d - 1, d - 2, d + 1, 2 * d - 1, d // 2 + 1, -(d - 1), 2 ** 62, 2 ** 63 - 1, -2 ** 63,
                    2 ** 63, 2 ** 64 + 3, -10 ** 25, rng.randrange(d), rng.randint(-2 ** 40, 2 ** 40)]
    terms = [f"{rng.choice(['+', '-'])} ({rng.choice(coefficients)}) * rec[{j}]" for j in range(n)]
    terms.append(f"+ ({rng.choice(coefficients)})")
    rng.shuffle(terms)
    return " ".join(terms)


def _ring_expression(rng, n, depth, d):
    """A random ring expression over rec[0 .. n - 1] with literals of every size and sign."""
    if depth <= 0 or rng.random() < 0.2:
        if rng.random() < 0.6:
            return f"rec[{rng.randrange(n)}]"
        return str(rng.choice([0, 1, 2, d - 1, d, d + 1, d // 2, 2 ** 31, 2 ** 62, 2 ** 63 - 1, 2 ** 63, -2 ** 63,
                               2 ** 64 + 7, 10 ** 30, rng.randrange(d), rng.randint(-10 ** 12, 10 ** 12)]))
    a, b = _ring_expression(rng, n, depth - 1, d), _ring_expression(rng, n, depth - 1, d)
    op = rng.choice(["+", "-", "*", "+", "-", "*", "neg", "pos", "pow", "mod"])
    if op == "neg":
        return f"-({a})"
    if op == "pos":
        return f"+({a})"
    if op == "pow":
        return f"({a}) ** {rng.choice([0, 1, 2, 3, 5])}"
    if op == "mod":
        return f"({a}) % {rng.choice([1, 2, -1, -3, 2 ** 40 + 1]) * d}"
    return f"({a}) {op} ({b})"


PRODUCTS_AND_POWERS = [
    "rec[0] * rec[1] * rec[2]",
    "rec[0] * rec[1] - rec[2] * rec[3]",
    "(rec[0] + rec[1]) * (rec[2] - rec[3]) * 1000000007",
    "-(rec[0] * rec[1] - rec[2] * rec[3]) * -1000000007",
    "+rec[0] * -rec[1] * +rec[2]",
    "rec[0] ** 3 - 5 * rec[1] ** 2 + rec[2] ** 5",
    "(rec[0] - rec[1]) ** 7 + rec[2] ** 0",
    "2 ** 70 * rec[0] - 3 ** 50 * rec[1] + rec[2]",
    "rec[0] + 2 ** 63 + 9223372036854775808 * rec[1]",
    "0x7fffffffffffffffffff * rec[0] - 1_000_000_000_000 * rec[1] * rec[2]",
    "(rec[0] * rec[1] * rec[2]) % {2d} + rec[3] % {d}",
    "(rec[0] * rec[1]) % {-d} - rec[2] * rec[3] % {big}",
]


@pytest.mark.parametrize("d", DIMENSIONS)
def test_ring_expressions_are_exact(d):
    """Linear forms with coefficients near d, huge literals (2**63 and above raised OverflowError) and
    products and powers of records used to leave int64 before the reduction mod d."""
    rng = random.Random(d)
    records = _records(d, 4, 64, d)
    rows = records.tolist()
    expressions = [_linear_form(rng, rng.randint(1, 4), d) for _ in range(40)]
    expressions += [_ring_expression(rng, 4, rng.randint(1, 5), d) for _ in range(60)]
    expressions += [e.format(d=d, **{"2d": 2 * d, "-d": -d, "big": (2 ** 40 + 1) * d}) for e in PRODUCTS_AND_POWERS]
    for expr in expressions:
        values = _evaluate(records, _compile_detector(expr, d))
        assert values.tolist() == _exact(expr, rows, d), expr


@pytest.mark.parametrize("d", DIMENSIONS)
def test_large_constant_powers_are_exact(d):
    """Powers that int64 cannot hold, by exponents up to 10**30, are taken with every product reduced mod d."""
    records = _records(d, 2, 32, d + 1)
    for k in [31, 62, 63, 64, 1000, 2 ** 64 + 1, 10 ** 30]:
        expr = f"rec[0] ** {k} - 3 * (rec[0] - rec[1]) ** {k} + 2 ** {k}"
        values = _evaluate(records, _compile_detector(expr, d))
        expected = [(pow(x, k, d) - 3 * pow(x - y, k, d) + pow(2, k, d)) % d for x, y in zip(*records.tolist())]
        assert values.tolist() == expected, expr


def test_long_observable_with_large_coefficients():
    """A long sum reduces its terms, not the running sum, so it stays on int64 rows."""
    n = 1500
    expr = " + ".join(f"{(j * 7919) % D}*rec[{j}]" for j in range(n))
    records = np.random.default_rng(3).integers(0, D, size=(n, 8)).astype(np.int32)
    assert _evaluate(records, _compile_detector(expr, D)).tolist() == _exact(expr, records.tolist(), D)


NON_RING = [
    "abs(rec[0] - rec[1]) * 2147483646 * 2147483646",
    "(rec[0] != rec[1]) * 2147483646 * 2147483646 * 2147483646",
    "(rec[0] < rec[1]) + (rec[0] == 0) * rec[2] * rec[2] * rec[2]",
    "(rec[0] * rec[1] * rec[2]) // 3",
    "rec[0] // rec[1]",
    "(rec[0] * rec[1] * rec[2]) % 1000",
    "rec[0] % rec[1] + 7",
    "~(rec[0] * rec[1] * rec[2])",
    "(rec[0] ^ rec[1]) * rec[2] * rec[2] * rec[2]",
    "(rec[0] << 40) * rec[1]",
    "rec[0] / 2",
    "rec[0] * 1e0 * rec[1] * rec[2]",
    "rec[0] * True * rec[1] * rec[2]",
    "rec[0] ** 2 ** 3",
    "rec[0] ** rec[1]",
    "rec[0] ** -1",
    "np.sqrt(rec[0] * rec[1])",
    "np.logical_xor(rec[0], rec[1])",
    "np.abs(rec[0] - rec[1]) * 4611686018427387904",
    "sum([rec[0], rec[1]]) * rec[2] * rec[2] * rec[2]",
    "[rec[0] * rec[1] * rec[2]][0]",
    "(lambda x: x * x * x)(rec[0])",
    "rec[0] if rec[1][0] else 2 ** 64",
    "missing * rec[0]",
    # calls and empty tuples, with literal exponents large enough for the syntax tree path
    "(rec[0])(2) ** 5000",
    "rec[0] * 2 ** 5000 + ()",
    "(rec[0]) ** 5000 + rec[1](2)",
    # % and ** by operands that are not literals, even when their values are
    "(rec[0] * rec[1] * rec[2]) % (3 * 2147483647)",
    "rec[0] * rec[1] * rec[2] * rec[0] ** (1 + 1)",
    "2 ** 64 * rec[0] + 10 % 9",
    # % by records with zeros, which warns
    "(rec[0] % rec[1]) * rec[2] * rec[2] * rec[2]",
    "(rec[0] % rec[1]) * rec[2] + 5",
]
if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    # A power by a computed exponent beyond int64, which NumPy 2 refuses at once (NumPy 1.x computes it on
    # Python ints, without end).  The ring pass used to take it mod d first, with a million squarings.
    NON_RING.append("rec[0] ** 2 ** 2 ** 20")


def _int64_values(function, records):
    """What the function gives on the int64 shift rows, or the type of the exception it raises."""
    events = np.zeros(records.shape[1], dtype=np.int64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events[:] = function([row.astype(np.int64) for row in records])
    except Exception as e:
        return type(e)
    return events.tolist()


@pytest.mark.parametrize("d", [7, D])
def test_other_expressions_keep_their_int64_values(d):
    """Comparisons, calls, bit operations, //, / and % by other literals, names, the SyntaxError fallback
    and hand-built functions give exactly what NumPy int64 arithmetic gives, overflow included."""
    records = _records(d, 3, 64, 5)
    for expr in NON_RING:
        pristine = eval("lambda rec : _detector_mod((" + expr + "), " + str(d) + ")", vars(program))
        expected = _int64_values(pristine, records)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = _evaluate(records, _compile_detector(expr, d)).tolist()
        except Exception as e:
            values = type(e)
        assert values == expected, expr
    fallback = _compile_detector("rec[0] * 2147483646) * (2147483646 * rec[1]", d)
    hand_built = lambda rec: (rec[0] * 2147483646 * 2147483646 * 2147483646) % d
    for function in (fallback, hand_built):
        assert _evaluate(records, function).tolist() == _int64_values(function, records)


def test_computed_exponents_end_the_ring_pass_at_once():
    """Every ring exponent of a compiled function is a literal of at most 2**12 (larger ones take the syntax tree
    path), so a larger exponent was computed and the source is not a ring expression.  The pass used to take
    rec[0] ** 2 ** 2 ** 20 mod d with a million squarings of the rows, about 20 seconds, before falling back."""
    d = D
    rows = [row.astype(np.int64) for row in _records(d, 1, 64, 23)]
    function = _compile_detector("rec[0] ** 2 ** 2 ** 20", d)
    start = time.perf_counter()
    with pytest.raises(program._NotRing):
        program._ring_values(function, rows, d)
    assert time.perf_counter() - start < 2
    # Literal exponents above 2**12 are still exact, from the syntax tree.
    for expr in ["rec[0] ** 4097", "rec[0] ** (4096) * rec[0]", f"(rec[0] - 1) ** +5000 % {2 * d}"]:
        records = _records(d, 1, 32, 37)
        assert _evaluate(records, _compile_detector(expr, d)).tolist() == _exact(expr, records.tolist(), d), expr


@pytest.mark.parametrize("expr, d, exact", [
    ("rec[0] - rec[1]", D, False),
    (" - ".join(f"9999 * rec[{j % 3}]" for j in range(2000)), D, False),
    (f"{D - 1}*rec[0] + {D - 1}*rec[1] - 5", D, False),
    (f"{D - 1} * rec[0] - - {D - 1}*rec[1] + 3 * 1000000", D, False),
    ("(rec[0] - rec[1]) % 3", 3, False),
    ("(rec[0] - rec[1]) % 5 - 2 ** 62", 3, False),
    ("rec[0] ** 62", 3, False),
    ("abs(rec[0]) * rec[1] * rec[2] * rec[0]", D, False),
    ("(rec[0] * rec[1] * rec[2]) % 7", D, False),
    ("(rec[0] * rec[1] * rec[2]) % (3 * 2147483647)", D, False),
    (f"{D - 1}*rec[0] + {D - 1}*rec[1] + {D - 1}*rec[2]", D, True),
    ("rec[0] ** 63", 3, True),
    ("rec[0] + 2 ** 63", 3, True),
    ("rec[0] * 9223372036854775808", 2, True),
    (f"{2 ** 62}*rec[0] + 2*rec[1]", 3, True),
])
def test_only_ring_expressions_that_could_leave_int64_change_evaluation(expr, d, exact):
    """Everything else gives its values with its compiled function, so it keeps its NumPy semantics; the
    frame sampler records which is which for each source, and later evaluations agree with the first."""
    records = _records(d, 3, 16, 7)
    function = _compile_detector(expr, d)
    program._RING_SOURCES.pop((expr, d), None)
    first = _evaluate(records, function).tolist()
    assert program._RING_SOURCES[(expr, d)] is exact
    assert _evaluate(records, function).tolist() == first
    rows = [row.astype(np.int64) for row in records]
    assert program._detector_values(function, function.compiled_from, rows).tolist() == first
    if exact:
        assert first == _exact(expr, records.tolist(), d)
    else:
        assert first == _int64_values(function, records)


def test_nesting_continuations_and_long_literals():
    """Ring expressions are evaluated by their compiled function, so nesting and line continuations
    that compile are evaluated exactly too, and literals of any length are only ever read by Python."""
    d = D
    records = _records(d, 3, 32, 11)
    big = f"({d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2])"
    for expr in ["-" * 1001 + big, "-" * 1001 + big + f" % {2 * d}", f"{d - 1}*rec[0] + \\\n {d - 1}*rec[1] + {d - 1}*rec[2]",
                 "0" * 4301 + " + " + big, f"{d - 1}*rec[0] + \\\r {d - 1}*rec[1] + {d - 1}*rec[2]"]:
        assert _evaluate(records, _compile_detector(expr, d)).tolist() == _exact(expr, records.tolist(), d), expr[-60:]


def test_line_continuations_before_large_exponents():
    """A backslash line continuation between ** and its literal hid the exponent from _large_powers, so the
    compiled function computed 3 ** 10**30 itself and never ended.  Python also takes a backslash and a lone
    carriage return as one."""
    d = D
    records = _records(d, 2, 32, 41)
    for expr in [f"rec[0] - 3 ** \\\n {10 ** 30} * rec[1]", "(rec[0] - rec[1]) ** \\\n ( \\\n 5000) + 7",
                 "2 **\\\r\n+ 70 * rec[0] ** \\\n 4097", f"rec[0] - 3 ** \\\r {10 ** 30} * rec[1]",
                 "rec[0] ** \\\r5000 + rec[1]"]:
        assert program._large_powers(expr), expr
        expected = [(x - pow(3, 10 ** 30, d) * y) % d for x, y in zip(*records.tolist())] if "3 **" in expr \
            else _exact(expr, records.tolist(), d)
        assert _evaluate(records, _compile_detector(expr, d)).tolist() == expected, expr


def test_line_breaks_outside_parentheses():
    """A line break outside parentheses failed the standalone compile, so these expressions got the fallback
    function, which kept int64 values; in the lambda's parentheses the line break is harmless.  A lone carriage
    return breaks lines too."""
    d = D
    records = _records(d, 3, 32, 31)
    big = f"{d - 1}*rec[0] +\n{d - 1}*rec[1] + {d - 1}*rec[2]"
    for expr in [big, f"(rec[0] -\n rec[1]) ** 5 *\n\t{d - 1}", f"\n{big}\n  % {2 * d}\n", f"\n ({big}\n) % {2 * d}",
                 f"{d - 1} * rec[0] * \\\n rec[1] +\r\n 2 ** 5000", f"rec[0] ** 3 +\n rec[1] **\n 4097 - rec[2]",
                 f"{d - 1}*rec[0] -\r({d - 1}*rec[1] + {d - 1}*rec[2])", f"{d - 1}*rec[0] +\r {d - 1}*rec[1] + \\\r {d - 1}*rec[2]",
                 f"{d - 1}*rec[0] -\r {d - 1}*rec[1] *\r(rec[2] + 1)"]:
        function = _compile_detector(expr, d)
        assert function.compiled_from == (expr, d)
        assert _evaluate(records, function).tolist() == _exact("(" + expr + ")", records.tolist(), d), repr(expr)


@pytest.mark.parametrize("d", [7, D])
def test_line_breaks_keep_other_values(d):
    """Other expressions broken over lines give the same values as the fallback function gave them, and so do
    those that close a parenthesis they did not open, which keep the fallback."""
    records = _records(d, 3, 64, 43)
    for expr in ["abs(rec[0] -\n rec[1]) *\n 2147483646 * 2147483646", "(rec[0] * rec[1] * rec[2]) %\n 1000",
                 "(rec[0] % rec[1]) *\n rec[2] * rec[2] * rec[2]", "rec[0] +\n len(')') * 2 ** 64",
                 "rec[0] * 2147483646) * (\n2147483646 * rec[1]"]:
        fallback = eval("lambda rec : (" + expr + ") % " + str(d), vars(program))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = _evaluate(records, _compile_detector(expr, d)).tolist()
        except Exception as e:
            values = type(e)
        assert values == _int64_values(fallback, records), repr(expr)
    assert hasattr(_compile_detector("rec[0] +\n len(')') * 2 ** 64", d), "compiled_from")
    assert not hasattr(_compile_detector("rec[0] * 2147483646) * (\n2147483646 * rec[1]", d), "compiled_from")


def test_leading_white_space():
    """compile() rejects leading white space as an unexpected indent, so these expressions got the fallback
    function, which kept int64 values; in the lambda's parentheses the white space is harmless."""
    d = D
    records = _records(d, 3, 32, 13)
    big = f"{d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2]"
    for expr in [" " + big, "\t" + big, "\n  (" + big + f") % {2 * d}", " (rec[0] - rec[1]) ** 5", "\t2 ** 5000 * rec[0]"]:
        function = _compile_detector(expr, d)
        assert _evaluate(records, function).tolist() == _exact(expr.strip(), records.tolist(), d), repr(expr)


def test_a_failed_evaluation_records_nothing(monkeypatch):
    """An evaluation that fails for a reason other than the expression (rows that do not fit it, running out
    of memory) must not record the source as not a ring expression, or later frame runs keep int64 values."""
    d = D
    records = _records(d, 3, 32, 17)
    expr = f"{d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2]"
    function = _compile_detector(expr, d)
    program._RING_SOURCES.pop((expr, d), None)
    with pytest.raises(IndexError):
        _evaluate(records[:2], function)
    assert (expr, d) not in program._RING_SOURCES

    def out_of_memory(*args):
        raise MemoryError

    with monkeypatch.context() as patch:
        patch.setattr(program, "_ring_values", out_of_memory)
        with pytest.raises(MemoryError):
            _evaluate(records, function)
    assert (expr, d) not in program._RING_SOURCES
    assert _evaluate(records, function).tolist() == _exact(expr, records.tolist(), d)
    assert program._RING_SOURCES[(expr, d)] is True


def test_many_sources_keep_their_decisions(monkeypatch):
    """_RING_SOURCES was emptied when it reached 2**14 sources, so a circuit with more distinct ones decided most
    of them again on every run."""
    monkeypatch.setattr(program, "_RING_SOURCES", {})
    c = Circuit(1, 7)
    c.add_gate("N1", 0, noise_channel="f", prob=0.5)
    c.add_gate("M", 0)
    for j in range(20000):
        c.add_gate("DETECTOR", expr=f"rec[-1] + {j}")
    np.random.seed(5)
    _, (first, _) = Program(c).simulate(shots=4, raw_detector_output=True)
    assert len(program._RING_SOURCES) == 20000
    decided = []
    could_leave_int64 = program._could_leave_int64

    def deciding(source, d):
        decided.append(source)
        return could_leave_int64(source, d)

    monkeypatch.setattr(program, "_could_leave_int64", deciding)
    np.random.seed(5)
    _, (second, _) = Program(c).simulate(shots=4, raw_detector_output=True)
    assert decided == []
    np.testing.assert_array_equal(second, first)


def test_ring_sources_forget_only_their_oldest_half(monkeypatch):
    """The decisions stay bounded in number, and a full _RING_SOURCES forgets its oldest half, not everything."""
    monkeypatch.setattr(program, "_RING_SOURCES", {})
    monkeypatch.setattr(program, "_RING_SOURCES_LIMIT", 8)
    records = _records(7, 1, 4, 29)
    for j in range(20):
        _evaluate(records, _compile_detector(f"rec[0] + {j}", 7))
        assert len(program._RING_SOURCES) <= 8
    assert [source for source, _ in program._RING_SOURCES] == [f"rec[0] + {j}" for j in range(12, 20)]


def test_ring_sources_forgotten_by_another_thread(monkeypatch):
    """Two threads can find _RING_SOURCES full at once, and both forget its oldest half; the second must not fail
    on the sources that the first has already removed."""
    class ForgottenMeanwhile(dict):
        def __iter__(self):
            keys = list(super().__iter__())
            for key in keys[:len(keys) // 2]:
                super().pop(key)  # as another thread does, after this one lists the keys
            return iter(keys)

    records = _records(7, 1, 4, 47)
    full = ForgottenMeanwhile((_compile_detector(f"rec[0] + {j}", 7).compiled_from, False) for j in range(8))
    monkeypatch.setattr(program, "_RING_SOURCES", full)
    monkeypatch.setattr(program, "_RING_SOURCES_LIMIT", 8)
    assert _evaluate(records, _compile_detector("rec[0] + 8", 7)).tolist() == [(x + 8) % 7 for x in records[0]]
    assert [source for source, _ in program._RING_SOURCES] == [f"rec[0] + {j}" for j in range(4, 9)]


@pytest.mark.parametrize("expr", ["(rec[0] % rec[1]) * rec[2] * rec[2] * rec[2]", "(rec[0] % rec[1]) * rec[2] + 5"])
def test_the_ring_pass_leaves_warnings_to_the_function(expr):
    """A % by records that are 0 warns.  The ring pass ran it before finding that the expression is not a ring
    expression, and then the function warned again; under the tests' filters, which turn warnings from sdim into
    errors, the pass's warning ended it without recording its decision, so every run started it again."""
    d = D
    records = _records(d, 3, 16, 19)
    plain = eval("lambda rec : _detector_mod((" + expr + "), " + str(d) + ")", vars(program))
    expected = _int64_values(plain, records)
    function = _compile_detector(expr, d)
    program._RING_SOURCES.pop((expr, d), None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _evaluate(records, function).tolist() == expected
    assert [str(w.message) for w in caught] == ["divide by zero encountered in remainder"]
    assert program._RING_SOURCES[(expr, d)] is False

    program._RING_SOURCES.pop((expr, d), None)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", module=r"sdim(\.|$)")  # as pyproject.toml sets for the tests
        with pytest.raises(RuntimeWarning, match="divide by zero"):
            _evaluate(records, function)
    assert program._RING_SOURCES[(expr, d)] is False


@pytest.mark.parametrize("source", [
    "rec[0] - rec[1]", "((rec[0] - rec[1]) % 6) * (-5 * (rec[2] - 3))", "rec[0] ** 2\t", "()", "rec[0](1)",
    " rec[0]", "\trec[0] - rec[1]", "rec[0]) + (rec[1]", "rec[0]) * (2", "rec[0] -", "% rec[0]",
    "(" * 17 + "rec[0]" + ")" * 17, "abs(rec[0])", "0x1f * rec[0]", "", " \t", " ()", "\n  rec[0] * 2", " rec[0]) - (2",
    "rec[0] -\n rec[1]", "rec[0] -\n  rec[1] *\n 2", "rec[0]) -\n (rec[1]", "abs(rec[0]\n) +\n[1][0]", "rec[0] -\n",
    "rec[0]\n rec[1]", "x for x in rec", "x := rec[0]", "x for x in\n rec", "rec[0] -\n\\\n rec[1]", "rec[0] -\r rec[1]",
    "rec[0] ]-[ rec[1]", "rec[0] -\r(rec[1]\r)", "rec[0] -\r rec[1]) * (\r2", "yield rec[0] -\n rec[1]",
])
def test_detectors_compile_to_the_same_function(source):
    """A plain source is compiled once instead of twice; it must still get _detector_mod exactly when it is
    a complete expression on its own past its leading white space, or in parentheses when it spans lines and
    does not close them early, and the same values."""
    d = 7
    try:
        compile(source.lstrip(), "<detector>", "eval")
        complete = True
    except SyntaxError:
        # The sources hold no strings or comments, so their characters give the depth.
        depths = itertools.accumulate({"(": 1, "[": 1, ")": -1, "]": -1}.get(c, 0) for c in source)
        try:
            compile("(" + source + ")", "<detector>", "eval")
            complete = ("\n" in source or "\r" in source) and min(depths) >= 0
        except SyntaxError:
            complete = False
    try:
        function = _compile_detector(source, d)
    except SyntaxError:
        assert not complete
        return
    assert ("_detector_mod" in function.__code__.co_names) is complete
    expected = eval("lambda rec : " + ("_detector_mod((" + source + "), 7)" if complete else "(" + source + ") % 7"),
                    vars(program))
    rows = [np.arange(7, dtype=np.int64), np.arange(7, dtype=np.int64)[::-1].copy(), np.ones(7, dtype=np.int64)]
    assert _int64_values(function, np.array(rows)) == _int64_values(expected, np.array(rows))


def test_frame_detectors_follow_the_detector_error_model():
    """At d = 2**31 - 1 every frame shot's detectors and observable must be a multiple of the single
    mechanism's generator in sdim.dem, which reads the same expressions exactly."""
    d = D
    c = Circuit(3, d)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("N1", 0, noise_channel="f", prob=0.9)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT_INV", 0, 2)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr=f"{d - 1}*rec[0] + {d - 1}*rec[1] + {d - 1}*rec[2]")
    c.add_gate("DETECTOR", expr=f"{d - 2}*rec[1] - {d - 3}*rec[2] + 2147483000*rec[0]")
    c.add_gate("DETECTOR", expr=f"{2 ** 62}*rec[0] + {2 ** 63}*rec[1] - {10 ** 20}*rec[2]")
    c.add_gate("LOGICAL_OBSERVABLE", expr=f"{d - 5}*rec[0] - {d - 7}*rec[1] + {d - 11}*rec[2]")
    dem = DetectorErrorModel.from_circuit(c)
    (mechanism,) = dem.mechanisms
    (generator,) = mechanism.generators
    g = [generator.get(t, 0) for t in range(4)]
    np.random.seed(4)
    _, (detectors, observables) = Program(c).simulate(shots=2001, raw_detector_output=True)
    vectors = np.vstack([detectors, observables]).T.tolist()
    lead = next(t for t in range(4) if g[t])
    for v in vectors:
        a = v[lead] * pow(g[lead], -1, d) % d
        assert v == [a * coefficient % d for coefficient in g]
    assert sum(any(v) for v in vectors) > 1500


@pytest.mark.parametrize("d", [D, D - 1])
def test_outcomes_add_the_reference_in_int64(d):
    """Under NumPy 1.x, the int32 frame records were added to the int64 reference outcome in int32,
    so every outcome with reference + shift >= 2**31 came out 2 * (2**31 - d) too small."""
    references = [d - 1, d // 2 + 5, 0]
    shifts = [[0, 1, d - 1, d // 2, 2 ** 30, 7], [d // 2 - 5, d // 2 - 3, d - 1, 2 ** 30, 1, 0], [d - 1, 1, 0, 2, 3, 4]]
    assert sum(r + s >= 2 ** 31 for r, row in zip(references, shifts) for s in row) == 7
    p = Program(Circuit(3, d))
    p.measurement_results = [[[MeasurementResult(q, q == 2, value)]] for q, value in enumerate(references)]
    reference = Program._results_to_array(p.measurement_results)
    run = program._FrameRun(np.array(shifts, dtype=np.int32), np.arange(3), np.zeros(3, dtype=np.int64), None)
    measurements = p._combine_frame_run(run, reference, d)
    for q, value in enumerate(references):
        assert [r.measurement_value for r in measurements[q][0]] == [value] + [(value + s) % d for s in shifts[q]]
        assert [(r.qudit_index, r.deterministic) for r in measurements[q][0]] == [(q, q == 2)] * 7

    # The frame sampler's own outcomes, seeded: the reference outcome comes from Python's random module.
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("CNOT_INV", 0, 1)
    c.add_gate("M", [0, 1])
    random.seed(2)
    np.random.seed(2)
    measurements, _ = Program(c).simulate(shots=300)
    first = [r.measurement_value for r in measurements[0][0]]
    second = [r.measurement_value for r in measurements[1][0]]
    assert all(0 <= v < d for v in first + second)
    assert all((a + b) % d == 0 for a, b in zip(first, second))
