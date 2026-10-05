"""Exact values from the Pauli frame sampler where int64 arithmetic used to overflow.

Ring expressions (integer literals, rec[j], +, -, unary + and -, *, ** by a non-negative literal and %
by a literal that d divides) must give their exact value mod d on the measurement shifts, for dimensions
near 2**31, literals of 2**63 or more, and products and powers of records alike.  Every other expression
must give the same values as before, from NumPy int64 arithmetic.  Measurement outcomes must be
(reference + shift) mod d, under NumPy 2 and under NumPy 1.x's value-based casting alike.
"""

import random
import warnings

import numpy as np
import pytest

import sdim.program as program
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import DetectorData, Program, _compile_detector, _evaluate_detectors

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
]


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
                 "0" * 4301 + " + " + big]:
        assert _evaluate(records, _compile_detector(expr, d)).tolist() == _exact(expr, records.tolist(), d), expr[-60:]


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


@pytest.mark.parametrize("source", [
    "rec[0] - rec[1]", "((rec[0] - rec[1]) % 6) * (-5 * (rec[2] - 3))", "rec[0] ** 2\t", "()", "rec[0](1)",
    " rec[0]", "\trec[0] - rec[1]", "rec[0]) + (rec[1]", "rec[0]) * (2", "rec[0] -", "% rec[0]",
    "(" * 17 + "rec[0]" + ")" * 17, "abs(rec[0])", "0x1f * rec[0]", "", " \t", " ()", "\n  rec[0] * 2", " rec[0]) - (2",
])
def test_detectors_compile_to_the_same_function(source):
    """A plain source is compiled once instead of twice; it must still get _detector_mod exactly when it is
    a complete expression on its own past its leading white space, and the same values."""
    d = 7
    try:
        compile(source.lstrip(), "<detector>", "eval")
        complete = True
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
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("CNOT_INV", 0, 1)
    c.add_gate("M", [0, 1])
    np.random.seed(2)
    measurements, _ = Program(c).simulate(shots=300)
    first = [r.measurement_value for r in measurements[0][0]]
    second = [r.measurement_value for r in measurements[1][0]]
    assert all(0 <= v < d for v in first + second)
    assert all((a + b) % d == 0 for a, b in zip(first, second))
    reference = first[0]
    assert any(reference + (v - reference) % d >= 2 ** 31 for v in first[1:])
