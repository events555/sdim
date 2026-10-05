"""A qudit index outside the circuit raises IndexError in `DetectorErrorModel.from_circuit`, as in the simulators.

Indices beyond int64 used to reach the int64 IR arrays and raise OverflowError instead, on every gate but the
identity.
"""

import pytest

from sdim import Circuit, Program
from sdim.dem import DetectorErrorModel, compile_unit_responses

_ONE_QUDIT = [("I", {}), ("X", {}), ("X_INV", {}), ("Z", {}), ("Z_INV", {}), ("H", {}), ("H_INV", {}), ("P", {}),
              ("P_INV", {}), ("M", {}), ("M_X", {}), ("RESET", {}), ("MUL", {"a": 2}), ("N1", {}),
              ("N1", {"noise_channel": "f"}), ("N1", {"noise_channel": "p"})]
_TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP", "N2"]
_GATES = ([(name, params, "qudit") for name, params in _ONE_QUDIT]
          + [(name, {}, where) for name in _TWO_QUDIT for where in ("control", "target")])
_IDS = [f"{name}{params.get('noise_channel', '')}-{where}" for name, params, where in _GATES]


def _circuit(name, params, where, index, dense_n2=False):
    """Three qudits at d = 5: `name` on qudit `index` (the control or target of a two-qudit gate), then noise,
    measurements and a detector."""
    c = Circuit(3, 5)
    c.add_gate("RESET", [0, 1, 2])
    if where == "qudit":
        c.add_gate(name, index, **params)
    elif where == "control":
        c.add_gate(name, index, 1, **params)
    else:
        c.add_gate(name, 1, index, **params)
    c.add_gate("N1", [0, 1, 2], noise_channel="f", prob=0.01)
    if dense_n2:
        # A prob_dist N2 makes from_circuit build the IR of the whole circuit, noise gates included.
        c.add_gate("N2", 0, 2, prob_dist=[0.99] + [0.01 / 624] * 624)
    c.add_gate("N2", 0, 1, prob=0.01)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr="rec[-1]")
    return c


@pytest.mark.parametrize("index", [2 ** 70, -2 ** 70, 2 ** 40, -2 ** 40, 3, -4])
@pytest.mark.parametrize("name, params, where", _GATES, ids=_IDS)
def test_qudit_outside_the_circuit_raises_index_error(name, params, where, index):
    c = _circuit(name, params, where, index)
    with pytest.raises(IndexError, match=f"acts on qudit {index}, but the circuit has 3 qudits"):
        DetectorErrorModel.from_circuit(c)
    with pytest.raises(IndexError, match=f"acts on qudit {index}, but the circuit has 3 qudits"):
        compile_unit_responses(c)
    with pytest.raises(IndexError, match=f"acts on qudit {index}, but the circuit has 3 qudits"):
        DetectorErrorModel.from_circuit(_circuit(name, params, where, index, dense_n2=True))
    # The simulators raise IndexError too, in tableau and in frame mode.
    for shots in (1, 4):
        with pytest.raises(IndexError):
            Program(c).simulate(shots=shots)


@pytest.mark.parametrize("name, params, where", _GATES, ids=_IDS)
def test_negative_qudits_inside_the_circuit_count_from_the_end(name, params, where):
    """An index from -3 to -1 is the qudit 3 on, as before: the same model, or the same error."""
    def outcome(index):
        try:
            return str(DetectorErrorModel.from_circuit(_circuit(name, params, where, index)))
        except ValueError as error:
            return repr(error)

    # The other qudit of a two-qudit gate is qudit 1.
    for index in ((-3, -2, -1) if where == "qudit" else (-3, -1)):
        assert outcome(index) == outcome(index + 3)
