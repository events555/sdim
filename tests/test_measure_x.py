"""M_X measures in the X basis and leaves the qudit in the matching X eigenstate."""

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import Program


def _values(circuit, shots, **kwargs):
    if shots == 1:
        return [[r.measurement_value for r in Program(circuit).simulate(**kwargs)]]
    out = Program(circuit).simulate(shots=shots, **kwargs)
    measurements = out if kwargs.get("force_tableau") else out[0]   # frame mode also returns detectors
    rows = [r for rounds in measurements for r in rounds]
    return [[row[s].measurement_value for row in rows] for s in range(shots)]


@pytest.mark.parametrize("d", [2, 3, 5, 4, 6])
@pytest.mark.parametrize("mode", [{"force_tableau": True}, {}])
def test_repeated_m_x_repeats_its_outcome(d, mode):
    c = Circuit(1, d)
    c.add_gate("RESET", 0)          # Z eigenstate, so the first M_X is uniformly random
    c.add_gate("M_X", 0)
    c.add_gate("M_X", 0)
    c.add_gate("M_X", 0)
    rows = _values(c, 300, **mode)
    firsts = set()
    for _, first, second, third in rows:   # the RESET records a round too
        assert first == second == third
        firsts.add(first)
    assert len(firsts) == d                # and the first outcome really is random


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("mode", [{"force_tableau": True}, {}])
def test_m_after_m_x_is_random(d, mode):
    """An X eigenstate gives a uniformly random Z measurement."""
    c = Circuit(1, d)
    c.add_gate("RESET", 0)
    c.add_gate("H", 0)
    c.add_gate("M_X", 0)             # deterministic
    c.add_gate("M", 0)               # uniform
    rows = _values(c, 3000, **mode)
    assert len({row[1] for row in rows}) == 1
    counts = np.bincount([row[2] for row in rows], minlength=d)
    assert counts.min() > 3000 / d * 0.75


def test_dem_treats_m_x_like_the_simulators():
    d = 5
    c = Circuit(1, d)
    c.add_gate("RESET", 0)
    c.add_gate("H", 0)
    c.add_gate("M_X", 0)
    c.add_gate("N1", 0, noise_channel="p", prob=0.2)   # Z errors flip later X measurements
    c.add_gate("M_X", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    dem = DetectorErrorModel.from_circuit(c)
    det, _ = dem.sample(40000, seed=2)
    _, (frame, _) = Program(c).simulate(shots=40001, raw_detector_output=True)
    for i in range(2):
        assert abs((det[:, i] != 0).mean() - (frame[i] != 0).mean()) < 0.015
        assert abs((det[:, i] != 0).mean() - 0.2) < 0.015


def test_dem_rejects_z_measurement_of_an_x_eigenstate():
    c = Circuit(1, 3)
    c.add_gate("RESET", 0)
    c.add_gate("H", 0)
    c.add_gate("M_X", 0)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    with pytest.raises(ValueError, match="not deterministic"):
        DetectorErrorModel.from_circuit(c)
