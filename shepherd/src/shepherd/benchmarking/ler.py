"""
ler.py -- the "ler" experiment: logical error rate of a Z-basis memory of a CSS code.

    from shepherd.experiment import Experiment
    from shepherd.noise_model import NoiseModel
    exp = Experiment.create("ler", 3, rounds="d")
    noiseless = exp.build(code)
    noisy = NoiseModel.phenomenological(3, 0.01).apply(noiseless)

What is measured (default behavior)
------------------------------------
The logical error rate of a Z-basis memory: all k logical qudits start in |0-bar>, and a shot
fails if the decoder's predicted shift of *any* of the k Z logicals is wrong. Only logical shift
(X-type) errors are counted; logical phase errors would need an X-basis memory, which is not
implemented (basis="X" raises). For k > 1 this is a per-shot "any logical wrong" rate, which tends
to 1 - 1/q**k at high p; a per-logical-qudit rate is about 1 - (1 - LER)**(1/k).

No encoding circuit: |0...0> stands in for |0-bar>
--------------------------------------------------
The data starts in the product state |0...0> (sdim's default) and is never encoded. This is
exact for this experiment, not an approximation:

    |0-bar>  is proportional to  sum over s in rowspace(H_X) of |s>,

the projection of |0...0> onto the +1 eigenspace of the X stabilizers. Every branch |s> has the
same Z-check values (H_Z s = 0, as H_X H_Z^T = 0) and the same Z-logical values (L_Z s = 0, as Z
logicals commute with X checks). A shift error moves every branch by the same vector, and phase
errors (including those a SUM copies from target to control) do not change Z-basis outcomes.
Since the circuit measures only Z checks and reads out in the Z basis, every outcome has the
same distribution from |0...0> as from |0-bar>. The projection would matter only if X checks
were measured: in a circuit that measures both, the first round of X-check outcomes is random
(that measurement performs the projection) and X detectors start in round 2.

This is the standard memory-experiment convention: prepare a product state in the memory basis,
measure, and read out transversally in the same basis. Examples: Fowler et al., PRA 86, 032324
(2012), arXiv:1208.0928 (logical initialization by product-state preparation plus stabilizer
measurement); stim's generated "surface_code:rotated_memory_z" circuits (Gidney, Quantum 5, 497
(2021)); Google Quantum AI's surface-code memory experiments (Nature 614, 676 (2023); Nature 638,
920 (2025)), which prepare product states and report X- and Z-basis memories separately. Counting
one error type for a CSS code, with the other handled the same way, goes back to Dennis, Kitaev,
Landahl and Preskill, J. Math. Phys. 43, 4452 (2002), arXiv:quant-ph/0110143.

The circuit (noiseless; noise comes from a NoiseModel)
------------------------------------------------------
Data qudits start in |0...0>, so every Z check and every Z logical is deterministic.

* rounds = 0: I on every data qudit (where the data waits), then a transversal Z measurement.
  Detectors are the rows of H_Z evaluated on the outcomes.
* rounds >= 1, each round: I on every data qudit; every Z check measured into a fresh ancilla
  with SUM gates (data control, ancilla target; a coefficient c is |c| SUM or SUM_INV gates,
  c taken in (-q/2, q/2]); M and RESET on the ancillas. Round 0 detectors are the outcomes, later
  ones the change since the previous round. A final transversal data measurement closes every
  check against its last outcome.
* Observables are the rows of L_Z (`CSSCode.logical_matrices`) on the final data outcomes.

TICKs separate the phases of a round (wait, SUMs, measure), so a model with idle noise sees
them as layers; the SUMs of a round share one layer, which is not a hardware schedule.

Only the Z checks are measured. With code-capacity or phenomenological noise nothing is lost:
X checks would add no noise to the data. With circuit-level noise it leaves out the X-check
circuits' noise on the data, so circuit-level results from this experiment are for noise in the
Z-syndrome circuit only.

Noise models (NoiseModel.code_capacity / phenomenological / circuit_level) give:
* code capacity: rounds = 0 and shift noise on I.
* phenomenological: shift noise on I once per round, outcome flips on every M. The final data
  measurement is noisy too, which is one more round of data noise.
* circuit-level: depolarizing on I and after each SUM, flips on M and RESET.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sdim import Circuit

from shepherd.codes.css import CSSCode
from shepherd.experiment import Experiment
from shepherd.noise_model import NoiseModel


def _signed(c: int, q: int) -> int:
    """The representative of c mod q in (-q/2, q/2]."""
    c %= q
    return c - q if c > q // 2 else c


def _expr(terms: Iterable[tuple[int, int]], q: int, num_meas: int) -> str:
    """'c*rec[-k] + ...' for (coefficient, absolute measurement index) terms."""
    parts = [f"{c % q}*rec[{idx - num_meas}]" for c, idx in terms if c % q]
    if not parts:
        raise ValueError("empty detector or observable expression")
    return " + ".join(parts)


class LogicalErrorRate(Experiment, name="ler"):
    """
    Z-basis memory of a CSS code.

    rounds: number of syndrome rounds, or "d" for the code's distance (0 means code capacity:
        no syndrome rounds, only the final transversal measurement).
    basis: "Z" (the only basis implemented).
    """

    def __init__(self, dimension: int, *, rounds: int | str = 0, basis: str = "Z"):
        if basis != "Z":
            raise NotImplementedError("only the Z-basis memory is implemented")
        if rounds != "d" and (not isinstance(rounds, int) or rounds < 0):
            raise ValueError(f"rounds must be a non-negative integer or 'd', got {rounds!r}")
        self.dimension = dimension
        self.rounds = rounds
        self.basis = basis

    def rounds_for(self, code: CSSCode) -> int:
        if self.rounds == "d":
            if not np.isfinite(code.distance):
                raise ValueError(f"{code.label} has no finite distance; give rounds explicitly")
            return int(code.distance)
        return int(self.rounds)

    def _check(self, code: CSSCode) -> None:
        if not isinstance(code, CSSCode):
            raise TypeError(f"the ler experiment needs a CSSCode, got {type(code).__name__}")
        if code.dimension != self.dimension:
            raise ValueError(f"experiment is for q = {self.dimension}, code has q = {code.dimension}")

    def build(self, code: CSSCode) -> Circuit:
        self._check(code)
        q, n, hz = code.dimension, code.n, code.H_Z
        _, lz = code.logical_matrices()
        m = hz.shape[0]
        rounds = self.rounds_for(code)
        data = list(range(n))
        anc = list(range(n, n + m))
        c = Circuit(n + (m if rounds else 0), q)
        num_meas = 0
        prev: list[int] | None = None       # absolute index of each check's last ancilla outcome

        for _ in range(rounds):
            c.add_gate("I", data)
            c.add_gate("TICK")
            for i in range(m):
                for j in np.nonzero(hz[i])[0]:
                    s = _signed(int(hz[i, j]), q)
                    for _ in range(abs(s)):
                        c.add_gate("CNOT" if s > 0 else "CNOT_INV", int(j), anc[i])
            c.add_gate("TICK")
            c.add_gate("M", anc)
            cur = list(range(num_meas, num_meas + m))
            num_meas += m
            c.add_gate("RESET", anc)
            for i in range(m):
                terms = [(1, cur[i])] if prev is None else [(1, cur[i]), (q - 1, prev[i])]
                c.add_gate("DETECTOR", expr=_expr(terms, q, num_meas))
            c.add_gate("TICK")
            prev = cur

        if rounds == 0:
            c.add_gate("I", data)
            c.add_gate("TICK")
        c.add_gate("M", data)
        first = num_meas
        num_meas += n
        for i in range(m):
            terms = [(int(hz[i, j]), first + int(j)) for j in np.nonzero(hz[i])[0]]
            if prev is not None:
                terms.append((q - 1, prev[i]))
            c.add_gate("DETECTOR", expr=_expr(terms, q, num_meas))
        for row in lz:
            terms = [(int(row[j]), first + int(j)) for j in np.nonzero(row)[0]]
            c.add_gate("LOGICAL_OBSERVABLE", expr=_expr(terms, q, num_meas))
        return c

    def metadata(self, code: CSSCode) -> dict:
        return {**super().metadata(code), "rounds": self.rounds_for(code), "basis": self.basis}

    def expected_observables(self, code: CSSCode) -> np.ndarray:
        return np.zeros(code.k, dtype=np.int64)


def z_memory_circuit(code: CSSCode, *, p: float, rounds: int = 0,
                     p_meas: float | None = None) -> Circuit:
    """
    Shortcut for the two simplest models: code capacity (rounds = 0) or phenomenological
    (rounds >= 1, measurement flips p_meas, default p), applied to the ler circuit.
    """
    q = code.dimension
    model = (NoiseModel.code_capacity(q, p) if rounds == 0
             else NoiseModel.phenomenological(q, p, p_meas))
    return model.apply(LogicalErrorRate(q, rounds=rounds).build(code))
