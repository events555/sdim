"""
Compact detector error models (DEMs) for qudit circuits.

`DetectorErrorModel.from_circuit` turns a circuit with N1/N2 noise, detectors
and logical observables into a list of independent error mechanisms, one per
noise gate that a detector or observable can see. The model is exact, and its
size does not depend on the qudit dimension. It works the same at d = 3 and at
d = 1000003.

The older model in `sdim.dem_legacy` makes one mechanism per Pauli error value
(`d**2 - 1` for each N1 and `d**4 - 1` for each N2) and treats them as
independent. At d = 1000003 that is 10**24 mechanisms per N2 gate. It is also
only approximate. A depolarizing channel applies exactly one of its errors, so
those errors are disjoint cases, not independent events.

## Example

```python
from sdim import Circuit
from sdim.dem import DetectorErrorModel

# Two data qudits and an ancilla that measures x0 - x1.
circuit = Circuit(3, 1000003)
circuit.add_gate("RESET", [0, 1, 2])
circuit.add_gate("N1", [0, 1], noise_channel="f", prob=0.01)
circuit.add_gate("CNOT", 0, 2)
circuit.add_gate("CNOT_INV", 1, 2)
circuit.add_gate("N2", 1, 2, prob=0.001)
circuit.add_gate("M", [2, 0, 1])
circuit.add_gate("DETECTOR", expr="rec[-3]")
circuit.add_gate("DETECTOR", expr="rec[-3] - rec[-2] + rec[-1]")
circuit.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")

dem = DetectorErrorModel.from_circuit(circuit)
print(dem)
detectors, observables = dem.sample(100_000)
```

Output:
```plaintext
DIMENSION 1000003
DETECTORS 2
OBSERVABLES 1
ERROR(0.01000000999998) D0=1 # N1[f]@3:q0
ERROR(0.01000000999998) D0=1 L0=1000002 # N1[f]@4:q1
ERROR(0.001) D1=1 L0=1 | D0=1 D1=1 # N2@7:q1,q2
```

An `ERROR(pi) g_1 | g_2 | ...` line means that with probability `pi` the
mechanism adds `a_1*g_1 + a_2*g_2 + ...` to the detectors and observables,
where each `a_j` is uniform on Z_d (zero included). `Dk=v` and `Lk=v` give
the coefficient, mod d, on detector k and observable k. The comment names the
noise gate(s) the mechanism came from. `write_to_file` and `read_from_file`
use the same format.

The N1 lines show `pi` slightly above 0.01 because one of the d equally likely
shifts is the identity, so `pi = p / (1 - 1/d)`. The X fault after the second
N1 reaches observable 0 with coefficient -1, which prints as 1000002.

## How it works

Write an n-qudit Pauli error, up to phase, as a vector
`(x_1, z_1, ..., x_n, z_n)` in G = Z_d^(2n). Call the channel "with
probability pi, apply a uniformly random element of the subgroup S" a
subgroup mechanism. Its Fourier transform is `1 - pi` on the characters of G
that are non-trivial on S, and 1 on the rest. Independent channels multiply in
the Fourier domain, so subgroup mechanisms combine by multiplying their
`1 - pi` factors. For prime d this gives three facts the module relies on.

1. Depolarizing noise of strength p on n qudits (a uniformly random
   non-identity Pauli with probability p) is a single subgroup mechanism on
   all of G, with `pi = p * d**(2n) / (d**(2n) - 1)`. The detector responses
   of its 2n unit faults, X and Z on each qudit, describe it completely. When
   it fires it adds `sum_j a_j * response_j` with each `a_j` uniform on Z_d.
   This is what `from_circuit` stores.
2. The same mechanism splits exactly into `(d**(2n) - 1) / (d - 1)`
   independent mechanisms, one per line (1-dimensional subgroup) of G, each
   with `pi_line = 1 - (1 - pi) ** (d ** (1 - 2n))`. This works because every
   non-trivial character is non-trivial on exactly `d**(2n-1)` lines. At
   d = 2 a line {I, P} that fires with probability pi applies P with
   probability pi / 2, and the formulas reduce to Gidney's decorrelated
   depolarization, `1/2 - 1/2 sqrt(1 - 4p/3)` and
   `1/2 - 1/2 (1 - 16p/15)**(1/8)` (https://algassert.com/post/2001).
   `DetectorErrorModel.to_lines` does this expansion.
3. Two mechanisms on the same subgroup merge into one with
   `1 - pi = (1 - pi_1)(1 - pi_2)`. At d = 2 this is the usual XOR rule
   `p_1 (1 - p_2) + (1 - p_1) p_2`. The XOR rule is wrong for d > 2, since a
   shift applied twice adds up to twice the shift instead of cancelling.

The unit-fault responses come from pushing each unit fault through the rest of
the circuit with the same frame update rules as `sdim.program.simulate_frame`.
Detector and observable coefficients are read from sdim's own compiled
detector expressions.

## Limitations

- The dimension must be prime, so that Z_d is a field.
- N2 gates must use `prob`. A custom `prob_dist` is not a subgroup mechanism.
  `sdim.dem_legacy` handles those for small d.
- Detector and observable expressions must be linear in their measurement
  records, with no constant term.
- A single fault can spread to at most 512 qudits at a time during
  propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import itertools
import math

import numpy as np
from numba import njit

from .circuit import Circuit
from .program import Program

# Gate ids, in the order GateData registers the gates.
_H, _H_INV, _P, _P_INV = 5, 6, 7, 8
_CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP = 9, 10, 11, 12, 13
_M, _M_X, _RESET = 14, 15, 16
_N1, _N2, _DETECTOR, _OBSERVABLE = 17, 18, 19, 20
_FRAME_GATES = {_H, _H_INV, _P, _P_INV, _CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP, _M, _M_X, _RESET}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def depolarizing_subgroup_probability(p: float, dimension: int, num_qudits: int) -> float:
    """
    Converts a depolarizing probability into a subgroup mechanism probability.

    Depolarizing noise applies a uniformly random non-identity Pauli with
    probability p. The equivalent subgroup mechanism applies a uniformly random
    Pauli, identity included, with probability pi. One in d**(2n) of those
    draws is the identity, so pi = p / (1 - d**(-2n)).

    Args:
        p (float): Depolarizing probability.
        dimension (int): Qudit dimension d.
        num_qudits (int): Number of qudits n the channel acts on.

    Returns:
        float: The subgroup mechanism probability pi.
    """
    size = float(dimension) ** (2 * num_qudits)
    return p / (1.0 - 1.0 / size)


def line_probability(pi: float, dimension: int, subgroup_rank: int) -> float:
    """
    Probability of each line mechanism when a subgroup mechanism is split into lines.

    A mechanism with probability pi on a rank-k subgroup is the same channel as
    (d**k - 1) / (d - 1) independent mechanisms, one per line of the subgroup,
    each with probability 1 - (1 - pi) ** (d ** (1 - k)). For d = 2 this is
    Gidney's decorrelated depolarization.

    Args:
        pi (float): Probability of the rank-k mechanism.
        dimension (int): Qudit dimension d.
        subgroup_rank (int): Rank k of the subgroup (2n for n-qudit depolarizing).

    Returns:
        float: Probability of each line mechanism.

    Raises:
        ValueError: If pi >= 1, which has no such decomposition.
    """
    if pi >= 1.0:
        raise ValueError("pi >= 1 (maximally mixing) cannot be decorrelated into independent lines")
    # expm1/log1p keep full precision when pi is tiny.
    exponent = float(dimension) ** (1 - subgroup_rank)
    return -math.expm1(exponent * math.log1p(-pi))


def merge_subgroup_probabilities(*pis: float) -> float:
    """
    Merges independent mechanisms that act on the same subgroup.

    The merged probability satisfies 1 - pi = (1 - pi_1)(1 - pi_2)...
    At d = 2, where a line that fires with probability pi applies its Pauli
    with probability pi / 2, this is the XOR rule p_1 (1 - p_2) + (1 - p_1) p_2.

    Args:
        *pis (float): Probabilities of the mechanisms to merge.

    Returns:
        float: Probability of the merged mechanism.
    """
    log_keep = sum(math.log1p(-pi) for pi in pis)
    return -math.expm1(log_keep)


@dataclass
class ErrorMechanism:
    """
    One independent error mechanism in a `DetectorErrorModel`.

    With probability `probability` the mechanism fires and adds
    a_1 * generators[0] + a_2 * generators[1] + ... to the detector and
    observable values, with each a_j uniform on Z_d. Zero is included, so a
    mechanism can fire and still change nothing.

    Attributes:
        probability (float): Firing probability pi.
        generators (list[dict[int, int]]): Sparse vectors {target: coefficient mod d}.
            Targets 0 to num_detectors - 1 are detectors, and target
            num_detectors + k is logical observable k.
        source (str): Name of the noise gate(s) the mechanism came from.
    """

    probability: float
    generators: list
    source: str = ""

    @property
    def rank(self) -> int:
        """Number of generators."""
        return len(self.generators)


@dataclass
class DetectorErrorModel:
    """
    A detector error model made of independent `ErrorMechanism`s.

    Build one from a circuit with `from_circuit`, or load one with
    `read_from_file`.

    Attributes:
        dimension (int): Qudit dimension d.
        num_detectors (int): Number of detectors.
        num_observables (int): Number of logical observables.
        mechanisms (list[ErrorMechanism]): The error mechanisms.
        detector_labels (list[str]): Label of each detector, "" if it has none.
        observable_labels (list[str]): Label of each logical observable, "" if it has none.
    """

    dimension: int
    num_detectors: int = 0
    num_observables: int = 0
    mechanisms: list = field(default_factory=list)
    detector_labels: list = field(default_factory=list)
    observable_labels: list = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_circuit(cls, circuit: Circuit, merge: bool = True, check_dimension_prime: bool = True) -> "DetectorErrorModel":
        """
        Builds the DEM of a noisy circuit.

        Every N1 or N2 gate with non-zero probability and a visible effect
        becomes one mechanism, so the size of the model does not depend on d.

        Args:
            circuit (Circuit): Circuit with N1/N2 noise and DETECTOR /
                LOGICAL_OBSERVABLE instructions.
            merge (bool): Merge rank-1 mechanisms that act on the same line of
                detector space, see `merge_lines`. Defaults to True.
            check_dimension_prime (bool): Raise if the dimension is not prime.
                Defaults to True.

        Returns:
            DetectorErrorModel: The compiled model.

        Raises:
            ValueError: If the dimension is not prime, an N2 gate uses
                `prob_dist`, or a detector expression is not linear.
        """
        d = circuit.dimension
        if check_dimension_prime and not _is_prime(d):
            raise ValueError("Compact qudit DEMs require a prime dimension (Z_d must be a field).")
        compiled = compile_unit_responses(circuit)
        dem = cls(dimension=d,
                  num_detectors=compiled.num_detectors,
                  num_observables=compiled.num_observables,
                  detector_labels=compiled.detector_labels,
                  observable_labels=compiled.observable_labels)
        for loc in compiled.locations:
            # A unit fault that no detector or observable sees adds nothing.
            generators = [g for g in loc.responses if g]
            if not generators or loc.probability <= 0.0:
                continue
            dem.mechanisms.append(ErrorMechanism(loc.subgroup_probability, generators, loc.source))
        if merge:
            dem.merge_lines()
        return dem

    def merge_lines(self) -> None:
        """
        Merges rank-1 mechanisms that act on the same line of detector space.

        Two generators are on the same line when one is a non-zero multiple of
        the other. Each generator is scaled so its first coefficient is 1, and
        mechanisms with equal scaled generators are merged with
        `merge_subgroup_probabilities`. Mechanisms of rank 2 or more are left
        as they are, since two of them almost never share a subgroup.
        """
        d = self.dimension
        merged: dict = {}
        order: list = []
        others: list = []
        for mech in self.mechanisms:
            if mech.rank != 1:
                others.append(mech)
                continue
            key, scaled = _canonical_line(mech.generators[0], d)
            if key in merged:
                prev = merged[key]
                prev.probability = merge_subgroup_probabilities(prev.probability, mech.probability)
                prev.source = prev.source + "+" + mech.source
            else:
                merged[key] = ErrorMechanism(mech.probability, [scaled], mech.source)
                order.append(key)
        self.mechanisms = [merged[k] for k in order] + others

    # -------------------------------------------------------- decorrelation
    def to_lines(self, max_lines_per_mechanism: int = 10 ** 6) -> "DetectorErrorModel":
        """
        Expands the model into independent rank-1 (line) mechanisms.

        A rank-k mechanism with probability pi becomes (d**k - 1) / (d - 1)
        line mechanisms with probability `line_probability(pi, d, k)`. Lines
        are enumerated in the error space Z_d^k and then mapped through the
        generators, so the expansion stays exact when the generators are
        linearly dependent. Lines with no effect are dropped and repeated
        lines are merged.

        The number of lines grows like d**(k-1), so this is only practical
        for small d. It is the form to use for decoders that expect one shift
        per mechanism, and for comparing with stim at d = 2.

        Args:
            max_lines_per_mechanism (int): Raise instead of expanding a
                mechanism into more lines than this. Defaults to 10**6.

        Returns:
            DetectorErrorModel: A new model in which every mechanism has rank 1.

        Raises:
            ValueError: If a mechanism needs more than `max_lines_per_mechanism` lines.
        """
        d = self.dimension
        out = DetectorErrorModel(d, self.num_detectors, self.num_observables, [],
                                 list(self.detector_labels), list(self.observable_labels))
        for mech in self.mechanisms:
            k = mech.rank
            num_lines = (d ** k - 1) // (d - 1)
            if num_lines > max_lines_per_mechanism:
                raise ValueError(f"{num_lines} lines for one mechanism; use the compact form for this dimension")
            pl = line_probability(mech.probability, d, k)
            for direction in _projective_points(d, k):
                combined: dict = {}
                for coeff, gen in zip(direction, mech.generators):
                    if coeff == 0:
                        continue
                    for t, v in gen.items():
                        combined[t] = (combined.get(t, 0) + coeff * v) % d
                combined = {t: v for t, v in combined.items() if v}
                if combined:
                    out.mechanisms.append(ErrorMechanism(pl, [combined], mech.source))
        out.merge_lines()
        return out

    # ---------------------------------------------------------------- sample
    def _flatten(self):
        """
        Packs the mechanisms into flat arrays for the numba sampler.

        Mechanism i has probability mech_prob[i] and generators
        gen_ptr[i]:gen_ptr[i + 1]. Generator g has entries
        ent_ptr[g]:ent_ptr[g + 1] in ent_tgt (targets) and ent_val (coefficients).
        """
        mech_prob = np.array([m.probability for m in self.mechanisms], dtype=np.float64)
        gen_ptr = np.zeros(len(self.mechanisms) + 1, dtype=np.int64)
        ent_ptr = [0]
        ent_tgt = []
        ent_val = []
        g = 0
        for i, m in enumerate(self.mechanisms):
            for gen in m.generators:
                for t, v in sorted(gen.items()):
                    ent_tgt.append(t)
                    ent_val.append(v)
                ent_ptr.append(len(ent_tgt))
                g += 1
            gen_ptr[i + 1] = g
        return (mech_prob, gen_ptr, np.array(ent_ptr, dtype=np.int64),
                np.array(ent_tgt, dtype=np.int64), np.array(ent_val, dtype=np.int64))

    def sample(self, shots: int, seed: int | None = None):
        """
        Samples detector and observable values.

        Mechanisms with the same probability are grouped, and within a group
        the sampler jumps straight to the next mechanism that fires. The cost
        grows with the number of firings, not with shots * len(mechanisms).

        Args:
            shots (int): Number of samples.
            seed (int, optional): Seed for the sampler. A random seed is used if None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Detector values with shape
                (shots, num_detectors) and observable values with shape
                (shots, num_observables), as int64 residues mod d.
        """
        mech_prob, gen_ptr, ent_ptr, ent_tgt, ent_val = self._flatten()
        order = np.argsort(mech_prob, kind="stable")
        sorted_prob = mech_prob[order]
        bounds = np.flatnonzero(np.diff(sorted_prob)) + 1
        class_start = np.concatenate(([0], bounds)).astype(np.int64)
        class_end = np.concatenate((bounds, [len(sorted_prob)])).astype(np.int64)
        class_prob = sorted_prob[class_start] if len(sorted_prob) else np.zeros(0)
        n_targets = self.num_detectors + self.num_observables
        out = np.zeros((shots, n_targets), dtype=np.int64)
        # The kernel seeds numba's own generator, which is separate from NumPy's global state.
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] & 0x7FFFFFFF)
        _sample_kernel(out, order.astype(np.int64), class_start, class_end, class_prob,
                       gen_ptr, ent_ptr, ent_tgt, ent_val, self.dimension, seed)
        return out[:, :self.num_detectors], out[:, self.num_detectors:]

    # -------------------------------------------------------------------- io
    def __str__(self) -> str:
        """The model in the format `read_from_file` reads, without the header comments."""
        lines = [f"DIMENSION {self.dimension}",
                 f"DETECTORS {self.num_detectors}",
                 f"OBSERVABLES {self.num_observables}"]
        for i, label in enumerate(self.detector_labels):
            if label:
                lines.append(f"DETECTOR D{i} {label}")
        for i, label in enumerate(self.observable_labels):
            if label:
                lines.append(f"LOGICAL_OBSERVABLE L{i} {label}")
        for m in self.mechanisms:
            gens = " | ".join(" ".join(self._target_name(t) + f"={v}" for t, v in sorted(g.items()))
                              for g in m.generators)
            tag = f" # {m.source}" if m.source else ""
            lines.append(f"ERROR({m.probability!r}) {gens}{tag}")
        return "\n".join(lines) + "\n"

    def _target_name(self, t: int) -> str:
        return f"D{t}" if t < self.num_detectors else f"L{t - self.num_detectors}"

    def write_to_file(self, path: str | Path, comment: str = "") -> None:
        """
        Writes the model to a text file.

        Args:
            path (str or Path): Output path.
            comment (str, optional): Extra text, written as `#` lines in the header.
        """
        header = [
            "# sdim compact qudit detector error model (format: qdem v1)",
            "# ERROR(pi) g_1 | g_2 | ... : with probability pi, add sum_j a_j g_j,",
            "#   a_j i.i.d. uniform on Z_d (identity included); mechanisms are independent.",
            "# Coefficients are residues mod DIMENSION.  Targets Dk are detectors, Lk observables.",
        ]
        if comment:
            header += ["# " + line for line in comment.splitlines()]
        Path(path).write_text("\n".join(header) + "\n#\n" + str(self))

    @classmethod
    def read_from_file(cls, path: str | Path) -> "DetectorErrorModel":
        """
        Reads a model written by `write_to_file`.

        Args:
            path (str or Path): Input path.

        Returns:
            DetectorErrorModel: The model in the file.

        Raises:
            ValueError: On a line that is not part of the format.
        """
        dem = None
        det_labels: dict = {}
        obs_labels: dict = {}
        mechanisms = []
        for raw in Path(path).read_text().splitlines():
            line, _, source = raw.partition("#")
            line, source = line.strip(), source.strip()
            if not line:
                continue
            head = line.split()[0]
            if head == "DIMENSION":
                dem = cls(int(line.split()[1]))
            elif head == "DETECTORS":
                dem.num_detectors = int(line.split()[1])
            elif head == "OBSERVABLES":
                dem.num_observables = int(line.split()[1])
            elif head == "DETECTOR":
                parts = line.split(maxsplit=2)
                det_labels[int(parts[1][1:])] = parts[2] if len(parts) > 2 else ""
            elif head == "LOGICAL_OBSERVABLE":
                parts = line.split(maxsplit=2)
                obs_labels[int(parts[1][1:])] = parts[2] if len(parts) > 2 else ""
            elif head.startswith("ERROR("):
                prob_text, rest = line[len("ERROR("):].split(")", 1)
                gens = []
                for chunk in rest.split("|"):
                    gen = {}
                    for item in chunk.split():
                        name, value = item.split("=")
                        idx = int(name[1:])
                        t = idx if name[0] == "D" else dem.num_detectors + idx
                        gen[t] = int(value) % dem.dimension
                    if gen:
                        gens.append(gen)
                mechanisms.append(ErrorMechanism(float(prob_text), gens, source))
            else:
                raise ValueError(f"unrecognized DEM line: {raw}")
        dem.mechanisms = mechanisms
        dem.detector_labels = [det_labels.get(i, "") for i in range(dem.num_detectors)]
        dem.observable_labels = [obs_labels.get(i, "") for i in range(dem.num_observables)]
        return dem


# ---------------------------------------------------------------------------
# Unit-response compilation


@dataclass
class NoiseLocation:
    """
    One N1 or N2 gate and the detector responses of its unit faults.

    Attributes:
        ir_index (int): Position of the gate in the program IR.
        gate_id (int): 17 for N1, 18 for N2.
        qudits (tuple): Qudits the gate acts on.
        channel (str): "d", "f" or "p" for N1, and "d2" for N2.
        probability (float): The gate's `prob` parameter.
        subgroup_probability (float): Probability of the equivalent subgroup mechanism.
        responses (list[dict[int, int]]): Sparse response of each unit fault. The
            order is X then Z on each qudit, with only X for "f" and only Z for "p".
        source (str): Name of the gate as written in DEM files.
    """
    ir_index: int
    gate_id: int
    qudits: tuple
    channel: str
    probability: float
    subgroup_probability: float
    responses: list
    source: str


@dataclass
class CompiledResponses:
    """
    Output of `compile_unit_responses`.

    Attributes:
        num_detectors (int): Number of detectors.
        num_observables (int): Number of logical observables.
        detector_labels (list[str]): Label of each detector.
        observable_labels (list[str]): Label of each logical observable.
        locations (list[NoiseLocation]): One entry per N1/N2 gate, in circuit order.
    """
    num_detectors: int
    num_observables: int
    detector_labels: list
    observable_labels: list
    locations: list


def _detector_coefficients(detector_info, dimension: int):
    """
    Reads the linear coefficients of each detector and logical observable.

    sdim compiles each DETECTOR / LOGICAL_OBSERVABLE expression into a
    function of its measurement records. Evaluating it on unit vectors gives
    the coefficient of each record, and one random input checks that the
    expression really is linear.

    Returns:
        tuple: Detector coefficients and observable coefficients (lists of
            {record index: coefficient mod d}), then detector labels and
            observable labels.

    Raises:
        ValueError: If an expression has a constant term or is not linear.
    """
    dets, obs, det_labels, obs_labels = [], [], [], []
    for unique_index, label, arguments, is_logical in detector_info.detector_data:
        fn = detector_info.detector_functions[unique_index]
        n = len(arguments)
        base = int(fn([0] * n)) % dimension
        if base != 0:
            raise ValueError(f"detector {label!r} has a non-zero constant term")
        coeffs = {}
        for j, rec in enumerate(arguments):
            unit = [0] * n
            unit[j] = 1
            c = int(fn(unit)) % dimension
            if c:
                coeffs[int(rec)] = (coeffs.get(int(rec), 0) + c) % dimension
        # Check linearity on one random input. Skipped when a record appears twice.
        rng = np.random.default_rng(1234 + unique_index)
        vals = [int(v) for v in rng.integers(0, dimension, size=n)]
        expected = sum(coeffs.get(int(r), 0) * v for r, v in zip(arguments, vals)) % dimension
        if len(set(arguments)) == n and int(fn(vals)) % dimension != expected:
            raise ValueError(f"detector {label!r} is not linear in its records")
        (obs if is_logical else dets).append(coeffs)
        (obs_labels if is_logical else det_labels).append(label or "")
    return dets, obs, det_labels, obs_labels


def compile_unit_responses(circuit: Circuit) -> CompiledResponses:
    """
    Computes the detector response of every unit fault in a circuit.

    For each N1/N2 gate, an X or Z fault is placed right after the gate on
    each qudit it acts on, then pushed through the rest of the circuit with
    the same update rules as `sdim.program.simulate_frame`. The response is
    the change the fault causes in each detector and logical observable.

    Args:
        circuit (Circuit): The noisy circuit.

    Returns:
        CompiledResponses: Detector counts and labels, plus one `NoiseLocation`
            per noise gate.

    Raises:
        ValueError: If an N2 gate uses `prob_dist`, or a detector expression
            is not linear.
    """
    d = circuit.dimension
    program = Program(circuit)
    ir_array, _, detector_info = program._build_ir([circuit], 1)
    gid = np.ascontiguousarray(ir_array["gate_id"], dtype=np.int64)
    qa = np.ascontiguousarray(ir_array["qudit_index"], dtype=np.int64)
    qb = np.ascontiguousarray(ir_array["target_index"], dtype=np.int64)
    n_ops = len(gid)
    n_qudits = circuit.num_qudits

    # Measurement record index of each measuring op (sdim counts M and M_X only).
    is_meas = (gid == _M) | (gid == _M_X)
    rec_of_op = np.full(n_ops, -1, dtype=np.int64)
    rec_of_op[is_meas] = np.arange(int(is_meas.sum()), dtype=np.int64)
    n_recs = int(is_meas.sum())

    # For each qudit, the IR ops that can change its frame, so the kernel only
    # visits ops on qudits a fault has reached. posa / posb hold an op's index
    # in the flattened lists of its first / second qudit.
    frame_mask = np.isin(gid, np.array(sorted(_FRAME_GATES)))
    per_qudit = [[] for _ in range(n_qudits)]
    posa = np.full(n_ops, -1, dtype=np.int64)
    posb = np.full(n_ops, -1, dtype=np.int64)
    for i in np.flatnonzero(frame_mask):
        a = int(qa[i])
        posa[i] = len(per_qudit[a])
        per_qudit[a].append(i)
        if qb[i] >= 0 and gid[i] in (_CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP):
            b = int(qb[i])
            posb[i] = len(per_qudit[b])
            per_qudit[b].append(i)
        else:
            # Single-qudit op, so the kernel takes its one-qudit branch.
            qb[i] = -1
    qptr = np.zeros(n_qudits + 1, dtype=np.int64)
    qptr[1:] = np.cumsum([len(l) for l in per_qudit])
    qops = np.array([i for l in per_qudit for i in l], dtype=np.int64)
    frame_ops = np.flatnonzero(frame_mask)
    posa[frame_ops] += qptr[qa[frame_ops]]
    two = frame_ops[qb[frame_ops] >= 0]
    posb[two] += qptr[qb[two]]

    # For each measurement record, the detectors / observables that use it and their coefficients.
    dets, obs, det_labels, obs_labels = _detector_coefficients(detector_info, d)
    n_det = len(dets)
    incidence = [[] for _ in range(n_recs)]
    for t, coeffs in enumerate(dets):
        for r, c in coeffs.items():
            incidence[r].append((t, c))
    for k, coeffs in enumerate(obs):
        for r, c in coeffs.items():
            incidence[r].append((n_det + k, c))
    rptr = np.zeros(n_recs + 1, dtype=np.int64)
    rptr[1:] = np.cumsum([len(l) for l in incidence])
    rtgt = np.array([t for l in incidence for t, _ in l], dtype=np.int64)
    rcoef = np.array([c for l in incidence for _, c in l], dtype=np.int64)

    # Noise gates in IR order and the unit faults to probe for each.
    # _build_ir drops identity gates (id 0), so they don't count toward ir_index.
    # A probe (qudit, 0) is an X fault and (qudit, 1) is a Z fault.
    locations = []
    probe_op, probe_qudit, probe_kind = [], [], []
    ir_index = -1
    for instr in circuit.operations:
        if instr.gate_id == 0:
            continue
        ir_index += 1
        if instr.gate_id == _N1:
            channel = instr.params.get("noise_channel", instr.params.get("channel", "d"))
            p = float(instr.params.get("prob", 0.0))
            q0 = int(instr.qudit_index)
            kinds = {"d": [(q0, 0), (q0, 1)], "f": [(q0, 0)], "p": [(q0, 1)]}[channel]
            rank = len(kinds)
            pi = p / (1.0 - float(d) ** (-rank))
            qudits = (q0,)
            name = f"N1[{channel}]@{ir_index}:q{q0}"
        elif instr.gate_id == _N2:
            if instr.params.get("prob_dist", None) is not None:
                raise ValueError("Compact DEMs support N2 with prob=... (uniform non-identity depolarizing); "
                                 "use sdim.dem_legacy for arbitrary prob_dist at small d.")
            p = float(instr.params.get("prob", 0.0))
            q0, q1 = int(instr.qudit_index), int(instr.target_index)
            kinds = [(q0, 0), (q0, 1), (q1, 0), (q1, 1)]
            channel = "d2"
            pi = p / (1.0 - float(d) ** (-4))
            qudits = (q0, q1)
            name = f"N2@{ir_index}:q{q0},q{q1}"
        else:
            continue
        first_probe = len(probe_op)
        for (qq, kind) in kinds:
            probe_op.append(ir_index)
            probe_qudit.append(qq)
            probe_kind.append(kind)
        locations.append(NoiseLocation(ir_index, instr.gate_id, qudits, channel, p, pi,
                                       list(range(first_probe, len(probe_op))), name))

    probe_op = np.array(probe_op, dtype=np.int64)
    probe_qudit = np.array(probe_qudit, dtype=np.int64)
    probe_kind = np.array(probe_kind, dtype=np.int64)
    n_targets = n_det + len(obs)
    responses = _run_probes(gid, qa, qb, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                            probe_op, probe_qudit, probe_kind, d, n_targets)
    # Until now loc.responses held probe indices. Swap in the actual responses.
    for loc in locations:
        loc.responses = [responses[i] for i in loc.responses]
    return CompiledResponses(n_det, len(obs), det_labels, obs_labels, locations)


def _run_probes(gid, qa, qb, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                probe_op, probe_qudit, probe_kind, d, n_targets, chunk=1 << 15):
    """
    Runs `_probe_kernel` over all probes in chunks.

    The kernel writes into preallocated output buffers sized for `cap` entries
    per probe. When it returns False the chunk is rerun with buffers 4x larger.

    Returns:
        list[dict[int, int]]: Sparse response {target: coefficient} of each probe.
    """
    out = []
    for start in range(0, len(probe_op), chunk):
        stop = min(start + chunk, len(probe_op))
        cap = 32
        while True:
            ptr = np.zeros(stop - start + 1, dtype=np.int64)
            tgt = np.zeros((stop - start) * cap, dtype=np.int64)
            val = np.zeros((stop - start) * cap, dtype=np.int64)
            ok = _probe_kernel(gid, qa, qb, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                               probe_op[start:stop], probe_qudit[start:stop], probe_kind[start:stop],
                               d, n_targets, ptr, tgt, val)
            if ok:
                break
            cap *= 4
        for i in range(stop - start):
            a, b = ptr[i], ptr[i + 1]
            out.append({int(t): int(v) for t, v in zip(tgt[a:b], val[a:b])})
    return out


@njit(cache=True)
def _probe_kernel(gid, qa, qb, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                  probe_op, probe_qudit, probe_kind, d, n_targets, out_ptr, out_tgt, out_val):
    """
    Pushes unit faults through the circuit, one probe at a time.

    The frame update rules are the ones in `sdim.program.simulate_frame`, with
    one difference. Measurement and reset set the Z frame to 0 instead of
    re-randomizing it. The random part cancels in every deterministic
    detector, so what is left is the fault's deterministic response.

    Only qudits the fault has reached are tracked. Each one gets a slot with
    the qudit index (sq), its X and Z frame (sx, sz) and its position in that
    qudit's op list (scur). Each step advances the slot whose next op comes
    first in the circuit. A slot is freed once its frame is back to zero, and
    the probe ends when no slots are left or no ops remain.

    Returns:
        bool: False if the output buffers fill up or a fault reaches more than
            512 qudits at once, True otherwise.
    """
    smax = 512
    sq = np.empty(smax, dtype=np.int64)
    sx = np.empty(smax, dtype=np.int64)
    sz = np.empty(smax, dtype=np.int64)
    scur = np.empty(smax, dtype=np.int64)
    acc = np.zeros(n_targets, dtype=np.int64)
    touched = np.empty(n_targets, dtype=np.int64)
    is_touched = np.zeros(n_targets, dtype=np.bool_)
    cap_total = out_tgt.shape[0]
    w = 0
    big = 1 << 62
    for p in range(probe_op.shape[0]):
        u = probe_qudit[p]
        nslot = 1
        sq[0] = u
        sx[0] = 1 if probe_kind[p] == 0 else 0
        sz[0] = 1 if probe_kind[p] == 1 else 0
        lo = qptr[u]
        hi = qptr[u + 1]
        # first op strictly after the noise op
        while lo < hi:
            mid = (lo + hi) // 2
            if qops[mid] <= probe_op[p]:
                lo = mid + 1
            else:
                hi = mid
        scur[0] = lo
        nt = 0
        while nslot > 0:
            best = -1
            bestop = big
            for s in range(nslot):
                if scur[s] < qptr[sq[s] + 1]:
                    o = qops[scur[s]]
                    if o < bestop:
                        bestop = o
                        best = s
            if best < 0:
                break
            op = bestop
            g = gid[op]
            if qb[op] < 0:
                x = sx[best]
                z = sz[best]
                if g == 5:  # H
                    nx = (d - z) % d
                    nz = x
                elif g == 6:  # H_INV
                    nx = z
                    nz = (d - x) % d
                elif g == 7:  # P
                    nx = x
                    nz = (z + x) % d
                elif g == 8:  # P_INV
                    nx = x
                    nz = (z + d - x) % d
                elif g == 14 or g == 15:  # M, M_X
                    if g == 15:
                        nx = z
                    else:
                        nx = x
                    nz = 0
                    if nx != 0:
                        r = rec_of_op[op]
                        for k in range(rptr[r], rptr[r + 1]):
                            t = rtgt[k]
                            acc[t] = (acc[t] + rcoef[k] * nx) % d
                            if not is_touched[t]:
                                is_touched[t] = True
                                touched[nt] = t
                                nt += 1
                elif g == 16:  # RESET
                    nx = 0
                    nz = 0
                else:
                    nx = x
                    nz = z
                sx[best] = nx
                sz[best] = nz
                scur[best] += 1
            else:
                # Two-qudit gate. Open a slot for whichever qudit the fault hasn't reached yet.
                a = qa[op]
                b = qb[op]
                sa = -1
                sb = -1
                for s in range(nslot):
                    if sq[s] == a:
                        sa = s
                    elif sq[s] == b:
                        sb = s
                if sa < 0:
                    if nslot >= smax:
                        return False
                    sa = nslot
                    sq[sa] = a
                    sx[sa] = 0
                    sz[sa] = 0
                    scur[sa] = posa[op]
                    nslot += 1
                if sb < 0:
                    if nslot >= smax:
                        return False
                    sb = nslot
                    sq[sb] = b
                    sx[sb] = 0
                    sz[sb] = 0
                    scur[sb] = posb[op]
                    nslot += 1
                xa = sx[sa]
                za = sz[sa]
                xb = sx[sb]
                zb = sz[sb]
                if g == 9:  # CNOT
                    xb = (xb + xa) % d
                    za = (za + d - zb) % d
                elif g == 10:  # CNOT_INV
                    xb = (xb + d - xa) % d
                    za = (za + zb) % d
                elif g == 11:  # CZ
                    nzb = (zb + xa) % d
                    za = (za + xb) % d
                    zb = nzb
                elif g == 12:  # CZ_INV
                    nzb = (zb + d - xa) % d
                    za = (za + d - xb) % d
                    zb = nzb
                elif g == 13:  # SWAP
                    xa, xb = xb, xa
                    za, zb = zb, za
                sx[sa] = xa
                sz[sa] = za
                sx[sb] = xb
                sz[sb] = zb
                scur[sa] += 1
                scur[sb] += 1
            # Free the slots whose frame went back to zero.
            s = 0
            while s < nslot:
                if sx[s] == 0 and sz[s] == 0:
                    nslot -= 1
                    sq[s] = sq[nslot]
                    sx[s] = sx[nslot]
                    sz[s] = sz[nslot]
                    scur[s] = scur[nslot]
                else:
                    s += 1
        # Write out this probe's non-zero totals and clear the accumulator.
        out_ptr[p] = w
        for i in range(nt):
            t = touched[i]
            if acc[t] != 0:
                if w >= cap_total:
                    return False
                out_tgt[w] = t
                out_val[w] = acc[t]
                w += 1
            acc[t] = 0
            is_touched[t] = False
    out_ptr[probe_op.shape[0]] = w
    return True


@njit(cache=True)
def _sample_kernel(out, order, class_start, class_end, class_prob, gen_ptr, ent_ptr, ent_tgt, ent_val, d, seed):
    """
    Fills `out` with sampled detector / observable values. See `DetectorErrorModel.sample`.

    A class of n mechanisms that share probability pi is n * shots independent
    coin flips. The number of misses before the next hit is geometric,
    floor(log(u) / log(1 - pi)), so the loop jumps from one firing to the next
    instead of flipping every coin.
    """
    np.random.seed(seed)
    shots = out.shape[0]
    for c in range(class_start.shape[0]):
        pi = class_prob[c]
        if pi <= 0.0:
            continue
        lo = class_start[c]
        n = class_end[c] - lo
        total = n * shots
        log_keep = np.log1p(-pi) if pi < 1.0 else -np.inf
        pos = -1
        while True:
            if pi >= 1.0:
                pos += 1
            else:
                u = np.random.random()
                while u <= 0.0:
                    u = np.random.random()
                pos += 1 + np.int64(np.floor(np.log(u) / log_keep))
            if pos >= total:
                break
            shot = pos // n
            m = order[lo + pos % n]
            for gi in range(gen_ptr[m], gen_ptr[m + 1]):
                a = np.random.randint(0, d)
                if a == 0:
                    continue
                for k in range(ent_ptr[gi], ent_ptr[gi + 1]):
                    t = ent_tgt[k]
                    out[shot, t] = (out[shot, t] + a * ent_val[k]) % d


def _canonical_line(gen: dict, d: int):
    """Scales a sparse vector so its first entry is 1. Returns (hashable key, scaled dict)."""
    items = sorted(gen.items())
    inv = pow(items[0][1], -1, d)
    scaled = {t: (v * inv) % d for t, v in items}
    return tuple(sorted(scaled.items())), scaled


def _projective_points(d: int, k: int):
    """Yields one point on each line through the origin of Z_d^k, scaled so its first non-zero coordinate is 1."""
    for lead in range(k):
        for tail in itertools.product(range(d), repeat=k - lead - 1):
            yield (0,) * lead + (1,) + tail
