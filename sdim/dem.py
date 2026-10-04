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
- Every detector and observable must be deterministic without noise.
  `from_circuit` checks this.
- Noise probabilities can go up to the fully mixing value (`1 - 1/d` for
  N1 'f' and 'p', `1 - 1/d**2` for N1 'd', `1 - 1/d**4` for N2), where
  pi = 1. Stronger noise biases away from the identity and has no subgroup
  form, so `from_circuit` rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import contextlib
import copy
import dis
import gc
import itertools
import math
import threading
import types

import numba
import numpy as np
from numba import njit

from .circuit import Circuit
from .program import Program, _detector_mod

# Gate ids, in the order GateData registers the gates.
_H, _H_INV, _P, _P_INV = 5, 6, 7, 8
_CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP = 9, 10, 11, 12, 13
_M, _M_X, _RESET = 14, 15, 16
_N1, _N2, _DETECTOR, _OBSERVABLE = 17, 18, 19, 20
_MUL = 22
_FRAME_GATES = {_H, _H_INV, _P, _P_INV, _CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP, _M, _M_X, _RESET, _MUL}
_TWO_QUDIT_FRAME_GATES = (_CNOT, _CNOT_INV, _CZ, _CZ_INV, _SWAP)

# Shots per independently seeded sampler block.
_SAMPLE_CHUNK = 256
# Below this estimated amount of work, sample on the calling thread rather than start worker threads.
_SAMPLE_PARALLEL_WORK = 200_000
# Sampler tasks per thread. Each task is a run of consecutive blocks; more tasks balance the load better.
_SAMPLE_TASKS_PER_THREAD = 8
# The sampler's arithmetic needs d < 2**31 (see `_sample_chunks`).
_MAX_SAMPLE_DIMENSION = 2 ** 31 - 1
# With fewer unit-fault probes than this, `from_circuit` propagates them on the calling thread.
_PROBE_PARALLEL_MIN = 2048


def _thread_count() -> int:
    """
    Number of threads for the unit-fault pass and the sampler.

    This is numba's thread count: `numba.get_num_threads()` once numba's own
    thread pool is running (so `numba.set_num_threads` applies), and
    otherwise `numba.config.NUMBA_NUM_THREADS`, which the NUMBA_NUM_THREADS
    environment variable sets. Reading it never starts numba's pool.
    """
    try:
        from numba.np.ufunc import parallel as numba_parallel
        if getattr(numba_parallel, "_is_initialized", False):
            return max(1, int(numba.get_num_threads()))
    except Exception:
        pass
    try:
        return max(1, int(numba.config.NUMBA_NUM_THREADS))
    except Exception:
        return 1


def _run_tasks(task, n_tasks: int, n_threads: int) -> None:
    """
    Calls task(i) for every i in range(n_tasks), on up to n_threads threads.

    Each task should spend its time in a `nogil` numba kernel so the threads
    run at the same time. Threads take the next task as they finish one, and
    the calling thread works too. Every task writes only its own output, so
    the result does not depend on the number of threads or on scheduling.

    These are plain Python threads, started for this call and joined before it
    returns. numba's `parallel=True` thread pool is not used: under the GNU
    OpenMP layer it aborts processes forked after it has run, and under the
    workqueue layer it aborts when two Python threads use it at once.

    Raises:
        BaseException: The first exception a task raised. The other threads
            stop taking tasks, and all of them are joined first.
    """
    n_threads = min(n_threads, n_tasks)
    if n_threads <= 1:
        for i in range(n_tasks):
            task(i)
        return
    lock = threading.Lock()
    next_task = [0]
    failures = []

    def work():
        try:
            while True:
                with lock:
                    i = next_task[0]
                    if i >= n_tasks or failures:
                        return
                    next_task[0] = i + 1
                task(i)
        except BaseException as e:
            with lock:
                failures.append(e)

    threads = []
    for _ in range(n_threads - 1):
        thread = threading.Thread(target=work, name="sdim-dem-worker", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            # No more threads can be started (for example at interpreter shutdown). The threads
            # already running and this one share the remaining tasks.
            break
        threads.append(thread)
    try:
        work()
    finally:
        try:
            for thread in threads:
                thread.join()
        except BaseException as e:
            # Interrupted while waiting: make the workers stop after their current task.
            with lock:
                failures.append(e)
            raise
    if failures:
        raise failures[0]


@contextlib.contextmanager
def _gc_paused():
    """
    Pauses Python's cyclic garbage collector.

    Building a model creates many small dicts and mechanisms and no reference
    cycles, and every full collection that this allocation triggers walks the
    whole heap. Pausing the collector changes nothing but the time taken.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


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
        ValueError: If pi > 1.
    """
    if pi > 1.0:
        raise ValueError(f"pi = {pi} is above 1")
    if pi == 1.0:
        # Uniform on the subgroup: every line is uniform too, and their sum is uniform again.
        return 1.0
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
    if any(pi >= 1.0 for pi in pis):
        return 1.0
    log_keep = sum(math.log1p(-pi) for pi in pis)
    return -math.expm1(log_keep)


def _merge_pair(a: float, b: float) -> float:
    """
    `merge_subgroup_probabilities(a, b)` for two floats, bit for bit.

    For 0 < a, b < 1 neither log1p is -0.0, so summing the two terms directly
    gives exactly what `sum` does; other inputs go through the general function.
    """
    if 0.0 < a < 1.0 and 0.0 < b < 1.0:
        return -math.expm1(math.log1p(-a) + math.log1p(-b))
    return merge_subgroup_probabilities(a, b)


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
                `prob_dist`, a noise probability is above the fully mixing
                value, or a detector or observable is not linear in its
                records or not deterministic without noise.
        """
        d = circuit.dimension
        if check_dimension_prime and not _is_prime(d):
            raise ValueError("Compact qudit DEMs require a prime dimension (Z_d must be a field).")
        with _gc_paused():
            compiled = _compile(circuit)
            dem = cls(dimension=d,
                      num_detectors=compiled.num_detectors,
                      num_observables=compiled.num_observables,
                      detector_labels=compiled.detector_labels,
                      observable_labels=compiled.observable_labels)
            # Same result as keeping every noise gate with prob > 0 and a visible unit fault, then
            # calling merge_lines(), but built from flat arrays.
            dem.mechanisms = compiled.mechanisms(merge)
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
        sources: dict = {}
        others: list = []
        with _gc_paused():
            for mech in self.mechanisms:
                if mech.rank != 1:
                    others.append(mech)
                    continue
                key, scaled = _canonical_line(mech.generators[0], d)
                prev = merged.get(key)
                if prev is not None:
                    prev.probability = _merge_pair(prev.probability, mech.probability)
                    parts = sources.get(key)
                    if parts is None:
                        sources[key] = [prev.source, mech.source]
                    else:
                        parts.append(mech.source)
                else:
                    merged[key] = ErrorMechanism(mech.probability, [scaled], mech.source)
            # Joining once gives the same string as appending "+" + source on every merge.
            for key, parts in sources.items():
                merged[key].source = "+".join(parts)
            self.mechanisms = list(merged.values()) + others

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
        mechs = self.mechanisms
        with _gc_paused():
            # The checks and line probabilities, mechanism by mechanism.
            pls = []
            for mech in mechs:
                k = mech.rank
                num_lines = (d ** k - 1) // (d - 1)
                if num_lines > max_lines_per_mechanism:
                    raise ValueError(f"{num_lines} lines for one mechanism; use the compact form for this dimension")
                pls.append(line_probability(mech.probability, d, k))
            lines = _lines_from_arrays(mechs, pls, d)
            if lines is not None:
                out.mechanisms = lines
                return out
            # Targets or coefficients that are not plain ints: expand one dict at a time.
            for mech, pl in zip(mechs, pls):
                for direction in _projective_points(d, mech.rank):
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

        Returns mech_prob (probability of each mechanism), n_gens (number of
        generators of each mechanism), sizes (number of entries of each
        generator), and ent_tgt / ent_val (targets and coefficients of all
        entries, generator by generator, in dict order).
        """
        mechs = self.mechanisms
        mech_prob = np.array([m.probability for m in mechs], dtype=np.float64)
        n_gens = np.fromiter((len(m.generators) for m in mechs), dtype=np.int64, count=len(mechs))
        gens = [g for m in mechs for g in m.generators]
        sizes = np.fromiter(map(len, gens), dtype=np.int64, count=len(gens))
        n_ent = int(sizes.sum())
        ent_tgt = np.fromiter(itertools.chain.from_iterable(gens), dtype=np.int64, count=n_ent)
        ent_val = np.fromiter(itertools.chain.from_iterable(g.values() for g in gens), dtype=np.int64, count=n_ent)
        return mech_prob, n_gens, sizes, ent_tgt, ent_val

    def sample(self, shots: int, seed: int | None = None):
        """
        Samples detector and observable values.

        Mechanisms are grouped into bins of similar probability. Within a bin
        the sampler jumps straight to the next candidate firing with a
        geometric skip, and a mechanism whose probability is below the bin's
        maximum keeps each candidate with probability pi / pi_max. The cost
        grows with the number of firings, not with shots * len(mechanisms).

        Shots are split into fixed blocks of 256, and large jobs spread the
        blocks over several threads (numba's thread count, see
        `_thread_count`). Each block has its own random stream
        (xoshiro256**) seeded from `seed` through NumPy's SeedSequence, so a
        given seed gives the same samples whatever the number of threads.

        Args:
            shots (int): Number of samples.
            seed (int, optional): Seed for the sampler. Fresh entropy is used if None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Detector values with shape
                (shots, num_detectors) and observable values with shape
                (shots, num_observables), as int64 residues mod d.

        Raises:
            ValueError: If the dimension is not between 1 and 2**31 - 1, a
                mechanism probability is NaN, or a generator refers to a
                target outside the model.
        """
        nd = self.num_detectors
        det = np.zeros((shots, nd), dtype=np.int64)
        obs = np.zeros((shots, self.num_observables), dtype=np.int64)
        if not self.mechanisms or shots == 0:
            return det, obs
        d = int(self.dimension)
        if not 1 <= d <= _MAX_SAMPLE_DIMENSION:
            raise ValueError(f"sample() needs a dimension between 1 and 2**31 - 1, not {d}")
        mech_prob, n_gens, sizes, ent_tgt, ent_val = self._flatten()
        n_targets = nd + self.num_observables
        montgomery = d % 2 == 1
        # int32 pack entries halve the sampler's memory traffic; they hold targets below n_targets and
        # residues below d.
        pack_like = np.empty(0, dtype=np.int32 if n_targets <= np.iinfo(np.int32).max else np.int64)
        status, pack, info, always_off, bin_ptr, bin_pmax, bin_log_keep, cost = _sample_plan(
            mech_prob, mech_prob.view(np.int64), n_gens, sizes, ent_tgt, ent_val, d, n_targets, montgomery,
            pack_like)
        if status == -2:
            raise ValueError("a mechanism probability is NaN")
        if status >= 0:
            raise ValueError(f"a generator refers to target {int(ent_tgt[status])}, but the model has "
                             f"{nd} detectors and {self.num_observables} observables")
        n_chunks = -(-shots // _SAMPLE_CHUNK)
        states = np.random.SeedSequence(seed).generate_state(4 * n_chunks, dtype=np.uint64).reshape(n_chunks, 4)
        # xoshiro256** must not start from the all-zero state.
        states[~states.any(axis=1), 0] = 1
        # -d^-1 mod 2**32 for Montgomery multiplication (odd d); 0 selects plain % for even d.
        nprime = np.uint64((-pow(d, -1, 1 << 32)) % (1 << 32) if montgomery else 0)
        thresh = np.uint64((1 << 32) % d)
        info_p = info.view(np.float64)
        n_threads = _thread_count() if n_chunks > 1 and shots * cost > _SAMPLE_PARALLEL_WORK else 1
        n_tasks = min(n_chunks, n_threads * _SAMPLE_TASKS_PER_THREAD) if n_threads > 1 else 1
        bounds = [n_chunks * i // n_tasks for i in range(n_tasks + 1)]

        def task(i):
            _sample_chunks(bounds[i], bounds[i + 1], det, obs, _SAMPLE_CHUNK, states, bin_ptr, bin_pmax,
                           bin_log_keep, info, info_p, always_off, pack, d, thresh, nprime)

        _run_tasks(task, n_tasks, n_threads)
        return det, obs

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
            lines.append(f"ERROR({float(m.probability)!r}) {gens}{tag}")
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
        for label in list(self.detector_labels) + list(self.observable_labels):
            if "\n" in (label or ""):
                raise ValueError(f"label {label!r} contains a newline")
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
            ValueError: If the file is not in the format, or refers to a
                detector or observable outside the declared counts.
        """
        dem = None
        det_labels: dict = {}
        obs_labels: dict = {}
        mechanisms = []
        for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            head = stripped.split()[0]
            if head != "DIMENSION" and dem is None:
                raise ValueError(f"line {number}: DIMENSION must come first")
            if head in ("DETECTOR", "LOGICAL_OBSERVABLE"):
                # Labels are read verbatim, so they may contain '#'.
                parts = stripped.split(maxsplit=2)
                prefix, labels = ("D", det_labels) if head == "DETECTOR" else ("L", obs_labels)
                if len(parts) < 2 or not parts[1].startswith(prefix) or not parts[1][1:].isdigit():
                    raise ValueError(f"line {number}: expected {head} {prefix}<index> <label>")
                labels[int(parts[1][1:])] = parts[2] if len(parts) > 2 else ""
                continue
            line, _, source = raw.partition("#")
            line, source = line.strip(), source.strip()
            if head == "DIMENSION":
                dem = cls(int(line.split()[1]))
            elif head == "DETECTORS":
                dem.num_detectors = int(line.split()[1])
            elif head == "OBSERVABLES":
                dem.num_observables = int(line.split()[1])
            elif head.startswith("ERROR("):
                prob_text, rest = line[len("ERROR("):].split(")", 1)
                probability = float(prob_text)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"line {number}: probability {probability} is not in [0, 1]")
                gens = []
                for chunk in rest.split("|"):
                    gen = {}
                    for item in chunk.split():
                        name, _, value = item.partition("=")
                        if name[:1] not in ("D", "L") or not name[1:].isdigit() or not value:
                            raise ValueError(f"line {number}: bad target {item!r}")
                        idx = int(name[1:])
                        limit = dem.num_detectors if name[0] == "D" else dem.num_observables
                        if idx >= limit:
                            raise ValueError(f"line {number}: {name} is out of range")
                        t = idx if name[0] == "D" else dem.num_detectors + idx
                        gen[t] = (gen.get(t, 0) + int(value)) % dem.dimension
                    gen = {t: v for t, v in gen.items() if v}
                    if gen:
                        gens.append(gen)
                mechanisms.append(ErrorMechanism(probability, gens, source))
            else:
                raise ValueError(f"line {number}: unrecognized DEM line: {raw}")
        if dem is None:
            raise ValueError("no DIMENSION line")
        dem.mechanisms = [m for m in mechanisms if m.generators]
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


class _LinearForm:
    """
    An affine form c_0 + c_1 rec[0] + ... + c_n rec[n - 1], coefficients mod d.

    `_detector_coefficients` calls a compiled detector expression once on a
    list of these forms. The operations below are the only ones defined, and
    each one turns values congruent mod d to its operands into a value
    congruent mod d to its result: + and -, multiplication by a constant (or
    by a form with no record terms), and % by a non-zero multiple of d. So
    when the call succeeds, the expression is congruent mod d to the returned
    form for every integer input. Any other operation raises TypeError, and
    the caller falls back to evaluating the expression on numeric probes.
    """

    __slots__ = ("c", "d")

    def __init__(self, c: tuple, d: int):
        self.c = c
        self.d = d

    def _coeffs(self, other) -> tuple:
        if type(other) is _LinearForm and len(other.c) == len(self.c):
            return other.c
        if type(other) is int:
            return (other % self.d,) + (0,) * (len(self.c) - 1)
        raise TypeError("not a linear operation")

    def __add__(self, other):
        d = self.d
        return _LinearForm(tuple((a + b) % d for a, b in zip(self.c, self._coeffs(other))), d)

    __radd__ = __add__

    def __sub__(self, other):
        d = self.d
        return _LinearForm(tuple((a - b) % d for a, b in zip(self.c, self._coeffs(other))), d)

    def __rsub__(self, other):
        d = self.d
        return _LinearForm(tuple((b - a) % d for a, b in zip(self.c, self._coeffs(other))), d)

    def __neg__(self):
        d = self.d
        return _LinearForm(tuple((-a) % d for a in self.c), d)

    def __pos__(self):
        return self

    def __mul__(self, other):
        d = self.d
        if type(other) is int:
            k = other % d
            form = self
        elif type(other) is _LinearForm and len(other.c) == len(self.c):
            if not any(other.c[1:]):
                k, form = other.c[0], self
            elif not any(self.c[1:]):
                k, form = self.c[0], other
            else:
                raise TypeError("product of two records")
        else:
            raise TypeError("not a linear operation")
        return _LinearForm(tuple((a * k) % d for a in form.c), d)

    __rmul__ = __mul__

    def __mod__(self, other):
        if type(other) is int and other != 0 and other % self.d == 0:
            return self
        raise TypeError("not a linear operation")

    # Anything that could branch on a value or turn it into something else is refused.
    def _refuse(self, *args):
        raise TypeError("not a linear operation")

    __bool__ = __index__ = __int__ = __float__ = __str__ = __format__ = _refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __hash__ = None


# Bytecode a detector expression may contain for the single symbolic evaluation: loading the
# record list and integer constants, indexing, unary minus, and the binary operators +, -, * and %.
# Anything else (calls, names, branches, comparisons, ...) takes the numeric path.
_LINEAR_OPNAMES = frozenset({
    "RESUME", "NOP", "CACHE", "EXTENDED_ARG", "RETURN_VALUE",
    "LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "LOAD_CONST", "LOAD_SMALL_INT",
    "BINARY_SUBSCR", "UNARY_NEGATIVE", "BINARY_OP",
    "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY", "BINARY_MODULO",
})
_LINEAR_BINARY_OPS = frozenset({"+", "-", "*", "%", "[]"})


_CALL_OPNAMES = frozenset({"LOAD_GLOBAL", "PUSH_NULL", "PRECALL", "CALL"})


def _is_straight_line_arithmetic(fn) -> bool:
    """True if `fn` is a one-argument function whose bytecode only uses `_LINEAR_OPNAMES`.

    sdim.program compiles detectors as `lambda rec : _detector_mod((expr), d)`, and
    `_detector_mod(x, d)` is `x % d` for anything but int64 arrays. Calls to that one helper are
    allowed too, when the name really refers to sdim's own function.
    """
    if type(fn) is not types.FunctionType or fn.__defaults__ or fn.__kwdefaults__ or fn.__closure__:
        return False
    code = fn.__code__
    wraps_mod = (code.co_names == ("_detector_mod",)
                 and fn.__globals__.get("_detector_mod") is _detector_mod)
    if (code.co_argcount != 1 or code.co_kwonlyargcount or (code.co_names and not wraps_mod) or code.co_freevars
            or code.co_cellvars or code.co_flags & (0x04 | 0x08)):   # *args, **kwargs
        return False
    for ins in dis.get_instructions(code):
        name = ins.opname
        if wraps_mod and name in _CALL_OPNAMES:
            if name == "LOAD_GLOBAL" and ins.argval != "_detector_mod":
                return False
            continue
        if name not in _LINEAR_OPNAMES:
            return False
        if name == "BINARY_OP" and ins.argrepr not in _LINEAR_BINARY_OPS:
            return False
        if name in ("LOAD_CONST", "LOAD_SMALL_INT") and type(ins.argval) is not int:
            return False
    return True


def _symbolic_coefficients(fn, n: int, dimension: int, cache: dict):
    """
    Coefficients (c_0, c_1, ..., c_n) mod d of a detector function, from one symbolic call.

    Returns None when the function is not plain straight-line arithmetic or uses an operation
    that `_LinearForm` refuses; the caller then probes it numerically. The result depends only
    on the code object and n, so it is cached on them.
    """
    key = (fn.__code__, n) if type(fn) is types.FunctionType else None
    if key is not None and key in cache:
        return cache[key]
    result = None
    if _is_straight_line_arithmetic(fn):
        zero = (0,) * (n + 1)
        rec = [_LinearForm(zero[:j + 1] + (1 % dimension,) + zero[j + 2:], dimension) for j in range(n)]
        try:
            value = fn(rec)
        except Exception:
            value = None
        if type(value) is _LinearForm:
            result = value.c
        elif type(value) is int:
            result = (value % dimension,) + (0,) * n
    if key is not None:
        cache[key] = result
    return result


def _probed_coefficients(fn, n: int, unique_index: int, label, dimension: int) -> list:
    """
    Coefficient of each record position, read by evaluating the detector function on probes.

    Evaluating on unit vectors gives the coefficients, and further probes check that the
    function really is linear.

    Raises:
        ValueError: If the function has a constant term or is not linear.
    """
    base = int(fn([0] * n)) % dimension
    if base != 0:
        raise ValueError(f"detector {label!r} has a non-zero constant term")

    def at(values):
        return int(fn(list(values))) % dimension

    position_coeffs = []
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        position_coeffs.append(at(unit))

    def linear(values):
        return sum(c * v for c, v in zip(position_coeffs, values)) % dimension

    # Linearity checks: every pair of positions (catches products of two records), doubled
    # unit vectors (catches squares), and random inputs (anything of higher degree).
    probes = []
    if n <= 40:
        for i in range(n):
            for j in range(i + 1, n):
                v = [0] * n
                v[i] = v[j] = 1
                probes.append(v)
    for j in range(n):
        v = [0] * n
        v[j] = 2 % dimension
        probes.append(v)
    rng = np.random.default_rng(1234 + unique_index)
    probes += [[int(x) for x in rng.integers(0, dimension, size=n)] for _ in range(16)]
    for v in probes:
        if at(v) != linear(v):
            raise ValueError(f"detector {label!r} is not linear in its records")
    return position_coeffs


def _detector_coefficients(detector_info, dimension: int):
    """
    Reads the linear coefficients of each detector and logical observable.

    sdim compiles each DETECTOR / LOGICAL_OBSERVABLE expression into a
    function of its measurement records. When the function is plain
    arithmetic on its records (+, -, * by constants, % by a multiple of d),
    one call on symbolic `_LinearForm` records gives its coefficients exactly
    and proves it linear mod d. Otherwise it is evaluated on unit vectors to
    get the coefficients, and on pairs, doubled unit vectors and random inputs
    to check that it really is linear.

    Returns:
        tuple: Detector coefficients and observable coefficients (lists of
            {record index: coefficient mod d}), then detector labels and
            observable labels.

    Raises:
        ValueError: If an expression has a constant term or is not linear.
    """
    dets, obs, det_labels, obs_labels = [], [], [], []
    cache: dict = {}
    for unique_index, label, arguments, is_logical in detector_info.detector_data:
        fn = detector_info.detector_functions[unique_index]
        n = len(arguments)
        form = _symbolic_coefficients(fn, n, dimension, cache)
        if form is None:
            position_coeffs = _probed_coefficients(fn, n, unique_index, label, dimension)
        else:
            # The same checks, in the same order, as the numeric path; every probe would agree.
            if form[0] != 0:
                raise ValueError(f"detector {label!r} has a non-zero constant term")
            position_coeffs = form[1:]

        coeffs = {}
        for rec, c in zip(arguments, position_coeffs):
            if c:
                coeffs[int(rec)] = (coeffs.get(int(rec), 0) + c) % dimension
        coeffs = {r: c for r, c in coeffs.items() if c}
        (obs if is_logical else dets).append(coeffs)
        (obs_labels if is_logical else det_labels).append(label or "")
    return dets, obs, det_labels, obs_labels


def _plain_noise_gate(instr) -> bool:
    """
    True if `Program._build_ir` handles this N1 / N2 gate without raising.

    Such gates only add to the IR's noise samples, which the DEM does not use,
    so `_compile` can leave them out of the circuit it hands to `_build_ir`.
    """
    try:
        params = instr.params
        if instr.gate_id == _N1:
            if params.get("noise_channel", params.get("channel", "d")) not in ("d", "f", "p"):
                return False
            float(params["prob"])
        else:
            if params.get("prob_dist", None) is not None:
                return False
            float(params.get("prob", 0.0))
    except Exception:
        return False
    return True


def _qudit_op_lists(gid, qa, qb, n_qudits: int):
    """
    For each qudit, the IR ops that can change its frame, in circuit order.

    Returns (qb, qptr, qops, posa, posb). Qudit q's ops are qops[qptr[q]:qptr[q + 1]],
    so the probe kernel only visits ops on qudits a fault has reached. posa / posb hold an
    op's position in qops within the list of its first / second qudit. qb is a copy with
    -1 for every single-qudit frame op, so the kernel takes its one-qudit branch.

    Raises:
        IndexError: If a frame op acts on a qudit outside 0 .. n_qudits - 1.
    """
    n_ops = len(gid)
    frame_mask = np.isin(gid, np.array(sorted(_FRAME_GATES), dtype=np.int64))
    two_mask = frame_mask & np.isin(gid, np.array(_TWO_QUDIT_FRAME_GATES, dtype=np.int64)) & (qb >= 0)
    qb = np.where(frame_mask & ~two_mask, -1, qb)
    frame_ops = np.flatnonzero(frame_mask)
    two_ops = np.flatnonzero(two_mask)
    ent_q = np.concatenate((qa[frame_ops], qb[two_ops]))
    if len(ent_q) and (ent_q.min() < 0 or ent_q.max() >= n_qudits):
        bad = int(ent_q.min() if ent_q.min() < 0 else ent_q.max())
        raise IndexError(f"a gate acts on qudit {bad}, but the circuit has {n_qudits} qudits")
    ent_op = np.concatenate((frame_ops, two_ops))
    # An op's first-qudit entry comes before its second-qudit entry, which matters only if they coincide.
    ent_second = np.concatenate((np.zeros(len(frame_ops), dtype=np.int64), np.ones(len(two_ops), dtype=np.int64)))
    order = np.lexsort((ent_second, ent_op, ent_q))
    qops = ent_op[order].astype(np.int64)
    qptr = np.zeros(n_qudits + 1, dtype=np.int64)
    qptr[1:] = np.cumsum(np.bincount(ent_q, minlength=n_qudits))
    where = np.empty(len(order), dtype=np.int64)
    where[order] = np.arange(len(order), dtype=np.int64)
    posa = np.full(n_ops, -1, dtype=np.int64)
    posb = np.full(n_ops, -1, dtype=np.int64)
    posa[frame_ops] = where[:len(frame_ops)]
    posb[two_ops] = where[len(frame_ops):]
    return np.ascontiguousarray(qb, dtype=np.int64), qptr, qops, posa, posb


# Unit faults of each noise-gate code: 0 = N1 'd', 1 = N1 'f', 2 = N1 'p', 3 = N2. Probe j of a
# gate is a fault of kind _PROBE_KIND (0 = X, 1 = Z) on its first (_PROBE_QUDIT = 0) or second qudit.
_N1_CODES = {"d": 0, "f": 1, "p": 2}
_PROBE_COUNT = np.array([2, 1, 1, 4], dtype=np.int64)
_PROBE_QUDIT = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]], dtype=np.int64)
_PROBE_KIND = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 1]], dtype=np.int64)


def _merge_groups(group, probs, sources, ct, cv, cp) -> list:
    """
    Merges lines with the same canonical form, as `DetectorErrorModel.merge_lines` does.

    Line j has group number group[j] (numbered in order of first appearance), probability
    probs[j], source sources[j] and canonical entries cp[j]:cp[j + 1] of ct / cv. Each group
    becomes one mechanism, in group order, whose probability merges its lines' in order and
    whose source joins theirs with "+".
    """
    merged = []        # [probability, first line, member sources or None]
    for j, g in enumerate(group):
        if g == len(merged):
            merged.append([probs[j], j, None])
        else:
            entry = merged[g]
            entry[0] = _merge_pair(entry[0], probs[j])
            if entry[2] is None:
                entry[2] = [sources[entry[1]]]
            entry[2].append(sources[j])
    return [ErrorMechanism(probability, [dict(zip(ct[cp[j]:cp[j + 1]], cv[cp[j]:cp[j + 1]]))],
                           sources[j] if parts is None else "+".join(parts))
            for probability, j, parts in merged]


def _lines_from_arrays(mechs: list, pls: list, d: int):
    """
    The mechanisms of `DetectorErrorModel.to_lines`, expanded and merged in numba.

    pls[i] is the line probability of mechs[i]. Returns None if a target or
    coefficient is not a Python int that fits in int64; the caller then
    expands one dict at a time.
    """
    gens = [g for m in mechs for g in m.generators]
    kinds = set(map(type, itertools.chain.from_iterable(gens)))
    kinds |= set(map(type, itertools.chain.from_iterable(g.values() for g in gens)))
    if not kinds <= {int}:
        return None
    gen_ptr = np.zeros(len(mechs) + 1, dtype=np.int64)
    gen_ptr[1:] = np.cumsum(np.fromiter((len(m.generators) for m in mechs), dtype=np.int64, count=len(mechs)))
    ent_ptr = np.zeros(len(gens) + 1, dtype=np.int64)
    ent_ptr[1:] = np.cumsum(np.fromiter(map(len, gens), dtype=np.int64, count=len(gens)))
    n_ent = int(ent_ptr[-1])
    try:
        ent_tgt = np.fromiter(itertools.chain.from_iterable(gens), dtype=np.int64, count=n_ent)
    except OverflowError:
        return None
    ent_val = np.fromiter((v % d for g in gens for v in g.values()), dtype=np.int64, count=n_ent)
    line_mech, lptr, ltgt, lval, bad = _expand_lines(gen_ptr, ent_ptr, ent_tgt, ent_val, d)
    if bad >= 0:
        # A leading coefficient with no inverse mod d (composite d). Raise what merge_lines raises.
        lead = int(lval[lptr[bad]])
        pow(lead, -1, d)
        raise ValueError(f"{lead} is not invertible mod {d}")
    group = np.empty(len(line_mech), dtype=np.int64)
    _group_lines(lptr, ltgt, lval, group)
    line_mech = line_mech.tolist()
    return _merge_groups(group.tolist(), [pls[i] for i in line_mech], [mechs[i].source for i in line_mech],
                         ltgt.tolist(), lval.tolist(), lptr.tolist())


class _Compiled:
    """
    Unit-fault responses of every noise gate, as flat arrays.

    Noise gate i has unit-fault probes probe_start[i]:probe_start[i + 1], and
    probe k has response entries ptr[k]:ptr[k + 1] in tgt (targets, in the
    order the kernel first touched them) and val (coefficients mod d).
    """

    def __init__(self, dimension, num_detectors, num_observables, detector_labels, observable_labels,
                 ir_index, gate_id, codes, channels, q0, q1, prob, pi, probe_start, ptr, tgt, val):
        self.dimension = dimension
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.detector_labels = detector_labels
        self.observable_labels = observable_labels
        self.ir_index = ir_index          # lists, one entry per noise gate
        self.gate_id = gate_id
        self.codes = codes
        self.channels = channels
        self.q0 = q0
        self.q1 = q1
        self.prob = prob
        self.pi = pi
        self.probe_start = probe_start    # arrays
        self.ptr = ptr
        self.tgt = tgt
        self.val = val

    def source(self, i: int) -> str:
        if self.codes[i] == 3:
            return f"N2@{self.ir_index[i]}:q{self.q0[i]},q{self.q1[i]}"
        return f"N1[{self.channels[i]}]@{self.ir_index[i]}:q{self.q0[i]}"

    def locations(self) -> list:
        """The `NoiseLocation` list that `compile_unit_responses` returns."""
        tgt, val, ptr = self.tgt.tolist(), self.val.tolist(), self.ptr.tolist()
        responses = [dict(zip(tgt[a:b], val[a:b])) for a, b in zip(ptr[:-1], ptr[1:])]
        start = self.probe_start.tolist()
        out = []
        for i in range(len(self.codes)):
            if self.codes[i] == 3:
                qudits, channel = (self.q0[i], self.q1[i]), "d2"
            else:
                qudits, channel = (self.q0[i],), self.channels[i]
            out.append(NoiseLocation(self.ir_index[i], self.gate_id[i], qudits, channel, self.prob[i], self.pi[i],
                                     responses[start[i]:start[i + 1]], self.source(i)))
        return out

    def mechanisms(self, merge: bool) -> list:
        """
        The mechanisms of `DetectorErrorModel.from_circuit`.

        Every noise gate with prob > 0 and a visible unit fault gives one mechanism whose
        generators are its non-empty responses. With `merge`, the result is the one
        `DetectorErrorModel.merge_lines` gives: rank-1 mechanisms scaled so their first
        coefficient is 1 and merged by line in order of first appearance, then the others.
        """
        n_loc = len(self.codes)
        if n_loc == 0:
            return []
        d = self.dimension
        nonempty = np.diff(self.ptr) > 0
        start = self.probe_start
        # Every noise gate has at least one probe, so reduceat sees no empty segment.
        rank = np.add.reduceat(nonempty.astype(np.int64), start[:-1])
        prob = np.array(self.prob, dtype=np.float64)
        keep = (rank > 0) & ~(prob <= 0.0)
        tgt, val, ptr = self.tgt.tolist(), self.val.tolist(), self.ptr.tolist()
        nonempty_list = nonempty.tolist()
        start_list = start.tolist()
        pi = self.pi

        def generators(i):
            return [dict(zip(tgt[ptr[k]:ptr[k + 1]], val[ptr[k]:ptr[k + 1]]))
                    for k in range(start_list[i], start_list[i + 1]) if nonempty_list[k]]

        if not merge:
            return [ErrorMechanism(pi[i], generators(i), self.source(i)) for i in np.flatnonzero(keep).tolist()]

        ones = np.flatnonzero(keep & (rank == 1))
        higher = np.flatnonzero(keep & (rank > 1)).tolist()
        # The one non-empty probe of each rank-1 gate (only read for those gates).
        probe_loc = np.repeat(np.arange(n_loc, dtype=np.int64), np.diff(start))
        nonempty_probe = np.full(n_loc, -1, dtype=np.int64)
        hits = np.flatnonzero(nonempty)
        nonempty_probe[probe_loc[hits]] = hits
        probes = nonempty_probe[ones]
        n_entries = int((self.ptr[probes + 1] - self.ptr[probes]).sum())
        cptr = np.zeros(len(probes) + 1, dtype=np.int64)
        ctgt = np.empty(n_entries, dtype=np.int64)
        cval = np.empty(n_entries, dtype=np.int64)
        group = np.empty(len(probes), dtype=np.int64)
        bad = _canonical_lines(self.ptr, self.tgt, self.val, probes, d, cptr, ctgt, cval, group)
        if bad >= 0:
            # Not invertible mod d (only possible for a composite d). Raise what merge_lines raises.
            k = int(probes[bad])
            lead = min(zip(tgt[ptr[k]:ptr[k + 1]], val[ptr[k]:ptr[k + 1]]))[1]
            pow(lead, -1, d)
            raise ValueError(f"{lead} is not invertible mod {d}")

        ones = ones.tolist()
        out = _merge_groups(group.tolist(), [pi[i] for i in ones], [self.source(i) for i in ones],
                            ctgt.tolist(), cval.tolist(), cptr.tolist())
        out += [ErrorMechanism(pi[i], generators(i), self.source(i)) for i in higher]
        return out


def _compile(circuit: Circuit) -> _Compiled:
    """
    Computes the unit-fault responses of every noise gate. See `compile_unit_responses`.

    The noise gates are left out of the circuit handed to `Program._build_ir`
    when that does not change what it returns or raises, since it would
    otherwise draw a noise sample for each of them.
    """
    d = circuit.dimension
    n_qudits = circuit.num_qudits

    # One pass over the circuit: the non-noise ops, and each noise gate with its IR index
    # (_build_ir drops identity gates, id 0) and the number of non-noise ops before it.
    kept = []
    keep = kept.append
    noise = []
    add_noise = noise.append
    plain = True
    checked = checked_gate = None
    ir_index = -1
    for instr in circuit.operations:
        g = instr.gate_id
        if g == 0:
            continue
        ir_index += 1
        if g == _N1 or g == _N2:
            add_noise((instr, ir_index, len(kept)))
            # add_gate gives all the gates of one call the same params dict; check it once.
            if plain and (instr.params is not checked or g != checked_gate):
                plain = _plain_noise_gate(instr)
                checked, checked_gate = instr.params, g
        else:
            keep(instr)

    # _build_ir also samples one shot of noise; keep the caller's global RNG state untouched.
    rng_state = np.random.get_state()
    try:
        if plain:
            stripped = copy.copy(circuit)
            stripped.operations = kept
            ir_array, _, detector_info = Program._build_ir([stripped], 1)
        else:
            ir_array, _, detector_info = Program._build_ir([circuit], 1)
    finally:
        np.random.set_state(rng_state)
    gid = np.ascontiguousarray(ir_array["gate_id"], dtype=np.int64)
    qa = np.ascontiguousarray(ir_array["qudit_index"], dtype=np.int64)
    qb = np.ascontiguousarray(ir_array["target_index"], dtype=np.int64)
    n_ops = len(gid)

    # MUL multiplies X by a and Z by a^-1 mod d. Only MUL ops use these; every other op holds 1.
    scalar = np.asarray(ir_array["scalar"], dtype=np.int64)
    mul_a = np.ones(n_ops, dtype=np.int64)
    mul_inv = np.ones(n_ops, dtype=np.int64)
    for i in np.flatnonzero(gid == _MUL):
        mul_a[i] = int(scalar[i]) % d
        mul_inv[i] = pow(int(mul_a[i]), -1, d)

    # Measurement record index of each measuring op (sdim counts M and M_X only).
    is_meas = (gid == _M) | (gid == _M_X)
    rec_of_op = np.full(n_ops, -1, dtype=np.int64)
    n_recs = int(is_meas.sum())
    rec_of_op[is_meas] = np.arange(n_recs, dtype=np.int64)

    qb, qptr, qops, posa, posb = _qudit_op_lists(gid, qa, qb, n_qudits)

    # For each measurement record, the detectors / observables that use it and their coefficients,
    # detectors first and then observables, each in order.
    dets, obs, det_labels, obs_labels = _detector_coefficients(detector_info, d)
    n_det = len(dets)
    inc_rec, inc_tgt, inc_coef = [], [], []
    for t, coeffs in enumerate(dets + obs):
        inc_rec += coeffs.keys()
        inc_tgt += [t] * len(coeffs)
        inc_coef += coeffs.values()
    inc_rec = np.array(inc_rec, dtype=np.int64)
    by_rec = np.argsort(inc_rec, kind="stable")
    rptr = np.zeros(n_recs + 1, dtype=np.int64)
    rptr[1:] = np.cumsum(np.bincount(inc_rec, minlength=n_recs))
    rtgt = np.array(inc_tgt, dtype=np.int64)[by_rec]
    rcoef = np.array(inc_coef, dtype=np.int64)[by_rec]

    # Noise gates in IR order. Raise on the first bad one, as compile_unit_responses always has.
    # Consecutive gates that share a params dict (one add_gate call) share its parsed values.
    den = {1: 1.0 - float(d) ** (-1), 2: 1.0 - float(d) ** (-2), 4: 1.0 - float(d) ** (-4)}
    rows = []
    add_row = rows.append
    parsed_params = parsed_gate = parsed = None
    for instr, ir_index, before in noise:
        params = instr.params
        g = instr.gate_id
        if params is not parsed_params or g != parsed_gate:
            parsed = None
        if g == _N1:
            if parsed is None:
                channel = params.get("noise_channel", params.get("channel", "d"))
                p = float(params.get("prob", 0.0))
                q0 = int(instr.qudit_index)
                if channel not in ("d", "f", "p"):
                    raise ValueError(f"N1 noise_channel must be 'd', 'f' or 'p', not {channel!r}.")
                code = _N1_CODES[channel]
                rank = 2 if code == 0 else 1
                pi = p / den[rank]
            else:
                q0 = int(instr.qudit_index)
            q1 = -1
        else:
            if parsed is None:
                if params.get("prob_dist", None) is not None:
                    raise ValueError("Compact DEMs support N2 with prob=... (uniform non-identity depolarizing); "
                                     "use sdim.dem_legacy for arbitrary prob_dist at small d.")
                p = float(params.get("prob", 0.0))
                channel, code, rank = "d2", 3, 4
                pi = p / den[rank]
            q0, q1 = int(instr.qudit_index), int(instr.target_index)
        if parsed is None:
            if pi > 1.0 + 1e-12:
                name = f"N2@{ir_index}:q{q0},q{q1}" if code == 3 else f"N1[{channel}]@{ir_index}:q{q0}"
                raise ValueError(f"{name}: prob={p} is above the fully mixing value {den[rank]}. "
                                 "The compact DEM can only represent noise up to full mixing.")
            parsed = (channel, code, p, min(pi, 1.0))
            parsed_params, parsed_gate = params, g
        # The fault sits right after the gate: the kernel starts at the first op after probe_op.
        add_row((ir_index, g, q0, q1, before - 1 if plain else ir_index) + parsed)
    if rows:
        ir_indices, gate_ids, q0s, q1s, probe_ops, channels, codes, probs, pis = (list(col) for col in zip(*rows))
    else:
        ir_indices, gate_ids, q0s, q1s, probe_ops, channels, codes, probs, pis = ([] for _ in range(9))

    # Unit-fault probes of the noise gates. A probe (qudit, 0) is an X fault and (qudit, 1) a Z fault.
    code_arr = np.array(codes, dtype=np.int64)
    probe_start = np.zeros(len(codes) + 1, dtype=np.int64)
    probe_start[1:] = np.cumsum(_PROBE_COUNT[code_arr])
    n_noise_probes = int(probe_start[-1])
    probe_loc = np.repeat(np.arange(len(codes), dtype=np.int64), _PROBE_COUNT[code_arr])
    local = np.arange(n_noise_probes, dtype=np.int64) - probe_start[probe_loc]
    probe_code = code_arr[probe_loc]
    noise_qudit = np.where(_PROBE_QUDIT[probe_code, local] == 0,
                           np.array(q0s, dtype=np.int64)[probe_loc], np.array(q1s, dtype=np.int64)[probe_loc])
    noise_kind = _PROBE_KIND[probe_code, local]
    noise_op = np.array(probe_ops, dtype=np.int64)[probe_loc]

    # Determinism probes.  The frame simulator randomizes the Z frame at the start and after every
    # M, M_X and RESET, and the unit-fault responses above assume those random parts cancel.  A unit
    # Z fault at each of those points must therefore reach no detector or observable.
    resets = np.flatnonzero((gid == _M) | (gid == _M_X) | (gid == _RESET))
    probe_op = np.concatenate((noise_op, np.full(n_qudits, -1, dtype=np.int64), resets)).astype(np.int64)
    probe_qudit = np.concatenate((noise_qudit, np.arange(n_qudits, dtype=np.int64), qa[resets])).astype(np.int64)
    probe_kind = np.concatenate((noise_kind, np.ones(n_qudits + len(resets), dtype=np.int64))).astype(np.int64)
    if len(probe_qudit) and (probe_qudit.min() < 0 or probe_qudit.max() >= n_qudits):
        bad = int(probe_qudit.min() if probe_qudit.min() < 0 else probe_qudit.max())
        raise IndexError(f"a gate acts on qudit {bad}, but the circuit has {n_qudits} qudits")

    n_targets = n_det + len(obs)
    ptr, tgt, val = _run_probes(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                                probe_op, probe_qudit, probe_kind, d, n_targets, n_qudits)
    random_targets = np.unique(tgt[ptr[n_noise_probes]:]).tolist()
    if random_targets:
        names = [(f"D{t}" if t < n_det else f"L{t - n_det}") for t in random_targets]
        raise ValueError("These detectors / observables are not deterministic without noise, so they "
                         f"have no detector error model: {', '.join(names[:20])}"
                         + (" ..." if len(names) > 20 else ""))
    end = int(ptr[n_noise_probes])
    return _Compiled(d, n_det, len(obs), det_labels, obs_labels, ir_indices, gate_ids, codes, channels,
                     q0s, q1s, probs, pis, probe_start, ptr[:n_noise_probes + 1], tgt[:end], val[:end])


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
    with _gc_paused():
        compiled = _compile(circuit)
        return CompiledResponses(compiled.num_detectors, compiled.num_observables, compiled.detector_labels,
                                 compiled.observable_labels, compiled.locations())


def _run_probes(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                probe_op, probe_qudit, probe_kind, d, n_targets, n_qudits, block=None, cap=16):
    """
    Runs `_probe_kernel` over all probes, in blocks of `block` probes.

    With at least `_PROBE_PARALLEL_MIN` probes, the blocks run on several
    threads (see `_run_tasks`), each thread taking the next block as it
    finishes one, since blocks differ a lot in cost (early faults travel
    further). By default a block is about 1/8 of a thread's share, between 64
    and 1024 probes. Each block starts with room for `cap` entries per probe
    (at most 2**20 in all) and grows its buffers as needed. The blocks only
    change how the work is split, not the result.

    Returns:
        tuple: (ptr, tgt, val), the response of probe k being the entries
            ptr[k]:ptr[k + 1] of tgt (targets) and val (coefficients mod d),
            in the order the kernel first touched each target.
    """
    n = len(probe_op)
    n_threads = _thread_count() if n >= _PROBE_PARALLEL_MIN else 1
    if block is None:
        block = max(n, 1) if n_threads == 1 else min(1024, max(64, -(-n // (8 * n_threads))))
    n_blocks = -(-n // block)
    max_slots = max(n_qudits, 1)
    results = [None] * n_blocks

    def task(b):
        lo, hi = b * block, min((b + 1) * block, n)
        results[b] = _probe_kernel(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt,
                                   rcoef, probe_op[lo:hi], probe_qudit[lo:hi], probe_kind[lo:hi], d, n_targets,
                                   max_slots, max(min((hi - lo) * cap, 1 << 20), 1))

    _run_tasks(task, n_blocks, n_threads)
    counts = [np.zeros(1, dtype=np.int64)]
    tgts, vals = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
    for status, bptr, btgt, bval in results:
        if status != 0:
            raise RuntimeError(f"unit-fault propagation failed with status {status}")
        counts.append(np.diff(bptr))
        tgts.append(btgt)
        vals.append(bval)
    return np.cumsum(np.concatenate(counts)), np.concatenate(tgts), np.concatenate(vals)


@njit(nogil=True, cache=True)
def _probe_kernel(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                  probe_op, probe_qudit, probe_kind, d, n_targets, max_slots, cap):
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

    `mul_a` and `mul_inv` hold, per op, the MUL scalar a mod d and its inverse
    (1 for every other op).

    The output starts with room for `cap` entries and doubles whenever a
    probe's response would not fit.

    Returns:
        tuple: (status, out_ptr, out_tgt, out_val). status is 0 on success and
            2 if a fault reached more than `max_slots` qudits (impossible when
            `max_slots` is the number of qudits). Probe p's response is the
            entries out_ptr[p]:out_ptr[p + 1] of out_tgt / out_val.
    """
    smax = max_slots
    n_probes = probe_op.shape[0]
    sq = np.empty(smax, dtype=np.int64)
    sx = np.empty(smax, dtype=np.int64)
    sz = np.empty(smax, dtype=np.int64)
    scur = np.empty(smax, dtype=np.int64)
    acc = np.zeros(n_targets, dtype=np.int64)
    touched = np.empty(n_targets, dtype=np.int64)
    is_touched = np.zeros(n_targets, dtype=np.int64)
    out_ptr = np.zeros(n_probes + 1, dtype=np.int64)
    out_tgt = np.empty(max(cap, 1), dtype=np.int64)
    out_val = np.empty(max(cap, 1), dtype=np.int64)
    w = 0
    big = 1 << 62
    for p in range(n_probes):
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
                elif g == 22:  # MUL
                    nx = (x * mul_a[op]) % d
                    nz = (z * mul_inv[op]) % d
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
                            if is_touched[t] == 0:
                                is_touched[t] = 1
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
                        return 2, out_ptr, out_tgt[:0], out_val[:0]
                    sa = nslot
                    sq[sa] = a
                    sx[sa] = 0
                    sz[sa] = 0
                    scur[sa] = posa[op]
                    nslot += 1
                if sb < 0:
                    if nslot >= smax:
                        return 2, out_ptr, out_tgt[:0], out_val[:0]
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
        # Write out this probe's non-zero totals and clear the accumulator. They are at most nt.
        out_ptr[p] = w
        if w + nt > out_tgt.shape[0]:
            size = 2 * out_tgt.shape[0]
            while size < w + nt:
                size *= 2
            grown_tgt = np.empty(size, dtype=np.int64)
            grown_val = np.empty(size, dtype=np.int64)
            for i in range(w):
                grown_tgt[i] = out_tgt[i]
                grown_val[i] = out_val[i]
            out_tgt = grown_tgt
            out_val = grown_val
        for i in range(nt):
            t = touched[i]
            if acc[t] != 0:
                out_tgt[w] = t
                out_val[w] = acc[t]
                w += 1
            acc[t] = 0
            is_touched[t] = 0
    out_ptr[n_probes] = w
    return 0, out_ptr, out_tgt[:w], out_val[:w]


@njit(cache=True)
def _sort_pairs(keys, vals, lo, hi):
    """Sorts keys[lo:hi] in place, moving vals[lo:hi] along with them (insertion sort, or heapsort)."""
    n = hi - lo
    if n <= 32:
        for k in range(lo + 1, hi):
            t = keys[k]
            v = vals[k]
            j = k
            while j > lo and keys[j - 1] > t:
                keys[j] = keys[j - 1]
                vals[j] = vals[j - 1]
                j -= 1
            keys[j] = t
            vals[j] = v
        return
    # Heapsort on the slice, in one loop so that it is one function to compile. The first n // 2
    # steps build the heap (sifting down roots n // 2 - 1 .. 0); each later step moves the largest
    # key to the end of the heap, which shrinks by one, and sifts down the new root.
    half = n // 2
    size = n
    for step in range(half + n - 1):
        if step < half:
            root = half - 1 - step
        else:
            size = n - 1 - (step - half)
            t = keys[lo]
            keys[lo] = keys[lo + size]
            keys[lo + size] = t
            v = vals[lo]
            vals[lo] = vals[lo + size]
            vals[lo + size] = v
            root = 0
        while True:
            child = 2 * root + 1
            if child >= size:
                break
            if child + 1 < size and keys[lo + child + 1] > keys[lo + child]:
                child += 1
            if keys[lo + root] >= keys[lo + child]:
                break
            t = keys[lo + root]
            keys[lo + root] = keys[lo + child]
            keys[lo + child] = t
            v = vals[lo + root]
            vals[lo + root] = vals[lo + child]
            vals[lo + child] = v
            root = child


@njit(cache=True)
def _mod_inverse(a, m):
    """The inverse of a mod m in 0 .. m - 1, or -1 if gcd(a, m) != 1."""
    t, new_t, r, new_r = 0, 1, m, a % m
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r != 1:
        return -1
    return t % m


@njit(inline="always")
def _line_hash(cptr, ctgt, cval, i):
    h = np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(cptr[i + 1] - cptr[i])
    for k in range(cptr[i], cptr[i + 1]):
        h = (h ^ np.uint64(ctgt[k])) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ np.uint64(cval[k])) * np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return np.int64(h >> np.uint64(1))


@njit(inline="always")
def _same_line(cptr, ctgt, cval, i, j):
    n = cptr[i + 1] - cptr[i]
    if cptr[j + 1] - cptr[j] != n:
        return False
    a = cptr[i]
    b = cptr[j]
    for k in range(n):
        if ctgt[a + k] != ctgt[b + k] or cval[a + k] != cval[b + k]:
            return False
    return True


@njit(cache=True)
def _canonical_lines(ptr, tgt, val, probes, d, cptr, ctgt, cval, group):
    """
    Canonical form and line class of rank-1 mechanisms, as in `_canonical_line` and `merge_lines`.

    Mechanism i is the response probes[i] (entries ptr[p]:ptr[p + 1] of tgt / val). Its
    entries are sorted by target and scaled so the first coefficient is 1, and written to
    cptr / ctgt / cval. group[i] numbers the distinct canonical forms in order of first
    appearance.

    Returns:
        int: -1 on success, or the first i whose leading coefficient is not invertible mod d.
    """
    m = probes.shape[0]
    w = 0
    cptr[0] = 0
    for i in range(m):
        a = ptr[probes[i]]
        n = ptr[probes[i] + 1] - a
        for k in range(n):
            ctgt[w + k] = tgt[a + k]
            cval[w + k] = val[a + k]
        _sort_pairs(ctgt, cval, w, w + n)
        inv = _mod_inverse(cval[w], d)
        if inv < 0:
            return i
        for k in range(n):
            cval[w + k] = (cval[w + k] * inv) % d
        w += n
        cptr[i + 1] = w
    _group_lines(cptr, ctgt, cval, group)
    return -1


@njit(cache=True)
def _group_lines(cptr, ctgt, cval, group):
    """
    Numbers equal lines (entries cptr[i]:cptr[i + 1] of ctgt / cval) in order of first appearance.

    Uses an open-addressing hash table, and compares entries on every hash match.
    """
    m = cptr.shape[0] - 1
    cap = 2
    while cap < 2 * m:
        cap *= 2
    mask = cap - 1
    table = np.empty(cap, dtype=np.int64)
    for s in range(cap):
        table[s] = -1
    hashes = np.empty(m, dtype=np.int64)
    n_groups = 0
    for i in range(m):
        h = _line_hash(cptr, ctgt, cval, i)
        hashes[i] = h
        s = h & mask
        while True:
            r = table[s]
            if r < 0:
                table[s] = i
                group[i] = n_groups
                n_groups += 1
                break
            if hashes[r] == h and _same_line(cptr, ctgt, cval, r, i):
                group[i] = group[r]
                break
            s = (s + 1) & mask
    return n_groups


@njit(cache=True)
def _expand_lines(gen_ptr, ent_ptr, ent_tgt, ent_val, d):
    """
    The line mechanisms of `DetectorErrorModel.to_lines`, in canonical form.

    For each mechanism (generators gen_ptr[i]:gen_ptr[i + 1], entries ent_ptr[g]:ent_ptr[g + 1]
    with coefficients already reduced mod d), every point of `_projective_points(d, k)` in the
    same order gives sum_j point_j * generator_j. Non-zero results are sorted by target and
    scaled so the first coefficient is 1, as `merge_lines` does.

    Returns:
        tuple: (line_mech, lptr, ltgt, lval, bad). Line l came from mechanism line_mech[l] and
            has entries lptr[l]:lptr[l + 1]. bad is -1, or the index of a line whose leading
            coefficient is not invertible mod d; that line is the last one, left unscaled.
    """
    n_mech = gen_ptr.shape[0] - 1
    line_mech = np.empty(16, dtype=np.int64)
    lptr = np.zeros(17, dtype=np.int64)
    ltgt = np.empty(64, dtype=np.int64)
    lval = np.empty(64, dtype=np.int64)
    n_lines = 0
    w = 0
    for i in range(n_mech):
        g0 = gen_ptr[i]
        k = gen_ptr[i + 1] - g0
        if k == 0:
            continue
        # The distinct targets of the mechanism, sorted.
        e0 = ent_ptr[g0]
        e1 = ent_ptr[g0 + k]
        cols = np.empty(e1 - e0, dtype=np.int64)
        for e in range(e1 - e0):
            cols[e] = ent_tgt[e0 + e]
        spare = np.zeros(e1 - e0, dtype=np.int64)
        _sort_pairs(cols, spare, 0, e1 - e0)
        m = 0
        for e in range(e1 - e0):
            if m == 0 or cols[e] != cols[m - 1]:
                cols[m] = cols[e]
                m += 1
        mat = np.zeros((k, m), dtype=np.int64)
        for j in range(k):
            for e in range(ent_ptr[g0 + j], ent_ptr[g0 + j + 1]):
                lo_u = 0
                hi_u = m
                while lo_u < hi_u:
                    mid = (lo_u + hi_u) // 2
                    if cols[mid] < ent_tgt[e]:
                        lo_u = mid + 1
                    else:
                        hi_u = mid
                mat[j, lo_u] = (mat[j, lo_u] + ent_val[e]) % d
        coeff = np.zeros(k, dtype=np.int64)
        vals = np.empty(m, dtype=np.int64)
        for lead in range(k):
            n_tail = 1
            for _ in range(k - lead - 1):
                n_tail *= d
            for tail in range(n_tail):
                # The point (0, ..., 0, 1, tail digits), last digit fastest as in itertools.product.
                for j in range(k):
                    coeff[j] = 0
                coeff[lead] = 1
                x = tail
                for pos in range(k - 1, lead, -1):
                    coeff[pos] = x % d
                    x //= d
                count = 0
                first = -1
                for u in range(m):
                    acc = 0
                    for j in range(lead, k):
                        if coeff[j] != 0:
                            acc = (acc + coeff[j] * mat[j, u]) % d
                    vals[u] = acc
                    if acc != 0:
                        count += 1
                        if first < 0:
                            first = u
                if count == 0:
                    continue
                inv = _mod_inverse(vals[first], d)
                # Room for one more line.
                if n_lines >= line_mech.shape[0]:
                    size = 2 * line_mech.shape[0]
                    grown = np.empty(size, dtype=np.int64)
                    grown_ptr = np.zeros(size + 1, dtype=np.int64)
                    for j in range(n_lines):
                        grown[j] = line_mech[j]
                        grown_ptr[j] = lptr[j]
                    grown_ptr[n_lines] = lptr[n_lines]
                    line_mech = grown
                    lptr = grown_ptr
                if w + count > ltgt.shape[0]:
                    size = 2 * ltgt.shape[0]
                    while w + count > size:
                        size *= 2
                    grown_t = np.empty(size, dtype=np.int64)
                    grown_v = np.empty(size, dtype=np.int64)
                    for j in range(w):
                        grown_t[j] = ltgt[j]
                        grown_v[j] = lval[j]
                    ltgt = grown_t
                    lval = grown_v
                for u in range(m):
                    if vals[u] != 0:
                        ltgt[w] = cols[u]
                        lval[w] = vals[u] if inv < 0 else (vals[u] * inv) % d
                        w += 1
                line_mech[n_lines] = i
                n_lines += 1
                lptr[n_lines] = w
                if inv < 0:
                    return line_mech[:n_lines], lptr[:n_lines + 1], ltgt[:w], lval[:w], n_lines - 1
    return line_mech[:n_lines], lptr[:n_lines + 1], ltgt[:w], lval[:w], -1


@njit(inline="always")
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(inline="always")
def _next_u64(s0, s1, s2, s3):
    """One step of xoshiro256**. Returns the output and the new state."""
    result = _rotl(s1 * np.uint64(5), 7) * np.uint64(9)
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    return result, s0, s1, s2, s3


@njit(inline="always")
def _open_unit(r):
    """Maps 64 random bits to a float in (0, 1): the top 53 bits plus one half, times 2**-53."""
    return (np.float64(r >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _sample_plan(mech_prob, bits, n_gens, sizes, ent_tgt, ent_val, d, n_targets, montgomery, pack_like):
    """
    Lays out the arrays of `DetectorErrorModel._flatten` for `_sample_chunks`.

    Mechanisms with pi >= 1 fire in every shot. Those with 0 < pi < 1 are
    grouped into bins by the exponent and top 3 mantissa bits of pi, so the
    probabilities in a bin are within a factor 9/8; bin_pmax is the largest
    one and bin_log_keep its log(1 - pmax). Mechanisms with pi <= 0 or no
    entries are left out.

    Each mechanism kept becomes one block of `pack`, so firing it reads one
    stretch of memory: the number of generators, then for each generator the
    number of entries followed by (target, value) pairs. The value is the
    coefficient reduced mod d, in Montgomery form v * 2**32 mod d when
    `montgomery` (odd d). info[k] = (block offset, probability bits) for the
    k-th mechanism in bin order (bins in increasing probability, mechanisms
    in index order within a bin); always_off holds the blocks of pi >= 1.

    `pack` has the dtype of `pack_like`. int32 is enough when every target is
    below 2**31; values are below d, which is below 2**31. `bits` is
    mech_prob viewed as int64.

    Returns:
        tuple: (status, pack, info, always_off, bin_ptr, bin_pmax, bin_log_keep, cost).
            status is -1 if all is well, -2 if a probability is NaN, or the
            index of an entry whose target is out of range. cost estimates the
            work per shot.
    """
    n_mech = mech_prob.shape[0]
    n_gen = sizes.shape[0]
    gen_ptr = np.zeros(n_mech + 1, dtype=np.int64)
    for i in range(n_mech):
        gen_ptr[i + 1] = gen_ptr[i] + n_gens[i]
    ent_ptr = np.zeros(n_gen + 1, dtype=np.int64)
    for g in range(n_gen):
        ent_ptr[g + 1] = ent_ptr[g] + sizes[g]
    # Allocations use few distinct (shape, dtype) forms, since numba compiles each form separately.
    empty = np.zeros(0, dtype=np.int64)
    no_pack = np.empty(0, dtype=pack_like.dtype)
    no_info = np.empty((0, 2), dtype=np.int64)
    no_float = np.empty(0, dtype=np.float64)
    for k in range(ent_tgt.shape[0]):
        if ent_tgt[k] < 0 or ent_tgt[k] >= n_targets:
            return k, no_pack, no_info, empty, empty, no_float, no_float, 0.0
        v = ent_val[k] % d
        ent_val[k] = (v << 32) % d if montgomery else v
    for i in range(n_mech):
        if np.isnan(mech_prob[i]):
            return -2, no_pack, no_info, empty, empty, no_float, no_float, 0.0
    # Classify the mechanisms and estimate the work per shot.
    n_always = 0
    n_live = 0
    keys = np.empty(n_mech, dtype=np.int64)
    cost = 0.0
    for i in range(n_mech):
        work = 0
        for g in range(gen_ptr[i], gen_ptr[i + 1]):
            work += 1 + ent_ptr[g + 1] - ent_ptr[g]
        if ent_ptr[gen_ptr[i + 1]] == ent_ptr[gen_ptr[i]]:
            keys[i] = -1
        elif mech_prob[i] >= 1.0:
            keys[i] = -2
            n_always += 1
            cost += work
        elif mech_prob[i] > 0.0:
            # For positive doubles the bit pattern grows with the value.
            keys[i] = bits[i] >> 49
            n_live += 1
            cost += mech_prob[i] * work
        else:
            keys[i] = -1
    # Counting sort by key (bits >> 49 of a positive double is below 2**14). It is stable, so
    # mechanisms stay in index order within a bin.
    counts = np.zeros((1 << 14) + 1, dtype=np.int64)
    for i in range(n_mech):
        if keys[i] >= 0:
            counts[keys[i] + 1] += 1
    n_bins = 0
    for key in range(1 << 14):
        if counts[key + 1] > 0:
            n_bins += 1
        counts[key + 1] += counts[key]
    order = np.empty(n_live + n_always, dtype=np.int64)
    ia = n_live
    for i in range(n_mech):
        if keys[i] >= 0:
            order[counts[keys[i]]] = i
            counts[keys[i]] += 1
        elif keys[i] == -2:
            order[ia] = i
            ia += 1
    # Pack the mechanisms in that order: bins first, then the ones that always fire.
    size = 0
    for j in range(n_live + n_always):
        i = order[j]
        size += 1 + (gen_ptr[i + 1] - gen_ptr[i]) + 2 * (ent_ptr[gen_ptr[i + 1]] - ent_ptr[gen_ptr[i]])
    pack = np.empty(size, dtype=pack_like.dtype)
    info = np.empty((n_live, 2), dtype=np.int64)
    always_off = np.empty(n_always, dtype=np.int64)
    w = 0
    for j in range(n_live + n_always):
        i = order[j]
        if j < n_live:
            info[j, 0] = w
            info[j, 1] = bits[i]
        else:
            always_off[j - n_live] = w
        pack[w] = gen_ptr[i + 1] - gen_ptr[i]
        w += 1
        for g in range(gen_ptr[i], gen_ptr[i + 1]):
            pack[w] = ent_ptr[g + 1] - ent_ptr[g]
            w += 1
            for e in range(ent_ptr[g], ent_ptr[g + 1]):
                pack[w] = ent_tgt[e]
                pack[w + 1] = ent_val[e]
                w += 2
    bin_ptr = np.zeros(n_bins + 1, dtype=np.int64)
    bin_pmax = np.empty(n_bins, dtype=np.float64)   # every bin has a first mechanism, which sets it
    b = -1
    for j in range(n_live):
        p = mech_prob[order[j]]
        if j == 0 or keys[order[j]] != keys[order[j - 1]]:
            b += 1
            bin_ptr[b] = j
            bin_pmax[b] = p
        elif p > bin_pmax[b]:
            bin_pmax[b] = p
    bin_ptr[n_bins] = n_live
    bin_log_keep = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        bin_log_keep[b] = math.log1p(-bin_pmax[b])
    cost += n_bins
    return -1, pack, info, always_off, bin_ptr, bin_pmax, bin_log_keep, cost


@njit(nogil=True, cache=True)
def _sample_chunks(c_lo, c_hi, det, obs, chunk, states, bin_ptr, bin_pmax, bin_log_keep, info, info_p, always_off,
                   pack, d, thresh, nprime):
    """
    Samples blocks c_lo .. c_hi - 1. Block c is rows c * chunk onwards, drawn from the xoshiro256** state states[c].

    See `DetectorErrorModel.sample`. A block only uses its own random state and
    only writes its own rows, so blocks can run on any thread in any order.

    The mechanisms of bin b (info[bin_ptr[b]:bin_ptr[b + 1]]) are n_b coins
    per shot, n_b * shots coins in a block. Each is a candidate with
    probability pmax = bin_pmax[b], and the number of misses before the next
    candidate is geometric, floor(log(u) / log(1 - pmax)), so the loop jumps
    from one candidate to the next. A candidate mechanism with probability
    pi < pmax fires with probability pi / pmax, so overall it fires with
    probability pi, independently of every other coin. nxt[b] holds the next
    candidate of bin b, so each shot's row is finished before the next. The
    mechanisms at always_off fire in every shot. info_p is info viewed as
    float64, so info_p[k, 1] is the probability of the k-th binned mechanism.

    Within a shot the always-on mechanisms come first, then the bins in
    order. Each candidate draws its thinning coin (when pi < pmax), then its
    coefficients if it fires, then the skip to the next candidate.

    A firing mechanism adds sum_j a_j * generator_j, with each a_j uniform on
    Z_d by Lemire's method on the top 32 bits of a draw: (x * d) >> 32 is
    uniform once draws whose low half falls below thresh = 2**32 mod d are
    rejected. For odd d, nprime is -d^-1 mod 2**32 and a * v mod d is a
    Montgomery product with R = 2**32 (the packed value is v * R mod d); for
    even d, nprime is 0 and the product uses %. With d < 2**31 every
    intermediate value stays below 2**64.

    The firing code is written out in this one function instead of calling a
    helper, which would cost reference-count updates on every array per firing.
    """
    shots = det.shape[0]
    nd = det.shape[1]
    n_bins = bin_pmax.shape[0]
    n_always = always_off.shape[0]
    d_u = np.uint64(d)
    low32 = np.uint64(0xFFFFFFFF)
    nxt = np.empty(n_bins, dtype=np.int64)
    for c in range(c_lo, c_hi):
        s0 = states[c, 0]
        s1 = states[c, 1]
        s2 = states[c, 2]
        s3 = states[c, 3]
        lo = c * chunk
        n_shots = min(chunk, shots - lo)
        for b in range(n_bins):
            total = (bin_ptr[b + 1] - bin_ptr[b]) * n_shots
            r, s0, s1, s2, s3 = _next_u64(s0, s1, s2, s3)
            skip = np.floor(np.log(_open_unit(r)) / bin_log_keep[b])
            # For tiny pi the skip can exceed int64; anything past the end just means "done".
            nxt[b] = np.int64(skip) if skip < total else total
        for s in range(n_shots):
            shot = lo + s
            ia = 0            # next always-on mechanism
            b = -1            # current bin, -1 before the first
            first = 0         # bin b: first mechanism, size, coins in the block, end of this shot's coins,
            n_b = 0           # largest probability, next candidate
            total = 0
            end = 0
            pmax = 0.0
            pos = 0
            pending = False   # a candidate was handled and the skip past it is not drawn yet
            while True:
                # The block of the next mechanism that fires, or -1 when the shot is done.
                off = -1
                if ia < n_always:
                    off = always_off[ia]
                    ia += 1
                else:
                    while True:
                        if pending:
                            pending = False
                            r, s0, s1, s2, s3 = _next_u64(s0, s1, s2, s3)
                            skip = np.floor(np.log(_open_unit(r)) / bin_log_keep[b])
                            if skip >= total - 1 - pos:
                                pos = total
                            else:
                                pos += 1 + np.int64(skip)
                        if b >= 0 and pos < end:
                            k = first + pos - (end - n_b)
                            pending = True
                            pm = info_p[k, 1]
                            if pm < pmax:
                                r, s0, s1, s2, s3 = _next_u64(s0, s1, s2, s3)
                                if not _open_unit(r) * pmax < pm:
                                    continue
                            off = info[k, 0]
                            break
                        if b >= 0:
                            nxt[b] = pos
                        b += 1
                        if b == n_bins:
                            break
                        first = bin_ptr[b]
                        n_b = bin_ptr[b + 1] - first
                        total = n_b * n_shots
                        end = (s + 1) * n_b
                        pmax = bin_pmax[b]
                        pos = nxt[b]
                    if off < 0:
                        break
                # Fire the mechanism at `off`.
                n_gen = np.int64(pack[off])
                kk = off + 1
                for _ in range(n_gen):
                    n_ent = np.int64(pack[kk])
                    kk += 1
                    while True:
                        r, s0, s1, s2, s3 = _next_u64(s0, s1, s2, s3)
                        prod = (r >> np.uint64(32)) * d_u
                        if (prod & low32) >= thresh:
                            break
                    a = prod >> np.uint64(32)
                    if a == 0:
                        kk += 2 * n_ent
                        continue
                    for _ in range(n_ent):
                        t = np.int64(pack[kk])
                        if nprime != 0:
                            prod = a * np.uint64(pack[kk + 1])
                            q = ((prod & low32) * nprime) & low32
                            x = (prod + q * d_u) >> np.uint64(32)
                            inc = np.int64(x - d_u if x >= d_u else x)
                        else:
                            inc = (np.int64(a) * np.int64(pack[kk + 1])) % d
                        if t < nd:
                            y = det[shot, t] + inc
                            det[shot, t] = y - d if y >= d else y
                        else:
                            y = obs[shot, t - nd] + inc
                            obs[shot, t - nd] = y - d if y >= d else y
                        kk += 2


def _canonical_line(gen: dict, d: int):
    """Scales a sparse vector so its first entry is 1. Returns (hashable key, scaled dict)."""
    items = sorted(gen.items())
    inv = pow(items[0][1], -1, d)
    scaled = {t: (v * inv) % d for t, v in items}
    # scaled is built in sorted order, so its items are already the sorted key.
    return tuple(scaled.items()), scaled


def _projective_points(d: int, k: int):
    """Yields one point on each line through the origin of Z_d^k, scaled so its first non-zero coordinate is 1."""
    for lead in range(k):
        for tail in itertools.product(range(d), repeat=k - lead - 1):
            yield (0,) * lead + (1,) + tail
