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
use the same format, with detector and observable labels written as JSON
string literals on `DETECTOR Dk "label"` and `LOGICAL_OBSERVABLE Lk "label"`
lines.

The N1 lines show `pi` slightly above 0.01 because one of the d equally likely
shifts is the identity, so `pi = p / (1 - 1/d)`. The X fault after the second
N1 changes detector 0 by -1 and observable 0 by +1. A mechanism with one
generator adds a uniformly random multiple of it, so `merge_lines` scales
every such generator to make its first coefficient 1. The line becomes
`D0=1 L0=-1`, which prints as `L0=1000002` since coefficients are written as
residues mod d.

`dem.sample` checks and packs the whole model on every call. To draw many
small batches (a decoder taking 256 shots at a time, say), compile a sampler
once. Its calls continue one random stream, so with a seed the batches are,
row for row, what one `dem.sample` call with that seed returns:

```python
sampler = dem.compile_sampler(seed=5)
for _ in range(100):
    detectors, observables = sampler.sample(256)  # all 100: dem.sample(25_600, seed=5)
```

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

The unit-fault responses follow the frame update rules of
`sdim.program.simulate_frame`. Rather than pushing every unit fault forward
to the end of the circuit, which costs time quadratic in the number of rounds,
`from_circuit` makes one backward sweep, as stim does: for each qudit it keeps
the linear maps from its X and Z frame components to the detectors and
observables, and updates them with the transpose of each gate's frame rule.
Each noise gate then reads the responses of its unit faults directly.
Every response, and every generator built from one, lists its targets in
increasing order (detectors first, then observables). Detector and
observable coefficients are read from sdim's own compiled detector
expressions.

## Limitations

- The dimension must be prime, so that Z_d is a field. `from_circuit` with
  `check_dimension_prime=False` still builds the model for a composite d
  (the frame rules and the subgroup argument hold over Z_d), but `merge_lines`
  then leaves lines whose leading coefficient is not a unit unmerged, and
  `to_lines` refuses composite dimensions.
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
import functools
import gc
import itertools
import json
import math
import opcode
import operator
import re
import threading
import types

import numba
import numpy as np
from numba import njit

from ._jit import _kernel
from .circuit import Circuit
from .program import Program, _detector_mod

# Gate ids, in the order GateData registers the gates.
_X, _X_INV, _Z, _Z_INV = 1, 2, 3, 4
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
# The sampler's arithmetic needs d < 2**31 (see `_sample_chunks`), and so does `_expand_lines`.
_MAX_SAMPLE_DIMENSION = 2 ** 31 - 1
# With fewer unit-fault probes than this, the forward kernel runs them on the calling thread.
_PROBE_PARALLEL_MIN = 2048
# Compute unit-fault responses with one backward sweep (True) or by pushing every unit fault forward
# to the end of the circuit (False). Both give the same model; the forward kernel is the reference.
_BACKWARD = True


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


# Miller-Rabin with these bases is exact for every n below _MILLER_RABIN_EXACT (Sorenson and Webster, 2015).
_MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MILLER_RABIN_EXACT = 3317044064679887385961981


def _is_prime(n) -> bool:
    """
    Whether the integer n is prime, in microseconds for any size of n.

    Miller-Rabin with the 13 smallest prime bases decides every n below
    3.3 * 10**24 exactly. Larger n get the Baillie-PSW test (Miller-Rabin to
    base 2 and a strong Lucas test), which no known composite passes. (Trial
    division up to sqrt(n) never finishes for a DIMENSION such as 2**89 - 1.)
    A value that is not an integer is not prime.
    """
    try:
        n = operator.index(n)
    except TypeError:
        if not (isinstance(n, float) and n.is_integer()):
            return False
        n = int(n)
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n % p == 0:
            return n == p
    odd, s = n - 1, 0
    while odd % 2 == 0:
        odd //= 2
        s += 1
    # Baillie-PSW needs base 2 only; below the bound, the 13 bases are exact without the Lucas test.
    for a in (_MILLER_RABIN_BASES if n < _MILLER_RABIN_EXACT else (2,)):
        x = pow(a, odd, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return n < _MILLER_RABIN_EXACT or _strong_lucas_probable_prime(n)


def _jacobi(a: int, n: int) -> int:
    """The Jacobi symbol (a / n) for odd n > 0."""
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def _strong_lucas_probable_prime(n: int) -> bool:
    """
    The strong Lucas probable prime test with Selfridge's parameters, for odd n > 41.

    D is the first of 5, -7, 9, -11, ... with Jacobi symbol (D / n) = -1, P = 1
    and Q = (1 - D) / 4. Writing n + 1 = k * 2**s with k odd, n passes if
    U_k = 0 or V_(k * 2**r) = 0 (mod n) for some 0 <= r < s.
    """
    root = math.isqrt(n)
    if root * root == n:
        return False                    # no D has (D / n) = -1
    D = 5
    while True:
        j = _jacobi(D, n)
        if j == -1:
            break
        if j == 0 and abs(D) != n:
            return False                # D shares a factor with n
        D = -D - 2 if D > 0 else -D + 2
    Q = (1 - D) // 4
    k, s = n + 1, 0
    while k % 2 == 0:
        k //= 2
        s += 1

    def half(x):
        # x / 2 mod n (n is odd).
        x %= n
        return (x + n) // 2 if x % 2 else x // 2

    # U_m, V_m and Q**m for m = 1, then the bits of k from the top: m -> 2m, and m -> m + 1 on a 1 bit.
    U, V, Qm = 1, 1, Q % n
    for bit in bin(k)[3:]:
        U, V, Qm = U * V % n, (V * V - 2 * Qm) % n, Qm * Qm % n
        if bit == "1":
            U, V, Qm = half(U + V), half(D * U + V), Qm * Q % n
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V, Qm = (V * V - 2 * Qm) % n, Qm * Qm % n
        if V == 0:
            return True
    return False


def _padded_labels(labels, count) -> list:
    """`labels` (None counts as empty) padded with "" up to `count` entries."""
    labels = [] if labels is None else labels
    try:
        missing = int(count) - len(labels)
    except (TypeError, ValueError):
        return labels
    return list(labels) + [""] * missing if missing > 0 else labels


# A high surrogate followed by a low one: JSON decodes their two escapes as one character.
_SURROGATE_PAIR = re.compile("[\ud800-\udbff][\udc00-\udfff]")
# The first line of a file written by format v1, which wrote labels verbatim, contains this.
_V1_HEADER_TAG = "(format: qdem v1)"


def _read_label(text: str) -> str:
    """A label in a format v2 file: a JSON string literal, or else the text itself (as format v1 wrote it)."""
    if text.startswith('"'):
        try:
            value = json.loads(text)
        except ValueError:
            return text
        if isinstance(value, str):
            return value
    return text


def _as_integer(x, what: str) -> int:
    """`operator.index(x)`, or a ValueError naming the generator entry."""
    try:
        return operator.index(x)
    except TypeError:
        raise ValueError(f"a generator has {what} {x!r}, which is not an integer") from None


def _plain_int(x):
    """x as a Python int if it is an integer of any type (a NumPy integer, say), else x itself."""
    if type(x) is int:
        return x
    try:
        return operator.index(x)
    except TypeError:
        return x


def _all_ints(gens: list) -> bool:
    """Whether every target and coefficient of these generators is a Python int."""
    return (set(map(type, itertools.chain.from_iterable(gens))) <= {int}
            and set(map(type, itertools.chain.from_iterable(g.values() for g in gens))) <= {int})


def _normalize_generators(mechanisms) -> None:
    """
    Makes every integer target and coefficient in the mechanisms' generators a Python int.

    A mechanism with an entry of another type gets new generator dicts, in the same order.
    Entries that are not integers (floats, say) stay as they are, for `sample` to reject.
    """
    if _all_ints([g for m in mechanisms for g in m.generators]):
        return
    for m in mechanisms:
        if any(type(t) is not int or type(v) is not int for g in m.generators for t, v in g.items()):
            m.generators = [{_plain_int(t): _plain_int(v) for t, v in g.items()} for g in m.generators]


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
        ValueError: If pi is not in [0, 1].
    """
    if pi > 1.0:
        raise ValueError(f"pi = {pi} is above 1")
    if not pi >= 0.0:
        raise ValueError(f"pi = {pi} is not in [0, 1]")
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
        *pis (float): Probabilities of the mechanisms to merge, each in [0, 1].

    Returns:
        float: Probability of the merged mechanism, 0.0 if there are none.

    Raises:
        ValueError: If a probability is not in [0, 1] (NaN included).
    """
    for pi in pis:
        if not 0.0 <= pi <= 1.0:
            raise ValueError(f"cannot merge a mechanism with probability {pi}: it is not in [0, 1]")
    if any(pi >= 1.0 for pi in pis):
        return 1.0
    log_keep = sum(math.log1p(-pi) for pi in pis)
    # 0.0 - x turns the -0.0 of an all-zero merge into 0.0 and leaves every other value alone.
    return 0.0 - math.expm1(log_keep)


def _merge_pair(a: float, b: float) -> float:
    """
    `merge_subgroup_probabilities(a, b)` for two floats, bit for bit.

    For 0 < a, b < 1 neither log1p is -0.0, so summing the two terms directly
    gives exactly what `sum` does; other inputs go through the general function.
    """
    if 0.0 < a < 1.0 and 0.0 < b < 1.0:
        return 0.0 - math.expm1(math.log1p(-a) + math.log1p(-b))
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
    `read_from_file`. The constructor turns NumPy integers in the dimension,
    the counts and the generators of the given mechanisms into Python ints
    (one pass over their entries), so every method computes with exact ints.
    Mechanisms added later may hold NumPy integers too: `str` and
    `write_to_file` convert their generators the same way, `merge_lines`
    those of the rank-1 mechanisms it merges, and `to_lines` all of them
    when it expands them in Python (d above 2**31 - 1); `sample` reads them
    as Python ints.

    `sample` checks and packs every mechanism on each call. To draw many
    batches from one model, compile a sampler once and call it instead; its
    calls continue one random stream:

        sampler = dem.compile_sampler(seed=5)
        for _ in range(100):
            detectors, observables = sampler.sample(256)

    The 100 batches are, row for row, `dem.sample(25_600, seed=5)`.

    Attributes:
        dimension (int): Qudit dimension d.
        num_detectors (int): Number of detectors.
        num_observables (int): Number of logical observables.
        mechanisms (list[ErrorMechanism]): The error mechanisms.
        detector_labels (list[str]): Label of each detector, "" if it has none.
            A shorter list (the default is empty) is padded with "" to
            num_detectors entries, as `read_from_file` and `from_circuit` give.
        observable_labels (list[str]): Label of each logical observable, "" if
            it has none, padded the same way.
    """

    dimension: int
    num_detectors: int = 0
    num_observables: int = 0
    mechanisms: list = field(default_factory=list)
    detector_labels: list = field(default_factory=list)
    observable_labels: list = field(default_factory=list)

    def __post_init__(self):
        # NumPy integers have no three-argument pow and overflow int64 (d ** k in to_lines, say).
        self.dimension = _plain_int(self.dimension)
        self.num_detectors = _plain_int(self.num_detectors)
        self.num_observables = _plain_int(self.num_observables)
        _normalize_generators(self.mechanisms)
        self.detector_labels = _padded_labels(self.detector_labels, self.num_detectors)
        self.observable_labels = _padded_labels(self.observable_labels, self.num_observables)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_circuit(cls, circuit: Circuit, merge: bool = True, check_dimension_prime: bool = True) -> "DetectorErrorModel":
        """
        Builds the DEM of a noisy circuit.

        Every N1 or N2 gate with non-zero probability and a visible effect
        becomes one mechanism, so the size of the model does not depend on d.
        Negative qudit indices count back from the end of the circuit's
        qudits, as in `Program`.

        Args:
            circuit (Circuit): Circuit with N1/N2 noise and DETECTOR /
                LOGICAL_OBSERVABLE instructions.
            merge (bool): Merge rank-1 mechanisms that act on the same line of
                detector space, see `merge_lines`. Defaults to True.
            check_dimension_prime (bool): Raise if the dimension is not prime.
                Defaults to True. With False, a composite d is compiled with
                the same frame rules, which hold over Z_d; merging then skips
                lines whose leading coefficient is not a unit mod d, and
                `to_lines` and `read_from_file` (by default) refuse the model.

        Returns:
            DetectorErrorModel: The compiled model. Every generator lists its
                targets in increasing order.

        Raises:
            ValueError: If the dimension is not prime, an N1 gate has an
                unknown noise channel, an N2 gate uses `prob_dist`, a noise
                probability is above the fully mixing value, a two-qudit gate
                acts on one qudit twice, or a detector or observable is not
                linear in its records or not deterministic without noise.
            IndexError: If a gate acts on a qudit outside the circuit.
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

        Two generators are on the same line when one is a unit multiple of
        the other. Each generator is reduced mod d, sorted by target, and
        scaled so its first coefficient is 1, and mechanisms with equal scaled
        generators are merged with `merge_subgroup_probabilities`, in order of
        first appearance. The merged source joins the sources of all the
        merged mechanisms with "+", in order; each source goes into exactly
        one merged source, so the sources take no more room than before.
        Mechanisms of rank 2 or more come after the lines, as they are,
        since two of them almost never share a subgroup.

        A generator with no non-zero coefficient, or (composite d only) one
        whose first coefficient is not a unit mod d, cannot be scaled that
        way; its mechanism stays in its place, reduced and sorted but not
        merged with any other.

        Raises:
            ValueError: If two merged mechanisms have probabilities outside [0, 1].
        """
        d = _plain_int(self.dimension)
        merged: dict = {}
        sources: dict = {}
        lines: list = []
        others: list = []
        with _gc_paused():
            for mech in self.mechanisms:
                (lines if len(mech.generators) == 1 else others).append(mech)
            # Mechanisms added after the constructor may hold NumPy integers, which overflow here.
            _normalize_generators(lines)
            for mech in lines:
                key, scaled = _canonical_line(mech.generators[0], d)
                if key is None:
                    merged[object()] = ErrorMechanism(mech.probability, [scaled], mech.source)
                    continue
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

        Each line mechanism, when it fires, adds a uniformly random multiple
        a * v (a uniform on Z_d, zero included) of its one vector v. At d = 2
        that is a single shift applied with probability pi / 2; for d > 2 the
        d - 1 non-zero multiples of v are equally likely. The number of lines
        grows like d**(k-1), so this is only practical for small d. It is the
        form to use for decoders that expect independent mechanisms that each
        move along one direction of detector space, and for comparing with
        stim at d = 2.

        Args:
            max_lines_per_mechanism (int): Raise instead of expanding a
                mechanism into more lines than this. Defaults to 10**6.

        Returns:
            DetectorErrorModel: A new model in which every mechanism has rank 1.

        Raises:
            ValueError: If the dimension is not prime (the line decomposition
                needs Z_d to be a field), a mechanism needs more than
                `max_lines_per_mechanism` lines, or a probability is not in [0, 1].
        """
        d = _plain_int(self.dimension)
        if not _is_prime(d):
            raise ValueError(f"to_lines needs a prime dimension, not {d}: the split into independent line "
                             "mechanisms only holds when Z_d is a field")
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
            # d above 2**31 - 1, or entries that are not integers: expand one dict at a time.
            _normalize_generators(mechs)
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
        Packs the mechanisms into flat arrays for the numba sampler, checking every entry.

        Returns mech_prob (probability of each mechanism), n_gens (number of
        generators of each mechanism), sizes (number of entries of each
        generator), and ent_tgt / ent_val (targets, and coefficients reduced
        mod d, of all entries, generator by generator, in dict order).

        Raises:
            ValueError: If a target or coefficient is not an integer, or a
                target is outside 0 .. num_detectors + num_observables - 1.
        """
        mechs = self.mechanisms
        d = int(self.dimension)
        n_targets = int(self.num_detectors) + int(self.num_observables)
        mech_prob = np.array([m.probability for m in mechs], dtype=np.float64)
        n_gens = np.fromiter((len(m.generators) for m in mechs), dtype=np.int64, count=len(mechs))
        gens = [g for m in mechs for g in m.generators]
        sizes = np.fromiter(map(len, gens), dtype=np.int64, count=len(gens))
        n_ent = int(sizes.sum())
        ent_tgt = ent_val = None
        targets = itertools.chain.from_iterable(gens)
        values = itertools.chain.from_iterable(g.values() for g in gens)
        if set(map(type, targets)) <= {int} and set(map(type, values)) <= {int}:
            # Plain ints: convert in bulk. Anything beyond int64 takes the entry-by-entry path.
            try:
                ent_tgt = np.fromiter(itertools.chain.from_iterable(gens), dtype=np.int64, count=n_ent)
                ent_val = np.fromiter(itertools.chain.from_iterable(g.values() for g in gens), dtype=np.int64,
                                      count=n_ent)
            except OverflowError:
                ent_tgt = None
        if ent_tgt is None:
            ent_tgt = np.empty(n_ent, dtype=np.int64)
            ent_val = np.empty(n_ent, dtype=np.int64)
            k = 0
            for g in gens:
                for t, v in g.items():
                    index = _as_integer(t, "target")
                    if not 0 <= index < n_targets:
                        raise ValueError(self._bad_target_message(t))
                    ent_tgt[k] = index
                    # Reduced in Python first, so coefficients of any size are exact.
                    ent_val[k] = _as_integer(v, "coefficient") % d
                    k += 1
        bad = np.flatnonzero((ent_tgt < 0) | (ent_tgt >= n_targets))
        if len(bad):
            raise ValueError(self._bad_target_message(int(ent_tgt[bad[0]])))
        # Coefficients as residues mod d (np.remainder by a positive int64 is never negative).
        np.remainder(ent_val, d, out=ent_val)
        return mech_prob, n_gens, sizes, ent_tgt, ent_val

    def _bad_target_message(self, target) -> str:
        return (f"a generator refers to target {target!r}, but the model has {self.num_detectors} detectors "
                f"and {self.num_observables} observables (targets 0 to "
                f"{int(self.num_detectors) + int(self.num_observables) - 1})")

    def sample(self, shots: int, seed: int | None = None):
        """
        Samples detector and observable values.

        Mechanisms are grouped into bins of similar probability. Within a bin
        the sampler jumps straight to the next candidate firing with a
        geometric skip, and a mechanism whose probability is below the bin's
        maximum keeps each candidate with probability pi / pi_max. Each shot
        only visits the bins that have a candidate in it (found in a bitmap
        of the bins), so the cost grows with the number of firings plus one
        skip per bin per block of shots, not with shots * len(mechanisms) or
        shots * (number of bins).

        Shots are split into fixed blocks of 256, and large jobs spread the
        blocks over several threads (numba's thread count, see
        `_thread_count`). Each block has its own random stream
        (xoshiro256**) seeded from `seed` through NumPy's SeedSequence, so a
        given seed gives the same samples whatever the number of threads.

        Targets and coefficients must be integers (Python or NumPy). Every
        target must be a detector or observable of the model, and
        coefficients are reduced mod d, whatever their size.

        Every call checks and packs the whole model before it draws. To
        draw many small batches, use `compile_sampler`, which does that once.

        Args:
            shots (int): Number of samples.
            seed (int, optional): Seed for the sampler. Fresh entropy is used if None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Detector values with shape
                (shots, num_detectors) and observable values with shape
                (shots, num_observables), as int64 residues mod d.

        Raises:
            ValueError: If the dimension is not between 1 and 2**31 - 1, a
                mechanism probability is NaN, a target or coefficient is not
                an integer, or a generator refers to a target outside
                0 .. num_detectors + num_observables - 1.
        """
        nd = self.num_detectors
        det = np.zeros((shots, nd), dtype=np.int64)
        obs = np.zeros((shots, self.num_observables), dtype=np.int64)
        if not self.mechanisms or shots == 0:
            return det, obs
        _draw_blocks(_sampler_arrays(self), _BlockStream(seed), 0, det, obs)
        return det, obs

    def compile_sampler(self, seed: int | None = None) -> "CompiledDemSampler":
        """
        Checks and packs the model once, for a sampler that draws many batches from it.

        The sampler's `sample(shots)` returns what `sample` does, without the
        per-call work on the model. Its calls continue one random stream:
        the rows of consecutive calls, put together, are the rows that
        `sample(total, seed)` returns. The sampler keeps its own packed copy
        of the model, so later changes to the model do not reach it.

        Args:
            seed (int, optional): Seed of the sampler's stream. Fresh entropy is used if None.

        Returns:
            CompiledDemSampler: The sampler.

        Raises:
            ValueError: As `sample`, if the model has a mechanism and cannot be sampled.
        """
        return CompiledDemSampler(self, seed)

    # -------------------------------------------------------------------- io
    def __str__(self) -> str:
        """
        The model in the format `read_from_file` reads, without the header comments.

        Labels are written as JSON string literals (with every non-ASCII
        character escaped), so any label, including one with line breaks,
        '#' or leading and trailing spaces, reads back exactly. A source is a
        comment: line breaks in it are written as spaces.
        """
        lines = [f"DIMENSION {self.dimension}",
                 f"DETECTORS {self.num_detectors}",
                 f"OBSERVABLES {self.num_observables}"]
        for i, label in enumerate(self.detector_labels):
            if label:
                lines.append(f"DETECTOR D{i} {json.dumps(str(label))}")
        for i, label in enumerate(self.observable_labels):
            if label:
                lines.append(f"LOGICAL_OBSERVABLE L{i} {json.dumps(str(label))}")
        # Mechanisms added after the constructor may hold NumPy integers (a uint64 target minus
        # num_detectors is a float under NumPy 1.x).
        _normalize_generators(self.mechanisms)
        nd = _plain_int(self.num_detectors)
        for m in self.mechanisms:
            gens = " | ".join([" ".join([f"D{t}={v}" if t < nd else f"L{t - nd}={v}"
                                         for t, v in sorted(g.items())]) for g in m.generators])
            tag = f" # {' '.join(str(m.source).splitlines())}" if m.source else ""
            lines.append(f"ERROR({float(m.probability)!r}) {gens}{tag}")
        return "\n".join(lines) + "\n"

    def write_to_file(self, path: str | Path, comment: str = "") -> None:
        """
        Writes the model to a UTF-8 text file, in the format of `str(self)` after a header.

        Args:
            path (str or Path): Output path.
            comment (str, optional): Extra text, written as `#` lines in the header.

        Raises:
            ValueError: If a label has a lone high surrogate right before a
                lone low surrogate (a Python str can hold them). A JSON
                string literal cannot tell them from the one character
                they encode in UTF-16, so the label would not read back.
        """
        for label in itertools.chain(self.detector_labels, self.observable_labels):
            if label and _SURROGATE_PAIR.search(str(label)):
                raise ValueError(f"label {label!r} has a lone surrogate pair, which a JSON string literal reads "
                                 "back as one character; it cannot be written")
        header = [
            "# sdim compact qudit detector error model (format: qdem v2)",
            "# ERROR(pi) g_1 | g_2 | ... : with probability pi, add sum_j a_j g_j,",
            "#   a_j i.i.d. uniform on Z_d (identity included); mechanisms are independent.",
            "# Coefficients are residues mod DIMENSION.  Targets Dk are detectors, Lk observables.",
            "# Labels are JSON string literals.",
        ]
        if comment:
            header += ["# " + line for line in comment.splitlines()]
        Path(path).write_text("\n".join(header) + "\n#\n" + str(self), encoding="utf-8")

    @classmethod
    def read_from_file(cls, path: str | Path, check_dimension_prime: bool = True) -> "DetectorErrorModel":
        """
        Reads a model written by `write_to_file`.

        Labels are JSON string literals (format v2). A file whose first line
        is the header that format v1 wrote ("... (format: qdem v1)") has its
        labels read verbatim, as v1 wrote them, quotes and backslashes
        included. In any other file, a label that is not a valid JSON string
        literal is read verbatim as well.

        Args:
            path (str or Path): Input path, a UTF-8 text file.
            check_dimension_prime (bool): Raise if DIMENSION is not prime.
                Defaults to True, since the compact model needs Z_d to be a
                field (see `from_circuit`).

        Returns:
            DetectorErrorModel: The model in the file.

        Raises:
            ValueError: If the file is not in the format, DIMENSION is not an
                integer of at least 2 (or not prime), a header line is
                repeated, or a line refers to a detector or observable outside
                the declared counts.
        """
        dem = None
        det_labels: dict = {}
        obs_labels: dict = {}
        mechanisms = []
        seen = set()
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        # Format v1 wrote labels verbatim; its files start with this header line.
        verbatim_labels = bool(lines) and lines[0].lstrip().startswith("#") and _V1_HEADER_TAG in lines[0]
        for number, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            head = stripped.split()[0]
            if head != "DIMENSION" and dem is None:
                raise ValueError(f"line {number}: DIMENSION must come first")
            if head in ("DETECTOR", "LOGICAL_OBSERVABLE"):
                # The label is the rest of the line, so it may contain '#'.
                parts = stripped.split(maxsplit=2)
                prefix, labels = ("D", det_labels) if head == "DETECTOR" else ("L", obs_labels)
                if len(parts) < 2 or not parts[1].startswith(prefix) or not parts[1][1:].isdigit():
                    raise ValueError(f"line {number}: expected {head} {prefix}<index> <label>")
                if len(parts) < 3:
                    label = ""
                else:
                    label = parts[2] if verbatim_labels else _read_label(parts[2])
                labels[int(parts[1][1:])] = label
                continue
            line, _, source = raw.partition("#")
            line, source = line.strip(), source.strip()
            if head in ("DIMENSION", "DETECTORS", "OBSERVABLES"):
                if head in seen:
                    raise ValueError(f"line {number}: {head} is given twice")
                seen.add(head)
                fields = line.split()
                value = int(fields[1]) if len(fields) == 2 and fields[1].isascii() and fields[1].isdigit() else None
                if head == "DIMENSION":
                    if value is None or value < 2:
                        raise ValueError(f"line {number}: DIMENSION must be an integer of at least 2, not "
                                         f"{' '.join(fields[1:])!r}")
                    if check_dimension_prime and not _is_prime(value):
                        raise ValueError(f"line {number}: DIMENSION {value} is not prime; compact qudit DEMs "
                                         "require a prime dimension (Z_d must be a field)")
                    dem = cls(value)
                elif value is None:
                    raise ValueError(f"line {number}: {head} must be a non-negative integer, not "
                                     f"{' '.join(fields[1:])!r}")
                elif head == "DETECTORS":
                    dem.num_detectors = value
                else:
                    dem.num_observables = value
            elif head.startswith("ERROR("):
                prob_text, closed, rest = line[len("ERROR("):].partition(")")
                try:
                    probability = float(prob_text) if closed else None
                except ValueError:
                    probability = None
                if probability is None:
                    raise ValueError(f"line {number}: expected ERROR(<probability>), not {line!r}")
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"line {number}: probability {probability} is not in [0, 1]")
                gens = []
                for chunk in rest.split("|"):
                    gen = {}
                    for item in chunk.split():
                        name, _, value = item.partition("=")
                        if (name[:1] not in ("D", "L") or not name[1:].isascii() or not name[1:].isdigit()
                                or not value.lstrip("-").isascii() or not value.lstrip("-").isdigit()):
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
        for name, labels, count in (("D", det_labels, dem.num_detectors), ("L", obs_labels, dem.num_observables)):
            if labels and max(labels) >= count:
                raise ValueError(f"a label is given for {name}{max(labels)}, but the model has only {count} "
                                 + ("detectors" if name == "D" else "observables"))
        dem.mechanisms = [m for m in mechanisms if m.generators]
        dem.detector_labels = [det_labels.get(i, "") for i in range(dem.num_detectors)]
        dem.observable_labels = [obs_labels.get(i, "") for i in range(dem.num_observables)]
        return dem


class CompiledDemSampler:
    """
    Draws batches of detector and observable values from a `DetectorErrorModel`.

    Made by `DetectorErrorModel.compile_sampler`, which checks the model and
    packs it into the sampler's arrays once, so each `sample` call only
    draws. The arrays are a copy: later changes to the model do not reach
    the sampler.

    The shots come in blocks of 256, block c drawn from the c-th state that
    the seed gives (see `DetectorErrorModel.sample`), so consecutive calls
    continue one stream: their rows, put together, are the rows of
    `DetectorErrorModel.sample(total, seed)`, however the shots are split
    between the calls and whatever the number of threads. The stream does
    not repeat, however many shots it gives. A call that raises leaves the
    stream where it was, or, if an interrupt (Ctrl-C) comes just as it
    returns, skips the call's rows: no row is ever returned twice.

    A call that ends inside a block draws the whole block and keeps the
    rows it does not return for the next call, so between calls a sampler
    holds up to one block: 256 rows of the model's detectors and
    observables. Calls from several threads take turns. A sampler cannot
    be pickled or deep-copied; to sample in several processes, compile one
    in each, with its own seed.

    Attributes:
        num_detectors (int): Number of detectors of the model (read-only).
        num_observables (int): Number of logical observables of the model (read-only).
    """

    def __init__(self, dem: DetectorErrorModel, seed: int | None = None):
        # The packed targets are fixed, so the widths of the arrays the kernel writes into must be too.
        self._num_detectors = dem.num_detectors
        self._num_observables = dem.num_observables
        self._arrays = _sampler_arrays(dem) if dem.mechanisms else None
        self._stream = _BlockStream(seed, ahead=64)
        self._next_block = 0
        # The last block drawn, while the calls have returned only its first rows: (det, obs, rows returned).
        self._rest = None
        self._lock = threading.Lock()

    @property
    def num_detectors(self) -> int:
        """Number of detectors of the model."""
        return self._num_detectors

    @property
    def num_observables(self) -> int:
        """Number of logical observables of the model."""
        return self._num_observables

    def sample(self, shots: int):
        """
        Draws the next `shots` samples of the stream.

        Args:
            shots (int): Number of samples, a Python or NumPy integer.

        Returns:
            tuple[np.ndarray, np.ndarray]: Detector values with shape
                (shots, num_detectors) and observable values with shape
                (shots, num_observables), as int64 residues mod d.

        Raises:
            TypeError: If shots is not an integer.
            ValueError: If shots is negative.
        """
        shots = operator.index(shots)
        if shots < 0:
            raise ValueError(f"cannot draw {shots} shots")
        det = np.zeros((shots, self._num_detectors), dtype=np.int64)
        obs = np.zeros((shots, self._num_observables), dtype=np.int64)
        if self._arrays is None or shots == 0:
            return det, obs
        with self._lock:
            block, rest = self._next_block, self._rest
            done = 0
            if rest is not None:
                rest_det, rest_obs, used = rest
                done = min(shots, _SAMPLE_CHUNK - used)
                det[:done] = rest_det[used:used + done]
                obs[:done] = rest_obs[used:used + done]
                rest = (rest_det, rest_obs, used + done) if used + done < _SAMPLE_CHUNK else None
            whole = (shots - done) // _SAMPLE_CHUNK * _SAMPLE_CHUNK
            if whole:
                block = _draw_blocks(self._arrays, self._stream, block, det[done:done + whole],
                                     obs[done:done + whole])
                done += whole
            if done < shots:
                rest_det = np.zeros((_SAMPLE_CHUNK, self._num_detectors), dtype=np.int64)
                rest_obs = np.zeros((_SAMPLE_CHUNK, self._num_observables), dtype=np.int64)
                block = _draw_blocks(self._arrays, self._stream, block, rest_det, rest_obs)
                det[done:] = rest_det[:shots - done]
                obs[done:] = rest_obs[:shots - done]
                rest = (rest_det, rest_obs, shots - done)
            # The stream moves on only here, so a call that raised above (an interrupt, say) left it as it was.
            self._next_block, self._rest = block, rest
        return det, obs


def _sampler_arrays(dem: DetectorErrorModel) -> tuple:
    """
    Checks a model that has mechanisms and packs it for `_sample_chunks`.

    Returns (cost, args): the estimated work per shot, and the arguments of
    `_sample_chunks` after `states`, which are the same for every block.

    Raises:
        ValueError: As `DetectorErrorModel.sample`.
    """
    d = int(dem.dimension)
    if not 1 <= d <= _MAX_SAMPLE_DIMENSION:
        raise ValueError(f"sample() needs a dimension between 1 and 2**31 - 1, not {d}")
    mech_prob, n_gens, sizes, ent_tgt, ent_val = dem._flatten()
    n_targets = dem.num_detectors + dem.num_observables
    montgomery = d % 2 == 1
    # int32 pack entries halve the sampler's memory traffic; they hold targets below n_targets and
    # residues below d.
    pack_dtype = np.int32 if n_targets <= np.iinfo(np.int32).max else np.int64
    pack, info, always_off, bin_ptr, bin_pmax, bin_log_keep, cost = _sample_plan(
        mech_prob, n_gens, sizes, ent_tgt, ent_val, d, montgomery, pack_dtype)
    # Every block draws one skip per bin, and each shot scans a bitmap of the bins.
    cost += len(bin_pmax) / _SAMPLE_CHUNK
    # -d^-1 mod 2**32 for Montgomery multiplication (odd d); 0 selects plain % for even d.
    nprime = np.uint64((-pow(d, -1, 1 << 32)) % (1 << 32) if montgomery else 0)
    thresh = np.uint64((1 << 32) % d)
    args = (bin_ptr, bin_pmax, bin_log_keep, info, info.view(np.float64), always_off, pack, d, thresh, nprime)
    return cost, args


def _draw_blocks(arrays: tuple, stream: _BlockStream, first: int, det, obs) -> int:
    """
    Fills det and obs, zero arrays with one row per shot, from blocks first, first + 1, ... of a stream.

    `arrays` is what `_sampler_arrays` returns, and `stream` gives the
    blocks' random states. A last block of k < 256 rows gets the first k
    rows of the whole block, since the kernel finishes each shot before the
    next and draws nothing for the shots past its end.

    Returns:
        int: The block after the last one used.
    """
    cost, (bin_ptr, bin_pmax, bin_log_keep, info, info_p, always_off, pack, d, thresh, nprime) = arrays
    shots = det.shape[0]
    n_chunks = -(-shots // _SAMPLE_CHUNK)
    states = stream.states(first, n_chunks)
    n_threads = _thread_count() if n_chunks > 1 and shots * cost > _SAMPLE_PARALLEL_WORK else 1
    n_tasks = min(n_chunks, n_threads * _SAMPLE_TASKS_PER_THREAD) if n_threads > 1 else 1
    bounds = [n_chunks * i // n_tasks for i in range(n_tasks + 1)]

    def task(i):
        _sample_chunks(bounds[i], bounds[i + 1], det, obs, _SAMPLE_CHUNK, states, bin_ptr, bin_pmax,
                       bin_log_keep, info, info_p, always_off, pack, d, thresh, nprime)

    _run_tasks(task, n_tasks, n_threads)
    return first + n_chunks


# ---------------------------------------------------------------------------
# Unit-response compilation


@dataclass
class NoiseLocation:
    """
    One N1 or N2 gate and the detector responses of its unit faults.

    Attributes:
        ir_index (int): Position of the gate in the program IR.
        gate_id (int): 17 for N1, 18 for N2.
        qudits (tuple): Qudits the gate acts on (a negative index as the qudit it counts back to).
        channel (str): "d", "f" or "p" for N1, and "d2" for N2.
        probability (float): The gate's `prob` parameter.
        subgroup_probability (float): Probability of the equivalent subgroup mechanism.
        responses (list[dict[int, int]]): Sparse response of each unit fault. The
            order is X then Z on each qudit, with only X for "f" and only Z for "p".
            Each dict lists its targets in increasing order.
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


# A product of two polynomials is only expanded when their numbers of terms multiply to at most
# this; a larger one sends the detector to the numeric path. This bounds the work for products
# and powers of long sums.
_MAX_PRODUCT_TERMS = 4096


@functools.lru_cache(maxsize=64)
def _fermat_period(d: int) -> int:
    """d - 1 if d is prime, else 0. For prime d, x ** e = x ** ((e - 1) % (d - 1) + 1) on Z_d for every e >= 1."""
    return d - 1 if _is_prime(d) else 0


def _powers(m) -> tuple:
    """A monomial of `_Polynomial.c` as its tuple of (position, exponent) pairs."""
    return ((m, 1),) if type(m) is int else m


def _monomial_product(a: tuple, b: tuple, period: int):
    """
    The `_Polynomial.c` key of the product of two monomials given as (position, exponent) pairs.

    With period = d - 1 (prime d) exponents are reduced to 1 .. d - 1, which keeps the function
    on Z_d the same, since x ** d = x there.
    """
    powers = dict(a)
    for j, e in b:
        e += powers.get(j, 0)
        powers[j] = (e - 1) % period + 1 if period else e
    m = tuple(sorted(powers.items()))
    return m[0][0] if len(m) == 1 and m[0][1] == 1 else m


class _Polynomial:
    """
    A polynomial k + s * (sum of c[m] * m over monomials m in the records), coefficients mod d.

    `_detector_coefficients` calls a compiled detector expression once, on a
    `_Records` sequence of these. The operations below are the only ones
    defined, and each one turns values congruent mod d to its operands into a
    value congruent mod d to its result: + and -, *, ** by a non-negative
    integer constant, and % by a non-zero multiple of d. So when the call
    succeeds, the expression is congruent mod d to the returned polynomial for
    every integer input. Any other operation raises TypeError, and the caller
    falls back to evaluating the expression on numeric probes.

    c is a sparse dict {monomial: non-zero coefficient}. The monomial rec[j - 1]
    is the int j (1 .. n), and one of degree 2 or more is the sorted tuple of
    its (j, exponent) pairs. For prime d every exponent stays in 1 .. d - 1
    (x ** d = x on Z_d), and two such polynomials with the same values on
    Z_d^n are the same polynomial, so the expression is affine on Z_d^n exactly
    when no monomial of degree 2 or more is left. The scale s is a unit mod d.

    In the expressions this is used on (straight-line arithmetic, see
    `_is_straight_line_arithmetic`) every value is used exactly once, and
    `_Records` hands out a new polynomial for every rec[j], so an operation may
    reuse its operands: a sum adds the smaller dict into the larger one, and
    negation and multiplication by a constant only change k and s. An
    expression over many records (an observable that reads every round, say)
    thus costs time about linear in its length, where dense coefficient tuples
    cost its length times the number of records. Products and powers of
    polynomials with records in both factors are expanded into new ones.
    """

    __slots__ = ("k", "s", "c", "n", "d")

    def __init__(self, k: int, s: int, c: dict, n: int, d: int):
        self.k = k
        self.s = s
        self.c = c
        self.n = n
        self.d = d

    def coefficients(self):
        """
        The dense tuple (c_0, c_1, ..., c_n) mod d, c_0 being the constant term, or None if a
        monomial of degree 2 or more is left.
        """
        out = [0] * (self.n + 1)
        out[0] = self.k
        s, d = self.s, self.d
        for m, v in self.c.items():
            if type(m) is not int:
                return None
            out[m] = v * s % d
        return tuple(out)

    def _form(self, other):
        if type(other) is _Polynomial and other.n == self.n:
            return other
        if type(other) is int:
            return _Polynomial(other % self.d, 1, {}, self.n, self.d)
        raise TypeError("not a polynomial operation")

    def _scaled(self, m: int):
        """The polynomial times the constant m, in place."""
        d = self.d
        self.k = self.k * m % d
        s = self.s * m % d
        if s == 0:
            self.c, self.s = {}, 1
        elif math.gcd(s, d) != 1:
            # Composite d only: a non-unit scale could turn entries into zeros, so apply it now.
            self.c = {j: v * s % d for j, v in self.c.items() if v * s % d}
            self.s = 1
        else:
            self.s = s
        return self

    def _product(self, other):
        """The product with another polynomial, expanded into a new one; neither factor changes."""
        if len(self.c) * len(other.c) > _MAX_PRODUCT_TERMS:
            raise TypeError("too many terms to expand")
        d = self.d
        period = _fermat_period(d)
        # (k1 + A)(k2 + B) = k1 k2 + k2 A + k1 B + A B
        c = {}
        for poly, k in ((self, other.k), (other, self.k)):
            if k:
                for m, v in poly.c.items():
                    c[m] = (c.get(m, 0) + v * poly.s * k) % d
        b = [(_powers(m), v * other.s % d) for m, v in other.c.items()]
        for m, v in self.c.items():
            a, v = _powers(m), v * self.s % d
            for bm, w in b:
                key = _monomial_product(a, bm, period)
                c[key] = (c.get(key, 0) + v * w) % d
        return _Polynomial(self.k * other.k % d, 1, {m: v for m, v in c.items() if v}, self.n, d)

    def __add__(self, other):
        other = self._form(other)
        if other is self:
            return self._scaled(2)
        big, small = (self, other) if len(self.c) >= len(other.c) else (other, self)
        d = self.d
        big.k = (big.k + small.k) % d
        if small.c:
            # small's entries in big's scale; m is a unit, so every term it adds is non-zero.
            m = small.s * pow(big.s, -1, d) % d
            c = big.c
            for j, v in small.c.items():
                x = (c.get(j, 0) + v * m) % d
                if x:
                    c[j] = x
                else:
                    del c[j]
        return big

    __radd__ = __add__

    def __sub__(self, other):
        other = self._form(other)
        if other is self:
            return self._scaled(0)
        return self + other._scaled(-1)

    def __rsub__(self, other):
        return self._scaled(-1) + other

    def __neg__(self):
        return self._scaled(-1)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if type(other) is int:
            return self._scaled(other)
        if type(other) is _Polynomial and other.n == self.n:
            if not other.c:
                return self._scaled(other.k)
            if not self.c:
                return other._scaled(self.k)
            return self._product(other)
        raise TypeError("not a polynomial operation")

    __rmul__ = __mul__

    def __pow__(self, e, mod=None):
        if type(e) is not int or e < 0 or mod is not None:
            raise TypeError("not a polynomial operation")
        d = self.d
        if e == 0 or not self.c:
            return _Polynomial(pow(self.k, e, d), 1, {}, self.n, d)
        period = _fermat_period(d)
        if period:
            # Every function f on Z_d^n has f ** d = f (Fermat), so only e mod d - 1 matters.
            e = (e - 1) % period + 1
        result, power = None, self
        while True:
            if e & 1:
                result = power if result is None else result._product(power)
            e >>= 1
            if not e:
                return result
            power = power._product(power)

    def __mod__(self, other):
        if type(other) is int and other != 0 and other % self.d == 0:
            return self
        raise TypeError("not a polynomial operation")

    # Anything that could branch on a value or turn it into something else is refused.
    def _refuse(self, *args):
        raise TypeError("not a polynomial operation")

    __bool__ = __index__ = __int__ = __float__ = __str__ = __format__ = _refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __hash__ = None


class _Records:
    """The `rec` argument of the symbolic call: each rec[j] is a new polynomial 1 * rec[j], as a list would index."""

    __slots__ = ("n", "d")

    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        if type(j) is not int:
            raise TypeError("not a polynomial operation")
        if j < 0:
            j += self.n
        if not 0 <= j < self.n:
            raise IndexError("record index out of range")
        return _Polynomial(0, 1, {j + 1: 1} if self.d > 1 else {}, self.n, self.d)


# Bytecode a detector expression may contain for the single symbolic evaluation: loading the
# record list and integer constants, indexing, unary minus, and the binary operators +, -, *, **
# and %. Anything else (calls, names, branches, comparisons, ...) takes the numeric path.
_ARITHMETIC_OPNAMES = frozenset({
    "RESUME", "NOP", "CACHE", "EXTENDED_ARG", "RETURN_VALUE",
    "LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "LOAD_CONST", "LOAD_SMALL_INT", "LOAD_COMMON_CONSTANT",
    "BINARY_SUBSCR", "UNARY_NEGATIVE", "BINARY_OP",
    "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY", "BINARY_MODULO",
})
_ARITHMETIC_BINARY_OPS = frozenset({"+", "-", "*", "**", "%", "[]"})


_CALL_OPNAMES = frozenset({"LOAD_GLOBAL", "PUSH_NULL", "PRECALL", "CALL"})


def _opcodes(names) -> bytes:
    """The opcodes this Python has for these instruction names, as the bytes `bytes.translate` deletes."""
    return bytes(sorted({dis.opmap[name] for name in names if name in dis.opmap}))


# `_is_straight_line_arithmetic` reads co_code itself: an opcode byte and an argument byte per
# instruction, with the inline caches after some instructions as zero bytes (CACHE). Opcodes
# differ between Python versions, so they are looked up by name, and the BINARY_OP arguments of
# the operators above are read off this Python's own bytecode for them.
_ARITHMETIC_OPCODES = _opcodes(_ARITHMETIC_OPNAMES)
_CALL_OR_ARITHMETIC_OPCODES = _opcodes(_ARITHMETIC_OPNAMES | _CALL_OPNAMES)
_BINARY_OP = dis.opmap.get("BINARY_OP")
_LOAD_CONST = dis.opmap["LOAD_CONST"]
_EXTENDED_ARG = dis.opmap["EXTENDED_ARG"]
# Python 3.15 loads some constants, -1 among them, with LOAD_COMMON_CONSTANT, whose argument
# indexes the table that dis reads its values from. Without that table the opcode is refused
# and such expressions take the numeric path.
_LOAD_COMMON_CONSTANT = dis.opmap.get("LOAD_COMMON_CONSTANT")
_INT_COMMON_CONSTANTS = frozenset(
    i for i, value in enumerate(getattr(opcode, "_common_constants", ())) if type(value) is int)
_ARGUMENT_OPCODES = frozenset({_BINARY_OP, _LOAD_CONST, _EXTENDED_ARG, _LOAD_COMMON_CONSTANT})
_ARITHMETIC_BINARY_OP_ARGS = frozenset(
    ins.arg for ins in dis.get_instructions(compile("a + a, a - a, a * a, a ** a, a % a, a[a]", "<ops>", "eval"))
    if ins.opname == "BINARY_OP" and ins.argrepr in _ARITHMETIC_BINARY_OPS)


def _is_straight_line_arithmetic(fn) -> bool:
    """True if `fn` is a one-argument function whose bytecode only uses `_ARITHMETIC_OPNAMES`.

    sdim.program compiles detectors as `lambda rec : _detector_mod((expr), d)`, and
    `_detector_mod(x, d)` is `x % d` for anything but int64 arrays. Calls to that one helper are
    allowed too, when the name really refers to sdim's own function.

    The scan walks co_code directly; `dis.get_instructions` takes microseconds per instruction,
    which made it most of the work of reading a long observable.
    """
    if type(fn) is not types.FunctionType or fn.__defaults__ or fn.__kwdefaults__ or fn.__closure__:
        return False
    code = fn.__code__
    wraps_mod = (code.co_names == ("_detector_mod",)
                 and fn.__globals__.get("_detector_mod") is _detector_mod)
    if (code.co_argcount != 1 or code.co_kwonlyargcount or (code.co_names and not wraps_mod) or code.co_freevars
            or code.co_cellvars or code.co_flags & (0x04 | 0x08)):   # *args, **kwargs
        return False
    # With calls allowed, co_names is ("_detector_mod",), the only name LOAD_GLOBAL can load.
    ops, args = code.co_code[::2], code.co_code[1::2]
    if ops.translate(None, _CALL_OR_ARITHMETIC_OPCODES if wraps_mod else _ARITHMETIC_OPCODES):
        return False
    # The arguments that matter: the operator of each BINARY_OP and the constant each LOAD_CONST
    # or LOAD_COMMON_CONSTANT loads. EXTENDED_ARG holds the high bits of the next instruction's
    # argument.
    consts = code.co_consts
    arg = 0
    for op, low in zip(ops, args):
        if op in _ARGUMENT_OPCODES:
            arg |= low
            if op == _EXTENDED_ARG:
                arg <<= 8
                continue
            if (op == _BINARY_OP and arg not in _ARITHMETIC_BINARY_OP_ARGS
                    or op == _LOAD_CONST and type(consts[arg]) is not int
                    or op == _LOAD_COMMON_CONSTANT and arg not in _INT_COMMON_CONSTANTS):
                return False
        arg = 0
    return True


def _symbolic_coefficients(fn, n: int, dimension: int, cache: dict, name: str | None = None):
    """
    Coefficients (c_0, c_1, ..., c_n) mod d of a detector function, from one symbolic call.

    Returns None when the function is not plain straight-line arithmetic or uses an operation
    that `_Polynomial` refuses; the caller then probes it numerically. The call gives the
    function as a polynomial mod d. For prime d that settles exactly whether it is affine on
    Z_d^n: if not, this raises the ValueError the numeric path raises, naming `name`, or
    without a name returns None. For composite d a polynomial of degree 2 or more can still be
    affine (2 x**2 = 2 x mod 4), so it gives None. The result depends only on the code object
    and n, so it is cached on them.
    """
    key = (fn.__code__, n) if type(fn) is types.FunctionType else None
    if key is not None and key in cache:
        result = cache[key]
    else:
        result = None
        if _is_straight_line_arithmetic(fn):
            try:
                value = fn(_Records(n, dimension))
            except Exception:
                value = None
            if type(value) is _Polynomial:
                result = value.coefficients()
                if result is None and _fermat_period(dimension):
                    # Not affine: the numeric path's error, which checks the constant term first.
                    result = "has a non-zero constant term" if value.k else "is not linear in its records"
            elif type(value) is int:
                result = (value % dimension,) + (0,) * n
        if key is not None:
            cache[key] = result
    if type(result) is str:
        if name is None:
            return None
        raise ValueError(f"{name} {result}")
    return result


def _probed_coefficients(fn, n: int, unique_index: int, name: str, dimension: int) -> list:
    """
    Coefficient of each record position, read by evaluating the detector function on probes.

    Evaluating on unit vectors gives the coefficients, and further probes check that the
    function really is linear. `name` names the detector or observable in error messages.

    Raises:
        ValueError: If the function has a constant term or is not linear.
    """
    base = int(fn([0] * n)) % dimension
    if base != 0:
        raise ValueError(f"{name} has a non-zero constant term")

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
            raise ValueError(f"{name} is not linear in its records")
    return position_coeffs


def _detector_coefficients(detector_info, dimension: int):
    """
    Reads the linear coefficients of each detector and logical observable.

    sdim compiles each DETECTOR / LOGICAL_OBSERVABLE expression into a
    function of the values of the records it reads (`rec[0]`, `rec[1]`, ...
    are the records listed in its arguments, in order), wrapped in
    `sdim.program._detector_mod` (normally). When the function is straight-line
    arithmetic (record lookups, integer constants, unary minus, +, -, *, ** by
    a constant, %) one call on symbolic `_Polynomial` records gives it exactly
    as a polynomial mod d. For prime d that proves it affine on Z_d^n, with its
    coefficients, or proves it is not. Otherwise (also for a composite d and a
    polynomial of degree 2 or more, or a product too long to expand, see
    `_MAX_PRODUCT_TERMS`) it is evaluated on the zero vector (the constant
    term) and on unit vectors (the coefficients), then checked for linearity on
    doubled unit vectors, random inputs and, for at most 40 records, every
    pair of unit vectors.

    Args:
        detector_info: The DetectorData returned by `Program._build_ir`.
        dimension (int): Qudit dimension d.

    Returns:
        tuple: (dets, obs, det_labels, obs_labels). dets and obs hold, for each
            detector and each observable in order, a dict {absolute measurement
            record index: coefficient mod d}, with the coefficients of a record
            read twice added up and zero coefficients left out. The labels are
            the DETECTOR / LOGICAL_OBSERVABLE labels, "" for none.

    Raises:
        ValueError: If an expression has a non-zero constant term or is not
            linear in its records. The message names the detector (Dk) or
            observable (Lk) by index, and by label if it has one.
    """
    dets, obs, det_labels, obs_labels = [], [], [], []
    cache: dict = {}
    for unique_index, label, arguments, is_logical in detector_info.detector_data:
        fn = detector_info.detector_functions[unique_index]
        n = len(arguments)
        if is_logical:
            name = f"logical observable L{len(obs)}"
        else:
            name = f"detector D{len(dets)}"
        if label:
            name += f" ({label!r})"
        form = _symbolic_coefficients(fn, n, dimension, cache, name)
        if form is None:
            position_coeffs = _probed_coefficients(fn, n, unique_index, name, dimension)
        else:
            # The same checks, in the same order, as the numeric path; every probe would agree.
            if form[0] != 0:
                raise ValueError(f"{name} has a non-zero constant term")
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


def _frame_qudits(gid, qa, qb, n_qudits: int):
    """
    Checks the qudits of the ops that can change a frame, and of the Paulis.

    qa and qb are the IR's qudit indices: `Program._build_ir` has counted negative indices from
    -n_qudits on back from the end, left the others as they were, and written -1 for no qudit.

    Returns (qb, frame_ops, two_ops): qb is a copy with -1 for every single-qudit frame op, so
    the kernels take their one-qudit branch, and frame_ops / two_ops are the frame ops and the
    two-qudit frame ops.

    Raises:
        IndexError: If a frame op or a Pauli acts on a qudit outside the circuit.
        ValueError: If a two-qudit frame op acts on one qudit twice (it has no frame rule).
    """
    frame_mask = np.isin(gid, np.array(sorted(_FRAME_GATES), dtype=np.int64))
    # Any other negative target is outside the circuit, and the range check below reports it.
    two_mask = frame_mask & np.isin(gid, np.array(_TWO_QUDIT_FRAME_GATES, dtype=np.int64)) & (qb != -1)
    qb = np.where(frame_mask & ~two_mask, -1, qb)
    frame_ops = np.flatnonzero(frame_mask)
    two_ops = np.flatnonzero(two_mask)
    # A Pauli leaves every frame as it is, but the simulators reject one outside the circuit too.
    pauli_ops = np.flatnonzero((gid >= _X) & (gid <= _Z_INV))
    ent_q = np.concatenate((qa[frame_ops], qb[two_ops], qa[pauli_ops]))
    if len(ent_q) and (ent_q.min() < 0 or ent_q.max() >= n_qudits):
        bad = int(ent_q.min() if ent_q.min() < 0 else ent_q.max())
        raise IndexError(f"a gate acts on qudit {bad}, but the circuit has {n_qudits} qudits")
    same = two_ops[qa[two_ops] == qb[two_ops]]
    if len(same):
        i = int(same[0])
        raise ValueError(f"IR op {i} (gate id {int(gid[i])}) acts on qudit {int(qa[i])} twice; "
                         "a two-qudit gate needs two different qudits")
    return np.ascontiguousarray(qb, dtype=np.int64), frame_ops, two_ops


def _qudit_op_lists(gid, qa, qb, frame_ops, two_ops, n_qudits: int):
    """
    For each qudit, the IR ops that can change its frame, in circuit order (for the forward kernel).

    qb, frame_ops and two_ops are what `_frame_qudits` returns. Returns (qptr, qops, posa, posb).
    Qudit q's ops are qops[qptr[q]:qptr[q + 1]], so the probe kernel only visits ops on qudits a
    fault has reached. posa / posb hold an op's position in qops within the list of its first /
    second qudit.
    """
    n_ops = len(gid)
    ent_q = np.concatenate((qa[frame_ops], qb[two_ops]))
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
    return qptr, qops, posa, posb


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

    pls[i] is the line probability of mechs[i], and d is prime. Returns None if
    a target or coefficient is not an integer (NumPy integers are converted) or
    a target does not fit in int64, or if d is above 2**31 - 1 (`_expand_lines`
    multiplies residues in int64, so it needs d * d to fit); the caller then
    expands one dict at a time, exactly.
    """
    if d > _MAX_SAMPLE_DIMENSION:
        return None
    gens = [g for m in mechs for g in m.generators]
    if not _all_ints(gens):
        # NumPy integers in mechanisms added after the constructor normalized the rest.
        gens = [{_plain_int(t): _plain_int(v) for t, v in g.items()} for g in gens]
        if not _all_ints(gens):
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
        # Every non-zero residue is invertible mod a prime, and to_lines only takes prime d.
        raise RuntimeError(f"{int(lval[lptr[bad]])} is not invertible mod {d}")
    group = _line_groups(lptr, ltgt, lval)
    line_mech = line_mech.tolist()
    return _merge_groups(group.tolist(), [pls[i] for i in line_mech], [mechs[i].source for i in line_mech],
                         ltgt.tolist(), lval.tolist(), lptr.tolist())


def _pow_mod(base, exponent: int, d: int) -> np.ndarray:
    """base ** exponent mod d, elementwise, for int64 residues and d < 2**31 (products stay below 2**62)."""
    result = np.ones(len(base), dtype=np.int64)
    square = np.asarray(base, dtype=np.int64) % d
    while exponent:
        if exponent & 1:
            result = result * square % d
        square = square * square % d
        exponent >>= 1
    return result


def _scale_lines(cptr, cval, d: int):
    """
    Scales each line (entries cptr[i]:cptr[i + 1] of cval, sorted by target, non-zero mod d) so its first value is 1.

    Returns (scaled values, solo): solo is None, or (composite d) a bool array marking the lines
    whose first value is not invertible mod d. Those are left unscaled.
    """
    n = len(cptr) - 1
    if n == 0:
        return cval, None
    lead = cval[cptr[:-1]]
    solo = None
    if _is_prime(d):
        inv = _pow_mod(lead, d - 2, d)
    else:
        leads, where = np.unique(lead, return_inverse=True)
        inverses = []
        for v in leads.tolist():
            inverses.append(pow(v, -1, d) if math.gcd(v, d) == 1 else 1)
        inv = np.array(inverses, dtype=np.int64)[where.reshape(-1)]
        solo = np.array([math.gcd(v, d) != 1 for v in leads.tolist()], dtype=bool)[where.reshape(-1)]
    return cval * np.repeat(inv, np.diff(cptr)) % d, solo


def _line_groups(cptr, ctgt, cval, solo=None) -> np.ndarray:
    """
    Numbers equal lines (entries cptr[i]:cptr[i + 1] of ctgt / cval) in order of first appearance.

    A line marked in `solo` gets a number of its own. A few lines are grouped exactly, by their
    entries. Many lines are compared by a 64-bit hash of their entries, and every line found
    equal to an earlier one is checked entry by entry; a hash collision (never seen in practice)
    falls back to exact grouping.
    """
    n = len(cptr) - 1
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n <= 64:
        return _exact_line_groups(cptr, ctgt, cval, solo)
    sizes = np.diff(cptr)
    with np.errstate(over="ignore"):
        x = ctgt.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x ^= cval.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(29)
        total = np.zeros(len(x) + 1, dtype=np.uint64)
        np.cumsum(x, out=total[1:])
        h = total[cptr[1:]] - total[cptr[:-1]] + sizes.astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
    line = np.arange(n, dtype=np.int64)
    rep = line.copy()
    normal = line if solo is None else np.flatnonzero(~solo)
    if len(normal):
        _, first, where = np.unique(h[normal], return_index=True, return_inverse=True)
        rep[normal] = normal[first[where.reshape(-1)]]
    dup = np.flatnonzero(rep != line)
    if len(dup):
        same = sizes[dup] == sizes[rep[dup]]
        if same.all():
            a = _segments(cptr[dup], sizes[dup])
            b = _segments(cptr[rep[dup]], sizes[dup])
            same = (ctgt[a] == ctgt[b]) & (cval[a] == cval[b])
        if not same.all():
            # A hash collision: group exactly instead.
            return _exact_line_groups(cptr, ctgt, cval, solo)
    return np.searchsorted(np.unique(rep), rep)


def _exact_line_groups(cptr, ctgt, cval, solo=None) -> np.ndarray:
    """`_line_groups`, by comparing the entries of the lines themselves."""
    cptr, ctgt, cval = cptr.tolist(), ctgt.tolist(), cval.tolist()
    solo = [False] * (len(cptr) - 1) if solo is None else solo.tolist()
    seen: dict = {}
    group = []
    for i in range(len(cptr) - 1):
        a, b = cptr[i], cptr[i + 1]
        key = i if solo[i] else (tuple(ctgt[a:b]), tuple(cval[a:b]))
        group.append(seen.setdefault(key, len(seen)))
    return np.array(group, dtype=np.int64)


def _segments(starts, counts):
    """The indices starts[i] + 0 .. counts[i] - 1 for every i, concatenated."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    return np.repeat(np.asarray(starts, dtype=np.int64) - offsets, counts) + np.arange(total, dtype=np.int64)


class _Compiled:
    """
    Unit-fault responses of every noise gate, as flat arrays.

    Noise gate i has unit-fault probes probe_start[i]:probe_start[i + 1], and
    probe k has response entries ptr[k]:ptr[k + 1] in tgt (targets, in
    increasing order) and val (non-zero coefficients mod d). Both kernels give
    the entries sorted by target, and that is the dict order of every response
    and generator built from them.
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
        # Their entries, already sorted by target within each mechanism.
        sizes = self.ptr[probes + 1] - self.ptr[probes]
        entries = _segments(self.ptr[probes], sizes)
        ctgt = self.tgt[entries]
        cval = self.val[entries]
        cptr = np.zeros(len(probes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=cptr[1:])
        cval, solo = _scale_lines(cptr, cval, d)
        group = _line_groups(cptr, ctgt, cval, solo)

        ones = ones.tolist()
        out = _merge_groups(group.tolist(), [pi[i] for i in ones], [self.source(i) for i in ones],
                            ctgt.tolist(), cval.tolist(), cptr.tolist())
        out += [ErrorMechanism(pi[i], generators(i), self.source(i)) for i in higher]
        return out


def _compile(circuit: Circuit, backward: bool | None = None) -> _Compiled:
    """
    Computes the unit-fault responses of every noise gate. See `compile_unit_responses`.

    With `backward` (the default is `_BACKWARD`), one backward sweep over the
    circuit gives every response (`_backward_kernel`); otherwise every unit
    fault is pushed forward to the end of the circuit (`_probe_kernel`), which
    costs time quadratic in the number of rounds of a memory circuit. Both give
    the same responses, with their entries sorted by target, and the same
    model.

    The noise gates are left out of the circuit handed to `Program._build_ir`
    when that does not change what it returns or raises, since it would
    otherwise draw a noise sample for each of them.
    """
    if backward is None:
        backward = _BACKWARD
    # A NumPy integer dimension would overflow in the coefficient arithmetic below.
    d = _plain_int(circuit.dimension)
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
            # The identity changes nothing, but the simulators reject one outside the circuit too.
            q = instr.qudit_index
            if q is not None and not -n_qudits <= q < n_qudits:
                raise IndexError(f"a gate acts on qudit {q}, but the circuit has {n_qudits} qudits")
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

    qb, frame_ops, two_ops = _frame_qudits(gid, qa, qb, n_qudits)

    # For each measurement record, the detectors / observables that use it and their coefficients,
    # detectors first and then observables, each in order. Within a record, targets are increasing.
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
    rtgt = np.ascontiguousarray(np.array(inc_tgt, dtype=np.int64)[by_rec])
    rcoef = np.ascontiguousarray(np.array(inc_coef, dtype=np.int64)[by_rec])

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
            if q1 < 0 and q1 >= -n_qudits:
                q1 += n_qudits
        # Negative indices count back from the end of the circuit (N2's second one just above), as
        # Program._build_ir counts them; the range check below reports any outside the circuit.
        if q0 < 0 and q0 >= -n_qudits:
            q0 += n_qudits
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
    q0_arr = np.array(q0s, dtype=np.int64)
    q1_arr = np.array(q1s, dtype=np.int64)
    noise_qudit = np.where(_PROBE_QUDIT[probe_code, local] == 0, q0_arr[probe_loc], q1_arr[probe_loc])
    noise_kind = _PROBE_KIND[probe_code, local]
    noise_op = np.array(probe_ops, dtype=np.int64)[probe_loc]

    # Determinism probes.  The frame simulator randomizes the Z frame at the start and after every
    # M and RESET, and the X frame after every M_X, and the unit-fault responses above assume those
    # random parts cancel.  A unit fault of that kind at each of those points must therefore reach
    # no detector or observable.
    resets = np.flatnonzero((gid == _M) | (gid == _M_X) | (gid == _RESET))
    probe_op = np.concatenate((noise_op, np.full(n_qudits, -1, dtype=np.int64), resets)).astype(np.int64)
    probe_qudit = np.concatenate((noise_qudit, np.arange(n_qudits, dtype=np.int64), qa[resets])).astype(np.int64)
    reset_kind = np.where(gid[resets] == _M_X, 0, 1).astype(np.int64)
    probe_kind = np.concatenate((noise_kind, np.ones(n_qudits, dtype=np.int64), reset_kind)).astype(np.int64)
    if len(probe_qudit) and (probe_qudit.min() < 0 or probe_qudit.max() >= n_qudits):
        bad = int(probe_qudit.min() if probe_qudit.min() < 0 else probe_qudit.max())
        raise IndexError(f"a gate acts on qudit {bad}, but the circuit has {n_qudits} qudits")
    # add_gate cannot see that two indices, one of them negative, are the same qudit.
    twice = np.flatnonzero((code_arr == 3) & (q0_arr == q1_arr))
    if len(twice):
        i = int(twice[0])
        raise ValueError(f"N2@{ir_indices[i]} acts on qudit {q0s[i]} twice; "
                         "a two-qudit gate needs two different qudits")

    n_targets = n_det + len(obs)
    if backward:
        visit = np.argsort(-probe_op, kind="stable")
        ptr, tgt, val = _backward_kernel(gid, qa, qb, mul_a, mul_inv, rec_of_op, rptr, rtgt, rcoef, visit, probe_op,
                                         probe_qudit, probe_kind, d, n_qudits)
    else:
        qptr, qops, posa, posb = _qudit_op_lists(gid, qa, qb, frame_ops, two_ops, n_qudits)
        ptr, tgt, val = _run_probes(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt,
                                    rcoef, probe_op, probe_qudit, probe_kind, d, n_targets, n_qudits)
    random_targets = np.unique(tgt[ptr[n_noise_probes]:]).tolist()
    if random_targets:
        names = [(f"D{t}" if t < n_det else f"L{t - n_det}") for t in random_targets]
        raise ValueError("These detectors / observables are not deterministic without noise, so they "
                         f"have no detector error model: {', '.join(names[:20])}"
                         + (" ..." if len(names) > 20 else ""))
    end = int(ptr[n_noise_probes])
    ptr = ptr[:n_noise_probes + 1]
    return _Compiled(d, n_det, len(obs), det_labels, obs_labels, ir_indices, gate_ids, codes, channels,
                     q0s, q1s, probs, pis, probe_start, ptr, tgt[:end], val[:end])


def compile_unit_responses(circuit: Circuit) -> CompiledResponses:
    """
    Computes the detector response of every unit fault in a circuit.

    For each N1/N2 gate, an X or Z fault is placed right after the gate on
    each qudit it acts on. Its response is the change it causes in each
    detector and logical observable when it follows the update rules of
    `sdim.program.simulate_frame` through the rest of the circuit. The
    responses come from one backward sweep over the circuit (see the module
    docstring), and each one lists its targets in increasing order.

    Args:
        circuit (Circuit): The noisy circuit.

    Returns:
        CompiledResponses: Detector counts and labels, plus one `NoiseLocation`
            per noise gate.

    Raises:
        ValueError: If an N1 gate has an unknown noise channel, an N2 gate
            uses `prob_dist`, a noise probability is above the fully mixing
            value, a two-qudit gate acts on one qudit twice, a detector or
            observable expression has a constant term or is not linear in its
            records, or a detector or observable is not deterministic without
            noise.
        IndexError: If a gate acts on a qudit outside the circuit.
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
            ptr[k]:ptr[k + 1] of tgt (targets, in increasing order) and val
            (non-zero coefficients mod d).
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


@_kernel(nogil=True)
def _probe_kernel(gid, qa, qb, mul_a, mul_inv, rec_of_op, qptr, qops, posa, posb, rptr, rtgt, rcoef,
                  probe_op, probe_qudit, probe_kind, d, n_targets, max_slots, cap):
    """
    Pushes unit faults through the circuit, one probe at a time.

    This is the forward reference for `_backward_kernel`, used when `_BACKWARD`
    is False: it costs time proportional to how far each fault travels, which
    for a memory circuit grows with the number of rounds after the fault.

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
            entries out_ptr[p]:out_ptr[p + 1] of out_tgt / out_val, sorted by
            target.
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
                    # M records x and leaves a random z.  M_X records z, keeps it, and leaves a
                    # random x (the qudit ends in an X eigenstate).  Random parts cancel in every
                    # deterministic detector, so they are dropped here.
                    if g == 15:
                        nx = z
                        nz = z
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
                    if g == 15:
                        nx = 0
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
        # Write out this probe's non-zero totals, sorted by target, and clear the accumulator. They are at most nt.
        touched[:nt].sort()
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


@_kernel(nogil=True)
def _backward_kernel(gid, qa, qb, mul_a, mul_inv, rec_of_op, rptr, rtgt, rcoef, visit, probe_op, probe_qudit,
                     probe_kind, d, n_qudits):
    """
    The response of every unit-fault probe, from one backward sweep over the circuit.

    Map 2q (2q + 1) is the response of a unit X (Z) fault on qudit q placed at
    the current point of the sweep: a sparse vector of targets, sorted, with
    non-zero values mod d. At the end of the circuit every map is zero. Going
    back over op i turns the maps after it into the maps before it, by the
    transpose of the op's frame rule in `_probe_kernel`: a fault before the op
    becomes some combination of unit faults after it, plus the record it
    changes if the op measures. Writing X, Z for a qudit's maps, the maps
    before the op are, in terms of the maps after it:

        H        X <- Z,       Z <- -X
        H_INV    X <- -Z,      Z <- X
        P        X <- X + Z
        P_INV    X <- X - Z
        MUL a    X <- a X,     Z <- a^-1 Z
        M        X <- X + R,   Z <- 0          (R: the targets reading the record)
        M_X      Z <- Z + R,   X <- 0
        RESET    X <- 0,       Z <- 0
        CNOT a b     Xa <- Xa + Xb,   Zb <- Zb - Za
        CNOT_INV a b Xa <- Xa - Xb,   Zb <- Zb + Za
        CZ a b       Xa <- Xa + Zb,   Xb <- Xb + Za
        CZ_INV a b   Xa <- Xa - Zb,   Xb <- Xb - Za
        SWAP a b     swap the maps of a and b

    A probe (op p, qudit, kind) reads its map after op p, that is just before
    going back over op p (p = -1: at the start of the circuit). The maps live
    in one pool: a map is rewritten at the end of the pool, and the pool is
    compacted when it fills up. `visit` lists the probes by decreasing op.

    The targets (the detectors and observables) are whatever the record rows
    rptr / rtgt / rcoef say. The sweep starts at the last op whose record some
    target reads (every map is zero after it) and stops once every probe has
    read its map.

    Returns:
        tuple: (ptr, tgt, val), probe k's response being the entries
            ptr[k]:ptr[k + 1] of tgt (targets, in increasing order) and val
            (non-zero coefficients mod d).
    """
    n_ops = gid.shape[0]
    n_probes = visit.shape[0]
    n_maps = 2 * n_qudits
    mstart = np.zeros(n_maps, dtype=np.int64)
    mlen = np.zeros(n_maps, dtype=np.int64)
    pool_t = np.empty(max(1024, 2 * n_maps), dtype=np.int64)
    pool_v = np.empty(pool_t.shape[0], dtype=np.int64)
    used = 0
    res_start = np.zeros(n_probes, dtype=np.int64)
    res_len = np.zeros(n_probes, dtype=np.int64)
    out_t = np.empty(1024, dtype=np.int64)
    out_v = np.empty(1024, dtype=np.int64)
    w = 0
    # Up to two merges per op: (destination map, source map or -1 - record, multiplier).
    act = np.empty((2, 3), dtype=np.int64)
    v = 0
    op = n_ops - 1
    while op >= 0 and (rec_of_op[op] < 0 or rptr[rec_of_op[op] + 1] == rptr[rec_of_op[op]]):
        op -= 1
    while True:
        while v < n_probes and probe_op[visit[v]] >= op:
            pr = visit[v]
            m = 2 * probe_qudit[pr] + probe_kind[pr]
            n = mlen[m]
            if w + n > out_t.shape[0]:
                size = 2 * out_t.shape[0]
                while size < w + n:
                    size *= 2
                grown_t = np.empty(size, dtype=np.int64)
                grown_v = np.empty(size, dtype=np.int64)
                for i in range(w):
                    grown_t[i] = out_t[i]
                    grown_v[i] = out_v[i]
                out_t = grown_t
                out_v = grown_v
            s = mstart[m]
            for i in range(n):
                out_t[w + i] = pool_t[s + i]
                out_v[w + i] = pool_v[s + i]
            res_start[pr] = w
            res_len[pr] = n
            w += n
            v += 1
        if op < 0 or v == n_probes:
            break
        g = gid[op]
        n_act = 0
        if qb[op] >= 0 and g >= 9 and g <= 13:
            xa = 2 * qa[op]
            xb = 2 * qb[op]
            if g == 13:  # SWAP
                for c in range(2):
                    t0 = mstart[xa + c]
                    mstart[xa + c] = mstart[xb + c]
                    mstart[xb + c] = t0
                    t0 = mlen[xa + c]
                    mlen[xa + c] = mlen[xb + c]
                    mlen[xb + c] = t0
            else:
                n_act = 2
                if g == 9 or g == 10:  # CNOT, CNOT_INV
                    act[0, 0] = xa
                    act[0, 1] = xb
                    act[1, 0] = xb + 1
                    act[1, 1] = xa + 1
                    act[0, 2] = 1 if g == 9 else d - 1
                    act[1, 2] = d - 1 if g == 9 else 1
                else:  # CZ, CZ_INV
                    act[0, 0] = xa
                    act[0, 1] = xb + 1
                    act[1, 0] = xb
                    act[1, 1] = xa + 1
                    act[0, 2] = 1 if g == 11 else d - 1
                    act[1, 2] = act[0, 2]
        elif g >= 5 and g <= 8 or g == 22 or g >= 14 and g <= 16:
            x = 2 * qa[op]
            z = x + 1
            neg = -1
            if g == 5 or g == 6:  # H, H_INV
                t0 = mstart[x]
                mstart[x] = mstart[z]
                mstart[z] = t0
                t0 = mlen[x]
                mlen[x] = mlen[z]
                mlen[z] = t0
                neg = z if g == 5 else x
            elif g == 7 or g == 8:  # P, P_INV
                n_act = 1
                act[0, 0] = x
                act[0, 1] = z
                act[0, 2] = 1 if g == 7 else d - 1
            elif g == 22:  # MUL
                for c in range(2):
                    f = mul_a[op] if c == 0 else mul_inv[op]
                    if f != 1:
                        for i in range(mstart[x + c], mstart[x + c] + mlen[x + c]):
                            pool_v[i] = (pool_v[i] * f) % d
            elif g == 16:  # RESET
                mlen[x] = 0
                mlen[z] = 0
            else:  # M records X, M_X records Z
                kept = x if g == 14 else z
                mlen[z if g == 14 else x] = 0
                n_act = 1
                act[0, 0] = kept
                act[0, 1] = -1 - rec_of_op[op]
                act[0, 2] = 1
            if neg >= 0 and d > 2:
                for i in range(mstart[neg], mstart[neg] + mlen[neg]):
                    pool_v[i] = d - pool_v[i]
        for a in range(n_act):
            dst = act[a, 0]
            src = act[a, 1]
            c = act[a, 2]
            if src >= 0:
                ns = mlen[src]
            else:
                ns = rptr[-src] - rptr[-1 - src]
            if ns == 0:
                continue
            need = mlen[dst] + ns
            if used + need > pool_t.shape[0]:
                # Compact the live maps into a pool with room for at least as much again.
                live = 0
                for m in range(n_maps):
                    live += mlen[m]
                size = max(2 * (live + need), 2 * n_maps, 1024)
                new_t = np.empty(size, dtype=np.int64)
                new_v = np.empty(size, dtype=np.int64)
                pos = 0
                for m in range(n_maps):
                    s = mstart[m]
                    for i in range(mlen[m]):
                        new_t[pos + i] = pool_t[s + i]
                        new_v[pos + i] = pool_v[s + i]
                    mstart[m] = pos
                    pos += mlen[m]
                pool_t = new_t
                pool_v = new_v
                used = pos
            if src >= 0:
                src_t = pool_t
                src_v = pool_v
                j = mstart[src]
            else:
                src_t = rtgt
                src_v = rcoef
                j = rptr[-1 - src]
            j_end = j + ns
            i = mstart[dst]
            i_end = i + mlen[dst]
            k = used
            # dst + c * src, both sorted; c is a unit, so only equal targets can cancel.
            while i < i_end or j < j_end:
                if j == j_end or (i < i_end and pool_t[i] < src_t[j]):
                    tt = pool_t[i]
                    y = pool_v[i]
                    i += 1
                elif i == i_end or src_t[j] < pool_t[i]:
                    tt = src_t[j]
                    y = (c * src_v[j]) % d
                    j += 1
                else:
                    tt = pool_t[i]
                    y = (pool_v[i] + c * src_v[j]) % d
                    i += 1
                    j += 1
                if y != 0:
                    pool_t[k] = tt
                    pool_v[k] = y
                    k += 1
            mstart[dst] = used
            mlen[dst] = k - used
            used = k
        op -= 1
    ptr = np.zeros(n_probes + 1, dtype=np.int64)
    for pr in range(n_probes):
        ptr[pr + 1] = ptr[pr] + res_len[pr]
    tgt = np.empty(ptr[n_probes], dtype=np.int64)
    val = np.empty(ptr[n_probes], dtype=np.int64)
    for pr in range(n_probes):
        s = res_start[pr]
        for i in range(res_len[pr]):
            tgt[ptr[pr] + i] = out_t[s + i]
            val[ptr[pr] + i] = out_v[s + i]
    return ptr, tgt, val


@_kernel
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


@_kernel
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


@_kernel
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


# Position of the single set bit of x, by the de Bruijn product (x * 0x07EDD5E59A4E28C2) >> 58.
_LOW_BIT = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _LOW_BIT[((0x07EDD5E59A4E28C2 << _i) % (1 << 64)) >> 58] = _i
del _i


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


# NumPy's SeedSequence.generate_state: its 32-bit word i is x ^ (x >> 16), where
# x = (pool[i % len(pool)] ^ h_i) * h_(i + 1) mod 2**32 and h_i = _SEED_INIT * _SEED_MULT**i mod 2**32.
_SEED_INIT = 0x8B51F9DD
_SEED_MULT = 0x58F38DED


def _block_states(pool, first: int, count: int) -> np.ndarray:
    """
    The xoshiro256** states of sampler blocks first .. first + count - 1, as a (count, 4) uint64 array.

    Block c starts from the uint64 words 4c .. 4c + 3 of `generate_state` of
    the SeedSequence whose pool is `pool`, or, if all four are zero (where
    xoshiro256** must not start), from 1, 0, 0, 0. generate_state always
    starts at word 0, so a sampler that continues its stream at block
    `first` computes the words itself, in time linear in `count`.
    """
    n = 8 * count
    hash_const = np.full(n + 1, _SEED_MULT, dtype=np.uint32)
    hash_const[0] = _SEED_INIT * pow(_SEED_MULT, 8 * first, 1 << 32) % (1 << 32)
    hash_const = np.multiply.accumulate(hash_const, dtype=np.uint32)
    # The pool has 4 words, so word 8 * first reads pool[0] and each row of 4 words reads the whole pool.
    words = hash_const[:-1].reshape(-1, len(pool)) ^ pool
    words *= hash_const[1:].reshape(words.shape)
    words ^= words >> 16
    # generate_state pairs the 32-bit words as little-endian uint64s on every machine.
    states = words.astype("<u4", copy=False).view("<u8").astype(np.uint64, copy=False).reshape(count, 4)
    states[~states.any(axis=1), 0] = 1
    return states


# _SEED_MULT has order 2**30 mod 2**32, so the words of one SeedSequence repeat after 2**30 words, 2**27 blocks.
_STREAM_EPOCH = 1 << 27


class _BlockStream:
    """
    The xoshiro256** states of the blocks of one sampler stream, from its seed.

    Blocks 0 .. 2**27 - 1 start from the words of `SeedSequence(seed)` (see
    `_block_states`). Those words would then repeat, so block c of epoch
    e = c // 2**27 >= 1 starts from the words, at block c % 2**27, of the
    SeedSequence with the same entropy and spawn key (e,), so a long-lived
    `CompiledDemSampler` does not start over after 2**35 shots.

    With `ahead`, `states` computes at least that many states at once and
    keeps them for the next calls, since computing them one at a time costs
    about as much as drawing the blocks of a small model.
    """

    def __init__(self, seed, ahead: int = 0):
        self._seq = np.random.SeedSequence(seed)
        self._epoch, self._pool = 0, self._seq.pool
        self._ahead = ahead
        # The states computed last, those of blocks _kept_first onwards.
        self._kept_first, self._kept = 0, np.zeros((0, 4), dtype=np.uint64)

    def states(self, first: int, count: int) -> np.ndarray:
        """The states of blocks first .. first + count - 1, as a (count, 4) uint64 array."""
        k = first - self._kept_first
        if 0 <= k and k + count <= len(self._kept):
            return self._kept[k:k + count]
        block, left = first, max(count, self._ahead)
        parts = []
        while True:
            epoch, start = divmod(block, _STREAM_EPOCH)
            n = min(left, _STREAM_EPOCH - start)
            if epoch != self._epoch:
                seq = self._seq
                if epoch:
                    seq = np.random.SeedSequence(seq.entropy, spawn_key=seq.spawn_key + (epoch,))
                # Making a SeedSequence costs about as much as a small call, so keep the pool for the epoch.
                self._epoch, self._pool = epoch, seq.pool
            parts.append(_block_states(self._pool, start, n))
            block, left = block + n, left - n
            if not left:
                break
        kept = parts[0] if len(parts) == 1 else np.concatenate(parts)
        # Both fields change together and only now, so a call that raised above (an interrupt, say) left
        # the kept states with their own blocks.
        self._kept_first, self._kept = first, kept
        return kept[:count]


def _sample_plan(mech_prob, n_gens, sizes, ent_tgt, ent_val, d, montgomery, pack_dtype):
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
    coefficient (already reduced mod d), in Montgomery form v * 2**32 mod d
    when `montgomery` (odd d). info[k] = (block offset, probability bits) for
    the k-th mechanism in bin order (bins in increasing probability,
    mechanisms in index order within a bin); always_off holds the blocks of
    pi >= 1. `pack` has dtype `pack_dtype`; int32 is enough when every target
    is below 2**31 (values are below d, which is below 2**31).

    This runs in NumPy rather than numba: it is linear work done once per
    `DetectorErrorModel.sample` call or `compile_sampler`, and leaving it
    out of numba saves its compile time on first use.

    Returns:
        tuple: (pack, info, always_off, bin_ptr, bin_pmax, bin_log_keep, cost),
            cost estimating the work per shot of the firings.

    Raises:
        ValueError: If a probability is NaN.
    """
    if np.isnan(mech_prob).any():
        raise ValueError("a mechanism probability is NaN")
    if montgomery:
        ent_val = (ent_val << 32) % d
    n_mech = len(mech_prob)
    gen_ptr = np.zeros(n_mech + 1, dtype=np.int64)
    np.cumsum(n_gens, out=gen_ptr[1:])
    ent_ptr = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=ent_ptr[1:])
    mech_ent = ent_ptr[gen_ptr[1:]] - ent_ptr[gen_ptr[:-1]]
    work = n_gens + mech_ent
    live = np.flatnonzero((mech_ent > 0) & (mech_prob > 0.0) & (mech_prob < 1.0))
    always = np.flatnonzero((mech_ent > 0) & (mech_prob >= 1.0))
    cost = float(work[always].sum()) + float((mech_prob[live] * work[live]).sum())
    # For positive doubles the bit pattern grows with the value; a stable sort keeps index order in a bin.
    keys = mech_prob[live].view(np.int64) >> 49
    by_key = np.argsort(keys, kind="stable")
    live = live[by_key]
    keys = keys[by_key]
    order = np.concatenate((live, always))
    # One block per mechanism, in that order.
    block = 1 + n_gens[order] + 2 * mech_ent[order]
    block_off = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(block, out=block_off[1:])
    pack = np.empty(int(block_off[-1]), dtype=pack_dtype)
    pack[block_off[:-1]] = n_gens[order]
    gens = _segments(gen_ptr[order], n_gens[order])
    gen_mech = np.repeat(order, n_gens[order])
    gen_off = np.repeat(block_off[:-1], n_gens[order])
    gen_pos = gen_off + 1 + (gens - gen_ptr[gen_mech]) + 2 * (ent_ptr[gens] - ent_ptr[gen_ptr[gen_mech]])
    pack[gen_pos] = sizes[gens]
    ents = _segments(ent_ptr[gens], sizes[gens])
    ent_pos = np.repeat(gen_pos + 1 - 2 * ent_ptr[gens], sizes[gens]) + 2 * ents
    pack[ent_pos] = ent_tgt[ents]
    pack[ent_pos + 1] = ent_val[ents]
    n_live = len(live)
    info = np.empty((n_live, 2), dtype=np.int64)
    info[:, 0] = block_off[:n_live]
    info[:, 1] = mech_prob[live].view(np.int64)
    always_off = np.ascontiguousarray(block_off[n_live:-1])
    starts = (np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1]))) if n_live
              else np.zeros(0, dtype=np.int64))
    bin_ptr = np.append(starts, n_live).astype(np.int64)
    bin_pmax = np.maximum.reduceat(mech_prob[live], starts) if n_live else np.zeros(0, dtype=np.float64)
    bin_log_keep = np.array([math.log1p(-p) for p in bin_pmax.tolist()], dtype=np.float64)
    return pack, info, always_off, bin_ptr, np.ascontiguousarray(bin_pmax, dtype=np.float64), bin_log_keep, cost


@_kernel(nogil=True)
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

    Each shot only visits the bins with a candidate in it: row s of the
    bitmap `active` (n_words 64-bit words) marks the bins whose next candidate
    falls in shot s, and a bin is marked in the row of its next candidate's
    shot when it is done with the current one. Reading a row clears it. A
    block thus costs one skip per bin, one pass over n_bins / 64 words per
    shot, and the work of its candidates, not shots * bins.

    Within a shot the always-on mechanisms come first, then the bins visited
    in increasing order. Each candidate draws its thinning coin (when
    pi < pmax), then its coefficients if it fires, then the skip to the next
    candidate. Bins with no candidate in a shot draw nothing in it, so the
    random stream is the same as visiting every bin in every shot.

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
    n_words = (n_bins + 63) // 64
    active = np.zeros(chunk * n_words, dtype=np.uint64)
    one = np.uint64(1)
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
            if nxt[b] < total:
                i = (nxt[b] // (bin_ptr[b + 1] - bin_ptr[b])) * n_words + (b >> 6)
                active[i] |= one << np.uint64(b & 63)
        for s in range(n_shots):
            shot = lo + s
            row = s * n_words
            word = -1         # the bins of this shot still to visit: the bits of `bits`, then words word + 1 ..
            bits = np.uint64(0)
            ia = 0            # next always-on mechanism
            in_bin = False    # inside bin b: first mechanism, size, coins in the block, end of this shot's coins,
            b = 0             # largest probability, next candidate
            first = 0
            n_b = 0
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
                        if in_bin and pos < end:
                            k = first + pos - (end - n_b)
                            pending = True
                            pm = info_p[k, 1]
                            if pm < pmax:
                                r, s0, s1, s2, s3 = _next_u64(s0, s1, s2, s3)
                                if not _open_unit(r) * pmax < pm:
                                    continue
                            off = info[k, 0]
                            break
                        if in_bin:
                            in_bin = False
                            nxt[b] = pos
                            if pos < total:
                                # pos >= end: the next candidate is in a later shot of the block.
                                i = (pos // n_b) * n_words + (b >> 6)
                                active[i] |= one << np.uint64(b & 63)
                        while bits == 0 and word + 1 < n_words:
                            word += 1
                            bits = active[row + word]
                            active[row + word] = 0
                        if bits == 0:
                            break
                        low = bits & (~bits + one)
                        bits ^= low
                        b = word * 64 + _LOW_BIT[np.int64((low * np.uint64(0x07EDD5E59A4E28C2)) >> np.uint64(58))]
                        in_bin = True
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
    """
    Reduces a sparse vector mod d, sorts it by target and scales it so its first entry is 1.

    Returns (key, scaled), key being the hashable tuple of scaled's items. If no entry is
    non-zero mod d, or the first one is not invertible mod d (composite d), the vector cannot
    be scaled that way: the result is (None, the reduced, sorted vector), and the caller does
    not merge it with any other. The entries are Python ints (`merge_lines` converts NumPy
    integers first).
    """
    items = sorted(gen.items())
    lead = items[0][1] % d if items else 0
    if lead and math.gcd(lead, d) == 1:
        # The usual case. A unit multiple of an entry is 0 mod d only if the entry is, so one
        # pass reduces, drops the zeros and scales.
        inv = pow(lead, -1, d)
        scaled = {t: r for t, v in items if (r := v * inv % d)}
    else:
        items = [(t, v % d) for t, v in items if v % d]
        if not items or math.gcd(items[0][1], d) != 1:
            return None, dict(items)
        inv = pow(items[0][1], -1, d)
        scaled = {t: (v * inv) % d for t, v in items}
    # scaled is built in sorted order, so its items are already the sorted key.
    return tuple(scaled.items()), scaled


def _projective_points(d: int, k: int):
    """Yields one point on each line through the origin of Z_d^k, scaled so its first non-zero coordinate is 1."""
    for lead in range(k):
        for tail in itertools.product(range(d), repeat=k - lead - 1):
            yield (0,) * lead + (1,) + tail
