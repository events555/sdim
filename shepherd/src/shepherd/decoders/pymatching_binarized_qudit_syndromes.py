"""
pymatching_binarized_qudit_syndromes.py -- the "pymatching" decoder: minimum-weight perfect
matching on binarized qudit syndromes, with the matched edges lifted back to Z_d.

PyMatching works over Z_2, so it is given only *which* detectors fired (value != 0 mod d), not by
how much. Compiling and decoding go like this:

1. `dem.to_lines()` splits every mechanism into independent line mechanisms: with probability pi
   a line adds a * v to the detectors and observables, with a uniform on Z_d.
2. A line touching one or two detectors is an edge (one detector: an edge to the boundary). A line
   touching three or more (a "hyperedge") is written, if possible, as a sum of multiples of edges
   that already exist, and its probability is added to theirs, like stim's decompose_errors.
   The `hyperedges` setting says what happens to lines that cannot be split this way.
3. Lines on the same pair of detectors become one matching edge. Its weight is
   log((1 - q) / q), where q is the chance that at least one of those lines makes a non-zero shift
   (a line does so with probability pi * (d - 1) / d). The most likely line gives the edge's
   coefficients: which multiple of the shift lands on each endpoint and on each observable.
4. To decode a shot, PyMatching matches the detectors that fired. The matched edges form a forest,
   and their powers are found by peeling it from the leaves. At a leaf detector with residual r and
   one remaining edge e, the power is a = r / e[leaf] mod d. That multiple of e is subtracted from
   the residuals of both of e's endpoints, the edge is removed, and the next leaf is peeled. The
   prediction is the sum of a * e[observables] over all matched edges.
5. Binarizing loses the syndrome values while matching. For d > 2 the detectors that fired need
   not pair up the way their values do (an edge moves +a to one end and -a to the other, so it
   cannot explain D3=2, D5=2), and peeling can then leave a non-zero residual. That residual is a
   syndrome too, so it is matched and peeled again, up to `matching_passes` times in all. On the
   qudit repetition code in the tests, a second pass cuts the logical error rate about 4x.

Limitations:
- Even with re-matching, this is a baseline, not an optimal qudit decoder. Detection events
  whose values add to zero mod d (D0=D1=D2=1 at d = 3) need no boundary, but their binarized
  count is odd, so matching sends one of them to the boundary anyway.
- A shot whose fired detectors have odd parity in a component with no boundary cannot be matched.
  It is predicted as "no logical shift" (and will usually count as an error).
- to_lines makes (d**k - 1) / (d - 1) lines from a rank-k mechanism, and an N2 mechanism has rank
  up to 4, so compiling gets slow for large d.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

try:
    import pymatching
except ImportError as e:
    raise ImportError(
        "The 'pymatching' decoder needs PyMatching. Install it with: pip install -e \".[mwpm]\""
    ) from e

from sdim.dem import merge_subgroup_probabilities

from shepherd.decoder import CompiledDecoder, Decoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel

HYPEREDGE_MODES = ("decompose", "drop", "raise")

# Probabilities are clipped to [_MIN_Q, 1/2] before taking weights, so that weights are finite and
# non-negative.
_MIN_Q = 1e-300


def _canonical(gen: dict[int, int], d: int) -> tuple[tuple[int, int], ...]:
    """The line spanned by `gen`, as a hashable key: sorted, reduced mod d, first coefficient 1."""
    items = sorted((t, v % d) for t, v in gen.items() if v % d)
    if not items:
        return ()
    inv = pow(items[0][1], -1, d)
    return tuple((t, v * inv % d) for t, v in items)


class CompiledMatchingDecoder(CompiledDecoder):
    """
    A PyMatching graph for one DEM, plus each edge's coefficients mod d for lifting the result.

    Equal syndromes always get equal predictions (PyMatching is deterministic), so the worker may
    decode each distinct syndrome once.
    """

    def __init__(self, *, dimension: int, num_detectors: int, num_observables: int,
                 edge_u: np.ndarray, edge_cu: np.ndarray, edge_v: np.ndarray, edge_cv: np.ndarray,
                 edge_obs: np.ndarray, edge_weight: np.ndarray, num_dropped_hyperedges: int = 0,
                 matching_passes: int = 4):
        self.dimension = dimension
        self.matching_passes = matching_passes
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        # Edge k joins detector edge_u[k] to detector edge_v[k] (-1 = the boundary). When its line
        # fires with power a, it adds a * edge_cu[k] to edge_u[k], a * edge_cv[k] to edge_v[k],
        # and a * edge_obs[k] to the observables.
        self.edge_u, self.edge_cu = edge_u, edge_cu
        self.edge_v, self.edge_cv = edge_v, edge_cv
        self.edge_obs = edge_obs
        self.edge_weight = edge_weight
        self.num_dropped_hyperedges = num_dropped_hyperedges   # for inspection only
        self._build()

    def _build(self) -> None:
        """Derived state: the PyMatching graph and the edge lookup. Rebuilt after unpickling."""
        self._index = {self._key(int(u), int(v)): k
                       for k, (u, v) in enumerate(zip(self.edge_u, self.edge_v))}
        self._matching = pymatching.Matching()
        for u, v, w in zip(self.edge_u.tolist(), self.edge_v.tolist(), self.edge_weight.tolist()):
            if v < 0:
                self._matching.add_boundary_edge(u, weight=w, merge_strategy="disallow")
            else:
                self._matching.add_edge(u, v, weight=w, merge_strategy="disallow")
        # PyMatching wants syndromes of exactly its own length, which ends at the last detector
        # that has an edge. Detectors without edges can never be matched, so they are left out.
        self._num_nodes = self._matching.num_detectors
        has_edge = np.zeros(self.num_detectors, dtype=bool)
        has_edge[self.edge_u] = True
        has_edge[self.edge_v[self.edge_v >= 0]] = True
        self._matchable = has_edge[:self._num_nodes]

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        if u < 0:
            u, v = v, u
        return (u, v) if v < 0 else (min(u, v), max(u, v))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_index", "_matching", "_num_nodes", "_matchable"):
            state.pop(k, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._build()

    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        shots = detections.shape[0]
        out = np.zeros((shots, self.num_observables), dtype=np.int64)
        if self._num_nodes == 0 or self.num_observables == 0:
            return out
        d = self.dimension
        residuals = np.asarray(detections, dtype=np.int64) % d   # a copy: _lift updates it
        for s in range(shots):
            residual = residuals[s]
            for _ in range(self.matching_passes):
                fired = ((residual[:self._num_nodes] != 0) & self._matchable).astype(np.uint8)
                if not fired.any():
                    break
                try:
                    pairs = self._matching.decode_to_edges_array(fired)
                except ValueError:
                    break   # odd parity in a component without a boundary: see the module docstring
                out[s] = (out[s] + self._lift(residual, pairs)) % d
        return out

    def _lift(self, residual: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """
        Finds the power of each matched edge by peeling leaves, and returns the observable shift.
        Subtracts what the edges explain from `residual` (in place), leaving what they do not.
        """
        d = self.dimension
        edges = [self._index[self._key(int(u), int(v))] for u, v in pairs]
        incident: dict[int, set[int]] = defaultdict(set)
        for pos, k in enumerate(edges):
            incident[int(self.edge_u[k])].add(pos)
            if self.edge_v[k] >= 0:
                incident[int(self.edge_v[k])].add(pos)
        prediction = np.zeros(self.num_observables, dtype=np.int64)
        leaves = [n for n, es in incident.items() if len(es) == 1]
        while leaves:
            n = leaves.pop()
            if len(incident[n]) != 1:
                continue
            pos = incident[n].pop()
            k = edges[pos]
            u, v = int(self.edge_u[k]), int(self.edge_v[k])
            coef_here = int(self.edge_cu[k] if n == u else self.edge_cv[k])
            a = int(residual[n]) * pow(coef_here, -1, d) % d
            if a:
                residual[u] = (residual[u] - a * int(self.edge_cu[k])) % d
                if v >= 0:
                    residual[v] = (residual[v] - a * int(self.edge_cv[k])) % d
                prediction = (prediction + a * self.edge_obs[k]) % d
            other = v if n == u else u
            if other >= 0:
                incident[other].discard(pos)
                if len(incident[other]) == 1:
                    leaves.append(other)
        # Edges left over would form a cycle. A minimum-weight solution has none when all weights
        # are positive, so they are ignored.
        return prediction


class MatchingDecoder(Decoder, name="pymatching"):
    """
    Minimum-weight perfect matching (PyMatching) on binarized syndromes, lifted back to Z_d.

    Settings:
        hyperedges: What to do with lines that touch 3 or more detectors.
            "decompose" (default): split each into multiples of existing edges, and leave out the
                ones that cannot be split (CompiledMatchingDecoder.num_dropped_hyperedges counts
                them).
            "raise": the same, but raise a ValueError for any line that cannot be split.
            "drop": leave every hyperedge out.
        matching_passes: How many times to match and peel each shot (at least 1). Every pass
            after the first works on the residual the passes before it left (see the module
            docstring).
    """

    def __init__(self, hyperedges: str = "decompose", matching_passes: int = 4):
        if hyperedges not in HYPEREDGE_MODES:
            raise ValueError(f"hyperedges must be one of {HYPEREDGE_MODES}, got {hyperedges!r}")
        if not isinstance(matching_passes, int) or isinstance(matching_passes, bool) or matching_passes < 1:
            raise ValueError(f"matching_passes must be an integer >= 1, got {matching_passes!r}")
        super().__init__(hyperedges=hyperedges, matching_passes=matching_passes)
        self.hyperedges = hyperedges
        self.matching_passes = matching_passes

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledMatchingDecoder:
        d = dem.dimension
        nd, no = dem.num_detectors, dem.num_observables
        lines = dem.to_lines()

        # 1. Edges (lines on 1 or 2 detectors), merged by line.
        edge_prob: dict[tuple, float] = {}
        hyper: list[tuple[tuple, float]] = []
        for mech in lines.mechanisms:
            key = _canonical(mech.generators[0], d)
            n_det = sum(1 for t, _ in key if t < nd)
            if n_det == 0:
                continue                  # moves only observables: no decoder can see it
            if n_det <= 2:
                prev = edge_prob.get(key)
                edge_prob[key] = (mech.probability if prev is None
                                  else merge_subgroup_probabilities(prev, mech.probability))
            else:
                hyper.append((key, mech.probability))

        # 2. Hyperedges, split into multiples of existing edges where possible.
        by_detector: dict[int, list[tuple]] = defaultdict(list)
        for key in edge_prob:
            for t, _ in key:
                if t < nd:
                    by_detector[t].append(key)
        dropped = 0
        for key, p in hyper:
            parts = None if self.hyperedges == "drop" else _decompose(dict(key), nd, d, by_detector)
            if parts is None:
                if self.hyperedges == "raise":
                    raise ValueError(f"cannot decompose the hyperedge {key} into edges of the "
                                     f"matching graph (hyperedges='decompose' leaves such "
                                     f"lines out)")
                dropped += 1
                continue
            for part in parts:
                edge_prob[part] = merge_subgroup_probabilities(edge_prob[part], p)

        # 3. One matching edge per detector pair.
        groups: dict[tuple[int, int], list[tuple[tuple, float]]] = defaultdict(list)
        for key, p in edge_prob.items():
            dets = [t for t, _ in key if t < nd]
            pair = (dets[0], dets[1]) if len(dets) == 2 else (dets[0], -1)
            groups[pair].append((key, p))

        m = len(groups)
        edge_u = np.empty(m, dtype=np.int64)
        edge_v = np.empty(m, dtype=np.int64)
        edge_cu = np.empty(m, dtype=np.int64)
        edge_cv = np.zeros(m, dtype=np.int64)
        edge_obs = np.zeros((m, no), dtype=np.int64)
        edge_weight = np.empty(m, dtype=np.float64)
        for k, ((u, v), members) in enumerate(sorted(groups.items())):
            key, _ = max(members, key=lambda kp: kp[1])
            coef = dict(key)
            edge_u[k], edge_v[k] = u, v
            edge_cu[k] = coef[u]
            if v >= 0:
                edge_cv[k] = coef[v]
            for t, c in key:
                if t >= nd:
                    edge_obs[k, t - nd] = c
            # Chance that at least one line on this pair makes a non-zero shift.
            q = 1.0 - math.prod(1.0 - p * (d - 1) / d for _, p in members)
            q = min(max(q, _MIN_Q), 0.5)
            edge_weight[k] = math.log((1.0 - q) / q)

        return CompiledMatchingDecoder(
            dimension=d, num_detectors=nd, num_observables=no,
            edge_u=edge_u, edge_cu=edge_cu, edge_v=edge_v, edge_cv=edge_cv,
            edge_obs=edge_obs, edge_weight=edge_weight, num_dropped_hyperedges=dropped,
            matching_passes=self.matching_passes)


def _decompose(vec: dict[int, int], nd: int, d: int,
               by_detector: dict[int, list[tuple]]) -> list[tuple] | None:
    """
    Writes `vec` (sparse, mod d) as a sum of multiples of edge lines, or returns None if it cannot.

    The lowest detector in `vec` has to be covered by some edge e within vec's detectors. The
    multiple of e is then fixed: it has to cancel that detector. The search recurses on what is left.
    """
    dets = sorted(t for t in vec if t < nd)
    if not dets:
        return [] if not vec else None   # an observable change left over cannot be explained
    first = dets[0]
    support = set(dets)
    for key in by_detector.get(first, ()):
        coef = dict(key)
        if not {t for t in coef if t < nd} <= support:
            continue
        c = vec[first] * pow(coef[first], -1, d) % d
        rest = dict(vec)
        for t, v in coef.items():
            r = (rest.get(t, 0) - c * v) % d
            if r:
                rest[t] = r
            else:
                rest.pop(t, None)
        sub = _decompose(rest, nd, d, by_detector)
        if sub is not None:
            return [key, *sub]
    return None
