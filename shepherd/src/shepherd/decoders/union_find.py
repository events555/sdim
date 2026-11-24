"""
union_find.py -- the "union_find" decoder: union-find (cluster growth) decoding over Z_q, for any
detector error model, hyperedges included.

After Delfosse and Nickerson's union-find decoder, with the cluster validity test done by linear
algebra mod q as in Delfosse, Londe and Beverland's union-find for LDPC codes, so it holds for
qudits and for hypergraphs alike.

The model
---------
Every line of `dem.to_lines()` is an edge: when it fires it adds a * v to the detectors and
observables, a uniform on Z_q. An edge may touch 1 detector (a boundary edge), 2, or more (a
hyperedge). Lines that move the same detectors by the same vector but the observables
differently are the same edge to the decoder; the most likely one is kept (`num_ambiguous` counts
the others). Lines that touch no detector are invisible to every decoder and are left out.

The algorithm, per syndrome
---------------------------
1. Each detector with a non-zero value starts a cluster.
2. A cluster is valid when its syndrome can be produced by its own edges: H_c x == s_c mod q has
   a solution, where H_c is the columns of the cluster's fully grown edges restricted to the
   cluster's detectors (detectors the cluster has absorbed without firing take part with value
   0). For qubits on a graph this is "an even number of detections, or a boundary edge"; for
   qudits it is "the charges cancel mod q, or a boundary edge, or a loop whose coefficients do
   not multiply to 1"; the linear algebra covers all of these, and hyperedges, at once.
3. While some cluster is invalid, every invalid cluster grows each edge it touches by half an
   edge (an edge touched by two invalid clusters grows a whole edge). A fully grown edge pulls all
   of its detectors into the cluster, merging every cluster it touches. Validity is re-tested for
   every cluster that changed.
4. Each valid cluster is solved for a low-weight x (`min_weight_solution`): the solutions are
   x0 + c @ K over a kernel basis K of H_c; when q**len(K) <= ENUM_CAP every one is tried and the
   one with the fewest non-zero entries kept, otherwise a greedy descent over the kernel basis
   lowers the weight. The prediction is the sum over clusters of x . (observable shifts of the
   cluster's edges).

Why step 4 must not take just any solution: two solutions differ by a kernel element of H_c,
which is a stabilizer or a logical. A cluster that crosses the code (near threshold, on any code)
or a moderately sized LaCross cluster contains logicals, and an arbitrary solution then fails
often. With the exact minimum, if the true fault (weight w <= (d-1)/2) lies inside the cluster,
the chosen x differs from it by weight <= 2w < d, i.e. by a stabilizer. (Growth can still
stop before a cluster contains the true fault, so off topological codes there is no (d-1)/2
guarantee.)

A cluster that is invalid and can no longer grow (its syndrome is outside what the whole model can
produce) contributes no correction; `decode_correction` reports it as incomplete.

Growth is uniform: edge probabilities are not used. Every compiled structure is fixed after
compilation, and equal syndromes always give equal predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shepherd import gf_q
from shepherd.decoder import CompiledDecoder, Decoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel

ENUM_CAP = 4096     # enumerate every solution of a cluster when there are at most this many


def min_weight_solution(x0: np.ndarray, kernel: np.ndarray, q: int,
                        cap: int = ENUM_CAP) -> np.ndarray:
    """
    A solution x0 + c @ kernel (mod q) with few non-zero entries. Exact (fewest non-zero entries,
    ties to the first c in lexicographic order) when q**len(kernel) <= cap; otherwise a greedy
    descent that adds a * kernel[i], in a fixed order, whenever that lowers the weight. Never
    heavier than x0, and deterministic.
    """
    m = kernel.shape[0]
    if m == 0:
        return x0
    if q ** m <= cap:
        coeffs = np.stack(np.meshgrid(*([np.arange(q)] * m), indexing="ij"), axis=-1).reshape(-1, m)
        candidates = (x0 + coeffs @ kernel) % q
        return candidates[int(np.argmin(np.count_nonzero(candidates, axis=1)))]
    x = x0 % q
    weight = int(np.count_nonzero(x))
    improved = True
    while improved:
        improved = False
        for k in kernel:
            for a in range(1, q):
                y = (x + a * k) % q
                wy = int(np.count_nonzero(y))
                if wy < weight:
                    x, weight, improved = y, wy, True
    return x


class CompiledUnionFind(CompiledDecoder):
    """
    Edge e touches detectors edge_dets[e] with coefficients edge_coefs[e] (both tuples), and shifts
    the observables by edge_obs[e] (row of a (num_edges, num_observables) array), per unit power.
    """

    def __init__(self, *, dimension: int, num_detectors: int, num_observables: int,
                 edge_dets: list[tuple[int, ...]], edge_coefs: list[tuple[int, ...]],
                 edge_obs: np.ndarray, edge_prob: np.ndarray, num_ambiguous: int = 0,
                 num_invisible: int = 0):
        self.dimension = dimension
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.edge_dets = edge_dets
        self.edge_coefs = edge_coefs
        self.edge_obs = edge_obs
        self.edge_prob = edge_prob
        self.num_ambiguous = num_ambiguous     # lines dropped for a likelier one with equal detectors
        self.num_invisible = num_invisible     # lines that touch no detector
        self._build()

    def _build(self) -> None:
        det_edges: list[list[int]] = [[] for _ in range(self.num_detectors)]
        for e, dets in enumerate(self.edge_dets):
            for t in dets:
                det_edges[t].append(e)
        self._det_edges = det_edges

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_det_edges", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._build()

    @property
    def num_edges(self) -> int:
        return len(self.edge_dets)

    # -- decoding -------------------------------------------------------------------------------

    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections, dtype=np.int64) % self.dimension
        out = np.zeros((detections.shape[0], self.num_observables), dtype=np.int64)
        for r in range(detections.shape[0]):
            if detections[r].any():
                edges, powers, _ = self.decode_correction(detections[r])
                if edges:
                    out[r] = powers @ self.edge_obs[edges] % self.dimension
        return out

    def decode_correction(self, syndrome: np.ndarray) -> tuple[list[int], np.ndarray, bool]:
        """
        The correction for one syndrome: edge indices, the power of each, and whether every
        cluster became valid. When it did, sum_k powers[k] * (edge vector of edges[k])
        reproduces the syndrome exactly.
        """
        q = self.dimension
        s = np.asarray(syndrome, dtype=np.int64) % q
        det_edges, edge_dets = self._det_edges, self.edge_dets

        parent: dict[int, int] = {}
        members: dict[int, list[int]] = {}     # root -> detectors of the cluster
        grown: dict[int, list[int]] = {}       # root -> fully grown edges of the cluster
        valid: dict[int, bool] = {}            # root -> validity
        growth: dict[int, int] = {}            # edge -> half-edges grown, 0..2

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:           # path compression
                parent[x], x = root, parent[x]
            return root

        def new_cluster(t: int, is_valid: bool) -> None:
            parent[t] = t
            members[t] = [t]
            grown[t] = []
            valid[t] = is_valid

        def union(a: int, b: int) -> int:
            if a == b:
                return a
            if len(members[a]) < len(members[b]):
                a, b = b, a
            parent[b] = a
            members[a].extend(members.pop(b))
            grown[a].extend(grown.pop(b))
            valid.pop(b)
            return a

        for t in np.nonzero(s)[0].tolist():
            new_cluster(t, False)

        while True:
            invalid = [r for r, ok in valid.items() if not ok]
            if not invalid:
                break
            # How many distinct invalid clusters touch each edge that is not fully grown.
            increments: dict[int, int] = {}
            for r in invalid:
                touched: set[int] = set()
                for t in members[r]:
                    for e in det_edges[t]:
                        if e not in touched and growth.get(e, 0) < 2:
                            touched.add(e)
                            increments[e] = increments.get(e, 0) + 1
            if not increments:
                break                          # invalid clusters that cannot grow any further
            changed: list[int] = []
            for e, inc in increments.items():
                g = min(2, growth.get(e, 0) + inc)
                growth[e] = g
                if g < 2:
                    continue
                root = None
                for t in edge_dets[e]:
                    if t not in parent:
                        new_cluster(t, True)   # a detector that did not fire: valid on its own
                    rt = find(t)
                    root = rt if root is None else union(find(root), rt)
                root = find(root)
                grown[root].append(e)
                changed.append(root)
            for r in {find(x) for x in changed}:
                valid[r] = self._solve_cluster(members[r], grown[r], s) is not None

        edges: list[int] = []
        powers: list[int] = []
        complete = all(valid.values())
        for r, ok in valid.items():
            if not ok or not grown[r]:
                continue
            x0, kernel = gf_q.solve_with_kernel(self._cluster_matrix(members[r], grown[r]),
                                                s[members[r]], q)
            x = min_weight_solution(x0, kernel, q)
            nz = np.nonzero(x)[0]
            edges.extend(grown[r][k] for k in nz.tolist())
            powers.extend(x[nz].tolist())
        return edges, np.asarray(powers, dtype=np.int64), complete

    def _cluster_matrix(self, dets: list[int], edges: list[int]) -> np.ndarray:
        """H_c: the cluster's edges as columns, restricted to the cluster's detectors."""
        row = {t: i for i, t in enumerate(dets)}
        h = np.zeros((len(dets), len(edges)), dtype=np.int64)
        for j, e in enumerate(edges):
            for t, c in zip(self.edge_dets[e], self.edge_coefs[e]):
                h[row[t], j] = c               # every detector of a grown edge is in the cluster
        return h

    def _solve_cluster(self, dets: list[int], edges: list[int], s: np.ndarray) -> np.ndarray | None:
        """A solution of H_c x == s_c mod q over the cluster's edges, or None (validity test)."""
        return gf_q.solve(self._cluster_matrix(dets, edges), s[dets], self.dimension)


class QuditUnionFindDecoder(Decoder, name="union_find"):
    """Union-find over Z_q with linear-algebra cluster validity. Takes no settings."""

    # 2: minimum-weight solution inside each cluster (see the module docstring, step 4).
    version = 2

    def __init__(self):
        super().__init__()

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledUnionFind:
        q, nd, no = dem.dimension, dem.num_detectors, dem.num_observables
        best: dict[tuple, tuple[float, tuple, np.ndarray]] = {}
        invisible = ambiguous = 0
        for mech in dem.to_lines().mechanisms:
            g = {t: v % q for t, v in mech.generators[0].items() if v % q}
            dets = sorted((t, v) for t, v in g.items() if t < nd)
            if not dets:
                invisible += 1
                continue
            inv = pow(dets[0][1], -1, q)       # scale so the first detector coefficient is 1
            key = tuple((t, v * inv % q) for t, v in dets)
            obs = np.zeros(no, dtype=np.int64)
            for t, v in g.items():
                if t >= nd:
                    obs[t - nd] = v * inv % q
            prev = best.get(key)
            if prev is None:
                best[key] = (mech.probability, key, obs)
            else:
                ambiguous += 1
                if not np.array_equal(prev[2], obs) and mech.probability > prev[0]:
                    best[key] = (mech.probability, key, obs)
        items = list(best.values())
        return CompiledUnionFind(
            dimension=q, num_detectors=nd, num_observables=no,
            edge_dets=[tuple(t for t, _ in key) for _, key, _ in items],
            edge_coefs=[tuple(c for _, c in key) for _, key, _ in items],
            edge_obs=(np.array([obs for _, _, obs in items], dtype=np.int64).reshape(len(items), no)),
            edge_prob=np.array([p for p, _, _ in items], dtype=np.float64),
            num_ambiguous=ambiguous, num_invisible=invisible)
