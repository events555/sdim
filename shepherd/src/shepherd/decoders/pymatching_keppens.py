"""
pymatching_keppens.py -- the "pymatching_keppens" decoder: minimum-weight perfect matching on
a graph with one node per measurable eigenvalue of each detector, after Keppens et al., "Qudit vs.
Qubit: Simulated performance of error correction codes in higher dimensions", arXiv:2502.05992,
section 4.1 and appendix A.

Their construction, built here from a detector error model instead of a parity-check matrix:

1. Every detector k gets d - 1 eigenvalue nodes, (k, 1) ... (k, d - 1), one per non-zero value
   it can take (the paper's "q - 1 nodes for every ancilla, representing all q - 1 possible
   measurable eigenvalues"). A shot fires node (k, s) when detector k reads s != 0, so each
   detector fires at most one of its nodes and the value is kept.
2. Every *power* of an error is its own edge. A line mechanism v (see `dem.to_lines()`) at power
   a moves detector u by a * v[u] and detector w by a * v[w], so it becomes the edge
   (u, a * v[u]) -- (w, a * v[w]), or an edge to the boundary if v touches one detector. The edge
   carries the observable shift a * v[observables].
3. A line fires with probability pi and then picks a power uniformly from Z_d, so each power
   a != 0 has probability pi / d and weight log((1 - q) / q) with q = pi / d. (The paper uses
   uniform weights, which is what depolarizing noise gives.) When several (line, power) pairs give
   the same pair of nodes, the most likely one is kept.
4. Lines touching 3 or more detectors are split into existing edges, as in the "pymatching"
   decoder (the paper leaves them to the matching; a DEM has to say something about them).

The prediction is the sum of the observable shifts of the matched edges, mod d.

Unlike the binarized decoder, matching sees the syndrome values. D3=2, D5=2 is matched to the
boundary, because an edge (3, 2) -- (5, 2) would need an error that moves both by the same
amount, which this circuit's errors do not do.

The cost is that paths are only right mod 2. A path through a node (k, a) that did not fire uses
two edges at that node, that is two errors that each move detector k by a, which leaves 2a != 0
for d > 2. So chains of two or more errors through an intermediate detector are not modeled. That
is fine for a distance-3 code decoded one error at a time, the paper's setting, but limits this
decoder on larger codes and on many rounds.
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
        "The 'pymatching_keppens' decoder needs PyMatching. Install it with: pip install -e \".[mwpm]\""
    ) from e

from sdim.dem import merge_subgroup_probabilities

from shepherd.decoder import CompiledDecoder, Decoder
from shepherd.decoders.pymatching_binarized_qudit_syndromes import (
    HYPEREDGE_MODES, _MIN_Q, _canonical, _decompose)

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel


class CompiledKeppensMatchingDecoder(CompiledDecoder):
    """
    A PyMatching graph whose eigenvalue node (detector k, value a) is node k * (d - 1) + a - 1.

    Edge j carries fault id j, so decoding a batch gives, for each shot, which edges were matched;
    the prediction is that selection times the edges' observable shifts, mod d.
    """

    def __init__(self, *, dimension: int, num_detectors: int, num_observables: int,
                 edge_a: np.ndarray, edge_b: np.ndarray, edge_obs: np.ndarray,
                 edge_weight: np.ndarray, num_dropped_hyperedges: int = 0):
        self.dimension = dimension
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.edge_a, self.edge_b = edge_a, edge_b      # node ids; edge_b = -1 for the boundary
        self.edge_obs = edge_obs                       # (edges, num_observables) mod d
        self.edge_weight = edge_weight
        self.num_dropped_hyperedges = num_dropped_hyperedges   # for inspection only
        self._build()

    def _build(self) -> None:
        self._matching = pymatching.Matching()
        for j, (a, b, w) in enumerate(zip(self.edge_a.tolist(), self.edge_b.tolist(),
                                          self.edge_weight.tolist())):
            if b < 0:
                self._matching.add_boundary_edge(a, fault_ids={j}, weight=w, merge_strategy="disallow")
            else:
                self._matching.add_edge(a, b, fault_ids={j}, weight=w, merge_strategy="disallow")
        self._num_nodes = self._matching.num_detectors
        has_edge = np.zeros(self.num_detectors * (self.dimension - 1), dtype=bool)
        has_edge[self.edge_a] = True
        has_edge[self.edge_b[self.edge_b >= 0]] = True
        self._matchable = has_edge[:self._num_nodes]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_matching", "_num_nodes", "_matchable"):
            state.pop(k, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._build()

    def fired_nodes(self, detections: np.ndarray) -> np.ndarray:
        """
        (shots, num_detectors) values mod d -> (shots, matching nodes) uint8: for each detector
        reading s != 0, its eigenvalue node (k, s) is 1.
        """
        d = self.dimension
        values = np.asarray(detections, dtype=np.int64) % d
        shots = values.shape[0]
        nodes = np.zeros((shots, self.num_detectors * (d - 1)), dtype=np.uint8)
        rows, dets = np.nonzero(values)
        nodes[rows, dets * (d - 1) + values[rows, dets] - 1] = 1
        nodes = nodes[:, :self._num_nodes]
        nodes[:, ~self._matchable] = 0     # nodes without edges can never be matched
        return nodes

    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        shots = detections.shape[0]
        out = np.zeros((shots, self.num_observables), dtype=np.int64)
        if self._num_nodes == 0 or self.num_observables == 0 or shots == 0:
            return out
        syndromes = self.fired_nodes(detections)
        try:
            chosen = self._matching.decode_batch(syndromes)
        except ValueError:
            # Some shot has odd parity in a component without a boundary (common for d >= 5,
            # where the graph splits into components). Decode one at a time, leaving those at 0.
            chosen = np.zeros((shots, len(self.edge_a)), dtype=np.uint8)
            for s in range(shots):
                if syndromes[s].any():
                    try:
                        chosen[s] = self._matching.decode(syndromes[s])
                    except ValueError:
                        pass
        return (chosen.astype(np.int64) @ self.edge_obs) % self.dimension


class KeppensMatchingDecoder(Decoder, name="pymatching_keppens"):
    """
    Keppens et al.'s qudit MWPM: one eigenvalue node per (detector, non-zero value), one edge per
    error power. See the module docstring.

    Settings:
        hyperedges: "decompose" (default), "raise" or "drop", as for the "pymatching" decoder.
    """

    def __init__(self, hyperedges: str = "decompose"):
        if hyperedges not in HYPEREDGE_MODES:
            raise ValueError(f"hyperedges must be one of {HYPEREDGE_MODES}, got {hyperedges!r}")
        super().__init__(hyperedges=hyperedges)
        self.hyperedges = hyperedges

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledKeppensMatchingDecoder:
        d = dem.dimension
        nd, no = dem.num_detectors, dem.num_observables

        # Lines on 1 or 2 detectors, with hyperedges split into them (same as "pymatching").
        line_prob: dict[tuple, float] = {}
        hyper: list[tuple[tuple, float]] = []
        for mech in dem.to_lines().mechanisms:
            key = _canonical(mech.generators[0], d)
            n_det = sum(1 for t, _ in key if t < nd)
            if n_det == 0:
                continue
            if n_det <= 2:
                prev = line_prob.get(key)
                line_prob[key] = (mech.probability if prev is None
                                  else merge_subgroup_probabilities(prev, mech.probability))
            else:
                hyper.append((key, mech.probability))
        by_detector: dict[int, list[tuple]] = defaultdict(list)
        for key in line_prob:
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
                line_prob[part] = merge_subgroup_probabilities(line_prob[part], p)

        # Every non-zero power of every line is an edge between (detector, value) nodes.
        def node(det: int, value: int) -> int:
            return det * (d - 1) + value - 1

        best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
        for key, p in line_prob.items():
            dets = [(t, c) for t, c in key if t < nd]
            obs = np.zeros(no, dtype=np.int64)
            for t, c in key:
                if t >= nd:
                    obs[t - nd] = c
            q = p / d
            for a in range(1, d):
                ends = sorted(node(t, a * c % d) for t, c in dets)
                pair = (ends[0], ends[1]) if len(ends) == 2 else (ends[0], -1)
                if pair not in best or q > best[pair][0]:
                    best[pair] = (q, a * obs % d)

        m = len(best)
        edge_a = np.empty(m, dtype=np.int64)
        edge_b = np.empty(m, dtype=np.int64)
        edge_obs = np.zeros((m, no), dtype=np.int64)
        edge_weight = np.empty(m, dtype=np.float64)
        for j, ((a, b), (q, obs)) in enumerate(sorted(best.items(), key=lambda kv: kv[0])):
            edge_a[j], edge_b[j] = a, b
            edge_obs[j] = obs
            q = min(max(q, _MIN_Q), 0.5)
            edge_weight[j] = math.log((1.0 - q) / q)

        return CompiledKeppensMatchingDecoder(
            dimension=d, num_detectors=nd, num_observables=no, edge_a=edge_a, edge_b=edge_b,
            edge_obs=edge_obs, edge_weight=edge_weight, num_dropped_hyperedges=dropped)
