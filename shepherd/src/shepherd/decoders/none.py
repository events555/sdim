"""
none.py -- the "none" decoder: no correction at all.

It ignores the syndrome and always predicts that no logical observable changed, so a shot counts
as a logical error exactly when the noise changed one. Its logical error rate is the baseline a
real decoder has to beat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shepherd.decoder import CompiledDecoder, Decoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel


class CompiledZeroDecoder(CompiledDecoder):
    # Every syndrome gets the same prediction, so finding the distinct ones would be wasted work.
    deduplicate_syndromes = False

    def __init__(self, num_observables: int):
        self.num_observables = num_observables

    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        return np.zeros((detections.shape[0], self.num_observables), dtype=np.int64)


class ZeroDecoder(Decoder, name="none"):
    """Always predicts no logical error. Takes no settings."""

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledZeroDecoder:
        return CompiledZeroDecoder(dem.num_observables)
