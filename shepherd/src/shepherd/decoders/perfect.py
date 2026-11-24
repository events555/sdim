"""
perfect.py -- the "perfect" decoder. Placeholder: registered so it can be looked up and
listed, but not implemented yet; compiling it raises NotImplementedError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shepherd.decoder import CompiledDecoder, Decoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel


class CompiledPerfectDecoder(CompiledDecoder):
    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        raise NotImplementedError("the perfect decoder is not implemented yet")


class PerfectDecoder(Decoder, name="perfect"):
    def __init__(self, **settings):
        super().__init__(**settings)

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledPerfectDecoder:
        raise NotImplementedError("the perfect decoder is not implemented yet")
