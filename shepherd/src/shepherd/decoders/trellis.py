"""
trellis.py -- the "trellis" decoder. Placeholder: registered so it can be looked up and
listed, but not implemented yet; compiling it raises NotImplementedError.

A trellis decoder would build a trellis over the detector error model once, in
`compile_decoder_for_dem`, and decode each syndrome against it. Compiling can be expensive, so
it is marked cacheable (see decoder_cache.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shepherd.decoder import CompiledDecoder, Decoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel


class CompiledTrellisDecoder(CompiledDecoder):
    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        raise NotImplementedError("the trellis decoder is not implemented yet")


class TrellisDecoder(Decoder, name="trellis"):
    cacheable = True

    def __init__(self, **settings):
        super().__init__(**settings)

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledTrellisDecoder:
        raise NotImplementedError("the trellis decoder is not implemented yet")
