"""
shepherd -- simulating and benchmarking qudit (q > 2) stabilizer circuits on top of sdim.

The main names can be imported from the package directly; each loads its module on first use,
so `import shepherd` itself is instant (sdim is only imported when something needs it):

    from shepherd import Study, run_study, NoiseModel, CSSCode, surface_code

See README.md, or `python -m shepherd --help`.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("shepherd")
except PackageNotFoundError:            # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

# public name -> module that defines it
_EXPORTS = {
    "Task": "shepherd.task",
    "collect": "shepherd.collect",
    "TaskStats": "shepherd.stats",
    "read_stats_file": "shepherd.stats",
    "Study": "shepherd.study",
    "run_study": "shepherd.study",
    "NoiseModel": "shepherd.noise_model",
    "QECC": "shepherd.qecc",
    "PauliString": "shepherd.qecc",
    "CSSCode": "shepherd.codes.css",
    "hgp": "shepherd.codes.css",
    "repetition_code": "shepherd.codes.css",
    "surface_code": "shepherd.codes.css",
    "lacross_code": "shepherd.codes.css",
    "Decoder": "shepherd.decoder",
    "CompiledDecoder": "shepherd.decoder",
    "Experiment": "shepherd.experiment",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'shepherd' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value             # later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
