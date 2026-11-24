from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sdim import Circuit

from shepherd.qecc import QECC


class Experiment(ABC):
    registry: ClassVar[dict[str, type[Experiment]]] = {}

    # name -> module that defines it; imported on first lookup only
    _lazy: ClassVar[dict[str, str]] = {
        "lrb":       "shepherd.benchmarking.lrb",
        "ler":       "shepherd.benchmarking.ler",
        "msd":       "shepherd.benchmarking.msd",
    }

    @classmethod
    def lookup(cls, name: str) -> type[Experiment]:
        if name not in cls.registry:
            module_path = cls._lazy.get(name)
            if module_path is None:
                raise KeyError(f"Unknown experiment {name!r}; available: {cls.available()}")
            importlib.import_module(module_path)   # class statement runs -> __init_subclass__ registers it
            if name not in cls.registry:
                raise RuntimeError(f"{module_path} was imported but did not register {name!r}")
        return cls.registry[name]

    @classmethod
    def create(cls, name: str, dimension: int, **params) -> Experiment:
        return cls.lookup(name)(dimension, **params)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(set(cls.registry) | set(cls._lazy))

    @classmethod
    def register_lazy(cls, name: str, module_path: str) -> None:
        """Let users add their own codes without editing this file."""
        if name in cls._lazy or name in cls.registry:
            raise ValueError(f"Duplicate key: {name}")
        cls._lazy[name] = module_path


    # The name an experiment class was registered under (set by __init_subclass__).
    experiment_name: ClassVar[str | None] = None

    def __init_subclass__(cls, name: str | None = None, **kwargs):
            super().__init_subclass__(**kwargs)
            if name is None:
                return
            if name in Experiment.registry:
                raise ValueError(f"Duplicate key: {name}")
            Experiment.registry[name] = cls
            cls.experiment_name = name


    @abstractmethod
    def build(self, code: QECC) -> Circuit:
        """
        Noiseless circuit with detectors and observables annotated. Noise is added afterwards by
        `NoiseModel.apply`, so an experiment marks where qudits wait with I gates (and layers with
        TICK) and leaves the rest to the model.
        """

    def metadata(self, code: QECC) -> dict:
        """What this experiment contributes to a task's json_metadata. Extend in subclasses."""
        return {"experiment": self.experiment_name}

    @abstractmethod
    def expected_observables(self, code: QECC) -> np.ndarray:
        """What each observable `build(code)` declares should read if no logical error occurred."""