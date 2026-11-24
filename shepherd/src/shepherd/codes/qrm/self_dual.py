"""
self_dual.py -- the "qrm_self_dual" code family (self-dual quantum Reed-Muller codes).
Placeholder: registered so it can be looked up, created and listed, but `stabilizers` raises
NotImplementedError (and so do `logical_operators` and `encoding`, which are built on it).
"""

from __future__ import annotations

from shepherd.qecc import QECC, PauliString


class SelfDualQuantumReedMuller(QECC, name="qrm_self_dual"):
    def stabilizers(self) -> list[PauliString]:
        raise NotImplementedError("the qrm_self_dual code family is not implemented yet")
