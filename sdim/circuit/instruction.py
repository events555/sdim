from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..gates.registry import (
    gate_id_to_name,
    gate_name_to_id,
    is_gate_has_no_targets,
    is_gate_two_qubit,
)
from ..gates.targets import GateTarget


@dataclass
class CircuitInstruction:
    """Represents a single instruction in a quantum circuit."""

    gate_type: int
    targets: list[GateTarget] = field(default_factory=list)
    args: list[float] = field(default_factory=list)

    def __init__(
        self,
        gate_type_or_name: str | int,
        targets: int | GateTarget | Iterable[int | GateTarget],
        args: Optional[float | Iterable[float]] = None,
    ):
        if isinstance(gate_type_or_name, int):
            self.gate_type = gate_type_or_name
        else:
            self.gate_type = gate_name_to_id(gate_type_or_name)
        target_list: list[int | GateTarget]
        if isinstance(targets, (int, GateTarget)):
            target_list = [targets]
        elif isinstance(targets, Iterable):
            target_list = list(targets)  # type: ignore[arg-type]
        else:
            target_list = [targets]

        self.targets = []
        for t in target_list:
            if not isinstance(t, GateTarget):
                t = GateTarget.qudit(t)
            self.targets.append(t)
        if args is None:
            self.args = []
        elif isinstance(args, (int, float)):
            self.args = [float(args)]
        elif isinstance(args, Iterable) and not isinstance(args, (str, bytes)):
            self.args = [float(a) for a in args]
        else:
            self.args = [float(args)]
        self.validate()

    def __str__(self) -> str:
        return f"{gate_id_to_name(self.gate_type)}"

    def validate(self) -> None:
        gate_name = gate_id_to_name(self.gate_type)
        if is_gate_two_qubit(self.gate_type):
            if not self.targets:
                raise ValueError(
                    f"Two-qubit gate '{gate_name}' requires targets."
                )
            if len(self.targets) % 2 != 0:
                raise ValueError(
                    f"Two-qubit gate '{gate_name}' requires an even number of targets. "
                    f"Got {len(self.targets)} targets: {self.targets}"
                )
        else:
            if not self.targets and not is_gate_has_no_targets(self.gate_type):
                raise ValueError(
                    f"Gate '{gate_name}' requires at least one target. Got 0 targets."
                )

    def copy(self) -> "CircuitInstruction":
        return CircuitInstruction(
            self.gate_type, self.targets.copy(), self.args.copy()
        )
