from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, overload

import numpy as np

from ..gates.registry import (
    GATE_DATA,
    gate_id_to_name,
    gate_name_to_id,
    is_gate_records,
    is_gate_two_qubit,
)
from ..gates.targets import GateTarget
from .instruction import CircuitInstruction

if TYPE_CHECKING:
    from ..compiler.sampler import (
        CompiledDetectorSampler,
        CompiledMeasurementSampler,
    )


@dataclass
class Circuit:
    """
    Represents a quantum circuit.

    This class encapsulates the structure and operations of a quantum circuit,
    including the number of qudits, their dimension, and the sequence of gate operations.

    Attributes:
        num_qudits (int): The number of qudits in the circuit.
        dimension (int): The dimension of each qudit (default is 2 for qubits).
        operations (list): A list of CircuitInstruction objects representing the circuit operations.

    Raises:
        ValueError: If num_qudits is less than 1 or dimension is less than 2.
    """

    num_qudits: int
    dimension: int = 2
    operations: List[CircuitInstruction] = field(default_factory=list)

    @overload
    def append(self, op: CircuitInstruction) -> "Circuit": ...

    @overload
    def append(
        self,
        name: str,
        targets: int | GateTarget | Iterable[int | GateTarget],
        target: None = None,
        args: Optional[float | Iterable[float]] = None,
    ) -> "Circuit": ...

    @overload
    def append(
        self,
        name: str,
        targets: int | GateTarget | Iterable[int | GateTarget],
        target: int | GateTarget | Iterable[int | GateTarget],
        args: Optional[float | Iterable[float]] = None,
    ) -> "Circuit": ...

    def append(
        self,
        name_or_op: str | CircuitInstruction,
        targets: Optional[int | GateTarget | Iterable[int | GateTarget]] = None,
        target: Optional[int | GateTarget | Iterable[int | GateTarget]] = None,
        args: Optional[float | Iterable[float]] = None,
    ) -> "Circuit":
        """
        Appends an operation to the circuit.

        Overloads:
        1. Append with a gate name and targets:
            - `append("X", 0)`
            - `append("X", [0, 1])`
            - `append("X", GateTarget.qudit(0))`
        2. Append with a gate name, control(s), and target(s) for two-qudit operations:
            - `append("CNOT", 0, 1)`
            - `append("CNOT", [0, 2], [1, 3])`
            - `append("CNOT", GateTarget.qudit(0), GateTarget.qudit(1))`
            - 'append("CNOT, [0, 1, 2, 3])`
        3. Append a pre-constructed CircuitInstruction.

        Args:
            name_or_op: Either the gate name (str) or a CircuitInstruction.
            targets: For single-qudit operations, the target(s). For two-qudit operations,
                    this represents the control(s).
            target: For two-qudit operations, the target(s). Leave as None for single-qudit operations.
            args: Optional parameter(s) for the gate.

        Returns:
            The Circuit object (self).
        """
        if isinstance(name_or_op, CircuitInstruction):
            name_or_op.validate()
            self.operations.append(name_or_op)
            return self

        gate_id = gate_name_to_id(name_or_op)
        instruction_targets: List[GateTarget] = []

        if is_gate_two_qubit(gate_id):
            if target is not None:
                # Form: append("GATE", controls_arg, targets_arg)
                # 'targets' parameter is controls, 'target' parameter is targets_values
                if (
                    targets is None
                ):  # Controls cannot be None if targets_arg is provided
                    raise ValueError(
                        f"For two-qubit gate '{name_or_op}' with explicit target argument, "
                        "the controls argument (targets) cannot be None."
                    )

                control_list = self._normalize_targets(targets)
                target_list = self._normalize_targets(target)

                n_controls = len(control_list)
                n_targets = len(target_list)

                if n_controls == n_targets:
                    if n_controls == 0:  # e.g. CNOT, [], []
                        raise ValueError(
                            f"For two-qubit gate '{name_or_op}', control and target lists cannot both be empty."
                        )
                    for c, t_val in zip(control_list, target_list):
                        instruction_targets.append(c)
                        instruction_targets.append(t_val)
                elif (
                    n_controls == 1 and n_targets > 0
                ):  # Allow broadcasting control
                    for t_val in target_list:
                        instruction_targets.append(control_list[0])
                        instruction_targets.append(t_val)
                elif (
                    n_targets == 1 and n_controls > 0
                ):  # Allow broadcasting target
                    for c in control_list:
                        instruction_targets.append(c)
                        instruction_targets.append(target_list[0])
                else:
                    raise ValueError(
                        f"For two-qubit gate '{name_or_op}', control and target arguments have incompatible "
                        f"lengths ({n_controls} and {n_targets}). They must be equal, or one must be singular "
                        "and the other non-empty."
                    )
            else:
                # Form: append("GATE", [c1, t1, c2, t2, ...])
                # 'targets' parameter is the flattened list, 'target' parameter is None
                if targets is None:
                    raise ValueError(
                        f"For two-qubit gate '{name_or_op}' shorthand, "
                        "the targets argument cannot be None."
                    )

                instruction_targets = self._normalize_targets(targets)
                if len(instruction_targets) % 2 != 0:
                    raise ValueError(
                        f"For two-qubit gate '{name_or_op}' shorthand (e.g., CNOT, [c1,t1,c2,t2,...]), "
                        "the list of targets must have an even number of elements. "
                        f"Got {len(instruction_targets)}."
                    )
                if not instruction_targets:
                    raise ValueError(
                        f"For two-qubit gate '{name_or_op}' shorthand, targets list cannot be empty."
                    )

        else:  # Single-qudit gate or other non-two-qubit gate
            if targets is None:
                raise ValueError(
                    f"Gate '{name_or_op}' requires a target (targets argument cannot be None)."
                )
            instruction_targets = self._normalize_targets(targets)
            # Validation for non-empty will be done by CircuitInstruction

        op = CircuitInstruction(gate_id, instruction_targets, args)
        self.operations.append(op)
        return self

    @staticmethod
    def _normalize_targets(targets):
        """Convert targets into a list of GateTarget objects."""
        if targets is None:
            return []

        if not isinstance(targets, Iterable) or isinstance(
            targets, (str, bytes)
        ):
            targets = [targets]

        return [
            t if isinstance(t, GateTarget) else GateTarget.qudit(t)
            for t in targets
        ]

    def __mul__(self, repetitions: int):
        """
        Replicates the circuit by the specified number of times.

        Args:
            repititions (int): The number of times to replicate the circuit.

        Returns:
            Circuit: A new Circuit object with the replicated operations.
        """
        original_operations = list(self.operations)
        for _ in range(repetitions - 1):
            self.operations.extend(original_operations)
        return self

    def __imul__(self, repetitions: int):
        return self.__mul__(repetitions)

    def __add__(self, other):
        """
        Adds two Circuit objects together.

        Args:
            other (Circuit): The Circuit object to add to this one.

        Returns:
            Circuit: A new Circuit object with combined operations.

        Raises:
            ValueError: If the dimensions of the two circuits don't match.
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot add circuits with different dimensions")

        new_num_qudits = max(self.num_qudits, other.num_qudits)
        new_circuit = Circuit(new_num_qudits, self.dimension)

        new_circuit.operations = list(self.operations)

        new_circuit.operations.extend(other.operations)

        return new_circuit

    def __iadd__(self, other):
        """
        In-place addition of another Circuit object.

        Args:
            other (Circuit): The Circuit object to add to this one.

        Returns:
            Circuit: The modified Circuit object with combined operations.

        Raises:
            ValueError: If the dimensions of the two circuits don't match.
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot add circuits with different dimensions")

        self.num_qudits = max(self.num_qudits, other.num_qudits)

        # Append operations from other
        self.operations.extend(other.operations)

        return self

    def __str__(self):
        """
        Returns a string representation of the circuit.

        Returns:
            str: A string representation of all operations in the circuit.
        """
        return "\n".join(str(op) for op in self.operations)

    def inverse(self):
        """
        Returns the inverse of the circuit by reversing the order of operations
        and substituting each gate with its inverse. Raises ValueError if any
        gate in the circuit is not invertible.

        Returns:
            Circuit: A new Circuit object representing the inverse circuit.
        """
        new_circuit = Circuit(self.num_qudits, self.dimension)
        new_ops = []
        for op in reversed(self.operations):
            gate_name = gate_id_to_name(op.gate_type)
            gate_data = GATE_DATA.get(gate_name)
            if gate_data is None or gate_data.get("inverse") is None:
                raise ValueError(
                    f"Gate {gate_name} is not invertible, cannot create inverse circuit."
                )
            inverse_name = gate_data["inverse"]
            new_gate_id = gate_name_to_id(inverse_name)
            new_op = op.copy()
            new_op.gate_type = new_gate_id
            new_ops.append(new_op)
        new_circuit.operations = new_ops
        return new_circuit

    @property
    def num_measurements(self) -> int:
        """
        Returns the number of measurements in the circuit.

        Returns:
            int: The number of measurements in the circuit.
        """
        return sum(
            1 * len(op.targets)
            for op in self.operations
            if is_gate_records(op.gate_type)
        )

    def _build_ir(self) -> tuple[np.ndarray, np.ndarray]:
        from ..compiler.lowering import lower_to_ir

        return lower_to_ir(self)

    def _build_noise(
        self, shots: int, xp=None, rng=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        from ..noise.sampling import build_noise_banks

        return build_noise_banks(self, shots, xp=xp or np, rng=rng)

    def reference_sample(
        self,
        ir: Optional[np.ndarray] = None,
        args_pool: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from ..compiler.lowering import lower_to_ir, reference_sample

        if ir is None:
            ir, args_pool = lower_to_ir(self)
        return reference_sample(ir, self.num_qudits, self.dimension, args_pool)

    def compile_sampler(
        self,
        *,
        skip_reference_sample: bool = False,
        seed: Optional[int] = None,
        reference_sample: Optional[np.ndarray] = None,
        ir_array: Optional[np.ndarray] = None,
        args_pool: Optional[np.ndarray] = None,
        backend: str = "numpy",
    ) -> "CompiledMeasurementSampler":
        from ..compiler import compile as _compile
        from ..compiler.sampler import CompiledMeasurementSampler

        compiled = _compile(
            self,
            skip_reference_sample=skip_reference_sample,
            ref_sample=reference_sample,
            ir_array=ir_array,
            args_pool=args_pool,
        )
        return CompiledMeasurementSampler(
            circuit_object=self,
            reference_sample=compiled.reference_sample,
            ir_array=compiled.ir_array,
            args_pool=compiled.args_pool,
            measurement_records=compiled.measurement_records,
            seed=seed,
            backend=backend,
        )

    def compile_detector_sampler(
        self,
        *,
        seed: Optional[int] = None,
        reference_sample: Optional[np.ndarray] = None,
        ir_array: Optional[np.ndarray] = None,
        args_pool: Optional[np.ndarray] = None,
    ) -> "CompiledDetectorSampler":
        from ..compiler import compile as _compile
        from ..compiler.sampler import CompiledDetectorSampler

        compiled = _compile(
            self,
            ref_sample=reference_sample,
            ir_array=ir_array,
            args_pool=args_pool,
        )
        return CompiledDetectorSampler(
            circuit_object=self,
            reference_sample=compiled.reference_sample,
            ir_array=compiled.ir_array,
            args_pool=compiled.args_pool,
            measurement_records=compiled.measurement_records,
            seed=seed,
        )

    @classmethod
    def from_operation_list(cls, operation_list, num_qudits, dimension):
        """
        Creates a Circuit object from a list of operations.

        Args:
            operation_list (list): A list of operations, either as tuples or CircuitInstructions.
            num_qudits (int): The number of qudits in the circuit.
            dimension (int): The dimension of each qudit.

        Returns:
            Circuit: A new Circuit object with the specified operations.

        Raises:
            ValueError: If an unsupported operation type is encountered.
        """
        circuit = cls(num_qudits, dimension)
        for op in operation_list:
            if isinstance(
                op, tuple
            ):  # If the operation is a tuple (gate, qudits)
                gate_name = op[0]
                qudits = op[1]
                if len(qudits) == 1:
                    circuit.append(gate_name, qudits[0])
                elif len(qudits) == 2:
                    circuit.append(gate_name, qudits[0], qudits[1])
                else:
                    raise ValueError(
                        f"Unsupported number of qudits for gate {gate_name}"
                    )
            elif isinstance(
                op, CircuitInstruction
            ):  # If the operation is a CircuitInstruction
                circuit.append(op.gate_name, op.qudit_index, op.target_index)
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return circuit
