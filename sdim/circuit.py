from .gatedata import *
from .simulators.extended_tableau_simulator import ExtendedTableauSimulator
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, overload, TYPE_CHECKING 
import numpy as np

if TYPE_CHECKING:
    from .sampler import CompiledMeasurementSampler, CompiledDetectorSampler

@dataclass
class CircuitInstruction:
    """
    Represents a single instruction in a quantum circuit.

    This class encapsulates the details of a quantum gate operation,
    including the gate type, target qudit(s), and associated metadata.

    Attributes:
        gate_type (int): The gate type (ID) for the instruction.
        targets (list[GateTarget]): A list of target qudits for the instruction.
        args (list[float]): A list of arguments for the instruction

    Raises:
        ValueError: If the specified gate is not found in gate_data.
    """
    gate_type: int
    targets: List[GateTarget] = field(default_factory=list)
    args: List[float] = field(default_factory=list)

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
        if not isinstance(targets, Iterable) or isinstance(targets, (str, bytes)):
            targets = [targets]

        self.targets = []
        for t in targets:
            if not isinstance(t, GateTarget):
                t = GateTarget.qudit(t)
            self.targets.append(t)
        if args is None:
            self.args = []
        elif not (isinstance(args, Iterable) and not isinstance(args, (str, bytes))):
            self.args = [args]
        else:
            self.args = list(args)
        self.validate()

    def __str__(self):
        """
        Returns the canonical name given the gate type

        Returns:
            str: A string representation of the instruction.
        """
        return f"{gate_id_to_name(self.gate_type)}"

    
    def validate(self):
        """
        Validates the instruction by checking if the gate name is valid.

        Raises:
            ValueError: If the gate name is not found in the gate data.
        """
        gate_name = gate_id_to_name(self.gate_type) # For clearer error messages
        if is_gate_two_qubit(self.gate_type):
            if not self.targets:
                raise ValueError(f"Two-qubit gate '{gate_name}' requires targets.")
            if len(self.targets) % 2 != 0:
                raise ValueError(
                    f"Two-qubit gate '{gate_name}' requires an even number of targets. "
                    f"Got {len(self.targets)} targets: {self.targets}"
                )
        else: 
            if not self.targets and not is_gate_has_no_targets(self.gate_type):
                raise ValueError(f"Gate '{gate_name}' requires at least one target. Got 0 targets.")
            
    def copy(self) -> "CircuitInstruction":
        return CircuitInstruction(self.gate_type, self.targets.copy(), self.args.copy())


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
    def append(self, name: str, control: int | Iterable[int], 
               target: int | Iterable[int], 
               arg: Optional[float | Iterable[float]] = None, *, 
               tag: str = "") -> "Circuit":
        ...

    @overload
    def append(self, op: CircuitInstruction) -> "Circuit":
        ...
    
    def append(
        self,
        name_or_op: str | CircuitInstruction,
        targets: Optional[int 
                          | GateTarget 
                          | Iterable[int | GateTarget]] = None,
        target: Optional[int 
                         | GateTarget 
                         | Iterable[int | GateTarget]] = None,
        args: Optional[float | Iterable[float]] = None
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
                if targets is None: # Controls cannot be None if targets_arg is provided
                    raise ValueError(f"For two-qubit gate '{name_or_op}' with explicit target argument, "
                                     "the controls argument (targets) cannot be None.")

                control_list = self._normalize_targets(targets)
                target_list = self._normalize_targets(target)

                n_controls = len(control_list)
                n_targets = len(target_list)

                if n_controls == n_targets:
                    if n_controls == 0: # e.g. CNOT, [], []
                        raise ValueError(f"For two-qubit gate '{name_or_op}', control and target lists cannot both be empty.")
                    for c, t_val in zip(control_list, target_list):
                        instruction_targets.append(c)
                        instruction_targets.append(t_val)
                elif n_controls == 1 and n_targets > 0: # Allow broadcasting control
                    for t_val in target_list:
                        instruction_targets.append(control_list[0])
                        instruction_targets.append(t_val)
                elif n_targets == 1 and n_controls > 0: # Allow broadcasting target
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
                    raise ValueError(f"For two-qubit gate '{name_or_op}' shorthand, "
                                     "the targets argument cannot be None.")

                instruction_targets = self._normalize_targets(targets)
                if len(instruction_targets) % 2 != 0:
                     raise ValueError(
                        f"For two-qubit gate '{name_or_op}' shorthand (e.g., CNOT, [c1,t1,c2,t2,...]), "
                        "the list of targets must have an even number of elements. "
                        f"Got {len(instruction_targets)}."
                    )
                if not instruction_targets:
                    raise ValueError(f"For two-qubit gate '{name_or_op}' shorthand, targets list cannot be empty.")

        else: # Single-qudit gate or other non-two-qubit gate
            if targets is None:
                raise ValueError(f"Gate '{name_or_op}' requires a target (targets argument cannot be None).")
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
            
        if not isinstance(targets, Iterable) or isinstance(targets, (str, bytes)):
            targets = [targets]
            
        return [t if isinstance(t, GateTarget) else GateTarget.qudit(t) for t in targets]

    def __mul__(self, repetitions:int):
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
    
    def __imul__(self, repetitions:int):
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
                raise ValueError(f"Gate {gate_name} is not invertible, cannot create inverse circuit.")
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
        return sum(1*len(op.targets) for op in self.operations if is_gate_records(op.gate_type))
    
    def _build_ir(self,) -> np.ndarray:
        """
        Builds an intermediate representation (IR) for the given circuits, flattening all instructions, 
        and (if applicable) precomputes an array of sampled Pauli noise outcomes for noise gates.

        Note: np.iinfo(np.int64).max is used as a placeholder for the target index of single-qudit gates.
        
        Args:
            circuits (list[Circuit]): A list of Circuit objects.
            shots (int): The number of extra shots for which noise outcomes
                            will be sampled.
        
        Returns:
            tuple:
                - A NumPy array of IR instructions with each element as a tuple
                (gate_id, qudit_index, target_index).
                - A NumPy array of shape (num_noise_gates, extra_shots, [x_block, z_block]) containing
                pre-sampled noise outcomes for each noise gate encountered.
                If no noise gate is present, an empty array is returned.
        """
        ir_list  = []

        for instruction in self.operations:
            gate_id = instruction.gate_type
            arg0 = float(instruction.args[0]) if instruction.args else np.nan
            if is_gate_two_qubit(gate_id):
                # Process targets in pairs for two-qubit gates
                for i in range(0, len(instruction.targets), 2):
                    if i + 1 < len(instruction.targets):
                        control = instruction.targets[i]
                        target = instruction.targets[i + 1]
                        
                        control_idx = control.value
                        target_idx = target.value
                        
                        ir_list.append((gate_id, control_idx, target_idx, arg0))
                        
            else:
                for target in instruction.targets:
                    qudit_idx = target.value
                    ir_list.append((gate_id, qudit_idx, np.iinfo(np.int64).max, arg0))

        ir_dtype = np.dtype([
            ('gate_id', np.int64),
            ('qudit_index', np.int64),
            ('target_index', np.int64),
            ('arg0', np.float64),
        ])
        ir_array = np.array(ir_list, dtype=ir_dtype)

        return ir_array
    
    def _build_noise(self, shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        noise1_list = []
        noise2_list = []
        erasure_list = []
        measurement_list = []
        for instruction in self.operations:
            gate_id = instruction.gate_type
            gate_name = gate_id_to_name(gate_id)
            if not is_gate_noisy(gate_id):
                continue
            if is_gate_two_qubit(gate_id):
                for i in range(0, len(instruction.targets), 2):
                    if i + 1 < len(instruction.targets):
                        if gate_name == "DEPOLARIZE2" and instruction.args:
                            noise_pair = self._sample_depolarize2_noise(shots, instruction.args[0])
                            noise2_list.append(noise_pair)
                        elif gate_name == "PAULI_CHANNEL_2" and len(instruction.args) >= 15:
                            noise_pair = self._sample_pauli_channel2_noise(shots, instruction.args)
                            noise2_list.append(noise_pair)
            else:
                for _ in instruction.targets:                                
                    if gate_name == "X_ERROR" and instruction.args:
                        noise_pair = self._sample_x_error(shots, instruction.args[0])
                        noise1_list.append(noise_pair)
                    elif gate_name == "Z_ERROR" and instruction.args:
                        noise_pair = self._sample_z_error(shots, instruction.args[0])
                        noise1_list.append(noise_pair)
                    elif gate_name == "Y_ERROR" and instruction.args:
                        noise_pair = self._sample_y_error(shots, instruction.args[0])
                        noise1_list.append(noise_pair)
                    elif gate_name == "DEPOLARIZE1" and instruction.args:
                        noise_pair = self._sample_depolarize1_noise(shots, instruction.args[0])
                        noise1_list.append(noise_pair)
                    elif gate_name == "PAULI_CHANNEL_1" and len(instruction.args) >= 3:
                        noise_pair = self._sample_pauli_channel1_noise(shots, instruction.args[:3])
                        noise1_list.append(noise_pair)
                    elif gate_name == "HERALDED_ERASURE" and instruction.args:
                        pauli, locations = self._sample_heralded_erasure_noise(shots, instruction.args[0])
                        noise1_list.append(pauli)
                        erasure_list.append(locations)
                    elif gate_name in ("M", "M_X", "MR", "MR_X"):
                        probability = instruction.args[0] if instruction.args else 0.0
                        measurement_array = self._sample_x_error(shots, probability)
                        measurement_list.append(measurement_array)

        noise1_array = np.array(noise1_list, dtype=np.int64) if noise1_list else np.empty((0, shots, 2), dtype=np.int64)
        noise2_array = np.array(noise2_list, dtype=np.int64) if noise2_list else np.empty((0, shots, 4), dtype=np.int64)
        erasure_array = np.array(erasure_list, dtype=np.int8) if erasure_list else None
        measurement_array = np.array(measurement_list, dtype=np.int64) if measurement_list else None

        return noise1_array, noise2_array, erasure_array, measurement_array
        
    def _sample_x_error(self, shots: int, error_prob: float) -> np.ndarray:
        """
        Samples a single-qudit X error.
        With probability 1 - error_prob no error is applied (exponent 0).
        With probability error_prob, a nonzero exponent is chosen uniformly.
        Returns an array of shape (shots, 2) where the first column is the X exponent.
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")
        noise = np.zeros((shots, 2), dtype=np.int64)
        rnd = np.random.rand(shots)
        mask = rnd < error_prob
        if d == 2:
            noise[mask, 0] = 1
        else:
            noise[mask, 0] = np.random.randint(1, d, size=np.sum(mask))
        return noise

    def _sample_z_error(self, shots: int, error_prob: float) -> np.ndarray:
        """
        Samples a single-qudit Z error.
        Returns an array of shape (shots, 2) where the second column is the Z exponent.
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")
        noise = np.zeros((shots, 2), dtype=np.int64)
        rnd = np.random.rand(shots)
        mask = rnd < error_prob
        if d == 2:
            noise[mask, 1] = 1
        else:
            noise[mask, 1] = np.random.randint(1, d, size=np.sum(mask))
        return noise

    def _sample_y_error(self, shots: int, error_prob: float) -> np.ndarray:
        """
        Samples a single-qudit Y error.
        Here we define Y error as a correlated X and Z error.
        Returns (a, a) for a randomly chosen nonzero a with probability error_prob.
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")
        noise = np.zeros((shots, 2), dtype=np.int64)
        rnd = np.random.rand(shots)
        mask = rnd < error_prob
        if d == 2:
            noise[mask, 0] = 1
            noise[mask, 1] = 1
        else:
            a = np.random.randint(1, d, size=np.sum(mask))
            noise[mask, 0] = a
            noise[mask, 1] = a
        return noise

    def _sample_depolarize1_noise(self, shots: int, error_prob: float) -> np.ndarray:
        """
        For a single qudit, with probability 1 - error_prob no error occurs.
        Otherwise, uniformly sample an error from the set of non-identity errors,
        i.e. from {(x, z) : x,z in {0,...,d-1}} \ {(0,0)}.
        Returns an array of shape (shots, 2).
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")
        noise = np.zeros((shots, 2), dtype=np.int64)
        rnd = np.random.rand(shots)
        mask = rnd < error_prob
        num_errors = np.sum(mask)
        if num_errors > 0:
            errors = [(x, z) for x in range(d) for z in range(d) if not (x == 0 and z == 0)]
            errors = np.array(errors, dtype=np.int64)
            indices = np.random.randint(0, len(errors), size=num_errors)
            selected = errors[indices]
            noise[mask, :] = selected
        return noise

    def _sample_depolarize2_noise(self, shots: int, error_prob: float) -> np.ndarray:
        """
        For a two-qudit depolarizing channel:
          - With probability 1 - error_prob, no error occurs on either qudit.
          - Otherwise, for each qudit sample a non-identity error uniformly.
        Returns an array of shape (shots, 4), where the first two entries
        correspond to the first qudit and the last two to the second.
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")
        noise = np.zeros((shots, 4), dtype=np.int64)
        rnd = np.random.rand(shots)
        mask = rnd < error_prob
        num_errors = np.sum(mask)
        if num_errors > 0:
            errors = [(x, z) for x in range(d) for z in range(d) if not (x == 0 and z == 0)]
            errors = np.array(errors, dtype=np.int64)
            indices1 = np.random.randint(0, len(errors), size=num_errors)
            indices2 = np.random.randint(0, len(errors), size=num_errors)
            selected1 = errors[indices1]
            selected2 = errors[indices2]
            noise[mask, :2] = selected1
            noise[mask, 2:] = selected2
        return noise
    
    def _sample_pauli_channel1_noise(self, shots: int, pmf: list[float]) -> np.ndarray:
        """
        Implements a single-qudit Pauli channel using a probability mass function (pmf)
        over the d² generalized Pauli operators with the ordering:
            index = x * d + z,  for x, z in {0, ..., d-1}.
        The pmf can be provided as a full specification (length d²) or as probabilities
        for non-identity errors (length d² - 1), in which case the identity probability is
        computed as 1 - sum(pmf).
        Returns an array of shape (shots, 2) where each row is the (x, z) exponent pair.
        """
        d = self.dimension
        pmf = np.array(pmf, dtype=float)
        if pmf.size == d**2 - 1:
            identity_prob = 1 - np.sum(pmf)
            pmf = np.concatenate(([identity_prob], pmf))
        elif pmf.size != d**2:
            raise ValueError(
                f"PMF for PAULI_CHANNEL_1 must have length {d**2} or {d**2 - 1} for dimension {d}, got {pmf.size}."
            )
        pmf = pmf / np.sum(pmf)  # normalize
        cum_probs = np.cumsum(pmf)
        r = np.random.rand(shots)
        indices = np.searchsorted(cum_probs, r)
        errors = np.zeros((shots, 2), dtype=np.int64)
        errors[:, 0] = indices // d  # x exponent
        errors[:, 1] = indices % d   # z exponent
        return errors
    
    def _sample_pauli_channel2_noise(self, shots: int, pmf: list[float]) -> np.ndarray:
        """
        Implements a two-qudit Pauli channel using a probability mass function (pmf)
        over the d⁴ generalized Pauli operators with the ordering:
            Let i₁ = (x₁ * d + z₁) and i₂ = (x₂ * d + z₂).
            Then index = i₁ * d² + i₂.
        The pmf can be provided as a full specification (length d⁴) or as probabilities
        for non-identity errors (length d⁴ - 1), in which case the identity probability is
        computed as 1 - sum(pmf).
        Returns an array of shape (shots, 4) where each row is (x₁, z₁, x₂, z₂).
        """
        d = self.dimension
        pmf = np.array(pmf, dtype=float)
        if pmf.size == d**4 - 1:
            identity_prob = 1 - np.sum(pmf)
            pmf = np.concatenate(([identity_prob], pmf))
        elif pmf.size != d**4:
            raise ValueError(
                f"PMF for PAULI_CHANNEL_2 must have length {d**4} or {d**4 - 1} for dimension {d}, got {pmf.size}."
            )
        pmf = pmf / np.sum(pmf)  # normalize
        cum_probs = np.cumsum(pmf)
        r = np.random.rand(shots)
        indices = np.searchsorted(cum_probs, r)
        errors = np.zeros((shots, 4), dtype=np.int64)
        # For the first qudit:
        i1 = indices // (d**2)
        errors[:, 0] = i1 // d      # x₁ exponent
        errors[:, 1] = i1 % d       # z₁ exponent
        # For the second qudit:
        i2 = indices % (d**2)
        errors[:, 2] = i2 // d      # x₂ exponent
        errors[:, 3] = i2 % d       # z₂ exponent
        return errors
    
    def _sample_heralded_erasure_noise(self, shots: int, error_prob: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Implements the heralded qudit Pauli channel for a probability p of erasure detection.
        With probability 1 - p, no error occurs (exponent 0) and the erasure index with value 0 is returned.
        With probability p, a nonzero exponent is chosen uniformly and the erasure index is set to 1.
        """
        d = self.dimension
        if not (0.0 <= error_prob <= 1.0):
            raise ValueError("Probability p must be in [0,1].")

        # (a, b) exponents; initialise as identity
        pauli   = np.zeros((shots, 2), dtype=np.int64)
        erased  = np.zeros(shots,        dtype=np.int8)

        rnd = np.random.rand(shots)

        # Which shots experience an erasure?
        erase_mask = rnd >= (1.0 - error_prob)
        n_erase    = np.sum(erase_mask)

        if n_erase:
            # Uniformly choose (a, b) in Z_d × Z_d
            pauli_erase = np.column_stack((
                np.random.randint(0, d, size=n_erase),
                np.random.randint(0, d, size=n_erase)
            ))
            pauli[erase_mask] = pauli_erase
            erased[erase_mask] = 1
        return pauli, erased



    def reference_sample(self, ir: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the reference sample for the circuit.

        Returns:
            np.ndarray: The reference sample for the circuit.
        """
        if ir is None:
            ir = self._build_ir()
        tableau = ExtendedTableauSimulator(self.num_qudits, self.dimension)
        measurements  = []
        gate_count = 0
        for inst in ir:
            gate_id = inst['gate_id']
            qudit_index = inst['qudit_index']
            target_index = inst['target_index']
            arg0 = inst['arg0']
            gate_name = gate_id_to_name(gate_id)

            if gate_name == "HERALDED_ERASURE":
                measurements.append(0)  # Assumed no noise in reference sample
            elif is_gate_collapsing(gate_id):
                gate_name = gate_id_to_name(gate_id)
                # Rotate to Z basis
                if gate_name in ("M_X", "MR_X"):
                    tableau.hadamard_inv(qudit_index)

                # Measure in Z basis
                measurement = tableau.measure(qudit_index)

                # Apply Reset Gate
                if gate_name in ("MR", "MR_X", "RESET"):
                    correction = (-measurement) % self.dimension
                    tableau.x(qudit_index, correction)
                    if gate_name == "MR_X":
                        tableau.hadamard(qudit_index)

                if is_gate_records(gate_id):
                    measurements.append(measurement)
            else:
                if qudit_index < 0:
                    gate_name = gate_id_to_name(gate_id)
                    if gate_name == "CNOT":
                        tableau.x(target_index, measurements[qudit_index])
                    elif gate_name == "CZ":
                        tableau.z(target_index, measurements[qudit_index])
                elif target_index < 0:
                    gate_name = gate_id_to_name(gate_id)
                    if gate_name == "CNOT":
                        raise ValueError("CNOT gate cannot be applied to measurement record target.")
                    elif gate_name == "CZ":
                        tableau.z(qudit_index, measurements[target_index])
                else:
                    tableau.apply_gate(gate_id, qudit_index, target_index, arg0)
            gate_count += 1
            if gate_count % 128 == 0:
                tableau.modulo()
        return np.array(measurements)

    def compile_sampler(self,
                        *,
                        skip_reference_sample: bool = False,
                        seed: Optional[int] = None,
                        reference_sample: Optional[np.ndarray] = None,
                        ir_array: Optional[np.ndarray] = None
                        ) -> "CompiledMeasurementSampler":
        """
        Compiles a measurement sampler that can be used to simulate the circuit.
        
        Returns:
            CompiledMeasurementSampler: A compiled measurement sampler for the circuit.
        """
        from .sampler import CompiledMeasurementSampler
        current_ir = ir_array if ir_array is not None else self._build_ir()
        
        current_ref_sample = reference_sample
        if not skip_reference_sample and current_ref_sample is None:
            if self.num_measurements > 0 :
                current_ref_sample = self.reference_sample(ir=current_ir)
            else:
                current_ref_sample = np.array([], dtype=np.int64)
        elif skip_reference_sample and current_ref_sample is None and self.num_measurements > 0:
             raise ValueError("Cannot skip reference sample if it's not provided and circuit has measurements.")
        elif self.num_measurements == 0 and current_ref_sample is None:
            current_ref_sample = np.array([], dtype=np.int64)


        return CompiledMeasurementSampler(
            circuit_object=self,
            reference_sample=current_ref_sample, # type: ignore
            ir_array=current_ir,
            seed=seed
        )

    def compile_detector_sampler(self,
                                 *,
                                 seed: Optional[int] = None,
                                 reference_sample: Optional[np.ndarray] = None,
                                 ir_array: Optional[np.ndarray] = None
                                ) -> "CompiledDetectorSampler":
        """
        Compiles a detector sampler that can be used to simulate the circuit.
        
        Returns:    
            CompiledDetectorSampler: A compiled detector sampler for the circuit.
        """
        from .sampler import CompiledDetectorSampler
        current_ir = ir_array if ir_array is not None else self._build_ir()
        
        current_ref_sample = reference_sample
        if current_ref_sample is None:
            if self.num_measurements > 0:
                current_ref_sample = self.reference_sample(ir=current_ir)
            else:
                current_ref_sample = np.array([], dtype=np.int64)
            
        return CompiledDetectorSampler(
            circuit_object=self,
            reference_sample=current_ref_sample,
            ir_array=current_ir,
            seed=seed
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
            if isinstance(op, tuple):  # If the operation is a tuple (gate, qudits)
                gate_name = op[0]
                qudits = op[1]
                if len(qudits) == 1:
                    circuit.append(gate_name, qudits[0])
                elif len(qudits) == 2:
                    circuit.append(gate_name, qudits[0], qudits[1])
                else:
                    raise ValueError(f"Unsupported number of qudits for gate {gate_name}")
            elif isinstance(op, CircuitInstruction):  # If the operation is a CircuitInstruction
                circuit.append(op.gate_name, op.qudit_index, op.target_index)
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return circuit