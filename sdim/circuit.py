from .gatedata import *
from .sampler import CompiledMeasurementSampler
from dataclasses import dataclass, field
from typing import List, Union, Optional, Iterable, overload
import numpy as np

@dataclass
class CircuitInstruction:
    """
    Represents a single instruction in a quantum circuit.

    This class encapsulates the details of a quantum gate operation,
    including the gate type, target qudit(s), and associated metadata.

    Attributes:
        gate_data (GateData): Contains information about available gates.
        gate_name (str): The name of the gate.
        qudit_index (int): The index of the primary qudit the gate acts on.
        target_index (int, optional): The index of the target qudit for two-qudit gates.
        gate_id (int): The unique identifier for the gate.
        name (str): The canonical name of the gate.
        params (dict, optional): Additional parameters for specific gates (i.e. noise gates)

    Raises:
        ValueError: If the specified gate is not found in gate_data.
    """
    name: str
    targets: List[GateTarget] = field(default_factory=list)
    args: List[float] = field(default_factory=list)

    def __str__(self):
        result = self.name
        if self.args:
            result += "(" + ",".join(str(arg) for arg in self.args) + ")"
        # Format for GateTargets
        result += " " + " ".join(str(t) for t in self.targets)
        return result
    
    def validate(self):
        """
        Validates the instruction by checking if the gate name is valid.

        Raises:
            ValueError: If the gate name is not found in the gate data.
        """
        if self.gate_id == -1:
            raise ValueError(f"Invalid gate name: {self.name}")


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
    def append(self, name: str, control: Union[int, Iterable[int]], 
               target: Union[int, Iterable[int]], 
               arg: Optional[Union[float, Iterable[float]]] = None, *, tag: str = "") -> "Circuit":
        ...

    @overload
    def append(self, op: CircuitInstruction) -> "Circuit":
        ...
    
    def append(
        self,
        name_or_op: Union[str, CircuitInstruction],
        targets: Optional[Union[int, Iterable[int]]] = None,
        target: Optional[Union[int, Iterable[int]]] = None,
        args: Optional[Union[float, Iterable[float]]] = None
    ) -> "Circuit":
        """
        Appends an operation to the circuit.

        Overloads:
        1. Append with a gate name and targets:
            - `append("X", 0)`
            - `append("X", [0, 1])`
        2. Append with a gate name, control(s), and target(s) for two-qudit operations:
            - `append("CNOT", 0, 1)`
            - `append("CNOT", [0, 2], [1, 3])`
        3. Append a pre-constructed CircuitInstruction.
        
        Args:
            name_or_op: Either the gate name (str) or a CircuitInstruction.
            targets: For single-qudit operations, the target(s). For two-qudit operations,
                    this represents the control(s).
            target: For two-qudit operations, the target(s). Leave as None for single-qudit operations.
            arg: Optional parameter(s) for the gate.
            tag: An optional string tag.
            **kwargs: Additional keyword arguments (e.g., prob, noise_channel) to include in the parameters.
        
        Returns:
            The Circuit object (self).
        """
        if isinstance(name_or_op, CircuitInstruction):
            self.operations.append(name_or_op)
            return self

        gate_name = name_or_op.upper()
        args_list = []
        if args is not None:
            if isinstance(args, float):
                args_list = [args]
            else:
                args_list = list(args)
        # Single Qubit
        if target is None:
            if isinstance(targets, int):
                targets = [GateTarget.qudit(targets)]
            elif isinstance(targets, GateTarget):
                targets = [targets]
            else:
                targets = [GateTarget.qudit(t) if isinstance(t, int) else t
                       for t in targets]

            instr = CircuitInstruction(name=gate_name,
                                        targets=targets,
                                        args=args_list)
            self.operations.append(instr)
        else:
        # Two Qubit
          # Process controls
          if isinstance(targets, int):
              controls = [GateTarget.qudit(targets)]
          elif isinstance(targets, GateTarget):
              controls = [targets]
          else:
              controls = [GateTarget.qudit(c) if isinstance(c, int) else c
                          for c in targets]

          # Process targets
          if isinstance(target, int):
              targets_list = [GateTarget.qudit(target)]
          elif isinstance(target, GateTarget):
              targets_list = [target]
          else:
              targets_list = [GateTarget.qudit(t) if isinstance(t, int) else t
                          for t in target]

          if len(controls) == 1:
              all_targets = []
              for t in targets_list:
                  all_targets += [controls[0], t]
              instr = CircuitInstruction(name=gate_name,
                                      targets=all_targets,
                                      args=args_list)
              self.operations.append(instr)
          elif len(targets_list) == 1:
              all_targets = []
              for c in controls:
                  all_targets += [c, targets_list[0]]
              instr = CircuitInstruction(name=gate_name,
                                    targets=all_targets,
                                    args=args_list)
              self.operations.append(instr)
          elif len(controls) == len(targets_list):
              all_targets = []
              for c,t in zip(controls, targets_list):
                  all_targets += [c, t]
              instr = CircuitInstruction(name=gate_name,
                                        targets=all_targets,
                                        args=args_list)
              self.operations.append(instr)
          else:
                raise ValueError("Invalid combination: control and target must be of equal length or one must be singular.")
        return self


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
        
        # Copy operations from self
        new_circuit.operations = list(self.operations)
        
        # Append operations from other
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
    
    def _build_ir(self, extra_shots: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds an intermediate representation (IR) for the given circuits and also precomputes
        an array of sampled Pauli noise outcomes for noise gates (if applicable)
        
        Args:
            circuits (list[Circuit]): A list of Circuit objects.
            extra_shots (int): The number of extra shots for which noise outcomes
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
        noise_list = []
        dimension = self.stabilizer_tableau.dimension
        for instruction in self.operations:
            if instruction.gate_id == 0:
                continue
            target_index = instruction.target_index if instruction.target_index is not None else -1
            ir_list.append((gate_name_to_id(instruction.name), instruction.qudit_index, target_index))
                            
            if is_gate_noisy(instruction.gate_id):
                # Always add a noise sample, but only actually sample non-identity with some probability.
                channel = instruction.params['noise_channel']
                if channel == 'd':
                    # Sample integer r from 1 to dimension**2 - 1 for each extra shot.
                    r = np.random.randint(1, dimension**2, size=extra_shots)
                    a = r % dimension
                    b = r // dimension
                elif channel == 'f':
                    a = np.random.randint(1, dimension, size=extra_shots)
                    b = np.zeros(extra_shots, dtype=np.int64)
                elif channel == 'p':
                    a = np.zeros(extra_shots, dtype=np.int64)
                    b = np.random.randint(1, dimension, size=extra_shots)

                # Roll to see if the channel applies on this each shot
                shot_dice_rolls = np.random.uniform(0.0, 1.0, size=extra_shots)
                # Mask that checks for failure to clear threshold, aka applying I = X^0 Z^0
                probability = float(instruction.params['prob'])
                mask = shot_dice_rolls < 1.0 - probability

                # Apply mask to both Pauli exponents
                a[mask] = 0
                b[mask] = 0

                pair = np.stack((a, b), axis=1)
                noise_list.append(pair)

        ir_dtype = np.dtype([
            ('gate_id', np.int64),
            ('qudit_index', np.int64),
            ('target_index', np.int64)
        ])

        ir_array = np.array(ir_list, dtype=ir_dtype)

        if noise_list:
            noise_array = np.array(noise_list, dtype=np.int64)
        else:
            noise_array = np.empty((1, extra_shots, 2), dtype=np.int64)


        return ir_array, noise_array

    def reference_sample(self) -> np.ndarray:
        """
        Returns the reference sample for the circuit.

        Returns:
            np.ndarray: The reference sample for the circuit.
        """

    def compile_sampler(self,
                        *,
                        skip_reference_sample: bool = False,
                        seed: Optional[int] = None,
                        reference_sample: Optional[np.ndarray] = None
                        ) -> "CompiledMeasurementSampler":
        """
        Compiles a measurement sampler that can be used to simulate the circuit.
        
        Returns:
            CompiledMeasurementSampler: A compiled measurement sampler for the circuit.
        """
        return CompiledMeasurementSampler(self, skip_reference_sample=skip_reference_sample, seed=seed, reference_sample=reference_sample)

    def compile_detector_sampler(self) -> "CompiledDetectorSampler":
        """
        Compiles a detector sampler that can be used to simulate the circuit.
        
        Returns:
            CompiledDetectorSampler: A compiled detector sampler for the circuit.
        """
        ...

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