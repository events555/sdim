from .gatedata import GateData
from dataclasses import dataclass
from typing import List, Union, Optional, Iterable, overload

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
    gate_data: GateData
    gate_name: str
    qudit_index: int
    target_index: int = None
    gate_id: int = None
    name: str = None
    params: Optional[dict] = None

    def __post_init__(self):
        self.gate_id = self.gate_data.get_gate_id(self.gate_name)
        if self.gate_id is None:
            raise ValueError(f"Gate {self.gate_name} not found")
        self.name = self.gate_data.get_gate_name(self.gate_id)

    def __str__(self):
        return f"{self.gate_id} {self.qudit_index} {self.target_index}"


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
        gate_data (GateData): Contains information about available gates.

    Raises:
        ValueError: If num_qudits is less than 1 or dimension is less than 2.
    """
    num_qudits: int
    dimension: int = 2
    operations: List[CircuitInstruction] = None
    gate_data: GateData = None

    def __post_init__(self):
        """
        Performs post-initialization checks and setups.
        """
        if self.num_qudits < 1:
            raise ValueError("Number of qudits must be greater than 0")
        if self.dimension < 2:
            raise ValueError("Dimension must be greater than 1")
        self.operations = self.operations or []
        self.gate_data = self.gate_data or GateData(self.dimension)
    
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
        arg: Optional[Union[float, Iterable[float]]] = None,
        *,
        tag: str = "",
        **kwargs
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
        # Case 3: Directly appending an existing instruction.
        if isinstance(name_or_op, CircuitInstruction):
            self.operations.append(name_or_op)
            return self

        gate_name = name_or_op.upper()
        # Start building the parameters dict.
        params = {}
        if arg is not None:
            params["arg"] = arg
        if tag:
            params["tag"] = tag
        # Merge any additional keyword arguments.
        params.update(kwargs)

        # Case 1: Single-qudit or broadcast operation.
        if target is None:
            if isinstance(targets, int):
                targets = [targets]
            else:
                targets = list(targets) if targets is not None else []
            for t in targets:
                instr = CircuitInstruction(self.gate_data, gate_name, t, target_index=None, params=params)
                self.operations.append(instr)
            return self

        # Case 2: Paired control and target operation.
        if isinstance(targets, int):
            controls = [targets]
        else:
            controls = list(targets)
        if isinstance(target, int):
            targets_list = [target]
        else:
            targets_list = list(target)
        
        if len(controls) == 1:
            qubit_pairs = [(controls[0], t) for t in targets_list]
        elif len(targets_list) == 1:
            qubit_pairs = [(c, targets_list[0]) for c in controls]
        elif len(controls) == len(targets_list):
            qubit_pairs = list(zip(controls, targets_list))
        else:
            raise ValueError("Invalid combination: control and target must be of equal length or one must be singular.")

        for c, t in qubit_pairs:
            instr = CircuitInstruction(self.gate_data, gate_name, c, target_index=t, params=params)
            self.operations.append(instr)
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
    
    def print_gateData(self):
        """
        Prints the gate data associated with this circuit.
        """
        print(self.gate_data)
    
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