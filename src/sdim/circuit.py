from .gatedata import GateData
from dataclasses import dataclass
from typing import Union, List

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

    Raises:
        ValueError: If the specified gate is not found in gate_data.
    """
    gate_data: GateData
    gate_name: str
    qudit_index: int
    target_index: int = None
    gate_id: int = None
    name: str = None

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
    operations: list = None
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
    
    def add_gate(self, gate_name: str, control: Union[int, List[int]], target: Union[int, List[int], None] = None):
        """
        Adds gate operation(s) to the circuit.

        Args:
            gate_name (str): The name of the gate to add.
            control (int or List[int]): The index or indices of the control qudit(s).
            target (int, List[int], or None, optional): The index or indices of the target qudit(s).

        Returns:
            Circuit: The current Circuit object with the added operation(s).

        Raises:
            ValueError: If the input combination is invalid.
        """
        # Convert single integers to lists for uniform processing
        control = [control] if isinstance(control, int) else control
        target = [target] if isinstance(target, int) else target

        # Handle single-qubit gates
        if target is None:
            for c in control:
                self.operations.append(CircuitInstruction(self.gate_data, gate_name.upper(), c, None))
            return self

        # Generate all combinations of control and target qubits
        qubit_pairs = []
        if len(control) == 1:
            qubit_pairs = [(control[0], t) for t in target]
        elif len(target) == 1:
            qubit_pairs = [(c, target[0]) for c in control]
        elif len(control) == len(target):
            qubit_pairs = list(zip(control, target))
        else:
            raise ValueError("Invalid combination of control and target qubits")

        # Add instructions for all qubit pairs
        for c, t in qubit_pairs:
            self.operations.append(CircuitInstruction(self.gate_data, gate_name.upper(), c, t))

        return self

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
                    circuit.add_gate(gate_name, qudits[0])
                elif len(qudits) == 2:
                    circuit.add_gate(gate_name, qudits[0], qudits[1])
                else:
                    raise ValueError(f"Unsupported number of qudits for gate {gate_name}")
            elif isinstance(op, CircuitInstruction):  # If the operation is a CircuitInstruction
                circuit.add_gate(op.gate_name, op.qudit_index, op.target_index)
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return circuit