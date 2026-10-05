from .gatedata import GateData
from dataclasses import dataclass
from typing import Union, Optional, List
import math
import numpy as np
import operator

# Keyword parameters accepted by gates that take any.  Catches typos such as probability=0.1,
# which would otherwise be stored silently while the default prob is used.
_GATE_PARAMS = {
    "N1": {"noise_channel", "prob"},
    "N2": {"prob", "prob_dist"},
    "MUL": {"a", "scalar"},
}

# A single qudit index.  NumPy integers come from indexing arrays and from random generators.
_INTEGER_TYPES = (int, np.integer)

@dataclass
class CircuitInstruction:
    """
    Represents a single instruction in a quantum circuit.

    This class encapsulates the details of a quantum gate operation,
    including the gate type, target qudit(s), and associated metadata.

    Attributes:
        gate_data (GateData): Contains information about available gates.
        gate_name (str): The name of the gate.
        qudit_index (int, optional): The index of the primary qudit the gate acts on.
        target_index (int, optional): The index of the target qudit for two-qudit gates.
        gate_id (int): The unique identifier for the gate.
        name (str): The canonical name of the gate.
        params (dict, optional): Additional parameters for specific gates (e.g. noise gates)

    Raises:
        ValueError: If the specified gate is not found in gate_data.
    """
    gate_data: GateData
    gate_name: str
    qudit_index: int = None
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
        ValueError: If num_qudits is less than 1, or dimension is less than 2 or at least 2**31.
        TypeError: If num_qudits or dimension is not an integer (floats and bools are not).
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
        if self.dimension >= 2 ** 31:
            # Products of two values mod d must fit in int64 throughout the simulators.
            raise ValueError("Dimension must be less than 2**31")
        # Store Python ints: a NumPy dimension breaks pow(a, -1, d) and overflows in products.
        if type(self.num_qudits) is not int or type(self.dimension) is not int:
            # Floats have no __index__; bools have one, but are not sizes either.
            for name in ("num_qudits", "dimension"):
                value = getattr(self, name)
                if isinstance(value, (bool, np.bool_)) or not hasattr(value, "__index__"):
                    raise TypeError(f"{name} must be an integer, not {value!r}")
                setattr(self, name, operator.index(value))
        self.operations = self.operations or []
        self.gate_data = self.gate_data or GateData(self.dimension)
    
    def add_gate(self, gate_name: str, control: Union[int, List[int], None] = None, target: Union[int, List[int], None] = None, **kwargs):
        """
        Adds gate operation(s) to the circuit.

        A one-qudit gate takes only control, and a list there adds the gate to each qudit.  A
        two-qudit gate (CNOT, CNOT_INV, CZ, CZ_INV, SWAP, N2) takes both control and target.  When
        one of them is a list, the other qudit is paired with each entry; two lists of the same
        length are paired up in order.  DETECTOR, LOGICAL_OBSERVABLE and TICK take no qudits.

        Args:
            gate_name (str): The name of the gate to add.
            control (int or List[int], or None, optional): The index or indices of the control qudit(s), or of the qudit(s) of a one-qudit gate.  Only DETECTOR, LOGICAL_OBSERVABLE and TICK are given without one.
            target (int, List[int], or None, optional): The index or indices of the target qudit(s).  Only for two-qudit gates.

        Optional parameters:
            noise_channel (str): Channel type for N1.  Valid channels are "f", "p", and "d" for flip errors, phase errors, and depolarizing noise, respectively.  Defaults to "d".  The older key `channel` is still accepted.
            prob (float): Error probability for N1, and for N2 when no prob_dist is given.  For N2 it is two-qudit depolarizing: with probability prob, a uniformly random non-identity two-qudit Pauli is applied.  Defaults to 0.01.
            prob_dist (List[float]): Probability distribution for a general two-qudit Pauli channel on N2, used instead of prob.  An n-qudit Pauli channel applies powers of Pauli X and Pauli Z to n distinct qudits, which can be written as a tuple of powers (x_1, z_1,   x_2, z_2,   ...,  x_n, z_n).  The j-th entry in the distribution is the probability that the channel applies an n-qudit Pauli corresponding to the j-th tuple in lexicographic order of n-qudit tuples of Pauli powers.  It has d**4 entries, so it is only practical for small d, and sdim.dem.DetectorErrorModel does not accept it (use sdim.dem_legacy).
            a (int): The scalar of MUL, which maps |j> to |a j mod d>.  Required.  It must be an integer coprime to the dimension d; only a mod d matters, so negative values and values of at least d are fine.  scalar is another name for it.

        Returns:
            Circuit: The current Circuit object with the added operation(s).

        Raises:
            ValueError: If the input combination is invalid: a two-qudit gate without both a control
                and a target, a one-qudit gate without a qudit or with a target, the same qudit
                twice, an unknown parameter, or an invalid parameter value (such as a MUL scalar
                that is not an integer coprime to d).
        """
        # Convert single integers to lists for uniform processing
        control = [control] if isinstance(control, _INTEGER_TYPES) else control
        target = [target] if isinstance(target, _INTEGER_TYPES) else target

        gate_name_upper = gate_name.upper()
        primary_name = self.gate_data.aliasMap.get(gate_name_upper, gate_name_upper)
        gate = self.gate_data.gateMap.get(primary_name)

        # A missing qudit used to be stored as None, which the frame sampler read as the last
        # qudit and the tableau could not apply.  A target given to a one-qudit gate was silently
        # ignored.
        if gate and gate.arg_count == 1:
            if target is not None:
                raise ValueError(f"{primary_name} acts on one qudit and takes no target; "
                                 "to apply it to several qudits, give them as a list.")
            if control is None:
                raise ValueError(f"{primary_name} acts on one qudit: give its qudit, or a list of qudits.")
        elif gate and gate.arg_count == 2 and (control is None or target is None):
            raise ValueError(f"{primary_name} acts on two qudits: give both a control and a target qudit.")

        # Accept the older N1 parameter name.  This has to happen before the defaults are filled in,
        # otherwise the default noise_channel would shadow it.
        if primary_name == "N1" and "channel" in kwargs and "noise_channel" not in kwargs:
            kwargs["noise_channel"] = kwargs.pop("channel")

        if gate and gate.defaults:
            for key, value in gate.defaults.items():
                kwargs.setdefault(key, value)

        allowed = _GATE_PARAMS.get(primary_name)
        if allowed is not None and set(kwargs) - allowed:
            raise ValueError(f"{primary_name} does not take {sorted(set(kwargs) - allowed)}; "
                             f"its parameters are {sorted(allowed)}.")

        if primary_name == "N1":
            if kwargs["noise_channel"] not in ("d", "f", "p"):
                raise ValueError(f"N1 noise_channel must be 'd', 'f' or 'p', not {kwargs['noise_channel']!r}.")
            if not 0.0 <= float(kwargs["prob"]) <= 1.0:
                raise ValueError(f"N1 prob must be between 0 and 1, not {kwargs['prob']}.")
        elif primary_name == "N2" and kwargs.get("prob_dist") is None:
            if not 0.0 <= float(kwargs["prob"]) <= 1.0:
                raise ValueError(f"N2 prob must be between 0 and 1, not {kwargs['prob']}.")
        elif primary_name == "N2":
            # Same checks (and messages) as the simulators.  A (d, d, d, d) array is accepted and
            # stored flat, in the (x1, z1, x2, z2) lexicographic order the simulators use.
            from .program import _prob_dist_cdf
            dist = np.asarray(kwargs["prob_dist"], dtype=float)
            if dist.ndim != 1:
                # Flat inputs are stored as given, so gates sharing one distribution keep sharing it.
                kwargs["prob_dist"] = dist = dist.reshape(-1)
            _prob_dist_cdf(dist, self.dimension)
        elif primary_name == "MUL":
            # The simulators read the scalar as int(a) % d and need it to be invertible mod d.
            scalar = kwargs.get("a", kwargs.get("scalar"))
            if scalar is None:
                raise ValueError("MUL needs its scalar, as a= (or scalar=).")
            try:
                value = int(scalar)
            except (TypeError, ValueError, OverflowError):
                value = None
            # int() would truncate 2.5 to 2.  Text such as "2" is fine, the simulators call int() too.
            if value is None or (not isinstance(scalar, str) and value != scalar):
                raise ValueError(f"MUL scalar must be an integer, not {scalar!r}.")
            if math.gcd(value % self.dimension, self.dimension) != 1:
                raise ValueError(f"MUL scalar {scalar} is not coprime with the dimension {self.dimension}.")

        if control is None and target is None: # Detectors only
            self.operations.append(CircuitInstruction(self.gate_data, gate_name_upper, None, None, params=kwargs))
            return
        elif target is None:
            for c in control:
                self.operations.append(CircuitInstruction(self.gate_data, gate_name_upper, c, None, params=kwargs))
            return

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

        if any(c == t for c, t in qubit_pairs):
            raise ValueError(f"{primary_name} needs two different qudits.")

        # Add instructions for all qubit pairs
        for c, t in qubit_pairs:
            self.operations.append(CircuitInstruction(self.gate_data, gate_name_upper, c, t, params=kwargs))
        return

    def __mul__(self, repetitions:int):
        """
        Replicates the circuit by the specified number of times.

        Args:
            repetitions (int): The number of times to replicate the circuit.

        Returns:
            Circuit: A new Circuit object with the replicated operations and this circuit's gate data
                (no operations for repetitions <= 0, as for lists).  This circuit is unchanged.
        """
        return Circuit(self.num_qudits, self.dimension, self.operations * repetitions, self.gate_data)
    
    __rmul__ = __mul__

    def __imul__(self, repetitions:int):
        """
        Replicates the circuit in place.

        Args:
            repetitions (int): The number of times to replicate the circuit.

        Returns:
            Circuit: This Circuit object, with its operations replicated.
        """
        self.operations *= repetitions
        return self
    
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