from dataclasses import dataclass, field

@dataclass(frozen=True)
class GateTarget:
    _value: int
    is_inverted: bool = field(default=False, compare=False)
    is_pauli: bool = field(default=False, compare=False)
    is_x_target: bool = field(default=False, compare=False)
    is_y_target: bool = field(default=False, compare=False)
    is_z_target: bool = field(default=False, compare=False)
    is_sweep_bit: bool = field(default=False, compare=False)
    is_combiner: bool = field(default=False, compare=False)

    @staticmethod
    def qudit(index: int, *, invert: bool = False) -> "GateTarget":
        return GateTarget(index, is_inverted=invert)

    @staticmethod
    def x(index: int, *, invert: bool = False) -> "GateTarget":
      return GateTarget(index, is_inverted=invert, is_pauli=True, is_x_target = True)

    @staticmethod
    def y(index: int, *, invert: bool = False) -> "GateTarget":
      return GateTarget(index, is_inverted=invert, is_pauli=True, is_y_target = True)

    @staticmethod
    def z(index: int, *, invert: bool = False) -> "GateTarget":
        return GateTarget(index, is_inverted=invert, is_pauli=True, is_z_target = True)

    @staticmethod
    def rec(lookback: int) -> "GateTarget":
        if lookback >= 0:
            raise ValueError("Lookback index for measurement record must be negative.")
        return GateTarget(lookback)

    @staticmethod
    def sweep_bit(index: int) -> "GateTarget":
      return GateTarget(index, is_sweep_bit = True)
    
    @staticmethod
    def combiner() -> "GateTarget":
        return GateTarget(-1, is_combiner=True)

    @property
    def value(self) -> int:
        return self._value
    
    @property
    def is_qudit_target(self) -> bool:
        return not (self.is_measurement_record_target or self.is_pauli or self.is_sweep_bit or self.is_combiner)

    @property
    def is_measurement_record_target(self) -> bool:
        return self._value < 0 and not self.is_combiner

    
def _pauli_gates():
    return {
        "I": {"arg_count": 1, "aliases": ["I"]},
        "X": {"arg_count": 1, "aliases": ["X", "NOT"], "inverse": "X_INV"},
        "X_INV": {"arg_count": 1, "aliases": ["X_INV"], "inverse": "X"},
        "Z": {"arg_count": 1, "aliases": ["Z"], "inverse": "Z_INV"},
        "Z_INV": {"arg_count": 1, "aliases": ["Z_INV"], "inverse": "Z"},
    }
def _hadamard_gates():
    return {
        "H": {"arg_count": 1, "aliases": ["H", "R", "DFT", "F"], "inverse": "H_INV"},
        "H_INV": {"arg_count": 1, "aliases": ["H_INV", "R_INV", "DFT_INV", "F_INV", "H_DAG", "R_DAG", "DFT_DAG", "F_DAG"], "inverse": "H"},
        "P": {"arg_count": 1, "aliases": ["PHASE", "S"], "inverse": "P_INV"},
        "P_INV": {"arg_count": 1, "aliases": ["PHASE_INV", "S_INV", "S_DAG"], "inverse": "P"},
    }
def _controlled_gates():
    return {
        "CNOT": {"arg_count": 2, "aliases": ["CNOT", "CX", "C", "SUM"], "inverse": "CNOT_INV"},
        "CNOT_INV": {"arg_count": 2, "aliases": ["CNOT_INV", "CX_INV", "C_INV", "CNOT_DAG", "CX_DAG", "C_DAG", "SUM_INV", "SUM_DAG"], "inverse": "CNOT"},
        "CZ": {"arg_count": 2, "aliases": ["CZ"], "inverse": "CZ_INV"},
        "CZ_INV": {"arg_count": 2, "aliases": ["CZ_INV"], "inverse": "CZ"},
        "SWAP": {"arg_count": 2, "aliases": ["SWAP"], "inverse": "SWAP"},
        # "ISWAP": {"arg_count": 2, "aliases": ["ISWAP"]},
        # "ISWAP_INV": {"arg_count": 2, "aliases": ["ISWAP_DAG", "ISWAP_INV"]},
    }
def _collapsing_gates():
    return {
        "M": {"arg_count": 1, "aliases": ["M", "MEASURE", "COLLAPSE", "MZ"], "inverse": None},
        "MR": {"arg_count": 1, "aliases": ["MR", "MEASURE_R"], "inverse": None},
        "M_X": {"arg_count": 1, "aliases": ["M_X", "MEASURE_X", "MX"], "inverse": None},
        "MR_X": {"arg_count": 1, "aliases": ["MR_X", "MEASURE_R_X", "MRX"], "inverse": None},
        "RESET": {"arg_count": 1, "aliases": ["RESET", "R"]},
        # Add other collapsing gates here (MRX, MRY, etc.)
    }
def _noise_gates():
  return {
        "X_ERROR": {"arg_count": 1, "aliases": ["X_ERROR"], "inverse": None},
        "Z_ERROR": {"arg_count": 1, "aliases": ["Z_ERROR"], "inverse": None},
        "Y_ERROR": {"arg_count": 1, "aliases": ["Y_ERROR"], "inverse": None},
        "DEPOLARIZE1": {"arg_count": 1, "aliases": ["DEPOLARIZE1", "DEPOLARIZE"], "inverse": None},
        "DEPOLARIZE2": {"arg_count": 2, "aliases": ["DEPOLARIZE2"], "inverse": None},
  }
def _annotation_gates():
  return {
          "REPEAT": {"arg_count": None, "aliases":[], "inverse": None},
          "DETECTOR": {"arg_count": None, "aliases":[], "inverse": None},
          "SHIFT_COORDS": {"arg_count": None, "aliases":[], "inverse": None},
          "OBSERVABLE_INCLUDE": {"arg_count": None, "aliases":[], "inverse": None},
        }

GATE_DATA = {}
GATE_DATA.update(_pauli_gates())
GATE_DATA.update(_hadamard_gates())
GATE_DATA.update(_controlled_gates())
GATE_DATA.update(_collapsing_gates())
GATE_DATA.update(_noise_gates())
GATE_DATA.update(_annotation_gates())


_GATE_NAME_TO_ID = {name: i for i, name in enumerate(GATE_DATA)}

def gate_name_to_id(gate_name: str) -> str:
    """
    Looks up the canonical gate name, then returns the id.
    """
    gate_name = gate_name.upper()
    for canonical_name, data in GATE_DATA.items():
        if gate_name == canonical_name or gate_name in data["aliases"]:
            return _GATE_NAME_TO_ID[canonical_name] 
    raise ValueError(f"Gate name '{gate_name}' doesn't exist.")

def gate_id_to_name(gate_id: int) -> str:
  """
  Returns the canonical gate name given its ID.
  """
  for name, id in _GATE_NAME_TO_ID.items():
        if id == gate_id:
            return name
  raise ValueError(f"Gate id '{gate_id}' doesn't exist.")


def is_not_a_gate(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["REPEAT", "DETECTOR", "SHIFT_COORDS", "OBSERVABLE_INCLUDE"]

def is_gate_collapsing_and_records(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["M", "M_X", "MR", "MR_X"]

def is_gate_collapsing(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["M", "M_X", "MR", "MR_X", "RESET"]

def is_gate_noisy(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["X_ERROR", "Z_ERROR","DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2","M", "M_X", "MPP"]

def is_gate_two_qubit(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["CNOT", "CZ", "SWAP", "CNOT_INV", "CX_INV", "CZ_INV","ISWAP", "ISWAP_DAG", "XCZ", "XCY", "XCX", "YCZ", "YCY", "YCX", "DEPOLARIZE2","PAULI_CHANNEL_2"]

def is_gate_pauli(gate_id: int):
    name = gate_id_to_name(gate_id)
    return name in ["X", "Z", "X_INV", "Z_INV"]