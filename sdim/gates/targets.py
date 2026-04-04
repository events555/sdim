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
        return GateTarget(
            index, is_inverted=invert, is_pauli=True, is_x_target=True
        )

    @staticmethod
    def y(index: int, *, invert: bool = False) -> "GateTarget":
        return GateTarget(
            index, is_inverted=invert, is_pauli=True, is_y_target=True
        )

    @staticmethod
    def z(index: int, *, invert: bool = False) -> "GateTarget":
        return GateTarget(
            index, is_inverted=invert, is_pauli=True, is_z_target=True
        )

    @staticmethod
    def rec(lookback: int) -> "GateTarget":
        if lookback >= 0:
            raise ValueError(
                "Lookback index for measurement record must be negative."
            )
        return GateTarget(lookback)

    @staticmethod
    def sweep_bit(index: int) -> "GateTarget":
        return GateTarget(index, is_sweep_bit=True)

    @staticmethod
    def combiner() -> "GateTarget":
        return GateTarget(-1, is_combiner=True)

    @property
    def value(self) -> int:
        return self._value

    @property
    def is_qudit_target(self) -> bool:
        return not (
            self.is_measurement_record_target
            or self.is_pauli
            or self.is_sweep_bit
            or self.is_combiner
        )

    @property
    def is_measurement_record_target(self) -> bool:
        return self._value < 0 and not self.is_combiner
