from typing import Tuple, Optional
from .tableau_simulator import apply_H, apply_P, MeasurementResult
from .tableau import Tableau
def apply_PauliX(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    if tableau.dimension % 2 == 0:
        apply_H(tableau, qudit_index, _)
        apply_P(tableau, qudit_index, _)
        apply_P(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
    else:
        apply_H(tableau, qudit_index, _)
        for _ in range(tableau.dimension-1):
            apply_P(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
        apply_P(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
    return tableau, None

def apply_PauliZ(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    if tableau.dimension % 2 == 0:
        apply_P(tableau, qudit_index, _)
        apply_P(tableau, qudit_index, _)
    else:
        for _ in range(tableau.dimension-1):
            apply_P(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
        apply_P(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
        apply_H(tableau, qudit_index, _)
    return tableau, None
