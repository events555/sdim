//! IR dispatch loop — runs a flat instruction array through the tableau simulator.
//!
//! Gate IDs match the quality branch registry ordering (sdim/gates/registry.py).

use crate::tableau::TableauSimulator;

// Gate ID constants (matching quality branch _GATE_NAME_TO_ID)
pub const GATE_I: i64 = 0;
pub const GATE_X: i64 = 1;
pub const GATE_X_INV: i64 = 2;
pub const GATE_Z: i64 = 3;
pub const GATE_Z_INV: i64 = 4;
pub const GATE_H: i64 = 5;
pub const GATE_H_INV: i64 = 6;
pub const GATE_P: i64 = 7;
pub const GATE_P_INV: i64 = 8;
pub const GATE_MULTIPLY: i64 = 9;
pub const GATE_MULTIPLY_INV: i64 = 10;
pub const GATE_CNOT: i64 = 11;
pub const GATE_CNOT_INV: i64 = 12;
pub const GATE_CZ: i64 = 13;
pub const GATE_CZ_INV: i64 = 14;
pub const GATE_SWAP: i64 = 15;
pub const GATE_M: i64 = 16;
pub const GATE_MR: i64 = 17;
pub const GATE_M_X: i64 = 18;
pub const GATE_MR_X: i64 = 19;
pub const GATE_RESET: i64 = 20;
// Noise gates 21..=26 are skipped
pub const GATE_HERALDED_ERASURE: i64 = 26;

/// Returns true if the gate is a noise gate (skipped during tableau simulation).
#[inline]
fn is_noise_gate(gate_id: i64) -> bool {
    (21..=26).contains(&gate_id)
}

/// Returns true if the gate is collapsing (measurement or reset).
#[inline]
pub fn is_collapsing(gate_id: i64) -> bool {
    matches!(gate_id, GATE_M | GATE_MR | GATE_M_X | GATE_MR_X | GATE_RESET)
}

/// Returns true if the gate records a measurement result.
#[inline]
pub fn is_recording(gate_id: i64) -> bool {
    matches!(gate_id, GATE_M | GATE_MR | GATE_M_X | GATE_MR_X | GATE_HERALDED_ERASURE)
}

/// An IR instruction: (gate_id, qudit_index, target_index, arg0).
#[derive(Clone, Copy, Debug)]
pub struct IrInstruction {
    pub gate_id: i64,
    pub qudit_index: i64,
    pub target_index: i64,
    pub arg0: i64,
}

/// Run an IR instruction sequence through the tableau simulator.
/// Returns the list of measurement outcomes.
///
/// This is the Rust equivalent of `reference_sample()` in
/// `sdim/compiler/lowering.py`.
pub fn run_ir(instructions: &[IrInstruction], n: usize, d: i64) -> Vec<i64> {
    let mut tab = TableauSimulator::new(n, d);
    let mut measurements: Vec<i64> = Vec::new();
    let mut gate_count: usize = 0;

    for inst in instructions {
        let gate_id = inst.gate_id;
        let qudit_idx = inst.qudit_index;
        let target_idx = inst.target_index;

        if is_noise_gate(gate_id) {
            gate_count += 1;
            continue;
        }

        if gate_id == GATE_HERALDED_ERASURE {
            measurements.push(0);
        } else if is_collapsing(gate_id) {
            let qi = qudit_idx as usize;

            // X-basis measurement: apply H^dagger first
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                tab.hadamard(qi, true);
            }

            let measurement = tab.measure(qi);

            // Reset: correct back to |0>
            if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                let correction = (-measurement).rem_euclid(d);
                if correction != 0 {
                    tab.pauli_x(qi, correction, false);
                }
                if gate_id == GATE_MR_X {
                    tab.hadamard(qi, false);
                }
            }

            if is_recording(gate_id) {
                measurements.push(measurement);
            }
        } else {
            // Non-collapsing gate: handle measurement record references
            if qudit_idx < 0 {
                // qudit_index is a measurement record reference
                let ti = target_idx as usize;
                let rec_idx = (measurements.len() as i64 + qudit_idx) as usize;
                let rec_val = measurements[rec_idx];
                if gate_id == GATE_CNOT {
                    tab.pauli_x(ti, rec_val, false);
                } else if gate_id == GATE_CZ {
                    tab.pauli_z(ti, rec_val, false);
                }
            } else if target_idx < 0 {
                let qi = qudit_idx as usize;
                let rec_idx = (measurements.len() as i64 + target_idx) as usize;
                let rec_val = measurements[rec_idx];
                if gate_id == GATE_CZ {
                    tab.pauli_z(qi, rec_val, false);
                }
            } else {
                tab.apply_gate(gate_id, qudit_idx as usize, target_idx as usize, inst.arg0);
            }
        }

        gate_count += 1;
        if gate_count % 128 == 0 {
            tab.modulo();
        }
    }

    measurements
}

/// Run IR and return both measurements and the final tableau snapshot.
/// Used for testing gate-by-gate equivalence with the Python implementation.
pub fn run_ir_with_snapshot(
    instructions: &[IrInstruction],
    n: usize,
    d: i64,
    stop_after: usize,
) -> (Vec<i64>, TableauSimulator) {
    let mut tab = TableauSimulator::new(n, d);
    let mut measurements: Vec<i64> = Vec::new();
    let mut gate_count: usize = 0;

    for (idx, inst) in instructions.iter().enumerate() {
        if idx >= stop_after {
            break;
        }

        let gate_id = inst.gate_id;
        let qudit_idx = inst.qudit_index;
        let target_idx = inst.target_index;

        if is_noise_gate(gate_id) {
            gate_count += 1;
            continue;
        }

        if gate_id == GATE_HERALDED_ERASURE {
            measurements.push(0);
        } else if is_collapsing(gate_id) {
            let qi = qudit_idx as usize;
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                tab.hadamard(qi, true);
            }
            let measurement = tab.measure(qi);
            if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                let correction = (-measurement).rem_euclid(d);
                if correction != 0 {
                    tab.pauli_x(qi, correction, false);
                }
                if gate_id == GATE_MR_X {
                    tab.hadamard(qi, false);
                }
            }
            if is_recording(gate_id) {
                measurements.push(measurement);
            }
        } else {
            if qudit_idx < 0 {
                let ti = target_idx as usize;
                let rec_idx = (measurements.len() as i64 + qudit_idx) as usize;
                let rec_val = measurements[rec_idx];
                if gate_id == GATE_CNOT {
                    tab.pauli_x(ti, rec_val, false);
                } else if gate_id == GATE_CZ {
                    tab.pauli_z(ti, rec_val, false);
                }
            } else if target_idx < 0 {
                let qi = qudit_idx as usize;
                let rec_idx = (measurements.len() as i64 + target_idx) as usize;
                let rec_val = measurements[rec_idx];
                if gate_id == GATE_CZ {
                    tab.pauli_z(qi, rec_val, false);
                }
            } else {
                tab.apply_gate(gate_id, qudit_idx as usize, target_idx as usize, inst.arg0);
            }
        }

        gate_count += 1;
        if gate_count % 128 == 0 {
            tab.modulo();
        }
    }

    tab.modulo();
    (measurements, tab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_ir_simple_measure() {
        // Measure two qubits in |00> state
        let ir = vec![
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 1, target_index: i64::MAX, arg0: -1 },
        ];
        let result = run_ir(&ir, 2, 2);
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_run_ir_x_then_measure() {
        let ir = vec![
            IrInstruction { gate_id: GATE_X, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
        ];
        let result = run_ir(&ir, 1, 2);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_run_ir_bell_state() {
        for _ in 0..50 {
            let ir = vec![
                IrInstruction { gate_id: GATE_H, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
                IrInstruction { gate_id: GATE_CNOT, qudit_index: 0, target_index: 1, arg0: -1 },
                IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
                IrInstruction { gate_id: GATE_M, qudit_index: 1, target_index: i64::MAX, arg0: -1 },
            ];
            let result = run_ir(&ir, 2, 2);
            assert_eq!(result[0], result[1], "Bell pair mismatch");
        }
    }

    #[test]
    fn test_run_ir_measure_reset() {
        let ir = vec![
            IrInstruction { gate_id: GATE_X, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_MR, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
        ];
        let result = run_ir(&ir, 1, 2);
        assert_eq!(result, vec![1, 0]); // MR records 1, resets to 0, M records 0
    }
}
