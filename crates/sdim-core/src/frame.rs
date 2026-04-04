//! Pauli frame simulator for multi-shot sampling.
//!
//! Tracks x_frame and z_frame arrays of shape (num_qudits, shots) and
//! applies Clifford gate conjugation rules. Measurements combine the
//! reference sample with the frame's x error.

use crate::ir::*;
use rand::Rng;

/// Run the Pauli frame simulation over an IR array.
///
/// # Arguments
/// * `ir` - Flat IR instructions (gate_id, qudit_index, target_index, arg0)
/// * `reference_sample` - Noiseless measurement outcomes from the tableau sim
/// * `num_qudits` - Number of qudits
/// * `dimension` - Local qudit dimension
/// * `shots` - Number of shots to simulate
/// * `noise1_bank` - Pre-sampled 1-qudit noise: shape (num_noise1, shots, 2) where [.., 0]=x, [.., 1]=z
/// * `noise2_bank` - Pre-sampled 2-qudit noise: shape (num_noise2, shots, 4) where [.., 0..4]=(x1,z1,x2,z2)
/// * `erased_bank` - Pre-sampled erasure outcomes: shape (num_erasures, shots)
/// * `measurement_bank` - Pre-sampled measurement noise: shape (num_meas_noise, shots, 1)
///
/// # Returns
/// Flat measurement results array of shape (num_measurements, shots), row-major.
pub fn run_frame(
    ir: &[IrInstruction],
    reference_sample: &[i64],
    num_qudits: usize,
    dimension: i64,
    shots: usize,
    noise1_bank: &[i64],   // flattened (n1, shots, 2)
    noise2_bank: &[i64],   // flattened (n2, shots, 4)
    erased_bank: &[i64],   // flattened (n_erasure, shots)
    measurement_bank: &[i64], // flattened (n_meas_noise, shots, 1)
) -> Vec<i64> {
    let d = dimension;

    // Allocate frames: x_frame[qudit][shot], z_frame[qudit][shot]
    let mut x_frame = vec![vec![0i64; shots]; num_qudits];
    let mut z_frame: Vec<Vec<i64>> = (0..num_qudits)
        .map(|_| {
            let mut rng = rand::thread_rng();
            (0..shots).map(|_| rng.gen_range(0..d)).collect()
        })
        .collect();

    // Count total measurements to pre-allocate results
    let num_measurements = reference_sample.len();
    let mut frame_results = vec![vec![0i64; shots]; num_measurements];

    let mut noise1_counter: usize = 0;
    let mut noise2_counter: usize = 0;
    let mut erasure_counter: usize = 0;
    let mut measurement_counter: usize = 0;
    let mut measurement_noise_counter: usize = 0;
    let mut gate_counter: usize = 0;

    for inst in ir {
        let gate_id = inst.gate_id;
        let qi = inst.qudit_index as usize;
        let ti = inst.target_index as usize;
        let arg0 = inst.arg0;

        if gate_counter % 128 == 0 {
            for q in 0..num_qudits {
                for s in 0..shots {
                    x_frame[q][s] = x_frame[q][s].rem_euclid(d);
                    z_frame[q][s] = z_frame[q][s].rem_euclid(d);
                }
            }
        }

        let collapsing = is_collapsing(gate_id);
        let recording = is_recording(gate_id);
        let is_pure_noise = (21..=26).contains(&gate_id); // X_ERROR..HERALDED_ERASURE
        // Measurement gates are "noisy" in the registry, but only use noise banks
        // when measurement_bank is non-empty.
        let has_meas_noise = !measurement_bank.is_empty();
        let noisy_collapsing = collapsing && has_meas_noise;

        if noisy_collapsing {
            // Noisy collapsing gate (M, MR, M_X, MR_X with noise)
            let ref_outcome = reference_sample[measurement_counter];

            // X-basis: apply H_INV first
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                frame_h_inv(&mut x_frame, &mut z_frame, qi, shots);
            }

            if recording {
                let meas_base = measurement_noise_counter * shots;
                for s in 0..shots {
                    let meas_noise = measurement_bank[meas_base + s];
                    frame_results[measurement_counter][s] =
                        (ref_outcome + x_frame[qi][s] + meas_noise).rem_euclid(d);
                }
                measurement_noise_counter += 1;
            }

            // Reset
            if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                for s in 0..shots {
                    x_frame[qi][s] = 0;
                    z_frame[qi][s] = 0;
                }
            }

            // X-basis: apply H after
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                frame_h(&mut x_frame, &mut z_frame, qi, shots);
            }
        } else if is_pure_noise {
            // Noisy non-collapsing gate (noise channels)
            let is_two_qubit = matches!(gate_id, 25); // DEPOLARIZE2 = 25
            if is_two_qubit {
                let base = noise2_counter * shots * 4;
                for s in 0..shots {
                    x_frame[qi][s] += noise2_bank[base + s * 4];
                    z_frame[qi][s] += noise2_bank[base + s * 4 + 1];
                    x_frame[ti][s] += noise2_bank[base + s * 4 + 2];
                    z_frame[ti][s] += noise2_bank[base + s * 4 + 3];
                }
                noise2_counter += 1;
            } else {
                let base = noise1_counter * shots * 2;
                for s in 0..shots {
                    x_frame[qi][s] += noise1_bank[base + s * 2];
                    z_frame[qi][s] += noise1_bank[base + s * 2 + 1];
                }
                noise1_counter += 1;
            }
            if recording {
                // HERALDED_ERASURE records from erased_bank
                let base = erasure_counter * shots;
                for s in 0..shots {
                    frame_results[measurement_counter][s] = erased_bank[base + s];
                }
                erasure_counter += 1;
            }
        } else if !is_annotation_gate(gate_id) && !is_pure_noise {
            // Standard quantum gates or feedforward
            let qi_signed = inst.qudit_index;
            let ti_signed = inst.target_index;

            if qi_signed < 0 {
                // Classical feedforward via q_idx (measurement record reference)
                let abs_rec_idx = (measurement_counter as i64 + qi_signed) as usize;
                let ref_val = reference_sample[abs_rec_idx];
                let ti_u = ti_signed as usize;
                if gate_id == GATE_CNOT {
                    for s in 0..shots {
                        let control_error = (frame_results[abs_rec_idx][s] - ref_val).rem_euclid(d);
                        x_frame[ti_u][s] = (x_frame[ti_u][s] + control_error).rem_euclid(d);
                    }
                } else if gate_id == GATE_CZ {
                    for s in 0..shots {
                        let control_error = (frame_results[abs_rec_idx][s] - ref_val).rem_euclid(d);
                        z_frame[ti_u][s] = (z_frame[ti_u][s] + control_error).rem_euclid(d);
                    }
                }
            } else if ti_signed < 0 {
                // Classical feedforward via t_idx
                let abs_rec_idx = (measurement_counter as i64 + ti_signed) as usize;
                let ref_val = reference_sample[abs_rec_idx];
                if gate_id == GATE_CZ {
                    for s in 0..shots {
                        let control_error = (frame_results[abs_rec_idx][s] - ref_val).rem_euclid(d);
                        z_frame[qi][s] = (z_frame[qi][s] + control_error).rem_euclid(d);
                    }
                }
            } else {
                // Standard gate dispatch
                match gate_id {
                    GATE_H => frame_h(&mut x_frame, &mut z_frame, qi, shots),
                    GATE_H_INV => frame_h_inv(&mut x_frame, &mut z_frame, qi, shots),
                    GATE_P => {
                        for s in 0..shots { z_frame[qi][s] += x_frame[qi][s]; }
                    }
                    GATE_P_INV => {
                        for s in 0..shots { z_frame[qi][s] -= x_frame[qi][s]; }
                    }
                    GATE_CNOT => {
                        for s in 0..shots {
                            x_frame[ti][s] += x_frame[qi][s];
                            z_frame[qi][s] -= z_frame[ti][s];
                        }
                    }
                    GATE_CNOT_INV => {
                        for s in 0..shots {
                            x_frame[ti][s] -= x_frame[qi][s];
                            z_frame[qi][s] += z_frame[ti][s];
                        }
                    }
                    GATE_CZ => {
                        for s in 0..shots {
                            let x1 = x_frame[qi][s];
                            let x2 = x_frame[ti][s];
                            z_frame[ti][s] += x1;
                            z_frame[qi][s] += x2;
                        }
                    }
                    GATE_CZ_INV => {
                        for s in 0..shots {
                            let x1 = x_frame[qi][s];
                            let x2 = x_frame[ti][s];
                            z_frame[ti][s] -= x1;
                            z_frame[qi][s] -= x2;
                        }
                    }
                    GATE_SWAP => {
                        // Can't use std::mem::swap with two indices into same Vec
                        for s in 0..shots {
                            let tmp = x_frame[qi][s];
                            x_frame[qi][s] = x_frame[ti][s];
                            x_frame[ti][s] = tmp;
                            let tmp = z_frame[qi][s];
                            z_frame[qi][s] = z_frame[ti][s];
                            z_frame[ti][s] = tmp;
                        }
                    }
                    GATE_MULTIPLY | GATE_MULTIPLY_INV => {
                        let mut a = arg0.rem_euclid(d);
                        if a == 0 { a = 1; }
                        let a_inv = mod_inv_positive(a, d);
                        let (mul_x, mul_z) = if gate_id == GATE_MULTIPLY {
                            (a, a_inv)
                        } else {
                            (a_inv, a)
                        };
                        for s in 0..shots {
                            x_frame[qi][s] = (mul_x * x_frame[qi][s]).rem_euclid(d);
                            z_frame[qi][s] = (mul_z * z_frame[qi][s]).rem_euclid(d);
                        }
                    }
                    // Pauli gates (X, X_INV, Z, Z_INV) don't change the frame
                    GATE_X | GATE_X_INV | GATE_Z | GATE_Z_INV | GATE_I => {}
                    // Noiseless collapsing gates
                    GATE_M | GATE_MR | GATE_M_X | GATE_MR_X | GATE_RESET => {
                        // Noiseless measurement path (no noise banks)
                        let ref_outcome = reference_sample[measurement_counter];
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            frame_h_inv(&mut x_frame, &mut z_frame, qi, shots);
                        }
                        if recording {
                            for s in 0..shots {
                                frame_results[measurement_counter][s] =
                                    (ref_outcome + x_frame[qi][s]).rem_euclid(d);
                            }
                        }
                        if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                            for s in 0..shots {
                                x_frame[qi][s] = 0;
                                z_frame[qi][s] = 0;
                            }
                        }
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            frame_h(&mut x_frame, &mut z_frame, qi, shots);
                        }
                    }
                    _ => {} // Unknown gates are ignored
                }
            }
        }

        if recording {
            measurement_counter += 1;
        }
        gate_counter += 1;
    }

    // Final modulo
    // Flatten results to (num_measurements * shots) row-major
    let mut output = vec![0i64; num_measurements * shots];
    for m in 0..num_measurements {
        for s in 0..shots {
            output[m * shots + s] = frame_results[m][s].rem_euclid(d);
        }
    }
    output
}

#[inline]
fn frame_h(x_frame: &mut [Vec<i64>], z_frame: &mut [Vec<i64>], qi: usize, shots: usize) {
    for s in 0..shots {
        let tmp = x_frame[qi][s];
        x_frame[qi][s] = -z_frame[qi][s];
        z_frame[qi][s] = tmp;
    }
}

#[inline]
fn frame_h_inv(x_frame: &mut [Vec<i64>], z_frame: &mut [Vec<i64>], qi: usize, shots: usize) {
    for s in 0..shots {
        let tmp = x_frame[qi][s];
        x_frame[qi][s] = z_frame[qi][s];
        z_frame[qi][s] = -tmp;
    }
}

/// Check if a gate is an annotation (REPEAT=27, DETECTOR=28, SHIFT_COORDS=29, OBSERVABLE_INCLUDE=30).
#[inline]
fn is_annotation_gate(gate_id: i64) -> bool {
    (27..=30).contains(&gate_id)
}

fn mod_inv_positive(a: i64, m: i64) -> i64 {
    crate::tableau::mod_inverse(a, m).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_deterministic_noiseless() {
        // No noise: measure |0> should give all zeros
        let ir = vec![
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 1, target_index: i64::MAX, arg0: -1 },
        ];
        let reference = vec![0, 0];
        let result = run_frame(&ir, &reference, 2, 2, 10,
                               &[], &[], &[], &[]);
        // All measurements should be 0
        for &v in &result {
            assert_eq!(v, 0, "Expected 0, got {v}");
        }
    }

    #[test]
    fn test_frame_gate_conjugation() {
        // H then measure: frame should track through hadamard
        let ir = vec![
            IrInstruction { gate_id: GATE_H, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
        ];
        // Reference: after H, measurement is random. Say ref=0.
        let reference = vec![0];
        let shots = 100;
        let result = run_frame(&ir, &reference, 1, 2, shots,
                               &[], &[], &[], &[]);
        // After H, x_frame and z_frame are swapped. The x error (originally 0)
        // becomes -z (random). So measurements should show some nonzero values.
        let nonzero = result.iter().filter(|&&v| v != 0).count();
        // With random z init, roughly half should be nonzero
        assert!(nonzero > 10, "Expected randomness from H conjugation, got {nonzero}/100 nonzero");
    }
}
