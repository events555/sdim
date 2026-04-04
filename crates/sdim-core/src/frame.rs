//! Pauli frame simulator for multi-shot sampling.
//!
//! Uses flat contiguous arrays with manual row indexing for zero-copy
//! two-row operations and SIMD autovectorization.

use crate::ir::*;
use rand::Rng;

/// Row-major 2D array backed by a single contiguous Vec.
struct FlatArray {
    data: Vec<i64>,
    cols: usize,
}

impl FlatArray {
    fn new(rows: usize, cols: usize) -> Self {
        FlatArray {
            data: vec![0i64; rows * cols],
            cols,
        }
    }

    #[inline(always)]
    fn row(&self, r: usize) -> &[i64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline(always)]
    fn row_mut(&mut self, r: usize) -> &mut [i64] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Get two disjoint mutable row slices. Panics if qi == ti.
    #[inline(always)]
    fn two_rows_mut(&mut self, qi: usize, ti: usize) -> (&mut [i64], &mut [i64]) {
        assert_ne!(qi, ti);
        let cols = self.cols;
        let ptr = self.data.as_mut_ptr();
        unsafe {
            let row_q = std::slice::from_raw_parts_mut(ptr.add(qi * cols), cols);
            let row_t = std::slice::from_raw_parts_mut(ptr.add(ti * cols), cols);
            (row_q, row_t)
        }
    }

    fn fill_row(&mut self, r: usize, val: i64) {
        self.row_mut(r).fill(val);
    }

    fn modulo_all(&mut self, d: i64) {
        for v in self.data.iter_mut() {
            *v = v.rem_euclid(d);
        }
    }
}

pub fn run_frame(
    ir: &[IrInstruction],
    reference_sample: &[i64],
    num_qudits: usize,
    dimension: i64,
    shots: usize,
    noise1_bank: &[i64],
    noise2_bank: &[i64],
    erased_bank: &[i64],
    measurement_bank: &[i64],
) -> Vec<i64> {
    let d = dimension;
    let num_measurements = reference_sample.len();

    let mut x_frame = FlatArray::new(num_qudits, shots);
    let mut z_frame = FlatArray::new(num_qudits, shots);
    {
        let mut rng = rand::thread_rng();
        for v in z_frame.data.iter_mut() {
            *v = rng.gen_range(0..d);
        }
    }

    let mut results = FlatArray::new(num_measurements, shots);

    let mut noise1_counter: usize = 0;
    let mut noise2_counter: usize = 0;
    let mut erasure_counter: usize = 0;
    let mut meas_counter: usize = 0;
    let mut meas_noise_counter: usize = 0;
    let mut gate_counter: usize = 0;

    for inst in ir {
        let gate_id = inst.gate_id;
        let qi = inst.qudit_index as usize;
        let ti = inst.target_index as usize;
        let arg0 = inst.arg0;

        if gate_counter % 128 == 0 {
            x_frame.modulo_all(d);
            z_frame.modulo_all(d);
        }

        let collapsing = is_collapsing(gate_id);
        let recording = is_recording(gate_id);
        let is_pure_noise = (21..=26).contains(&gate_id);
        let has_meas_noise = !measurement_bank.is_empty();
        let noisy_collapsing = collapsing && has_meas_noise;

        if noisy_collapsing {
            let ref_out = reference_sample[meas_counter];
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, true);
            }
            if recording {
                let mb = meas_noise_counter * shots;
                let xr = x_frame.row(qi);
                let rr = results.row_mut(meas_counter);
                for s in 0..shots {
                    rr[s] = (ref_out + xr[s] + measurement_bank[mb + s]).rem_euclid(d);
                }
                meas_noise_counter += 1;
            }
            if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                x_frame.fill_row(qi, 0);
                z_frame.fill_row(qi, 0);
            }
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, false);
            }
        } else if is_pure_noise {
            if gate_id == 25 { // DEPOLARIZE2
                let base = noise2_counter * shots * 4;
                let chunk = &noise2_bank[base..base + shots * 4];
                let xq = x_frame.row_mut(qi);
                for s in 0..shots { xq[s] += chunk[s * 4]; }
                let zq = z_frame.row_mut(qi);
                for s in 0..shots { zq[s] += chunk[s * 4 + 1]; }
                let xt = x_frame.row_mut(ti);
                for s in 0..shots { xt[s] += chunk[s * 4 + 2]; }
                let zt = z_frame.row_mut(ti);
                for s in 0..shots { zt[s] += chunk[s * 4 + 3]; }
                noise2_counter += 1;
            } else {
                let base = noise1_counter * shots * 2;
                let chunk = &noise1_bank[base..base + shots * 2];
                let xq = x_frame.row_mut(qi);
                for s in 0..shots { xq[s] += chunk[s * 2]; }
                let zq = z_frame.row_mut(qi);
                for s in 0..shots { zq[s] += chunk[s * 2 + 1]; }
                noise1_counter += 1;
            }
            if recording {
                let base = erasure_counter * shots;
                results.row_mut(meas_counter).copy_from_slice(&erased_bank[base..base + shots]);
                erasure_counter += 1;
            }
        } else if !is_annotation_gate(gate_id) && !is_pure_noise {
            let qi_signed = inst.qudit_index;
            let ti_signed = inst.target_index;

            if qi_signed < 0 {
                let abs_rec = (meas_counter as i64 + qi_signed) as usize;
                let ref_val = reference_sample[abs_rec];
                let ti_u = ti_signed as usize;
                let rec_row = results.row(abs_rec);
                if gate_id == GATE_CNOT {
                    let xr = x_frame.row_mut(ti_u);
                    for s in 0..shots {
                        let ce = (rec_row[s] - ref_val).rem_euclid(d);
                        xr[s] = (xr[s] + ce).rem_euclid(d);
                    }
                } else if gate_id == GATE_CZ {
                    let zr = z_frame.row_mut(ti_u);
                    for s in 0..shots {
                        let ce = (rec_row[s] - ref_val).rem_euclid(d);
                        zr[s] = (zr[s] + ce).rem_euclid(d);
                    }
                }
            } else if ti_signed < 0 {
                let abs_rec = (meas_counter as i64 + ti_signed) as usize;
                let ref_val = reference_sample[abs_rec];
                let rec_row = results.row(abs_rec);
                if gate_id == GATE_CZ {
                    let zr = z_frame.row_mut(qi);
                    for s in 0..shots {
                        let ce = (rec_row[s] - ref_val).rem_euclid(d);
                        zr[s] = (zr[s] + ce).rem_euclid(d);
                    }
                }
            } else {
                match gate_id {
                    GATE_H => swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, false),
                    GATE_H_INV => swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, true),
                    GATE_P => {
                        // z[qi] += x[qi] — same-row read+write, need copy
                        // Actually both are different arrays, so no conflict
                        let xr = x_frame.row(qi);
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] += xr[s]; }
                    }
                    GATE_P_INV => {
                        let xr = x_frame.row(qi);
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] -= xr[s]; }
                    }
                    GATE_CNOT => {
                        // x[ti] += x[qi]; z[qi] -= z[ti]
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { xt[s] += xq[s]; }
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { zq[s] -= zt[s]; }
                    }
                    GATE_CNOT_INV => {
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { xt[s] -= xq[s]; }
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { zq[s] += zt[s]; }
                    }
                    GATE_CZ => {
                        // z[ti] += x[qi]; z[qi] += x[ti]
                        // Read from x, write to z — no conflict between x and z
                        let xq = x_frame.row(qi);
                        let xt = x_frame.row(ti);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots {
                            zt[s] += xq[s];
                            zq[s] += xt[s];
                        }
                    }
                    GATE_CZ_INV => {
                        let xq = x_frame.row(qi);
                        let xt = x_frame.row(ti);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots {
                            zt[s] -= xq[s];
                            zq[s] -= xt[s];
                        }
                    }
                    GATE_SWAP => {
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        xq.swap_with_slice(xt);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        zq.swap_with_slice(zt);
                    }
                    GATE_MULTIPLY | GATE_MULTIPLY_INV => {
                        let mut a = arg0.rem_euclid(d);
                        if a == 0 { a = 1; }
                        let a_inv = mod_inv_positive(a, d);
                        let (mul_x, mul_z) = if gate_id == GATE_MULTIPLY { (a, a_inv) } else { (a_inv, a) };
                        let xr = x_frame.row_mut(qi);
                        for s in 0..shots { xr[s] = (mul_x * xr[s]).rem_euclid(d); }
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] = (mul_z * zr[s]).rem_euclid(d); }
                    }
                    GATE_X | GATE_X_INV | GATE_Z | GATE_Z_INV | GATE_I => {}
                    GATE_M | GATE_MR | GATE_M_X | GATE_MR_X | GATE_RESET => {
                        let ref_out = reference_sample[meas_counter];
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, true);
                        }
                        if recording {
                            let xr = x_frame.row(qi);
                            let rr = results.row_mut(meas_counter);
                            for s in 0..shots {
                                rr[s] = (ref_out + xr[s]).rem_euclid(d);
                            }
                        }
                        if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                            x_frame.fill_row(qi, 0);
                            z_frame.fill_row(qi, 0);
                        }
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            swap_negate_rows(&mut x_frame, &mut z_frame, qi, shots, false);
                        }
                    }
                    _ => {}
                }
            }
        }

        if recording {
            meas_counter += 1;
        }
        gate_counter += 1;
    }

    // Transpose results from (num_measurements, shots) to (shots, num_measurements)
    // Write in output-row-major order for best cache behavior on the write side
    let mut output = vec![0i64; shots * num_measurements];
    for s in 0..shots {
        for m in 0..num_measurements {
            output[s * num_measurements + m] = results.data[m * shots + s].rem_euclid(d);
        }
    }
    output
}

/// For Hadamard: swap x[qi] and z[qi] rows, negating one.
/// H:     x[qi] <- -z[qi], z[qi] <- x[qi]   (inv=false)
/// H_INV: x[qi] <- z[qi],  z[qi] <- -x[qi]  (inv=true)
#[inline]
fn swap_negate_rows(x: &mut FlatArray, z: &mut FlatArray, qi: usize, shots: usize, inv: bool) {
    let xr = x.row_mut(qi);
    let zr = z.row_mut(qi);
    if inv {
        // H_INV: x <- z, z <- -x
        for s in 0..shots {
            let tmp = xr[s];
            xr[s] = zr[s];
            zr[s] = -tmp;
        }
    } else {
        // H: x <- -z, z <- x
        for s in 0..shots {
            let tmp = xr[s];
            xr[s] = -zr[s];
            zr[s] = tmp;
        }
    }
}

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
        let ir = vec![
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 1, target_index: i64::MAX, arg0: -1 },
        ];
        let reference = vec![0, 0];
        let result = run_frame(&ir, &reference, 2, 2, 10, &[], &[], &[], &[]);
        for &v in &result {
            assert_eq!(v, 0, "Expected 0, got {v}");
        }
    }

    #[test]
    fn test_frame_gate_conjugation() {
        let ir = vec![
            IrInstruction { gate_id: GATE_H, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
        ];
        let reference = vec![0];
        let shots = 100;
        let result = run_frame(&ir, &reference, 1, 2, shots, &[], &[], &[], &[]);
        let nonzero = result.iter().filter(|&&v| v != 0).count();
        assert!(nonzero > 10, "Expected randomness from H conjugation, got {nonzero}/100 nonzero");
    }
}
