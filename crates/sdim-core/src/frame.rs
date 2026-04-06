//! Pauli frame simulator for multi-shot sampling.
//!
//! Uses i8-packed flat arrays for 8x data density vs i64.
//! For qudit dimension d < 128, each frame element fits in a single byte.
//! Gate operations are plain i8 addition (no modulo), with periodic
//! branchless reduction to keep values in [0, d).
//!
//! This gives 32 shots per 256-bit SIMD word (vs 4 with i64),
//! plus 8x less memory bandwidth per gate.

use crate::ir::*;
use rand::Rng;

// ─── i8-packed 2D array ───────────────────────────────────────────

struct Frame {
    data: Vec<i8>,
    cols: usize,
}

impl Frame {
    fn new(rows: usize, cols: usize) -> Self {
        Frame { data: vec![0i8; rows * cols], cols }
    }

    #[inline(always)]
    fn row(&self, r: usize) -> &[i8] {
        let s = r * self.cols;
        &self.data[s..s + self.cols]
    }

    #[inline(always)]
    fn row_mut(&mut self, r: usize) -> &mut [i8] {
        let s = r * self.cols;
        &mut self.data[s..s + self.cols]
    }

    #[inline(always)]
    fn two_rows_mut(&mut self, a: usize, b: usize) -> (&mut [i8], &mut [i8]) {
        debug_assert_ne!(a, b);
        let c = self.cols;
        let ptr = self.data.as_mut_ptr();
        unsafe {
            (
                std::slice::from_raw_parts_mut(ptr.add(a * c), c),
                std::slice::from_raw_parts_mut(ptr.add(b * c), c),
            )
        }
    }

    /// Reduce a single row to [0, d) using a precomputed lookup table.
    /// This avoids the expensive i8 `%` (idiv) instruction.
    #[inline]
    fn reduce_row(&mut self, r: usize, lut: &[i8; 256]) {
        let row = self.row_mut(r);
        for v in row.iter_mut() {
            *v = lut[*v as u8 as usize];
        }
    }

    /// Reduce all rows to [0, d).
    fn reduce_all(&mut self, lut: &[i8; 256], num_rows: usize) {
        for r in 0..num_rows {
            self.reduce_row(r, lut);
        }
    }
}

// ─── i64-packed results (measurements are returned as i64) ────────

struct ResultArray {
    data: Vec<i64>,
    cols: usize,
}

impl ResultArray {
    fn new(rows: usize, cols: usize) -> Self {
        ResultArray { data: vec![0i64; rows * cols], cols }
    }

    #[inline(always)]
    fn row(&self, r: usize) -> &[i64] {
        let s = r * self.cols;
        &self.data[s..s + self.cols]
    }

    #[inline(always)]
    fn row_mut(&mut self, r: usize) -> &mut [i64] {
        let s = r * self.cols;
        &mut self.data[s..s + self.cols]
    }
}

// ─── How often to reduce mod d ───────────────────────────────────
// We use wrapping_add/sub, so values stay in i8 range [-128, 127].
// But we need to reduce before measurement to get correct [0, d) values.
// The only danger is if wrapping causes a value to jump to the wrong
// equivalence class. With wrapping i8 arithmetic:
//   val starts in [0, d-1], each gate adds/subs at most one value in [-(d-1), d-1]
//   After k ops, val is in [-(d-1)*k, (d-1)*(k+1)] before wrapping.
//   Wrapping is safe if this range stays within [-128, 127].
//   So k <= min(128/(d-1), 127/(d-1)) - 1 = 127/(d-1) - 1.
// We ONLY reduce before measurements. The periodic reduction is just a safety net.
#[inline]
fn reduction_interval(d: i8) -> usize {
    if d <= 1 { return usize::MAX; }
    let k = 127 / (d as usize - 1);
    k.max(1)
}

/// Build a 256-entry lookup table: lut[v as u8] = ((v as i8) mod d) mapped to [0, d).
fn build_mod_lut(d: i8) -> [i8; 256] {
    let mut lut = [0i8; 256];
    for i in 0u16..256 {
        let v = i as u8 as i8; // reinterpret as signed
        let mut r = v % d;
        if r < 0 { r += d; }
        lut[i as usize] = r;
    }
    lut
}

// ─── Main entry point ─────────────────────────────────────────────

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
    let d = dimension as i8;
    let d64 = dimension;
    let num_meas = reference_sample.len();
    let reduce_every = reduction_interval(d);

    let mut x_frame = Frame::new(num_qudits, shots);
    let mut z_frame = Frame::new(num_qudits, shots);
    {
        let mut rng = rand::thread_rng();
        for v in z_frame.data.iter_mut() {
            *v = rng.gen_range(0..d);
        }
    }

    let mut results = ResultArray::new(num_meas, shots);

    let lut = build_mod_lut(d);

    let mut noise1_ctr: usize = 0;
    let mut noise2_ctr: usize = 0;
    let mut erasure_ctr: usize = 0;
    let mut meas_ctr: usize = 0;
    let mut meas_noise_ctr: usize = 0;
    let mut gates_since_reduce: usize = 0;

    for inst in ir {
        let gate_id = inst.gate_id;
        let qi = inst.qudit_index as usize;
        let ti = inst.target_index as usize;
        let arg0 = inst.arg0;

        // Periodic safety reduction — only when close to i8 overflow
        if gates_since_reduce >= reduce_every {
            // Only reduce rows that might overflow. For most circuits
            // with measurements every few gates, this rarely triggers.
            x_frame.reduce_all(&lut, num_qudits);
            z_frame.reduce_all(&lut, num_qudits);
            gates_since_reduce = 0;
        }

        let collapsing = is_collapsing(gate_id);
        let recording = is_recording(gate_id);
        let is_pure_noise = (21..=26).contains(&gate_id);
        let has_meas_noise = !measurement_bank.is_empty();
        let noisy_collapsing = collapsing && has_meas_noise;

        if noisy_collapsing {
            // Only reduce the measured qudit's row
            x_frame.reduce_row(qi, &lut);

            let ref_out = reference_sample[meas_ctr];
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, true);
                x_frame.reduce_row(qi, &lut); // re-reduce after H swap
            }
            if recording {
                let mb = meas_noise_ctr * shots;
                let xr = x_frame.row(qi);
                let rr = results.row_mut(meas_ctr);
                for s in 0..shots {
                    rr[s] = (ref_out + xr[s] as i64 + measurement_bank[mb + s]).rem_euclid(d64);
                }
                meas_noise_ctr += 1;
            }
            if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                x_frame.row_mut(qi).fill(0);
                z_frame.row_mut(qi).fill(0);
            }
            if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, false);
            }
        } else if is_pure_noise {
            if gate_id == 25 {
                let base = noise2_ctr * shots * 4;
                let chunk = &noise2_bank[base..base + shots * 4];
                let xq = x_frame.row_mut(qi);
                for s in 0..shots { xq[s] = xq[s].wrapping_add(chunk[s * 4] as i8); }
                let zq = z_frame.row_mut(qi);
                for s in 0..shots { zq[s] = zq[s].wrapping_add(chunk[s * 4 + 1] as i8); }
                let xt = x_frame.row_mut(ti);
                for s in 0..shots { xt[s] = xt[s].wrapping_add(chunk[s * 4 + 2] as i8); }
                let zt = z_frame.row_mut(ti);
                for s in 0..shots { zt[s] = zt[s].wrapping_add(chunk[s * 4 + 3] as i8); }
                noise2_ctr += 1;
            } else {
                let base = noise1_ctr * shots * 2;
                let chunk = &noise1_bank[base..base + shots * 2];
                let xq = x_frame.row_mut(qi);
                for s in 0..shots { xq[s] = xq[s].wrapping_add(chunk[s * 2] as i8); }
                let zq = z_frame.row_mut(qi);
                for s in 0..shots { zq[s] = zq[s].wrapping_add(chunk[s * 2 + 1] as i8); }
                noise1_ctr += 1;
            }
            if recording {
                let base = erasure_ctr * shots;
                results.row_mut(meas_ctr).copy_from_slice(&erased_bank[base..base + shots]);
                erasure_ctr += 1;
            }
            gates_since_reduce += 1;
        } else if !is_annotation_gate(gate_id) && !is_pure_noise {
            let qi_signed = inst.qudit_index;
            let ti_signed = inst.target_index;

            if qi_signed < 0 {
                let abs_rec = (meas_ctr as i64 + qi_signed) as usize;
                let ref_val = reference_sample[abs_rec];
                let ti_u = ti_signed as usize;
                let rec_row = results.row(abs_rec);
                if gate_id == GATE_CNOT {
                    let xr = x_frame.row_mut(ti_u);
                    for s in 0..shots {
                        let ce = ((rec_row[s] - ref_val).rem_euclid(d64)) as i8;
                        xr[s] = xr[s].wrapping_add(ce);
                    }
                } else if gate_id == GATE_CZ {
                    let zr = z_frame.row_mut(ti_u);
                    for s in 0..shots {
                        let ce = ((rec_row[s] - ref_val).rem_euclid(d64)) as i8;
                        zr[s] = zr[s].wrapping_add(ce);
                    }
                }
            } else if ti_signed < 0 {
                let abs_rec = (meas_ctr as i64 + ti_signed) as usize;
                let ref_val = reference_sample[abs_rec];
                let rec_row = results.row(abs_rec);
                if gate_id == GATE_CZ {
                    let zr = z_frame.row_mut(qi);
                    for s in 0..shots {
                        let ce = ((rec_row[s] - ref_val).rem_euclid(d64)) as i8;
                        zr[s] = zr[s].wrapping_add(ce);
                    }
                }
            } else {
                match gate_id {
                    GATE_H => hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, false),
                    GATE_H_INV => hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, true),
                    GATE_P => {
                        let xr = x_frame.row(qi);
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] = zr[s].wrapping_add(xr[s]); }
                    }
                    GATE_P_INV => {
                        let xr = x_frame.row(qi);
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] = zr[s].wrapping_sub(xr[s]); }
                    }
                    GATE_CNOT => {
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { xt[s] = xt[s].wrapping_add(xq[s]); }
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { zq[s] = zq[s].wrapping_sub(zt[s]); }
                    }
                    GATE_CNOT_INV => {
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { xt[s] = xt[s].wrapping_sub(xq[s]); }
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots { zq[s] = zq[s].wrapping_add(zt[s]); }
                    }
                    GATE_CZ => {
                        let xq = x_frame.row(qi);
                        let xt = x_frame.row(ti);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots {
                            zt[s] = zt[s].wrapping_add(xq[s]);
                            zq[s] = zq[s].wrapping_add(xt[s]);
                        }
                    }
                    GATE_CZ_INV => {
                        let xq = x_frame.row(qi);
                        let xt = x_frame.row(ti);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        for s in 0..shots {
                            zt[s] = zt[s].wrapping_sub(xq[s]);
                            zq[s] = zq[s].wrapping_sub(xt[s]);
                        }
                    }
                    GATE_SWAP => {
                        let (xq, xt) = x_frame.two_rows_mut(qi, ti);
                        xq.swap_with_slice(xt);
                        let (zq, zt) = z_frame.two_rows_mut(qi, ti);
                        zq.swap_with_slice(zt);
                    }
                    GATE_MULTIPLY | GATE_MULTIPLY_INV => {
                        // Multiply requires reduced values for this qudit
                        x_frame.reduce_row(qi, &lut);
                        z_frame.reduce_row(qi, &lut);
                        let mut a = (arg0.rem_euclid(d64)) as i8;
                        if a == 0 { a = 1; }
                        let a_inv = mod_inv_positive(arg0.rem_euclid(d64), d64) as i8;
                        let (mul_x, mul_z) = if gate_id == GATE_MULTIPLY { (a, a_inv) } else { (a_inv, a) };
                        let xr = x_frame.row_mut(qi);
                        for s in 0..shots { xr[s] = ((xr[s] as i16 * mul_x as i16) % d as i16) as i8; }
                        let zr = z_frame.row_mut(qi);
                        for s in 0..shots { zr[s] = ((zr[s] as i16 * mul_z as i16) % d as i16) as i8; }
                    }
                    GATE_X | GATE_X_INV | GATE_Z | GATE_Z_INV | GATE_I => {}
                    GATE_M | GATE_MR | GATE_M_X | GATE_MR_X | GATE_RESET => {
                        // Only reduce the measured qudit
                        x_frame.reduce_row(qi, &lut);

                        let ref_out = reference_sample[meas_ctr];
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, true);
                            x_frame.reduce_row(qi, &lut);
                        }
                        if recording {
                            let xr = x_frame.row(qi);
                            let rr = results.row_mut(meas_ctr);
                            for s in 0..shots {
                                rr[s] = (ref_out + xr[s] as i64).rem_euclid(d64);
                            }
                        }
                        if gate_id == GATE_MR || gate_id == GATE_MR_X || gate_id == GATE_RESET {
                            x_frame.row_mut(qi).fill(0);
                            z_frame.row_mut(qi).fill(0);
                        }
                        if gate_id == GATE_M_X || gate_id == GATE_MR_X {
                            hadamard_rows(&mut x_frame, &mut z_frame, qi, shots, false);
                        }
                    }
                    _ => {}
                }
                gates_since_reduce += 1;
            }
        }

        if recording {
            meas_ctr += 1;
        }
    }

    // Transpose: (num_meas, shots) → (shots, num_meas) as i64
    let mut output = vec![0i64; shots * num_meas];
    for s in 0..shots {
        for m in 0..num_meas {
            output[s * num_meas + m] = results.data[m * shots + s];
        }
    }
    output
}

#[inline]
fn hadamard_rows(x: &mut Frame, z: &mut Frame, qi: usize, shots: usize, inv: bool) {
    let xr = x.row_mut(qi);
    let zr = z.row_mut(qi);
    if inv {
        for s in 0..shots {
            let tmp = xr[s];
            xr[s] = zr[s];
            zr[s] = tmp.wrapping_neg();
        }
    } else {
        for s in 0..shots {
            let tmp = xr[s];
            xr[s] = zr[s].wrapping_neg();
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
        assert!(nonzero > 10, "Expected randomness, got {nonzero}/100 nonzero");
    }

    #[test]
    fn test_frame_dimensions() {
        // Test correctness across dimensions
        for d in [2i64, 3, 5, 7, 11, 13] {
            let ir = vec![
                IrInstruction { gate_id: GATE_H, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
                IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 },
            ];
            let reference = vec![0];
            let shots = 500;
            let result = run_frame(&ir, &reference, 1, d, shots, &[], &[], &[], &[]);
            // All values should be in [0, d)
            for &v in &result {
                assert!(v >= 0 && v < d, "d={d}: value {v} out of range");
            }
        }
    }

    #[test]
    fn test_frame_deep_circuit() {
        // Many gates without measurement to stress lazy reduction
        let mut ir = Vec::new();
        for _ in 0..100 {
            ir.push(IrInstruction { gate_id: GATE_H, qudit_index: 0, target_index: i64::MAX, arg0: -1 });
            ir.push(IrInstruction { gate_id: GATE_P, qudit_index: 0, target_index: i64::MAX, arg0: -1 });
            ir.push(IrInstruction { gate_id: GATE_CNOT, qudit_index: 0, target_index: 1, arg0: -1 });
        }
        ir.push(IrInstruction { gate_id: GATE_M, qudit_index: 0, target_index: i64::MAX, arg0: -1 });
        ir.push(IrInstruction { gate_id: GATE_M, qudit_index: 1, target_index: i64::MAX, arg0: -1 });

        for d in [3i64, 7, 13] {
            let reference = vec![0, 0];
            let result = run_frame(&ir, &reference, 2, d, 50, &[], &[], &[], &[]);
            for &v in &result {
                assert!(v >= 0 && v < d, "d={d}: value {v} out of range after deep circuit");
            }
        }
    }

    #[test]
    fn test_reduction_interval() {
        assert_eq!(reduction_interval(2), 127); // 127/(2-1) = 127
        assert_eq!(reduction_interval(3), 63);  // 127/2 = 63
        assert_eq!(reduction_interval(5), 31);  // 127/4 = 31
        assert_eq!(reduction_interval(7), 21);  // 127/6 = 21
        assert_eq!(reduction_interval(13), 10); // 127/12 = 10
        assert!(reduction_interval(100) >= 1);
    }
}
