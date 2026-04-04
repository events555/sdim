//! Measurement protocol for the stabilizer tableau.
//!
//! Implements the unified Steps 1-5 from algorithm.md:
//! 1. Compute commutator row c and eta
//! 2. SNF of c to apply column transform
//! 3. Eigenvalue of P^s via linear solve
//! 4. Sample outcome h
//! 5. Collapse: scale generator 0, insert measurement result

use crate::tableau::{mod_inverse, TableauSimulator};
use ndarray::Array2;
use rand::Rng;

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Call modularsnf crate: smith_normal_form(matrix, modulus) -> (U, V, S)
fn snf_mod(matrix: &[Vec<i64>], d: i64) -> (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let rows = matrix.len();
    let cols = if rows > 0 { matrix[0].len() } else { 0 };

    let mut arr = Array2::<i64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = matrix[i][j].rem_euclid(d);
        }
    }

    let (u, v, s) = modularsnf::smith_normal_form(&arr, d)
        .expect("SNF computation failed");

    let to_vecs = |a: &Array2<i64>| -> Vec<Vec<i64>> {
        (0..a.nrows())
            .map(|i| (0..a.ncols()).map(|j| a[[i, j]]).collect())
            .collect()
    };

    (to_vecs(&s), to_vecs(&u), to_vecs(&v))
}

impl TableauSimulator {
    /// Measure qudit q in the Z basis. Returns outcome in [0, d).
    pub fn measure(&mut self, q: usize) -> i64 {
        let n = self.n;
        let mut a = vec![0i64; n];
        let b = vec![0i64; n];
        a[q] = 1;
        self.measure_pauli(&a, &b, 0)
    }

    /// Measure Pauli P = tau^{-delta} Z^a X^b.
    /// Returns outcome h in [0, d).
    pub fn measure_pauli(&mut self, a: &[i64], b: &[i64], delta: i64) -> i64 {
        let d = self.d;
        let l = self.l;
        let n = self.n;

        // Step 1: compute commutator row c
        // c[j] = (a . X[j] - b . Z[j]) mod d
        let mut c = vec![0i64; l];
        for j in 0..l {
            let mut val = 0i64;
            for k in 0..n {
                val += a[k] * self.x[j][k] - b[k] * self.z[j][k];
            }
            c[j] = val.rem_euclid(d);
        }

        // eta = gcd(d, c[0], c[1], ...)
        let mut eta = d;
        for &cj in &c {
            eta = gcd(eta, cj);
        }
        let s = d / eta;

        // Step 2: if eta < d, apply column transform via SNF of c
        if eta < d {
            let c_matrix = vec![c.clone()]; // 1 x l matrix
            let (_s_mat, _u_mat, v_mat) = snf_mod(&c_matrix, d);
            self.apply_column_transform(&v_mat);
        }

        // Step 3: find eigenvalue of P^s
        let (t, f1) = self.eigenvalue_ps(a, b, delta, s);

        // Step 4: sample outcome
        let h = self.sample_outcome(t, s, eta);

        // Step 5: collapse if measurement was non-deterministic
        if eta < d {
            self.collapse(a, b, delta, h, s, f1);
        }

        h
    }

    /// Step 3: find the eigenvalue of P^s by decomposing it over current generators.
    /// Returns (t, f1) where 2*s*h ≡ t (mod D).
    fn eigenvalue_ps(&self, a: &[i64], b: &[i64], delta: i64, s: i64) -> (i64, i64) {
        let n = self.n;
        let d = self.d;
        let big_d = self.big_d();
        let l = self.l;

        // P^s
        let mut ps_z = vec![0i64; n];
        let mut ps_x = vec![0i64; n];
        let mut ab_dot = 0i64;
        for k in 0..n {
            ps_z[k] = (s * a[k]).rem_euclid(d);
            ps_x[k] = (s * b[k]).rem_euclid(d);
            ab_dot += a[k] * b[k];
        }
        let ps_t = (s * delta + s * (s - 1) * ab_dot).rem_euclid(big_d);

        // Build generator Weyl block G (2n x l)
        // G = [gen_Z; gen_X] where gen_Z[k][j] = Z[j][k], gen_X[k][j] = X[j][k]
        let mut g_matrix = vec![vec![0i64; l]; 2 * n];
        for j in 0..l {
            for k in 0..n {
                g_matrix[k][j] = self.z[j][k];
                g_matrix[n + k][j] = self.x[j][k];
            }
        }

        let ps_vec: Vec<i64> = ps_z.iter().chain(ps_x.iter()).copied().collect();

        // SNF of G
        let (s_mat, u_mat, v_mat) = snf_mod(&g_matrix, d);

        let f1 = if !s_mat.is_empty() && !s_mat[0].is_empty() {
            s_mat[0][0].rem_euclid(d)
        } else {
            1
        };

        // rhs = U @ ps_vec mod d
        let u_rows = u_mat.len();
        let u_cols = if u_rows > 0 { u_mat[0].len() } else { 0 };
        let mut rhs = vec![0i64; u_rows];
        for i in 0..u_rows {
            let mut val = 0i64;
            for j in 0..u_cols.min(ps_vec.len()) {
                val += u_mat[i][j] * ps_vec[j];
            }
            rhs[i] = val.rem_euclid(d);
        }

        // Solve diagonal system S @ y = rhs (mod d)
        let s_rows = s_mat.len();
        let s_cols = if s_rows > 0 { s_mat[0].len() } else { 0 };
        let mut y = vec![0i64; l];
        for i in 0..s_rows.min(s_cols) {
            let dii = s_mat[i][i].rem_euclid(d);
            let bi = rhs[i].rem_euclid(d);
            if dii == 0 {
                continue;
            }
            let g = gcd(dii, d);
            if g == 0 {
                continue;
            }
            let dii_g = dii / g;
            let d_g = d / g;
            if let Some(inv) = mod_inverse(dii_g, d_g) {
                y[i] = ((bi / g) * inv).rem_euclid(d_g);
            }
        }

        // coeffs = V @ y mod d
        let v_rows = v_mat.len();
        let v_cols = if v_rows > 0 { v_mat[0].len() } else { 0 };
        let mut coeffs = vec![0i64; l];
        for i in 0..v_rows.min(l) {
            let mut val = 0i64;
            for j in 0..v_cols.min(l) {
                val += v_mat[i][j] * y[j];
            }
            coeffs[i] = val.rem_euclid(d);
        }

        // Accumulate product of generators^coeffs
        let mut acc_x = vec![0i64; n];
        let mut acc_z = vec![0i64; n];
        let mut acc_t: i64 = 0;
        for k in 0..l {
            let ck = coeffs[k].rem_euclid(d);
            if ck == 0 {
                continue;
            }
            let (px, pz, pt) = self.power(&self.x[k], &self.z[k], self.tau_exp[k], ck);
            let (nx, nz, nt) = self.product(&acc_x, &acc_z, acc_t, &px, &pz, pt);
            acc_x = nx;
            acc_z = nz;
            acc_t = nt;
        }

        let t = (acc_t - ps_t).rem_euclid(big_d);
        (t, f1)
    }

    /// Step 4: solve 2*s*h ≡ t (mod D), sample uniformly from coset.
    fn sample_outcome(&self, t: i64, s: i64, eta: i64) -> i64 {
        let d = self.d;
        let big_d = self.big_d();
        let g = gcd(2 * s, big_d);
        let two_s_g = 2 * s / g;
        let d_g = big_d / g;
        let h0 = if let Some(inv) = mod_inverse(two_s_g, d_g) {
            ((t / g) * inv).rem_euclid(d_g)
        } else {
            0
        };

        let mut rng = rand::thread_rng();
        let num_outcomes = d / eta;
        let k = if num_outcomes > 1 {
            rng.gen_range(0..num_outcomes)
        } else {
            0
        };
        ((h0 + eta * k) % d) as i64
    }

    /// Step 5: scale generator 0, insert measurement result R.
    fn collapse(
        &mut self,
        a: &[i64],
        b: &[i64],
        delta: i64,
        h: i64,
        s: i64,
        f1: i64,
    ) {
        let d = self.d;
        let big_d = self.big_d();
        let l = self.l;
        let n = self.n;

        let (scaled_x, scaled_z, scaled_t) =
            self.power(&self.x[0].clone(), &self.z[0].clone(), self.tau_exp[0], s);

        let trivial = scaled_x.iter().all(|&v| v.rem_euclid(d) == 0)
            && scaled_z.iter().all(|&v| v.rem_euclid(d) == 0)
            && scaled_t.rem_euclid(big_d) == 0;

        let keep = !trivial && f1 % s != 0;

        if keep {
            // Insert at row l
            for j in 0..n {
                self.x[l][j] = scaled_x[j].rem_euclid(d);
                self.z[l][j] = scaled_z[j].rem_euclid(d);
            }
            self.tau_exp[l] = scaled_t.rem_euclid(big_d);
            self.l = l + 1;
        }

        // Replace generator 0 with the measurement result
        for j in 0..n {
            self.x[0][j] = b[j].rem_euclid(d);
            self.z[0][j] = a[j].rem_euclid(d);
        }
        self.tau_exp[0] = (delta + 2 * h).rem_euclid(big_d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_deterministic_zero() {
        for d in [2, 3, 4, 5, 6] {
            let mut t = TableauSimulator::new(1, d);
            assert_eq!(t.measure(0), 0, "d={d}");
        }
    }

    #[test]
    fn test_measure_after_pauli_x() {
        let mut t = TableauSimulator::new(1, 2);
        t.pauli_x(0, 1, false);
        assert_eq!(t.measure(0), 1);
    }

    #[test]
    fn test_measure_idempotent() {
        for d in [2, 3, 5] {
            let mut t = TableauSimulator::new(1, d as i64);
            t.hadamard(0, false);
            let r1 = t.measure(0);
            let r2 = t.measure(0);
            assert_eq!(r1, r2, "Measurement not idempotent for d={d}");
        }
    }

    #[test]
    fn test_measure_bell_pair() {
        for _ in 0..50 {
            let mut t = TableauSimulator::new(2, 2);
            t.hadamard(0, false);
            t.cnot(0, 1, false);
            let r0 = t.measure(0);
            let r1 = t.measure(1);
            assert_eq!(r0, r1, "Bell pair measurement mismatch");
        }
    }

    #[test]
    fn test_measure_random_qubit() {
        let mut counts = [0u32; 2];
        for _ in 0..200 {
            let mut t = TableauSimulator::new(1, 2);
            t.hadamard(0, false);
            counts[t.measure(0) as usize] += 1;
        }
        assert!(counts[0] > 20 && counts[1] > 20, "Measurement not random: {:?}", counts);
    }

    #[test]
    fn test_measure_composite_deterministic() {
        for d in [4, 6] {
            let mut t = TableauSimulator::new(2, d);
            assert_eq!(t.measure(0), 0);
            assert_eq!(t.measure(1), 0);
        }
    }

    #[test]
    fn test_measure_composite_random() {
        for d in [4i64, 6] {
            let mut counts = vec![0u32; d as usize];
            for _ in 0..500 {
                let mut t = TableauSimulator::new(1, d);
                t.hadamard(0, false);
                let outcome = t.measure(0);
                counts[outcome as usize] += 1;
            }
            for (v, &count) in counts.iter().enumerate() {
                assert!(count > 10, "d={d}, outcome {v} count={count} too low");
            }
        }
    }
}
