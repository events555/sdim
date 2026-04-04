//! Qudit stabilizer tableau over the Weyl-Heisenberg group.
//!
//! Reference: de Beaudrap, QIC 13.1-2 (2013), arXiv:1102.3354v4.

/// Stabilizer tableau for qudits of dimension d.
///
/// Generators are stored in rows `0..l`. Arrays are pre-allocated to `2n` rows
/// (the maximum after composite-d measurements); only the first `l` rows are active.
#[derive(Clone, Debug)]
pub struct TableauSimulator {
    /// Number of qudits.
    pub n: usize,
    /// Local dimension.
    pub d: i64,
    /// Current number of generators (<= 2n).
    pub l: usize,
    /// X block: (2n, n) array, entries mod d.
    pub x: Vec<Vec<i64>>,
    /// Z block: (2n, n) array, entries mod d.
    pub z: Vec<Vec<i64>>,
    /// Phase exponents: (2n,) array, entries mod 2d.
    pub tau_exp: Vec<i64>,
}

impl TableauSimulator {
    /// Create a new tableau in the computational basis state |0...0>.
    pub fn new(n: usize, d: i64) -> Self {
        let cap = 2 * n;
        let x = vec![vec![0i64; n]; cap];
        let mut z = vec![vec![0i64; n]; cap];
        let tau_exp = vec![0i64; cap];

        // Z[:n] = identity
        for i in 0..n {
            z[i][i] = 1;
        }

        TableauSimulator {
            n,
            d,
            l: n,
            x,
            z,
            tau_exp,
        }
    }

    #[inline]
    pub fn even(&self) -> bool {
        self.d % 2 == 0
    }

    #[inline]
    pub fn big_d(&self) -> i64 {
        2 * self.d
    }

    /// Reduce all entries modulo d (X, Z) and 2d (tau_exp).
    pub fn modulo(&mut self) {
        let d = self.d;
        let big_d = self.big_d();
        for i in 0..self.l {
            for j in 0..self.n {
                self.x[i][j] = self.x[i][j].rem_euclid(d);
                self.z[i][j] = self.z[i][j].rem_euclid(d);
            }
            self.tau_exp[i] = self.tau_exp[i].rem_euclid(big_d);
        }
    }

    /// Ordered product of two Weyl vectors.
    pub fn product(
        &self,
        x1: &[i64],
        z1: &[i64],
        t1: i64,
        x2: &[i64],
        z2: &[i64],
        t2: i64,
    ) -> (Vec<i64>, Vec<i64>, i64) {
        let d = self.d;
        let big_d = self.big_d();
        let n = self.n;
        let mut rx = vec![0i64; n];
        let mut rz = vec![0i64; n];
        let mut cross: i64 = 0;
        for j in 0..n {
            rx[j] = (x1[j] + x2[j]).rem_euclid(d);
            rz[j] = (z1[j] + z2[j]).rem_euclid(d);
            cross += z1[j] * x2[j];
        }
        let rt = (t1 + t2 + 2 * cross).rem_euclid(big_d);
        (rx, rz, rt)
    }

    /// Closed-form power rule: (tau^{-t} Z^z X^x)^c.
    pub fn power(
        &self,
        base_x: &[i64],
        base_z: &[i64],
        base_t: i64,
        c: i64,
    ) -> (Vec<i64>, Vec<i64>, i64) {
        let d = self.d;
        let big_d = self.big_d();
        let n = self.n;
        let mut rx = vec![0i64; n];
        let mut rz = vec![0i64; n];
        let mut dot: i64 = 0;
        for j in 0..n {
            rx[j] = (c * base_x[j]).rem_euclid(d);
            rz[j] = (c * base_z[j]).rem_euclid(d);
            dot += base_z[j] * base_x[j];
        }
        let rt = (c * base_t + c * (c - 1) * dot).rem_euclid(big_d);
        (rx, rz, rt)
    }

    /// Row k <- row k * (row p)^m.
    pub fn row_add_mult(&mut self, k: usize, p: usize, m: i64) {
        if m == 0 {
            return;
        }
        let (px, pz, pt) = self.power(&self.x[p].clone(), &self.z[p].clone(), self.tau_exp[p], m);
        let (rx, rz, rt) = self.product(
            &self.x[k].clone(),
            &self.z[k].clone(),
            self.tau_exp[k],
            &px,
            &pz,
            pt,
        );
        self.x[k] = rx;
        self.z[k] = rz;
        self.tau_exp[k] = rt;
    }

    /// Apply unimodular matrix V to generator rows 0..l.
    pub fn apply_column_transform(&mut self, v: &[Vec<i64>]) {
        let d = self.d;
        let big_d = self.big_d();
        let l = self.l;
        let n = self.n;

        // Save old state
        let z_old: Vec<Vec<i64>> = self.z[..l].to_vec();
        let x_old: Vec<Vec<i64>> = self.x[..l].to_vec();
        let tau_old: Vec<i64> = self.tau_exp[..l].to_vec();

        // G = Z_old @ X_old^T  (l x l matrix)
        let mut g = vec![vec![0i64; l]; l];
        for i in 0..l {
            for j in 0..l {
                let mut s = 0i64;
                for k in 0..n {
                    s += z_old[i][k] * x_old[j][k];
                }
                g[i][j] = s;
            }
        }

        // new Z = V^T @ Z_old, new X = V^T @ X_old
        for i in 0..l {
            for k in 0..n {
                let mut sz = 0i64;
                let mut sx = 0i64;
                for j in 0..l {
                    sz += v[j][i] * z_old[j][k];
                    sx += v[j][i] * x_old[j][k];
                }
                self.z[i][k] = sz.rem_euclid(d);
                self.x[i][k] = sx.rem_euclid(d);
            }
        }

        // Phase: t1 = V^T @ tau_old
        // t2 = (V * (V - 1))^T @ diag(G)
        // t3 = 2 * diag(V^T @ triu(G, 1) @ V)
        for i in 0..l {
            let mut t1: i64 = 0;
            let mut t2: i64 = 0;
            for j in 0..l {
                t1 += v[j][i] * tau_old[j];
                t2 += v[j][i] * (v[j][i] - 1) * g[j][j];
            }

            let mut t3: i64 = 0;
            for r in 0..l {
                for c in (r + 1)..l {
                    t3 += v[r][i] * g[r][c] * v[c][i];
                }
            }
            t3 *= 2;

            self.tau_exp[i] = (t1 + t2 + t3).rem_euclid(big_d);
        }
    }

    // ---- Gate methods ----

    /// Hadamard gate on qudit q. If dagger=true, apply H^dagger.
    pub fn hadamard(&mut self, q: usize, dagger: bool) {
        let d = self.d;
        let big_d = self.big_d();
        let dir: i64 = if dagger { -1 } else { 1 };
        for i in 0..self.l {
            let x_old = self.x[i][q];
            let z_old = self.z[i][q];
            self.tau_exp[i] = (self.tau_exp[i] - 2 * z_old * x_old).rem_euclid(big_d);
            self.x[i][q] = (-dir * z_old).rem_euclid(d);
            self.z[i][q] = (dir * x_old).rem_euclid(d);
        }
    }

    /// Phase gate on qudit q. If dagger=true, apply P^dagger.
    pub fn phase_gate(&mut self, q: usize, dagger: bool) {
        let d = self.d;
        let big_d = self.big_d();
        let dir: i64 = if dagger { -1 } else { 1 };
        let even = self.even();
        for i in 0..self.l {
            let xj = self.x[i][q];
            self.z[i][q] = (self.z[i][q] + dir * xj).rem_euclid(d);
            if even {
                self.tau_exp[i] = (self.tau_exp[i] + dir * xj * xj).rem_euclid(big_d);
            } else {
                self.tau_exp[i] = (self.tau_exp[i] + dir * xj * (xj + 1)).rem_euclid(big_d);
            }
        }
    }

    /// Pauli X gate on qudit q with given power. If dagger=true, apply X^{-power}.
    pub fn pauli_x(&mut self, q: usize, power: i64, dagger: bool) {
        let big_d = self.big_d();
        let dir: i64 = if dagger { -1 } else { 1 };
        for i in 0..self.l {
            self.tau_exp[i] = (self.tau_exp[i] + dir * 2 * power * self.z[i][q]).rem_euclid(big_d);
        }
    }

    /// Pauli Z gate on qudit q with given power. If dagger=true, apply Z^{-power}.
    pub fn pauli_z(&mut self, q: usize, power: i64, dagger: bool) {
        let big_d = self.big_d();
        let dir: i64 = if dagger { -1 } else { 1 };
        for i in 0..self.l {
            self.tau_exp[i] = (self.tau_exp[i] - dir * 2 * power * self.x[i][q]).rem_euclid(big_d);
        }
    }

    /// CNOT gate with control c and target t. If dagger=true, apply CNOT^dagger.
    pub fn cnot(&mut self, c: usize, t: usize, dagger: bool) {
        let d = self.d;
        let dir: i64 = if dagger { -1 } else { 1 };
        for i in 0..self.l {
            self.x[i][t] = (self.x[i][t] + dir * self.x[i][c]).rem_euclid(d);
            self.z[i][c] = (self.z[i][c] - dir * self.z[i][t]).rem_euclid(d);
        }
    }

    /// CZ gate on qudits q1 and q2. If dagger=true, apply CZ^dagger.
    pub fn cz(&mut self, q1: usize, q2: usize, dagger: bool) {
        let d = self.d;
        let big_d = self.big_d();
        let dir: i64 = if dagger { -1 } else { 1 };
        for i in 0..self.l {
            let x1 = self.x[i][q1];
            let x2 = self.x[i][q2];
            self.z[i][q1] = (self.z[i][q1] + dir * x2).rem_euclid(d);
            self.z[i][q2] = (self.z[i][q2] + dir * x1).rem_euclid(d);
            self.tau_exp[i] = (self.tau_exp[i] + 2 * dir * x1 * x2).rem_euclid(big_d);
        }
    }

    /// SWAP gate on qudits q1 and q2.
    pub fn swap(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for i in 0..self.l {
            self.x[i].swap(q1, q2);
            self.z[i].swap(q1, q2);
        }
    }

    /// Multiplier gate M_a on qudit q. If dagger=true, apply M_a^dagger.
    pub fn multiply(&mut self, q: usize, a: i64, dagger: bool) {
        let d = self.d;
        let mut a = a;
        let mut a_inv = mod_inverse(a, d).expect("gcd(a, d) must be 1");
        if dagger {
            std::mem::swap(&mut a, &mut a_inv);
        }
        for i in 0..self.l {
            self.x[i][q] = (a * self.x[i][q]).rem_euclid(d);
            self.z[i][q] = (a_inv * self.z[i][q]).rem_euclid(d);
        }
    }

    /// Apply a gate by its numeric ID (matching quality branch registry ordering).
    pub fn apply_gate(&mut self, gate_id: i64, qudit_idx: usize, target_idx: usize, arg0: i64) {
        let power = if arg0 < 0 { 1 } else { arg0.rem_euclid(self.d) };
        match gate_id {
            0 => {} // I
            1 => self.pauli_x(qudit_idx, power, false),       // X
            2 => self.pauli_x(qudit_idx, power, true),         // X_INV
            3 => self.pauli_z(qudit_idx, power, false),        // Z
            4 => self.pauli_z(qudit_idx, power, true),         // Z_INV
            5 => self.hadamard(qudit_idx, false),              // H
            6 => self.hadamard(qudit_idx, true),               // H_INV
            7 => self.phase_gate(qudit_idx, false),            // P
            8 => self.phase_gate(qudit_idx, true),             // P_INV
            9 => self.multiply(qudit_idx, power, false),       // MULTIPLY
            10 => self.multiply(qudit_idx, power, true),       // MULTIPLY_INV
            11 => self.cnot(qudit_idx, target_idx, false),     // CNOT
            12 => self.cnot(qudit_idx, target_idx, true),      // CNOT_INV
            13 => self.cz(qudit_idx, target_idx, false),       // CZ
            14 => self.cz(qudit_idx, target_idx, true),        // CZ_INV
            15 => self.swap(qudit_idx, target_idx),            // SWAP
            // Noise gates (21..=26) are skipped by the caller
            _ => {}
        }
    }
}

/// Compute modular inverse of a mod m using extended Euclidean algorithm.
pub fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (mut old_r, mut r) = (a.rem_euclid(m), m);
    let (mut old_s, mut s) = (1i64, 0i64);
    while r != 0 {
        let q = old_r / r;
        let tmp = r;
        r = old_r - q * r;
        old_r = tmp;
        let tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }
    if old_r != 1 {
        None
    } else {
        Some(old_s.rem_euclid(m))
    }
}

/// Extended GCD: returns (g, x, y) such that a*x + b*y = g.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a.abs(), if a >= 0 { 1 } else { -1 }, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7), Some(5)); // 3*5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(2, 6), None); // gcd(2,6) = 2 ≠ 1
        assert_eq!(mod_inverse(1, 5), Some(1));
    }

    #[test]
    fn test_new_tableau() {
        let t = TableauSimulator::new(3, 2);
        assert_eq!(t.n, 3);
        assert_eq!(t.d, 2);
        assert_eq!(t.l, 3);
        // Z should be identity
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.z[i][j], if i == j { 1 } else { 0 });
            }
        }
        // X should be zero
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.x[i][j], 0);
            }
        }
    }
}
