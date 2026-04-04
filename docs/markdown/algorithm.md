# Qudit Stabilizer Simulation

Reference: de Beaudrap, arXiv:1102.3354v4.

## Representation

A Pauli operator on $n$ qudits of dimension $d$ is

$$P = \tau^{-\phi}\, Z^{\mathbf{z}}\, X^{\mathbf{x}}, \qquad \mathbf{z}, \mathbf{x} \in \mathbb{Z}_d^n, \quad \phi \in \mathbb{Z}_D$$

where $\omega = e^{2\pi i/d}$, $\tau = e^{i\pi(d^2+1)/d}$, $\tau^2 = \omega$, and

$$D = \begin{cases} d & d \text{ odd} \\ 2d & d \text{ even} \end{cases}$$

Two Paulis satisfy the commutation relation

$$P_1 P_2 = \tau^{2\langle 1,2\rangle}\, P_2 P_1$$

where $\langle 1,2\rangle$ is shorthand for the symplectic inner product of the Pauli vectors of $P_1$ and $P_2$:

$$\langle \mathbf{z}_1 \oplus \mathbf{x}_1,\, \mathbf{z}_2 \oplus \mathbf{x}_2 \rangle = \mathbf{z}_1 \cdot \mathbf{x}_2 - \mathbf{x}_1 \cdot \mathbf{z}_2 \pmod{d}$$

with $\oplus$ denoting the direct sum of vectors and $(\cdot)$ an inner product. Thus the Paulis commute iff $\langle 1,2\rangle = 0$. Throughout, two identities will be used. First the *ordered product rule*:

$$(\tau^{-\phi_1} Z^{\mathbf{z}_1} X^{\mathbf{x}_1})(\tau^{-\phi_2} Z^{\mathbf{z}_2} X^{\mathbf{x}_2}) = \tau^{-(\phi_1 + \phi_2 + 2\,\mathbf{x}_1 \cdot \mathbf{z}_2)}\, Z^{\mathbf{z}_1+\mathbf{z}_2}\, X^{\mathbf{x}_1+\mathbf{x}_2}$$

and the *power rule*:

$$\bigl(\tau^{-\phi} Z^{\mathbf{z}} X^{\mathbf{x}}\bigr)^c = \tau^{-(c\,\phi + c(c-1)\,\mathbf{z} \cdot \mathbf{x})}\, Z^{c\mathbf{z}}\, X^{c\mathbf{x}}$$

## Stabilizer Tableau

The state is encoded in a column-major tableau with ZX ordering. Each column is one stabilizer generator $S_j = \tau^{-\phi_j} Z^{\mathbf{z}_j} X^{\mathbf{x}_j}$:

$$T = \left[ \begin{array}{c|c|c} \phi_1 & \cdots & \phi_\ell \\ \hline z_{1,1} & \cdots & z_{1,\ell} \\ \vdots & \ddots & \vdots \\ z_{n,1} & \cdots & z_{n,\ell} \\ \hline x_{1,1} & \cdots & x_{1,\ell} \\ \vdots & \ddots & \vdots \\ x_{n,1} & \cdots & x_{n,\ell} \end{array} \right] \in \mathbb{Z}_D^{1 \times \ell} \oplus \mathbb{Z}_d^{n \times \ell} \oplus \mathbb{Z}_d^{n \times \ell}, \qquad \ell \leq 2n$$

For composite $d$, the number of generators $\ell$ can exceed $n$ (up to $2n$) after measurements. For prime $d$, $\ell = n$ always.

**Initial state** $|0\rangle^{\otimes n}$: $\ell = n$, with $(\boldsymbol\phi, \mathbf{Z}, \mathbf{X}) = (\mathbf{0}, \mathbf{1}_n, \mathbf{0}_n)$.

## Generator Arithmetic

Two column-level operations are needed. To *multiply* generator $k$ by $c$ copies of generator $p$ (implementing $S_k \leftarrow S_p^c \cdot S_k$):

$$\phi_k \leftarrow c\,\phi_p + \phi_k + 2c\,(\mathbf{x}_p \cdot \mathbf{z}_k) + c(c{-}1)\,(\mathbf{z}_p \cdot \mathbf{x}_p)$$
$$\mathbf{z}_k \leftarrow c\,\mathbf{z}_p + \mathbf{z}_k, \qquad \mathbf{x}_k \leftarrow c\,\mathbf{x}_p + \mathbf{x}_k$$

where $\mathbf{z}_k$ in the cross term is the value *before* update.

To *scale* generator $p$ by $c$ (implementing $S_p \leftarrow S_p^c$), the power rule gives:

$$\phi_p \leftarrow c\,\phi_p + c(c{-}1)\,(\mathbf{z}_p \cdot \mathbf{x}_p), \qquad \mathbf{z}_p \leftarrow c\,\mathbf{z}_p, \qquad \mathbf{x}_p \leftarrow c\,\mathbf{x}_p$$

## Unimodular Column Transformation

When we need to change basis among the generators (e.g. during measurement), we apply a unimodular matrix $V$ to the tableau columns. Column $j$ of the result represents the ordered product $\prod_k S_k^{V_{kj}}$. The Z and X blocks transform linearly:

$$\mathbf{Z}' = \mathbf{Z}V, \qquad \mathbf{X}' = \mathbf{X}V$$

and the phases pick up cross terms from the ordering:

$$\phi'_j = \sum_k V_{kj}\,\phi_k \;+\; \sum_k V_{kj}(V_{kj}{-}1)\, G_{kk} \;+\; 2\!\sum_{m < k} V_{mj}\, V_{kj}\, G_{km}$$

where $G_{ab} = \mathbf{z}_a \cdot \mathbf{x}_b$ is the Gram matrix of the original (pre-transformation) generators.

## Gate Conjugation

Each Clifford gate is applied by conjugating every column of the tableau: $S_j \leftarrow U S_j U^\dagger$. All phase corrections below use Z/X values *before* the update.

| Gate | Z/X Update (qudit $q$ or pair $c,t$) | $\Delta\phi$ |
|---|---|---|
| $S_q$ | $z_q \leftarrow z_q + x_q$ | See below |
| $F_q$ | $(z_q, x_q) \leftarrow (x_q, -z_q)$ | $-2\,z_q\, x_q$ |
| $F_q^\dagger$ | $(z_q, x_q) \leftarrow (-x_q, z_q)$ | $-2\,z_q\, x_q$ |
| $M_{a,q}$ | $z_q \leftarrow a^{-1} z_q,\; x_q \leftarrow a\, x_q$ | $0$ |
| $\mathrm{SUM}_{c \to t}$ | $x_t \mathrel{+}= x_c,\; z_c \mathrel{-}= z_t$ | $0$ |
| $\mathrm{CZ}_{1,2}$ | $z_1 \mathrel{+}= x_2,\; z_2 \mathrel{+}= x_1$ | $2\,x_1\, x_2$ |
| $X_q$ | — | $+2\, z_q$ |
| $Z_q$ | — | $-2\, x_q$ |
| $\mathrm{SWAP}_{q_1,q_2}$ | swap rows $q_1 \leftrightarrow q_2$ | $0$ |

The Fourier gate $F_q$ swaps the Z and X roles (with a sign flip and phase correction), the multiplier gate $M_{a,q}$ rescales by $a \in \mathbb{Z}_d^\times$, and the two-qudit gates SUM and CZ act on their respective control/target pairs.

The phase gate on qudit $q$ is defined as

$$S = \begin{cases} \displaystyle\sum_{q=0}^{d-1} \tau^{q^2} |q\rangle\langle q| & d \text{ even} \\[6pt] \displaystyle\sum_{q=0}^{d-1} \omega^{q(q-1)/2} |q\rangle\langle q| & d \text{ odd} \end{cases}$$

Both forms satisfy $SZS^\dagger = Z$ (Z is unchanged) and map $X \mapsto ZX$ up to a phase absorbed into the definition. Per column:

$$z_q \leftarrow z_q + x_q, \qquad \Delta\phi = \begin{cases} x_q^2 & d \text{ even} \\ x_q(x_q + 1) & d \text{ odd} \end{cases}$$

## Measurement of a Pauli Operator $P$

We want to measure a Pauli operator $P = \tau^{-\delta}\, Z^{\mathbf{a}} X^{\mathbf{b}}$. For a computational-basis measurement of qudit $r$, this is just $Z_r$: set $\mathbf{a} = \hat{e}_r,\ \mathbf{b} = \mathbf{0},\ \delta = 0$.

### Step 1 — Commutation Vector

Compute the symplectic inner product of $P$ against each stabilizer generator:

$$c_k = \mathbf{a} \cdot \mathbf{x}_k - \mathbf{b} \cdot \mathbf{z}_k \pmod{d}$$

Set $\eta = \gcd(d, c_1, \ldots, c_\ell)$.

### Step 2 — Isolate via SNF

Compute the Smith Normal Form of the row vector $C = [c_1 \;\cdots\; c_\ell]$ mod $d$, yielding a unimodular $V$ such that

$$CV = [\eta \quad 0 \quad \cdots \quad 0]$$

Apply $V$ to the full tableau using the unimodular column transformation. After this change of basis, only column 1 has a nonzero commutation value $\eta$. Set $s = d/\eta$.

### Step 3 — Eigenvalue of $P^s$

The operator $P^s$ commutes with the entire stabilizer group (since $s \cdot \eta \equiv 0$ mod $d$), so the stabilizer already pins down its eigenvalue. Compute $P^s$ via the power rule:

$$\phi_{P^s} = s\,\delta + s(s{-}1)\,(\mathbf{a} \cdot \mathbf{b}), \qquad \mathbf{z}_{P^s} = s\,\mathbf{a}, \qquad \mathbf{x}_{P^s} = s\,\mathbf{b}$$

To find the eigenvalue, decompose $P^s$ over the generators by solving $G \mathbf{c} \equiv \mathbf{v}_{P^s} \pmod{d}$, where $G$ is the $2n \times \ell$ Weyl block and $\mathbf{v}_{P^s} = \mathbf{z}_{P^s} \oplus \mathbf{x}_{P^s}$. The SNF of $G$ yields $UGV = D$, reducing to a diagonal system $D\mathbf{y} \equiv U\mathbf{v}_{P^s} \pmod{d}$ with $\mathbf{c} = V\mathbf{y}$. Record the first invariant factor $f_1 = D_{11}$.

Evaluate the ordered product $Q = \prod_k S_k^{c_k}$ using the product and power rules. The Weyl part of $Q$ equals $P^s$; the phase difference $t = \phi_Q - \phi_{P^s} \pmod{D}$ encodes the eigenvalue.

### Step 4 — Sample the Outcome

The measurement outcome $h$ satisfies

$$2\,s\,h \equiv t \pmod{D}$$

Sample $h$ uniformly from the solution set $h_0 + \eta\,\mathbb{Z}_d$, which has $\eta$ equally likely outcomes. When $\eta = d$, there is exactly one solution (deterministic).

### Step 5 — Collapse the Tableau

Update the tableau to reflect the post-measurement state. Scale column 1 by $s$ (via generator scaling), which zeroes out its commutation with $P$, and insert the measurement result $R = \tau^{-(\delta + 2h)}\, Z^{\mathbf{a}}\, X^{\mathbf{b}}$ as a new generator.

If $S_1^s = \mathbb{1}$ (the scaled column is all zeros mod $d$), drop it. Otherwise, check whether $f_1 \mid s$. If so, $S_1^s$ is already in the span of the other generators and can be dropped; if not, keep it (this increases $\ell$ by 1, up to a maximum of $2n$).

When $\eta = d$, column 1 already commutes with $P$, the scaling is trivial ($s = 1$), and $P$ is already in the stabilizer group. The tableau is unchanged.

## Appendix: Syndrome Extraction Circuit

Alternatively, $P = \bigotimes_j X_j^{a_j} Z_j^{b_j}$ can be measured indirectly via an ancilla $r$ in $|0\rangle$ using the circuit

$$\Lambda P = \left(S_r^{-\mathbf{a}\cdot\mathbf{b}} \otimes \mathbf{1}\right) \prod_j \mathrm{CZ}_{r,j}^{a_j} \prod_j \mathrm{SUM}_{r \to j}^{b_j}$$

sandwiched by $F_r, F_r^\dagger$ on the ancilla, followed by measuring $Z_r$. The direct protocol above avoids this decomposition.

## References

1. S. Aaronson and D. Gottesman, "Improved Simulation of Stabilizer Circuits," *Phys. Rev. A*, vol. 70, no. 5, 052328, 2004. doi:10.1103/PhysRevA.70.052328.

2. N. de Beaudrap, "A linearized stabilizer formalism for systems of finite dimension," *Quantum Inf. Comput.*, vol. 13, no. 1–2, pp. 73–115, 2013. doi:10.5555/2481591.2481597.

3. C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum*, vol. 5, p. 497, 2021. doi:10.22331/q-2021-04-06-433.
