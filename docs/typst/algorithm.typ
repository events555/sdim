#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Qudit Stabilizer Simulation Reference],
  authors: (
    (
      name: "Steven Nguyen",
      department: [Department of Electrical and Computer Engineering],
      organization: [Rutgers University],
      email: "steven.n.nguyen@rutgers.edu",
    ),
  ),
  abstract: [
    A self-contained reference for tableau-based simulation of Clifford circuits on qudits of prime or composite dimension $d$. Covers the stabilizer representation, gate conjugation rules, and a measurement protocol using Smith Normal Form reduction.
  ],
  index-terms: ("Clifford Circuit", "Smith Normal Form", "Tableau Representation", "Qudit"),
)

#show figure.caption: set align(center)
#set math.cases(gap: 0.45em)
#show math.equation: it => {
  show "!": h(1pt)
  it
}
= Representation

#h(1em) A Pauli operator on $n$ qudits of dimension $d$ is defined as
$ P = tau^(-phi) Z^(bold(upright(z))) X^(bold(upright(x))), quad bold(upright(z)), bold(upright(x)) in ZZ_d^n, quad phi in ZZ_D $
where $omega = e^(2 pi i \/ d)$, $tau = e^(i pi(d^2+1)\/d)$, $tau^2 = omega$, and
$ D = cases(
  d quad & d "odd",
  2d quad & d "even".
) $
Two Paulis thus satisfy the commutation relation
$ P_1 P_2 = tau^(2 chevron.l 1,2 chevron.r) P_2 P_1, $ where $chevron.l 1,2 chevron.r$ is shorthand for the symplectic inner product of the Pauli vectors of $P_1$ and $P_2$:
$ chevron.l 1,2 chevron.r :&= chevron.l bold(upright(z))_1 ! plus.o bold(upright(x))_1, bold(upright(z))_2 ! plus.o bold(upright(x))_2 chevron.r \
 &= bold(upright(z))_1 dot bold(upright(x))_2 - bold(upright(x))_1 dot bold(upright(z))_2 #h(4pt) (mod d) $
with $plus.o$ denoting the direct sum of vectors and $(dot.c)$ an inner product. Thus the Paulis commute iff $chevron.l 1,2 chevron.r=0$. Throughout, two identities will be used. First the _ordered product rule_:
$ (tau^(-phi_1) Z^(bold(upright(z))_1) X^(bold(upright(x))_1)) (tau^(-phi_2) Z^(bold(upright(z))_2) X^(bold(upright(x))_2)) \
 = tau^(-(phi_1 + phi_2 + 2 bold(upright(x))_1 dot bold(upright(z))_2)) Z^(bold(upright(z))_1 + bold(upright(z))_2) X^(bold(upright(x))_1 + bold(upright(x))_2) $
and the _power rule_:
$ (tau^(-phi) Z^(bold(upright(z))) X^(bold(upright(x))))^c = tau^(-(c phi + c(c-1) bold(upright(z)) dot bold(upright(x)))) Z^(c bold(upright(z))) X^(c bold(upright(x))). $

The state is encoded in a column-major tableau with ZX ordering shown in @tableau. Each column is one stabilizer generator $S_j = tau^(-phi_j) Z^(bold(upright(z))_j) X^(bold(upright(x))_j)$. Equivalently,
$ T in ZZ_D^(1 times ell) plus.o ZZ_d^(n times ell) plus.o ZZ_d^(n times ell), quad ell lt.eq 2n, $
Writing those blocks explicitly:

#figure({
  let pad = -.5pt
  let m = $mat(
    augment: #(hline: (2, 7)),
    delim: "[",
    gap: #0pt,
    phi_1, dots.c, phi_ell;
    #v(pad), , ;
    #v(pad), , ;
    z_(1,1), dots.c, z_(1,ell);
    dots.v, , dots.v;
    z_(n,1), dots.c, z_(n,ell);
    #v(pad), , ;
    #v(pad), , ;
    x_(1,1), dots.c, x_(1,ell);
    dots.v, , dots.v;
    x_(n,1), dots.c, x_(n,ell);
  )$

  context {
    let total = measure(m).height
    let phase-h = total / 6
    let block-h = total * 5 / 11

    let brace-label(ht, label) = box(
      height: ht,
      align(horizon, {
        $lr(size: #ht, brace.r)$
        h(4pt)
        label
      }),
    )
    grid(
      columns: (auto, auto, auto),
      column-gutter: 6pt,
      align: (right + horizon, center, left + horizon),
      stack(
        dir: ttb,
        box(height: phase-h, align(right + horizon, move(dy: -4pt, [phase]))),
        box(height: block-h, align(right + horizon, $bold(Z)$)),
        box(height: block-h, align(right + horizon, $bold(X)$)),
      ),
      m,
      stack(
        dir: ttb,
        box(height: phase-h, align(left + horizon, move(dy: -4pt, $in ZZ_D^(1 times ell)$))),
        brace-label(block-h, $n "rows" in ZZ_d^(n times ell)$),
        brace-label(block-h, $n "rows" in ZZ_d^(n times ell)$),
      ),
    )
  }
}, caption: [Stabilizer tableau layout.]) <tableau>

For the initial state $|0 chevron.r^(times.o n)$ with $ell = n$, we set the tableau to have the following form:
$ T: quad bold(phi) = bold(0), quad bold(Z) = bold(1)_n, quad bold(X) = bold(0)_n $
For composite $d$, the number of generators $ell$ can grow beyond $n$ (up to $2n$) after measurements. For prime $d$, $ell = n$ always.

== Generator Arithmetic

#h(1em)Two column-level operations are needed. To _multiply_ generator $k$ by $c$ copies of generator $p$ (implementing $S_k arrow.l S_p^c S_k$), apply
$ phi_k arrow.l & c phi_p + phi_k + 2c (bold(upright(x))_p dot bold(upright(z))_k) \ & + c(c-1)(bold(upright(z))_p dot bold(upright(x))_p) $
$ bold(upright(z))_k arrow.l c bold(upright(z))_p + bold(upright(z))_k, quad bold(upright(x))_k arrow.l c bold(upright(x))_p + bold(upright(x))_k $
where $bold(upright(z))_k$ in the cross term is the value _before_ update.

To _scale_ generator $p$ by $c$ (implementing $S_p arrow.l S_p^c$), the power rule gives
$ phi_p arrow.l c phi_p + c(c-1)(bold(upright(z))_p dot bold(upright(x))_p) $
$ bold(upright(z))_p arrow.l c bold(upright(z))_p, quad bold(upright(x))_p arrow.l c bold(upright(x))_p. $

== Unimodular Column Transformation

#h(1em)When we need to change basis among the generators (e.g.~during measurement), we apply a unimodular matrix $V$ to the tableau columns. Column $j$ of the result represents the ordered product $product_k S_k^(V_(k j))$. The Z and X blocks transform linearly:
$ bold(Z)' = bold(Z) V, quad bold(X)' = bold(X) V $
and the phases pick up cross terms from the ordering:
$ phi'_j = & sum_k V_(k j) phi_k + sum_k V_(k j)(V_(k j) - 1) G_(k k) \ & + 2 sum_(m < k) V_(m j) V_(k j) G_(k m) $
where $G_(a b) = bold(upright(z))_a dot bold(upright(x))_b$ is the Gram matrix of the original (pre-transformation) generators.

= Gate Conjugation

#h(1em)Each Clifford gate is applied by conjugating every column of the tableau: $S_j arrow.l U S_j U^dagger$. All phase corrections below use Z\/X values _before_ the update.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.4pt,
    inset: 5pt,
    table.header[*Gate*][*Z\/X Update*][$Delta phi$],
    [$S_q$], [$z_q arrow.l z_q + x_q$], [Refer to @phase],
    [$F_q$], [$(z_q, x_q) arrow.l (x_q, -z_q)$], [$-2 z_q x_q$],
    [$F_q^dagger$], [$(z_q, x_q) arrow.l (-x_q, z_q)$], [$-2 z_q x_q$],
    [$M_(a,q)$], [$z_q arrow.l a^(-1) z_q, #h(3pt) x_q arrow.l a x_q$], [$0$],
    [$"SUM"_(c arrow.r t)$], [$x_t arrow.l x_t + x_c, #h(3pt) z_c arrow.l z_c - z_t$], [$0$],
    [$"CZ"_(1,2)$], [$z_1 arrow.l z_1 + x_2, #h(3pt) z_2 arrow.l z_2 + x_1$], [$2 x_1 x_2$],
    [$X_q$], [---], [$+2 z_q$],
    [$Z_q$], [---], [$-2 x_q$],
    [$"SWAP"_(q_1, q_2)$], [swap rows $q_1 arrow.l.r q_2$], [$0$],
  ),
  caption: [Clifford gate conjugation rules.],
)

The Fourier gate $F_q$ swaps the Z and X roles (with a sign flip and phase correction), the multiplier gate $M_(a,q)$ rescales by $a in ZZ_d^times$, and the two-qudit gates SUM and CZ act on their respective control/target pairs.

The phase gate on qudit $q$ is defined as
$ S = cases(
  display(sum_(q=0)^(d-1) tau^(q^2) |q chevron.r chevron.l q|) & d "even",
  display(sum_(q=0)^(d-1) omega^(q(q-1)\/2) |q chevron.r chevron.l q|) & d "odd".
) $
Both forms satisfy $S Z S^dagger = Z$ (Z is unchanged) and map $X arrow.r.bar Z X$ up to a phase absorbed into the definition. Per column:
$ z_q arrow.l z_q + x_q, quad Delta phi = cases(
  x_q^2 & d "even",
  x_q (x_q + 1) & d "odd".
) $<phase>

= Measurement

We want to measure a Pauli operator $P = tau^(-delta) Z^(bold(a)) X^(bold(b))$. For a computational-basis measurement of qudit $r$, this is just:
$ Z_r : "set" bold(a) = hat(e)_r,#h(0.5em) bold(b) = bold(0),#h(0.5em) delta = 0. $

The following procedure handles both random and deterministic outcomes uniformly.

*Step 1 --- Commutation Vector.*
Compute the symplectic inner product of $P$ against each stabilizer generator:
$ c_k = bold(a) dot bold(upright(x))_k - bold(b) dot bold(upright(z))_k quad (mod d). $
Set $eta = gcd(d, c_1, dots, c_ell)$.

*Step 2 --- Isolate via SNF.*
Compute the Smith Normal Form of the row vector $C = [c_1 space dots.c space c_ell]$ mod $d$, yielding a unimodular $V$ such that
$ C V = [eta quad 0 quad dots.c quad 0]. $
Apply $V$ to the full tableau using the unimodular column transformation. After this change of basis, only column~1 has a nonzero commutation value $eta$. Set $s = d \/ eta$.

*Step 3 --- Eigenvalue of $P^s$.*
The operator $P^s$ commutes with the entire stabilizer group (since $s dot eta equiv 0$ mod $d$), so the stabilizer already pins down its eigenvalue. Compute $P^s$ via the power rule:
$ phi_(P^s) = s delta + s(s-1)(bold(a) dot bold(b)) $
$ bold(upright(z))_(P^s) = s bold(a), quad bold(upright(x))_(P^s) = s bold(b). $
To find the eigenvalue, decompose $P^s$ over the generators by solving $G bold(c) equiv bold(v)_(P^s) #h(4pt) (mod d)$, where $G$ is the $2n times ell$ Weyl block and $bold(v)_(P^s) = bold(upright(z))_(P^s) plus.o bold(upright(x))_(P^s)$. The SNF of $G$ yields $U G V = D$, reducing to a diagonal system $D bold(y) equiv U bold(v)_(P^s) #h(4pt) (mod d)$ with $bold(c) = V bold(y)$. Record the first invariant factor $f_1 = D_(1 1)$.

Evaluate the ordered product $Q = product_k S_k^(c_k)$ using the product and power rules. The Weyl part of $Q$ equals $P^s$; the phase difference $t = phi_Q - phi_(P^s) #h(4pt) (mod D)$ encodes the eigenvalue.

*Step 4 --- Sample the Outcome.*
The measurement outcome $h$ satisfies
$ 2 s h equiv t quad (mod D). $
Sample $h$ uniformly from the solution set $h_0 + eta ZZ_(d \/ eta)$, which has $d \/ eta = s$ equally likely outcomes. When $eta = d$ (so $s = 1$), there is exactly one solution (deterministic).

*Step 5 --- Collapse the Tableau.*
Update the tableau to reflect the post-measurement state. Scale column~1 by $s$ (via generator scaling), which zeroes out its commutation with $P$, and insert the measurement result $R = tau^(-(delta + 2h)) Z^(bold(a)) X^(bold(b))$ as a new generator.

If $S_1^s = bb(1)$ (the scaled column is all zeros mod $d$), drop it. Otherwise, check whether $f_1 | s$. If so, $S_1^s$ is already in the span of the other generators and can be dropped; if not, keep it (this increases $ell$ by 1, up to a maximum of $2n$).

When $eta = d$, column~1 already commutes with $P$, the scaling is trivial ($s = 1$), and $P$ is already in the stabilizer group. The tableau is unchanged.

= References

+ S. Aaronson and D. Gottesman, "Improved Simulation of Stabilizer Circuits," _Phys. Rev. A_, vol. 70, no. 5, 052328, 2004. doi:10.1103/PhysRevA.70.052328.

+ N. de Beaudrap, "A linearized stabilizer formalism for systems of finite dimension," _Quantum Inf. Comput._, vol. 13, no. 1–2, pp. 73–115, 2013. doi:10.5555/2481591.2481597.

+ C. Gidney, "Stim: a fast stabilizer circuit simulator," _Quantum_, vol. 5, p. 497, 2021. doi:10.22331/q-2021-04-06-433.

= Appendix: Syndrome Extraction Circuit

Alternatively, $P = times.o.big_j X_j^(a_j) Z_j^(b_j)$ can be measured indirectly via an ancilla $r$ in $|0 chevron.r$ using the circuit
$ Lambda P = (S_r^(-bold(a) dot bold(b)) times.o bold(1)) product_j "CZ"_(r,j)^(a_j) product_j "SUM"_(r arrow.r j)^(b_j) $
sandwiched by $F_r, F_r^dagger$ on the ancilla, followed by measuring $Z_r$. The direct protocol above avoids this decomposition.
