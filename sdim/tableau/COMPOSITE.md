## Issues with Composite Dimensions

In general, it should be possible to solve for measurement outcomes quickly and efficiently in composite dimensions. There are known algorithms that will solve system of linear congruences in time bound by $O(n^3)$ for an $n\times n$ matrix.

Furthermore, a system of linear congruences can be adapted to a system of linear Diophantine equations. 

The immediate hurdle is that the stabilizer formalism requires using only unimodular column operations, but Gaussian-Jordan elimination does not guarantee reducing the measurement operator over composite dimensions. 

One can then turn towards algorithms to compute the Smith-Normal form or Hermite-Normal form. These present further issues, where the Smith-Normal form requires unimodular row operations (which are not allowed when working with stabilizer generators) and bounded-term algorithms for the HNF are not readily implemented. There is, however, an available Python package ([Diophantine](https://pypi.org/project/Diophantine/)) that can solve a system of linear Diophantine equations.

This solver suffers from a common issue with simple to implement solvers—intermediate values may grow exponentially. This leads to integer overflow issues when working with fixed-point representations like NumPy. Therefore the solver relies strongly on [SymPy](https://www.sympy.org/en/index.html) for infinite-precision arithmetic, which is *very, very* slow.

### Present implementation

`WeylTableau.measure_z` (`tableau_composite.py`) no longer uses the Diophantine solver. It eliminates one row at a time over $\mathbb{Z}_d$ with operations that keep the generated group: a generator is multiplied by a power of another, and when no single entry of the row has the gcd of the whole row and $d$ as its gcd with $d$, a pair of generators $(a, b)$ with row entries $(a_r, b_r)$ is replaced by $(a^x b^y, a^{-b_r/g} b^{a_r/g})$, where $x a_r + y b_r = g$ (Bezout's identity). That replacement has determinant 1, so the pair generates the same group. Once the pivot $P$ of a row is taken out, $P^{d/\gcd(P_r, d)}$ is added back, so the remaining generators span exactly the elements that vanish in that row. Every value is reduced mod $d$ (phases) or mod the order $2d$ or $d$ (vectors) after each step, so nothing grows. The arithmetic is int64 when it provably cannot overflow, and Python integers otherwise (large $d$).

The measurement is exact, and `tests/test_composite_tableau.py` checks it against a statevector simulation. `program.simulate(exact=True)` is still accepted for compatibility, but it has no effect.


### References

<a id="1">[1]
</a> de Beaudrap, Niel. “On the Complexity of Solving Linear Congruences and Computing Nullspaces modulo a Constant.” Chicago Journal of Theoretical Computer Science, vol. 19, no. 1, 2013, pp. 1–18. arXiv.org, https://doi.org/10.4086/cjtcs.2013.010.

<a id="2">[2]
</a> Antonio Hernando, Luis de Ledesma, Luis M. Laita,
Showing the non-existence of solutions in systems of linear Diophantine equations,
Mathematics and Computers in Simulation,
Volume 79, Issue 11,
2009,
Pages 3211-3220,
ISSN 0378-4754,
https://doi.org/10.1016/j.matcom.2009.03.004.

<a id="3">[3]
</a> Arne Storjohann. 1996. Near optimal algorithms for computing Smith normal forms of integer matrices. In Proceedings of the 1996 international symposium on Symbolic and algebraic computation (ISSAC '96). Association for Computing Machinery, New York, NY, USA, 267–274. https://doi.org/10.1145/236869.237084

<a id="4">[4]
</a> Extended gcd and Hermite normal form algorithms via lattice basis reduction, G. Havas, B.S. Majewski, K.R. Matthews, Experimental Mathematics, Vol 7 (1998) 125-136

<a id="5">[5]
</a> Koukouvinos, C.; Mitrouli, M.; and Seberry, Jennifer: Numerical algorithms for the computation of the
Smith normal form of integral matrices, 1998.
https://ro.uow.edu.au/infopapers/1156 