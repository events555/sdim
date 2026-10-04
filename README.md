# sdim

## Project Overview

Despite the growing research interest in qudits as an alternative way to scale certain quantum architectures, no publicly available stabilizer circuit simulators for **qudits** (multi-level quantum systems) are available. The two most prominent ones are [Cirq](https://quantumai.google/cirq/build/qudits) which is a statevector simulation and [True-Q™](https://trueq.quantumbenchmark.com/index.html) which is a licensed program.

The following are relevant details for the project:
- Supports **only Clifford** operations. 
- **Prime** dimensions are strongly tested while the "fast" solver for composite dimensions is known to have possible errors
    - The issue lies in math implementation details that can be found inside the [markdown](sdim/tableau/COMPOSITE.md) located in `sdim/tableau`
- Does not currently `.stim` circuit notation, only a variant based on Scott Aaronson's original `.chp`

## Project Installation
You can install the `sdim` Python module directly from [PyPI](https://pypi.org/project/sdim/) using `pip install sdim`

## How to use sdim?
Take a look at the Python notebooks for an in-depth examples.
```python
from sdim import Circuit, Program

# Create a new quantum circuit
circuit = Circuit(4, 2) # Create a circuit with 4 qubits and dimension 2

# Add gates to the circuit
circuit.add_gate('H', 0)  # Hadamard gate on qubit 0
circuit.add_gate('CNOT', 0, 1)  # CNOT gate with control on qubit 0 and target on qubit 1
circuit.add_gate('CNOT', 0, [2, 3]) # Short-hand for multiple target qubits, applies CNOT between 0 -> 2 and 0 -> 3
circuit.add_gate('MEASURE', [0, 1, 2, 3]) # Short-hand for multiple single-qubit gates

# Create a program and add the circuit
program = Program(circuit) # Must be given an initial circuit as a constructor argument

# Execute the program
result = program.simulate(show_measurement=True) # Runs the program and prints the measurement results. Also returns the results as a list of MeasurementResult objects.
```

## Detector error models
`sdim.dem` turns a circuit with noise, detectors and logical observables into a detector error model (DEM). Each noise gate becomes one independent error mechanism, so the model is the same size for any qudit dimension. The tests run it up to d = 1000003.
```python
from sdim import Circuit
from sdim.dem import DetectorErrorModel

circuit = Circuit(3, 1000003)
circuit.add_gate('RESET', [0, 1, 2])
circuit.add_gate('N1', [0, 1], noise_channel='f', prob=0.01) # Flip errors on the two data qudits
circuit.add_gate('CNOT', 0, 2)
circuit.add_gate('CNOT_INV', 1, 2) # Ancilla 2 now holds x0 - x1
circuit.add_gate('N2', 1, 2, prob=0.001) # Two-qudit depolarizing
circuit.add_gate('M', [2, 0, 1])
circuit.add_gate('DETECTOR', expr='rec[-3]')
circuit.add_gate('DETECTOR', expr='rec[-3] - rec[-2] + rec[-1]')
circuit.add_gate('LOGICAL_OBSERVABLE', expr='rec[-1]')

dem = DetectorErrorModel.from_circuit(circuit)
detectors, observables = dem.sample(100_000) # int64 arrays of values mod d, one row per shot
dem.write_to_file('model.qdem')
```

Noise gates take these parameters:
- `N1`: `noise_channel` (`'d'` depolarizing, `'f'` flip, `'p'` phase) and `prob`.
- `N2`: `prob` for two-qudit depolarizing. A full `prob_dist` over all d^4 Paulis also works for small d, but `sdim.dem` rejects it. Use the older `sdim.dem_legacy` model for those circuits.

The dimension has to be prime, every detector and observable must be deterministic without noise, and noise can go up to full mixing (for example `prob <= 1 - 1/d` for flip errors). `from_circuit` checks all three and raises a `ValueError` otherwise. For small d, `dem.to_lines()` splits every mechanism into independent single-shift mechanisms. At d = 2 these match stim's error models for `DEPOLARIZE1` and `DEPOLARIZE2`. The docstring at the top of [`sdim/dem.py`](sdim/dem.py) explains the math and the file format.

## Primary References
<a id="1">[1]
</a> Aaronson, Scott, and Daniel Gottesman. “Improved Simulation of Stabilizer Circuits.” Physical Review A, vol. 70, no. 5, Nov. 2004, p. 052328. arXiv.org, https://doi.org/10.1103/PhysRevA.70.052328.

<a id="2">[2]
</a>de Beaudrap, Niel. “A Linearized Stabilizer Formalism for Systems of Finite Dimension.” Quantum Information and Computation, vol. 13, no. 1 & 2, Jan. 2013, pp. 73–115. arXiv.org, https://doi.org/10.26421/QIC13.1-2-6.

<a id="3">[3]
</a>Gottesman, Daniel. “Fault-Tolerant Quantum Computation with Higher-Dimensional Systems.” Chaos, Solitons & Fractals, vol. 10, no. 10, Sept. 1999, pp. 1749–58. arXiv.org, https://doi.org/10.1016/S0960-0779(98)00218-5.

## Secondary References

<a id="1a">[4]
</a>Farinholt, J. M. “An Ideal Characterization of the Clifford Operators.” Journal of Physics A: Mathematical and Theoretical, vol. 47, no. 30, Aug. 2014, p. 305303. arXiv.org, https://doi.org/10.1088/1751-8113/47/30/305303.

<a id="2a">[5]
</a>Greenberg, H. (1971). *Integer Programming*. Academic Press. Chapter 6, Sections 2 and 3.

<a id="3a">[6]
</a>Extended gcd and Hermite normal form algorithms via lattice basis reduction, G. Havas, B.S. Majewski, K.R. Matthews, Experimental Mathematics, Vol 7 (1998) 125-136


