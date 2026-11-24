# sdim

## Project Overview

Despite the growing research interest in qudits as an alternative way to scale certain quantum architectures, no publicly available stabilizer circuit simulators for **qudits** (multi-level quantum systems) are available. The two most prominent ones are [Cirq](https://quantumai.google/cirq/build/qudits) which is a statevector simulation and [True-Q™](https://trueq.quantumbenchmark.com/index.html) which is a licensed program.

The following are relevant details for the project:
- Supports **only Clifford** operations. 
- Works in any dimension below 2**31. Prime dimensions use an extended tableau; composite dimensions use a Weyl tableau whose measurements are computed exactly, see the [notes on composite dimensions](https://github.com/events555/sdim/blob/main/sdim/tableau/COMPOSITE.md).
- Does not currently support `.stim` circuit notation, only a variant based on Scott Aaronson's original `.chp`
- Runs on Python 3.11 to 3.14, tested with the newest releases of its dependencies and with the oldest ones it supports.

## Project Installation
You can install the `sdim` Python module directly from [PyPI](https://pypi.org/project/sdim/) using `pip install sdim`. It needs Python 3.11 or newer and installs numpy, sympy, numba and cirq-core. sdim only uses cirq-core; install `cirq` as well if you want Cirq's hardware vendor packages.

The first run after installing compiles sdim's numba kernels, a few seconds for each part of sdim you use (up to about 20 seconds for all of them). Later runs load them from numba's cache.

## How to use sdim?
Take a look at the notebooks in [`examples/`](https://github.com/events555/sdim/tree/main/examples) for in-depth examples.
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

Besides the Clifford gates (`H`, `P`, `CNOT`, `CZ`, `SWAP`, `X`, `Z` and their inverses), `M` measures in the computational basis, `M_X` in the X basis and `RESET` returns a qudit to |0>. `MUL` multiplies a qudit by a scalar `a` coprime to d, for example `circuit.add_gate('MUL', 0, a=2)` at d = 5. Negative qudit indices count from the end, like Python lists.

## Detector error models
`sdim.dem` turns a circuit with noise, detectors and logical observables into a detector error model (DEM). Each noise gate becomes one error mechanism, so the model does not grow with the qudit dimension. A mechanism adds a random combination of its generators, the changes that the gate's unit X and Z faults make to the detectors and observables. At prime d, mechanisms with a single generator (from N1 `'f'` and `'p'` gates, or from any gate only one of whose unit faults reaches a detector or observable) are merged when their generators are multiples of each other; mechanisms with more generators, such as most N1 `'d'` and N2 gates, are kept as they are. The tests run it up to d = 1000003.
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
detectors, observables = dem.sample(100_000) # int64 arrays of values mod d, shape (shots, detectors)
dem.write_to_file('model.qdem')

sampler = dem.compile_sampler(seed=7) # Packs the model once, for drawing many small batches
detectors, observables = sampler.sample(256)
```

Noise gates take these parameters:
- `N1`: `noise_channel` (`'d'` depolarizing, `'f'` flip, `'p'` phase) and `prob`.
- `N2`: `prob` for two-qudit depolarizing. A full `prob_dist` over all d^4 Paulis also works for small d, but `sdim.dem` rejects it. Use the older `sdim.dem_legacy` model for those circuits.

The dimension has to be prime, every detector and observable must be deterministic without noise, and noise can go up to full mixing (for example `prob <= 1 - 1/d` for flip errors). `from_circuit` checks all three and raises a `ValueError` otherwise. For small d, `dem.to_lines()` splits every mechanism into independent line mechanisms, each adding a uniformly random multiple of a single vector. At d = 2 these match stim's error models for `DEPOLARIZE1` and `DEPOLARIZE2` up to a factor of two: a line adds a uniformly random multiple of its vector, zero included, so a line printed with probability p flips its detectors with probability p/2, which is the number stim prints. The docstring at the top of [`sdim/dem.py`](https://github.com/events555/sdim/blob/main/sdim/dem.py) explains the math and the file format.

`Program(circuit).simulate(shots=n, raw_detector_output=True)` samples detectors with the Pauli frame simulator instead. Its arrays are indexed the other way round, (detector, shot), and cover the n - 1 shots after the noiseless reference shot.

## shepherd: large-scale studies
shepherd is sdim's large-scale study tool. It builds qudit error-correction experiments for codes over a prime dimension q, turns them into detector error models with `sdim.dem`, samples and decodes them in parallel, keeps the statistics in resumable results files, and turns those into threshold plots and crossing estimates. It lives in [`shepherd/`](https://github.com/events555/sdim/tree/main/shepherd), next to the `sdim` package, as its own project: sdim is the core simulator, shepherd the task automation on top of it.

shepherd is not on PyPI yet. Install it from a checkout of this repository, after sdim:
```bash
pip install -e .                    # sdim
pip install -e "./shepherd[plot]"   # shepherd, with plotting; add mwpm for the pymatching decoders:
                                    #   pip install -e "./shepherd[plot,mwpm]"
cd shepherd && python -m pytest     # shepherd's tests (sdim's run from the repository root)
```

It can then be invoked with the `shepherd` command (the same as `python -m shepherd`), or from Python, where the main names import from the package directly: `from shepherd import Study, run_study, NoiseModel, surface_code`.

### Running a threshold study
```bash
shepherd list                                   # built-in studies
shepherd run surface_cc --dry-run               # the tasks, and what is already done
shepherd run surface_cc --workers 8 --max-errors 50   # quick coarse pass
shepherd plot surface_cc                        # look; adjust the grid if needed
shepherd run surface_cc --workers 8             # full budget (tops up the coarse pass)
shepherd status surface_cc                      # per-task shots, errors, LER
```

- Results go to `results/<study>.jsonl`; plots, point tables and crossing tables go to `plots/<study>/`, one PNG per (family, noise, q, decoder).
- `run` can be stopped at any time and restarted; it resumes from the results file.
- `--where q=3 --where n=12,16` restricts a run to matching tasks (any metadata key, or `decoder`). `--ps 0.02,0.03` replaces the study's noise strengths. `--max-shots` and `--max-errors` override the study's budget.
- Run one process per results file at a time.

### What the logical error rate means
The `"ler"` experiment is a **Z-basis memory**. The data starts in the product state `|0...0>` with no encoding circuit, Z checks are measured for the given number of rounds, and the data is read out in the Z basis. A shot fails if the predicted shift of *any* of the k Z logicals is wrong. So:

- Only logical shift (X-type) errors are counted. Logical phase errors need an X-basis memory, which is not implemented yet. Label results "Z-basis memory".
- Starting from `|0...0>` instead of an encoded `|0-bar>` is exact here, because only Z-type quantities are measured. This is the standard memory-experiment convention; the argument and references are in the docstring of [`benchmarking/ler.py`](https://github.com/events555/sdim/blob/main/shepherd/src/shepherd/benchmarking/ler.py).
- For k > 1 the rate is per shot ("any logical wrong") and tends to 1 - 1/q^k at high p.

### The pieces
Modules are in [`shepherd/src/shepherd/`](https://github.com/events555/sdim/tree/main/shepherd/src/shepherd).

| Module | What it is |
|---|---|
| `qecc.py`, `codes/` | `QECC` registry and `PauliString`. A code supplies its stabilizers; logical operators are derived from them by default. `codes/css.py`: `CSSCode` (any H_X, H_Z mod q), the hypergraph product, and the repetition, surface and LaCross families. Also `five_qudit`, `rotated_memory_z` (work in progress: no stabilizers yet) and the placeholder `qrm_self_dual` |
| `experiment.py`, `benchmarking/` | `Experiment` registry; `"ler"` builds the noiseless Z-memory circuit of a CSS code for 0 (code capacity) or more rounds; `"lrb"` and `"msd"` are placeholders |
| `noise_model.py` | `NoiseModel`: a JSON-serializable noise spec, `apply(circuit)` to make a circuit noisy, and `code_capacity`, `phenomenological`, `circuit_level` |
| `study.py`, `studies.py` | `Study` (codes x noise strengths x decoders, with per-task budgets), `run_study`, and the built-in studies |
| `task.py`, `collect.py`, `stats.py` | one task (circuit + decoder), the parallel resumable collector, and its statistics |
| `decoder.py`, `decoders/` | `Decoder` registry: `union_find` (qudit union-find, hyperedges included), `pymatching`, `pymatching_keppens`, `none`; `trellis` and `perfect` are registered placeholders that raise `NotImplementedError` |
| `analysis.py` | results to tables, threshold crossings, and plots |

### Your own study
```python
# mystudy.py
from shepherd.study import Study
from shepherd.noise_model import NoiseModel
from shepherd.codes.css import CSSCode

S = Study(
    name="my_code_phenom", description="my code, phenomenological",
    experiment="ler", experiment_params={"rounds": "d"},
    codes=lambda: [CSSCode(3, H_X, H_Z, distance=6, label="mine_n100", family="mine")],
    noise=NoiseModel.phenomenological,            # (q, p) -> NoiseModel
    ps=[0.005, 0.01, 0.02],
    decoders=["union_find"], max_shots=50_000, max_errors=300)
```

```bash
shepherd run mystudy.py:S --workers 8
```

`noise` can be any function returning a `NoiseModel`, including one loaded from JSON (`NoiseModel.load("model.json")`). Each task's metadata records the noise model's name and fingerprint, so results from different models never mix.

### Results that must not be mixed
A task's id (its `strong_id`) hashes its detector error model, its decoder and the decoder's `version`, its postselection and its metadata. Results are added together only when all of these match, so:

- bump a decoder's `version` whenever a change alters its output;
- a new sdim release that changes DEM text starts new ids, and so new results;
- adding a metadata key orphans the results collected without it.

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


