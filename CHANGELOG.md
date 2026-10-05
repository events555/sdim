# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-09-19

### Added

- Initial release of sdim
- Added MEASURE_RESET gate.

### Changed

- Move `sdim` module to parent level
- Update `README.md` to include PyPI information and example circuit.

### Removed

- Remove `/src/` folder


[1.0.0]: https://github.com/events555/sdim/releases/tag/v1.0.0

## [1.1.0] - 2024-09-19

### Added

- Allow `Program.simulate()` to take argument `shots`
- Modified return of `Program.simulate()` to be multidimensional list when `shots>1`

### Changed
- Update `print_measurements()` to support `shots==1` and `shots>1`
- Rename `composite.md` to `COMPOSITE.md`

[1.1.0]: https://github.com/events555/sdim/releases/tag/v1.1.0

## [1.2.0] - 2024-11-19

### Fixed

- Fix incorrect phase calculation on H_INV implementation in `tableau_prime.py`
- Fix incorrect tableau conjugations for P_INV and H_INV for documentation
- Fix implementation in prime dimensions for SWAP gate 

### Added
- Add empty `surface_code.ipynb` to examples
- Add Optional `stabilizer_tableau` to `MeasurementResult` dataclass 
- Add `record_tableau` argument to `simulate()` to record final stabilizer tableau before measurement
- Add unit test for SWAP gate
- Add automated GitHub action workflow for PyPi deployment from `main` and `dev` branch to PyPi and TestPyPi respectively

[1.2.0]: https://github.com/events555/sdim/releases/tag/v1.2.0

## [1.3.0] - 2025-02-18

### Added
- Add noise validation with `test_noise_and_io.py` using PyTest framework.
- Add Cirq definitions for CZ and its inverse gates.
- Add PauliFrame simulator to sample from measurement distribution faster
- Add tests for syndrome extraction
- Add arguments to GateData for noise support
- Add TODO.md
- Add helper functions for integrated PauliFrame simulator (ie. convert to numpy array to MeasurementResult list)

### Changed
- Update `circuit_io.py`  to support named parameters single-qudit gates.
- Included support for inverse symbol of Hadamard in the Cirq circuit diagram.
- Update .gitignore
- Change numpy version requirements

### Fixed
- Fix reset gate not properly working for dimensions greater than 2
- Fix sign convention on CZ, where prior CZ_INV was being treated as CZ and vice versa

[1.3.0]: https://github.com/events555/sdim/releases/tag/v1.3.0

## [1.3.1] - 2025-04-05

### Added
- Add `CITATION.cff`
- Add a test for RESET in the Pauli frame sampler

### Fixed
- Fix RESET in frame mode (`shots > 1`) not recording its measurement round, so later rounds of that qudit were shifted by one and the last round held uninitialized values

[1.3.1]: https://pypi.org/project/sdim/1.3.1/

## [1.3.2] - 2026-01-23

### Changed
- Two-qudit gates keep their keyword parameters in `params`; they were dropped before

### Added
- Add the two-qudit Pauli noise gate `N2` (alias `NOISE2`). Its `prob_dist` gives the probability of each of the d**4 two-qudit Paulis (x1, z1, x2, z2) in lexicographic order. Like `N1`, it is only sampled by the Pauli frame simulator

### Fixed
- Fix `write_circuit` printing "Gate is ..." for every gate

[1.3.2]: https://pypi.org/project/sdim/1.3.2/

## [1.3.3] - 2026-01-26

### Changed
- `N2` raises a ValueError when `prob_dist` does not have d**4 entries, instead of acting as the identity

### Fixed
- Fix the default `N2` distribution, which summed to d**8 instead of 1, so `N2` without `prob_dist` failed in frame mode. It is now uniform over all d**4 two-qudit Paulis

[1.3.3]: https://pypi.org/project/sdim/1.3.3/

## [1.3.4] - 2026-06-01

### Changed
- **Breaking:** `Program.simulate(shots > 1)` in frame mode returns `(measurements, detectors)`. `detectors` is a dict whose `'detectors'` and `'logicals'` lists hold a `{'label', 'data'}` entry per gate, with one value per frame shot (the reference shot is not included)
- `Circuit.add_gate` can add a gate without qudits, for the detector gates
- `write_circuit` quotes parameter values and `read_circuit` splits lines with `shlex`, so detector expressions with spaces and gates without qudits can be read back
- `N2` with a `prob_dist` of the wrong length acts as the identity again instead of raising

### Added
- Add `DETECTOR` (aliases `DETECT`, `D`) and `LOGICAL_OBSERVABLE` (aliases `LO`, `LOGICALOBSERVABLE`, `OBSERVABLE`) gates. They take an `expr` over earlier M and M_X records, such as `"rec[-1] - rec[-2]"`, and an optional `label`, and are evaluated mod d on the measurement flips of each frame shot
- Add a `TICK` gate, which does nothing

### Fixed
- Fix frame mode (`shots > 1`) failing with "Empty or invalid measurement results format" when qudit 0 is never measured

[1.3.4]: https://pypi.org/project/sdim/1.3.4/

## [1.4.0] - 2026-10-05

### Changed
- **Breaking:** `M_X` leaves the measured qudit in the matching X eigenstate (H_INV, M, H), so repeating `M_X` repeats its outcome. A single `M_X` gives the same outcomes as before
- **Breaking:** `N2` takes `prob=` for two-qudit depolarizing noise (a uniformly random non-identity two-qudit Pauli with probability `prob`) and defaults to `prob=0.01`. Without parameters it used to apply a uniform distribution over all d**4 Paulis. A dense `prob_dist` is still accepted
- **Breaking:** `circuit * n` returns a new circuit and leaves `circuit` unchanged (it used to extend `circuit` in place and return it); `circuit *= n` repeats in place
- Depend on `cirq-core` instead of the `cirq` metapackage and drop the unused `Diophantine` and `networkx` dependencies; install `cirq` yourself if you need its vendor packages. The dependency floors are the oldest releases that pass the tests: numpy 1.23.2, sympy 1.9, cirq-core 1.0 and numba 0.57
- `add_gate` rejects unknown noise parameters, malformed `prob_dist`, out-of-range probabilities, a two-qudit gate given one qudit or the same qudit twice, a one-qudit gate given no qudit, and a `MUL` scalar that is not an integer coprime to d
- Compile the prime-dimension tableau, the Pauli frame sampler and the DEM sampler with numba. A 400-qudit, 13k-gate noisy circuit at d = 1000003 takes about 0.1 s per tableau shot and about 0.5 s for 2000 frame shots. The first run after installing compiles the kernels once (about 10 s); later runs load them from numba's cache, and an install where numba cannot write a cache compiles them in each process
- Sample noise lazily in the frame sampler, so memory no longer grows with shots times noise gates
- Detector expressions reject out-of-range `rec[...]` indices with a ValueError instead of wrapping them around the number of records
- Negative qudit indices count from the end of the program's qudits in every simulation mode, and indices outside the program raise
- `Program(circuit, tableau=...)` with an initial state other than a computational basis state samples the shots after the reference shot with the tableau, with the same output as the frame sampler
- `read_circuit` takes a path as given (absolute or relative to the working directory) before the package's circuits folder, `write_circuit` writes to `circuits/` in the working directory instead of next to the installed package, and both read and write .chp files as UTF-8
- Dimensions must be below 2**31

### Added
- Add `sdim.dem`: compact, exact detector error models for prime dimensions (`DetectorErrorModel.from_circuit`, `sample`, `compile_sampler`, `to_lines`, `merge_lines`, `write_to_file`, `read_from_file`), one mechanism per noise gate independent of the dimension
- Add `sdim.dem_legacy`, the enumerating detector error model for small dimensions
- Add the multiplication gate `MUL` (aliases `MULT`, `MULTIPLY`), which takes the scalar as `a`
- Add `raw_detector_output=True` to `Program.simulate`, which returns the detector and observable data as two arrays indexed (index, shot)
- Add a GitHub Actions workflow that runs the tests on Python 3.11 to 3.14, at the declared dependency floors, from the built wheel, weekly, and on the next Python with pre-releases; warnings from sdim's own code fail the tests
- Add a surface-code memory example to `examples/surface_code.ipynb`, which was empty

### Removed
- Remove the column-reduction heuristic and the Diophantine solver from composite-dimension measurement, which is now always exact; `simulate(exact=...)` is accepted but has no effect

### Fixed
- Fix `N1` and `N2` being ignored by the tableau simulator (`shots == 1`, `force_tableau=True` or `record_tableau=True`)
- Fix `N1` without `noise_channel`, or with the older `channel` key, raising a KeyError in frame mode
- Fix `Circuit` building a d**4 array for the default `N2` distribution, which took about 10 s at d = 101 and needed 32 GiB at d = 257
- Fix int64 overflow in the tableau and the frame sampler at large dimensions, which silently changed measurement outcomes
- Fix composite-dimension tableaus: periodic reduction mod d flipped signs for even d, measurement could return impossible outcomes, and M, M_X and RESET on a negative qudit index measured in the wrong basis
- Fix detector expressions that combined absolute indices (`rec[2] + rec[0]` read one record twice) or, in appended circuits, used the first circuit's detectors
- Fix frame mode for `Program(circuit, tableau=...)`, which assumed the all-zero state and returned random outcomes flagged deterministic
- Fix `append_circuit` with a circuit that has more qudits, which raised IndexError in every mode, and stop it from modifying the appended circuits
- Fix `simulate(show_measurement=True)` running a one-shot circuit a second time to print it, and reprinting every earlier shot with `force_tableau=True`
- Fix a two-qudit gate given one qudit, which made the frame sampler apply the missing half of the gate to the last qudit
- Fix NumPy integer dimensions and qudit indices breaking `MUL`, composite-dimension measurement and `add_gate`
- Fix `import sdim` failing on a read-only install with no writable cache directory, and overflow warnings with `NUMBA_DISABLE_JIT=1`
- Fix RESET recording the noiseless outcome in frame mode
- Fix frame mode (`shots > 1`) failing on circuits without measurements
- Fix circuit files losing `prob_dist` and other gate parameters on a write/read round trip
- Fix `read_circuit` dropping negative qudit indices, `Circuit.from_operation_list` dropping gate parameters, and SWAP and N2 in `generate_random_clifford_circuit(gate_set=...)`
- Fix `circuit_to_cirq_circuit` for SWAP, M_X, RESET, N2 and the detector gates, and the inverses and powers of sdim's cirq gates (`cirq.inverse` of Z gave Z)
- Fix `dem_legacy.from_circuit` printing progress lines and failing on circuits with no noise or a single noise outcome
- Fix `pip install sdim` failing on Python 3.15, and package metadata that named only the first author and listed contradictory GPL classifiers
- Fix the documentation workflow

[1.4.0]: https://github.com/events555/sdim/releases/tag/v1.4.0
