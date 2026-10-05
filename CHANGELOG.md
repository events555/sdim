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

## [1.4.0] - 2026-10-05

### Changed
- **Breaking:** `Program.simulate(shots > 1)` in frame mode returns `(measurements, detectors)`; `detectors` holds detector and logical observable data
- **Breaking:** `M_X` leaves the measured qudit in the matching X eigenstate (H_INV, M, H), so repeating `M_X` repeats its outcome. A single `M_X` gives the same outcomes as before
- `N2` takes `prob=` for two-qudit depolarizing noise; a dense `prob_dist` over all d**4 Paulis is optional and no longer built by default
- `N1` reads `noise_channel`; the older `channel` key is still accepted
- `add_gate` rejects unknown noise parameters, malformed `prob_dist`, out-of-range probabilities and two-qudit gates on a single qudit
- Compile the prime-dimension tableau, the Pauli frame sampler and the DEM sampler with numba. A 400-qudit, 13k-gate noisy circuit at d = 1000003 now takes about 0.1 s per tableau shot (was about 11 s) and about 0.75 s for 2000 frame shots (was about 30 s)
- Sample noise lazily in the frame sampler, so memory no longer grows with shots times noise gates
- Detector expressions accept relative (`rec[-k]`) and absolute (`rec[k]`) record indices and reject out-of-range ones
- Dimensions must be below 2**31

### Added
- Add `sdim.dem`: compact, exact detector error models for prime dimensions (`DetectorErrorModel.from_circuit`, `sample`, `to_lines`, `write_to_file`, `read_from_file`), one mechanism per noise gate independent of the dimension
- Add `sdim.dem_legacy`, the enumerating detector error model for small dimensions
- Add `DETECTOR`, `LOGICAL_OBSERVABLE` and `TICK` gates
- Add the multiplication gate `MUL` (aliases `MULT`, `MULTIPLY`)
- Add a GitHub Actions workflow that runs the tests on Python 3.11 and 3.12 and on numpy 1.26

### Removed
- Remove the Diophantine heuristic from composite-dimension measurement; `simulate(exact=...)` is accepted but has no effect

### Fixed
- Fix int64 overflow in the tableau and the frame sampler at large dimensions, which silently changed measurement outcomes
- Fix composite-dimension tableaus: periodic reduction mod d flipped signs for even d, measurement could return impossible outcomes, and large dimensions hung
- Fix detector expressions that wrapped out-of-range record indices, dropped the letter "p", or reused the first circuit's detectors in appended circuits
- Fix RESET recording the noiseless outcome in frame mode
- Fix circuit files losing `prob_dist` and other gate parameters on a write/read round trip
- Fix `circuit_to_cirq_circuit` for SWAP, M_X, RESET and the detector gates
- Fix the documentation workflow

[1.4.0]: https://github.com/events555/sdim/releases/tag/v1.4.0
