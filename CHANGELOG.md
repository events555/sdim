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


## [2.0.0] - 2025-

### Added
- Add `GateTarget` class for referential measurement control   

### Changed

- **Change `append()` --> `append()` to match stim**
- **Change `Program` --> `Sampler()` to match stim**
- Decouple `GataData` from `CircuitInstruction` and `Circuit` classes
- **Adjust backend to use `REPEAT_BLOCKS` and only flatten during the `_build_ir()` compilation step**