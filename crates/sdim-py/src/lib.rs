//! PyO3 bindings for sdim-core.
//!
//! Exposes two functions:
//! - `run_ir(ir_array, num_qudits, dimension)` -> numpy array of measurements
//! - `snapshot(ir_array, num_qudits, dimension, stop_after)` -> (X, Z, tau_exp)

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use sdim_core::ir::{IrInstruction, run_ir as core_run_ir, run_ir_with_snapshot};

/// Run an IR instruction array through the Rust tableau simulator.
///
/// Args:
///     ir_array: numpy int64 array of shape (N, 4) with columns
///               [gate_id, qudit_index, target_index, arg0]
///     num_qudits: number of qudits in the circuit
///     dimension: local qudit dimension
///
/// Returns:
///     numpy int64 array of measurement outcomes
#[pyfunction]
fn run_ir<'py>(
    py: Python<'py>,
    ir_array: PyReadonlyArray2<'py, i64>,
    num_qudits: usize,
    dimension: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let arr = ir_array.as_array();
    let n_instr = arr.nrows();

    let mut instructions = Vec::with_capacity(n_instr);
    for i in 0..n_instr {
        instructions.push(IrInstruction {
            gate_id: arr[[i, 0]],
            qudit_index: arr[[i, 1]],
            target_index: arr[[i, 2]],
            arg0: arr[[i, 3]],
        });
    }

    let measurements = core_run_ir(&instructions, num_qudits, dimension);
    let result = Array1::from_vec(measurements);
    Ok(result.into_pyarray(py))
}

/// Run IR up to `stop_after` instructions and return the tableau state.
///
/// Args:
///     ir_array: numpy int64 array of shape (N, 4)
///     num_qudits: number of qudits
///     dimension: local qudit dimension
///     stop_after: number of instructions to execute before stopping
///
/// Returns:
///     Tuple of (X, Z, tau_exp) numpy arrays representing the active
///     portion of the tableau (rows 0..l).
#[pyfunction]
fn snapshot<'py>(
    py: Python<'py>,
    ir_array: PyReadonlyArray2<'py, i64>,
    num_qudits: usize,
    dimension: i64,
    stop_after: usize,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let arr = ir_array.as_array();
    let n_instr = arr.nrows();

    let mut instructions = Vec::with_capacity(n_instr);
    for i in 0..n_instr {
        instructions.push(IrInstruction {
            gate_id: arr[[i, 0]],
            qudit_index: arr[[i, 1]],
            target_index: arr[[i, 2]],
            arg0: arr[[i, 3]],
        });
    }

    let (_measurements, tab) =
        run_ir_with_snapshot(&instructions, num_qudits, dimension, stop_after);

    let l = tab.l;
    let n = tab.n;

    let mut x_arr = Array2::<i64>::zeros((l, n));
    let mut z_arr = Array2::<i64>::zeros((l, n));
    let mut tau_arr = Array1::<i64>::zeros(l);

    for i in 0..l {
        for j in 0..n {
            x_arr[[i, j]] = tab.x[i][j];
            z_arr[[i, j]] = tab.z[i][j];
        }
        tau_arr[i] = tab.tau_exp[i];
    }

    Ok((
        x_arr.into_pyarray(py),
        z_arr.into_pyarray(py),
        tau_arr.into_pyarray(py),
    ))
}

/// Compute the Smith Normal Form of an integer matrix modulo d.
///
/// Args:
///     matrix: list of lists of ints (the input matrix)
///     d: modulus
///
/// Returns:
///     Tuple of (S, U, V) as lists of lists of ints, where S = U @ A @ V (mod d).
#[pyfunction]
fn snf_mod(matrix: Vec<Vec<i64>>, d: i64) -> PyResult<(Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    let rows = matrix.len();
    let cols = if rows > 0 { matrix[0].len() } else { 0 };

    let mut arr = ndarray::Array2::<i64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = matrix[i][j].rem_euclid(d);
        }
    }

    let (u, v, s) = modularsnf::smith_normal_form(&arr, d)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let to_vecs = |a: &ndarray::Array2<i64>| -> Vec<Vec<i64>> {
        (0..a.nrows())
            .map(|i| (0..a.ncols()).map(|j| a[[i, j]]).collect())
            .collect()
    };

    // Return (S, U, V) to match modularsnf Python API ordering
    Ok((to_vecs(&s), to_vecs(&u), to_vecs(&v)))
}

/// Python module: sdim._sdim_rs
#[pymodule]
fn _sdim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ir, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(snf_mod, m)?)?;
    Ok(())
}
