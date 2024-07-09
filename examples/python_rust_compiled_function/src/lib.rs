mod ffi;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn hamming_distance(series_a: &Bound<PyAny>, series_b: &Bound<PyAny>) -> PyResult<PyObject> {
    let series_a = ffi::py_series_to_rust_series(series_a)?;
    let series_b = ffi::py_series_to_rust_series(series_b)?;

    let out = hamming_distance_impl(&series_a, &series_b)
        .map_err(|e| PyValueError::new_err(format!("Something went wrong: {:?}", e)))?;
    ffi::rust_series_to_py_series(&out.into_series())
}

/// This function iterates over 2 `StringChunked` arrays and computes the hamming distance between the values .
fn hamming_distance_impl(a: &Series, b: &Series) -> PolarsResult<UInt32Chunked> {
    Ok(a.str()?
        .into_iter()
        .zip(b.str()?)
        .map(|(lhs, rhs)| hamming_distance_strs(lhs, rhs))
        .collect())
}

/// Compute the hamming distance between 2 string values.
fn hamming_distance_strs(a: Option<&str>, b: Option<&str>) -> Option<u32> {
    match (a, b) {
        (None, _) => None,
        (_, None) => None,
        (Some(a), Some(b)) => {
            if a.len() != b.len() {
                None
            } else {
                Some(
                    a.chars()
                        .zip(b.chars())
                        .map(|(a_char, b_char)| (a_char != b_char) as u32)
                        .sum::<u32>(),
                )
            }
        },
    }
}

#[pymodule]
fn my_polars_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(hamming_distance)).unwrap();
    Ok(())
}
