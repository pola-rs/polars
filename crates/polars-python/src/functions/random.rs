use pyo3::prelude::*;

#[pyfunction]
pub fn set_random_seed(seed: u64) -> PyResult<()> {
    polars_core::random::set_global_random_seed(seed);
    Ok(())
}
