use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{wrap_pyfunction, Bound, PyResult, Python};
use pyo3_polars::PolarsAllocator;

mod distances;
mod expressions;

mod byte_rev;
mod min_by;
mod rolling_product;
mod vertical_scan;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// A Python module implemented in Rust.
#[pyo3::pymodule]
fn plugin_v2(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(min_by::min_by, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_product::rolling_product, m)?)?;
    m.add_function(wrap_pyfunction!(byte_rev::byte_rev, m)?)?;
    m.add_function(wrap_pyfunction!(vertical_scan::vertical_scan, m)?)?;
    Ok(())
}
