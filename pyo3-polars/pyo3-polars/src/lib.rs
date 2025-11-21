//! This crate offers a [`PySeries`] and a [`PyDataFrame`] which are simple wrapper around `Series` and `DataFrame`. The
//! advantage of these wrappers is that they can be converted to and from python as they implement `FromPyObject` and `IntoPy`.
//!
//! # Example
//!
//! From `src/lib.rs`.
//! ```rust
//! # use polars::prelude::*;
//! # use pyo3::prelude::*;
//! # use pyo3_polars::PyDataFrame;
//!
//! #[pyfunction]
//! fn my_cool_function(pydf: PyDataFrame) -> PyResult<PyDataFrame> {
//!     let df: DataFrame = pydf.into();
//!     let df = {
//!         // some work on the dataframe here
//!         todo!()
//!     };
//!
//!     // wrap the dataframe and it will be automatically converted to a python polars dataframe
//!     Ok(PyDataFrame(df))
//! }
//!
//! /// A Python module implemented in Rust.
//! #[pymodule]
//! fn expression_lib(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(my_cool_function, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! Compile your crate with `maturin` and then import from python.
//!
//! From `my_python_file.py`.
//! ```python
//! from expression_lib import my_cool_function
//!
//! df = pl.DataFrame({
//!     "foo": [1, 2, None],
//!     "bar": ["a", None, "c"],
//! })
//! out_df = my_cool_function(df)
//! ```
mod alloc;
#[cfg(feature = "derive")]
pub mod derive;
pub mod error;
#[cfg(feature = "derive")]
pub mod export;

pub mod types;

pub use crate::alloc::PolarsAllocator;
mod ffi;

use once_cell::sync::Lazy;
use pyo3::prelude::*;
pub use types::*;

pub(crate) static POLARS: Lazy<Py<PyModule>> =
    Lazy::new(|| Python::attach(|py| PyModule::import(py, "polars").unwrap().unbind()));

pub(crate) static POLARS_INTERCHANGE: Lazy<Py<PyModule>> =
    Lazy::new(|| Python::attach(|py| PyModule::import(py, "polars.interchange").unwrap().unbind()));

pub(crate) static SERIES: Lazy<Py<PyAny>> =
    Lazy::new(|| Python::attach(|py| POLARS.getattr(py, "Series").unwrap()));
