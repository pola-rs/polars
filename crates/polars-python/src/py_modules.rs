use once_cell::sync::Lazy;
use pyo3::prelude::*;

pub(crate) static POLARS: Lazy<Py<PyModule>> =
    Lazy::new(|| Python::with_gil(|py| PyModule::import(py, "polars").unwrap().unbind()));

pub(crate) static UTILS: Lazy<PyObject> =
    Lazy::new(|| Python::with_gil(|py| POLARS.getattr(py, "_utils").unwrap()));

pub(crate) static SERIES: Lazy<PyObject> =
    Lazy::new(|| Python::with_gil(|py| POLARS.getattr(py, "Series").unwrap()));
