use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

static POLARS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
static POLARS_PLR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static UTILS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SERIES: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static DATAFRAME: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

pub fn polars(py: Python<'_>) -> &Py<PyModule> {
    POLARS.get_or_init(py, || py.import("polars").unwrap().unbind())
}

pub fn polars_rs(py: Python<'_>) -> &Py<PyAny> {
    POLARS_PLR.get_or_init(py, || polars(py).getattr(py, "_plr").unwrap())
}

pub fn pl_utils(py: Python<'_>) -> &Py<PyAny> {
    UTILS.get_or_init(py, || polars(py).getattr(py, "_utils").unwrap())
}

pub fn pl_series(py: Python<'_>) -> &Py<PyAny> {
    SERIES.get_or_init(py, || polars(py).getattr(py, "Series").unwrap())
}

pub fn pl_df(py: Python<'_>) -> &Py<PyAny> {
    DATAFRAME.get_or_init(py, || polars(py).getattr(py, "DataFrame").unwrap())
}
