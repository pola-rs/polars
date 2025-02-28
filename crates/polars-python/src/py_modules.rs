use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

static POLARS: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static UTILS: GILOnceCell<PyObject> = GILOnceCell::new();
static SERIES: GILOnceCell<PyObject> = GILOnceCell::new();

pub(crate) fn polars(py: Python<'_>) -> &Py<PyModule> {
    POLARS.get_or_init(py, || py.import("polars").unwrap().unbind())
}

pub(crate) fn pl_utils(py: Python<'_>) -> &PyObject {
    UTILS.get_or_init(py, || polars(py).getattr(py, "_utils").unwrap())
}

pub(crate) fn pl_series(py: Python<'_>) -> &PyObject {
    SERIES.get_or_init(py, || polars(py).getattr(py, "Series").unwrap())
}
