use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

static POLARS: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static POLARS_RS: GILOnceCell<PyObject> = GILOnceCell::new();
static UTILS: GILOnceCell<PyObject> = GILOnceCell::new();
static SERIES: GILOnceCell<PyObject> = GILOnceCell::new();
static DATAFRAME: GILOnceCell<PyObject> = GILOnceCell::new();

pub(crate) fn polars(py: Python<'_>) -> &Py<PyModule> {
    POLARS.get_or_init(py, || py.import("polars").unwrap().unbind())
}

pub(crate) fn polars_rs(py: Python<'_>) -> &PyObject {
    POLARS_RS.get_or_init(py, || polars(py).getattr(py, "polars").unwrap())
}

pub(crate) fn pl_utils(py: Python<'_>) -> &PyObject {
    UTILS.get_or_init(py, || polars(py).getattr(py, "_utils").unwrap())
}

pub(crate) fn pl_series(py: Python<'_>) -> &PyObject {
    SERIES.get_or_init(py, || polars(py).getattr(py, "Series").unwrap())
}

pub(crate) fn pl_df(py: Python<'_>) -> &PyObject {
    DATAFRAME.get_or_init(py, || polars(py).getattr(py, "DataFrame").unwrap())
}
