use polars_stream::nodes::unique::UniqueKeepStrategy;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::prelude::Wrap;

/// Python enum class for the streaming unique keep strategy options
#[pyclass(module = "polars._internal.streaming", name = "PyStreamingUniqueKeepStrategy")]
#[derive(Clone, Debug)]
pub struct PyStreamingUniqueKeepStrategy {
    pub inner: UniqueKeepStrategy,
}

#[pymethods]
impl PyStreamingUniqueKeepStrategy {
    #[classattr]
    const FIRST: &'static str = "first";
    #[classattr]
    const LAST: &'static str = "last";
    #[classattr]
    const ANY: &'static str = "any";

    #[classmethod]
    #[pyo3(name = "from_str")]
    fn from_str_py(_cls: &PyType, s: &str) -> PyResult<Self> {
        let keep_strategy = match s {
            "first" => UniqueKeepStrategy::First,
            "last" => UniqueKeepStrategy::Last,
            "any" => UniqueKeepStrategy::Any,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid keep strategy: {s}. Expected one of ['first', 'last', 'any']"
                )));
            }
        };
        Ok(PyStreamingUniqueKeepStrategy {
            inner: keep_strategy,
        })
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            UniqueKeepStrategy::First => "First",
            UniqueKeepStrategy::Last => "Last",
            UniqueKeepStrategy::Any => "Any",
        }
    }
}

impl From<UniqueKeepStrategy> for PyStreamingUniqueKeepStrategy {
    fn from(inner: UniqueKeepStrategy) -> Self {
        PyStreamingUniqueKeepStrategy { inner }
    }
}

impl From<PyStreamingUniqueKeepStrategy> for UniqueKeepStrategy {
    fn from(py_keep: PyStreamingUniqueKeepStrategy) -> Self {
        py_keep.inner
    }
}

impl Wrap<UniqueKeepStrategy> for PyStreamingUniqueKeepStrategy {
    fn inner(&self) -> &UniqueKeepStrategy {
        &self.inner
    }
}