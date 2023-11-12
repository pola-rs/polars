use polars::prelude::*;
use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn name_keep(&self) -> Self {
        self.inner.clone().name().keep().into()
    }

    fn name_map(&self, lambda: PyObject) -> Self {
        self.inner
            .clone()
            .name()
            .map(move |name| {
                let out = Python::with_gil(|py| lambda.call1(py, (name,)));
                match out {
                    Ok(out) => Ok(out.to_string()),
                    Err(e) => Err(PolarsError::ComputeError(
                        format!("Python function in 'name.map' produced an error: {e}.").into(),
                    )),
                }
            })
            .into()
    }

    fn name_prefix(&self, prefix: &str) -> Self {
        self.inner.clone().name().prefix(prefix).into()
    }

    fn name_suffix(&self, suffix: &str) -> Self {
        self.inner.clone().name().suffix(suffix).into()
    }

    fn name_to_lowercase(&self) -> Self {
        self.inner.clone().name().to_lowercase().into()
    }

    fn name_to_uppercase(&self) -> Self {
        self.inner.clone().name().to_uppercase().into()
    }
}
