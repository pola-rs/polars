use polars::prelude::*;
use pyo3::prelude::*;

use super::PyLazyFrame;
use crate::utils::EnterPolarsExt;
use crate::PyDataFrame;

#[pymethods]
#[cfg(not(target_arch = "wasm32"))]
impl PyLazyFrame {
    fn collect_concurrently(&self, py: Python) -> PyResult<PyInProcessQuery> {
        let ipq = py.enter_polars(|| {
            let ldf = self.ldf.clone();
            ldf.collect_concurrently()
        })?;
        Ok(PyInProcessQuery { ipq })
    }
}

#[pyclass]
#[cfg(not(target_arch = "wasm32"))]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyInProcessQuery {
    pub ipq: InProcessQuery,
}

#[pymethods]
#[cfg(not(target_arch = "wasm32"))]
impl PyInProcessQuery {
    pub fn cancel(&self, py: Python) -> PyResult<()> {
        py.enter_polars_ok(|| self.ipq.cancel())
    }

    pub fn fetch(&self, py: Python) -> PyResult<Option<PyDataFrame>> {
        let out = py.enter_polars(|| self.ipq.fetch().transpose())?;
        Ok(out.map(|df| df.into()))
    }

    pub fn fetch_blocking(&self, py: Python) -> PyResult<PyDataFrame> {
        let out = py.enter_polars(|| self.ipq.fetch_blocking())?;
        Ok(out.into())
    }
}
