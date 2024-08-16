use polars::prelude::*;
use pyo3::prelude::*;

use super::PyLazyFrame;
use crate::error::PyPolarsErr;
use crate::PyDataFrame;

#[pymethods]
impl PyLazyFrame {
    fn collect_concurrently(&self, py: Python) -> PyResult<PyInProcessQuery> {
        let ipq = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.collect_concurrently().map_err(PyPolarsErr::from)
        })?;
        Ok(PyInProcessQuery { ipq })
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyInProcessQuery {
    pub ipq: InProcessQuery,
}

#[pymethods]
impl PyInProcessQuery {
    pub fn cancel(&self, py: Python) {
        py.allow_threads(|| self.ipq.cancel())
    }

    pub fn fetch(&self, py: Python) -> PyResult<Option<PyDataFrame>> {
        let out = py.allow_threads(|| self.ipq.fetch().transpose().map_err(PyPolarsErr::from))?;
        Ok(out.map(|df| df.into()))
    }

    pub fn fetch_blocking(&self, py: Python) -> PyResult<PyDataFrame> {
        let out = py.allow_threads(|| self.ipq.fetch_blocking().map_err(PyPolarsErr::from))?;
        Ok(out.into())
    }
}
