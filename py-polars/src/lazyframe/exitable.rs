use super::*;

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
    pub fn cancel(&self) {
        self.ipq.cancel();
    }

    pub fn fetch(&self) -> PyResult<Option<PyDataFrame>> {
        let out = self.ipq.fetch().transpose().map_err(PyPolarsErr::from)?;
        Ok(out.map(|df| df.into()))
    }

    pub fn fetch_blocking(&self) -> PyResult<PyDataFrame> {
        let out = self.ipq.fetch_blocking().map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}
