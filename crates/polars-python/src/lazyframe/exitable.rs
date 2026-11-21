use std::time::Duration;

use polars::prelude::*;
use pyo3::prelude::*;

use super::PyLazyFrame;
use crate::PyDataFrame;
use crate::utils::EnterPolarsExt;

#[pymethods]
#[cfg(not(target_arch = "wasm32"))]
impl PyLazyFrame {
    fn collect_concurrently(&self, py: Python) -> PyResult<PyInProcessQuery> {
        let ipq = py.enter_polars(|| {
            let ldf = self.ldf.read().clone();
            ldf.collect_concurrently()
        })?;
        Ok(PyInProcessQuery { ipq })
    }
}

#[pyclass(frozen)]
#[cfg(not(target_arch = "wasm32"))]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyInProcessQuery {
    pub ipq: InProcessQuery,
}

#[pymethods]
#[cfg(not(target_arch = "wasm32"))]
impl PyInProcessQuery {
    /// Cancels the query and waits until it has stopped (see `InProcessQuery::wait_finished`),
    /// unless called on one of Polars' own threads.
    ///
    /// The wait is in slices, checking Python's signals in between: a Ctrl-C raises
    /// `KeyboardInterrupt` from here (the query still stops, without waiting for it). It isn't
    /// run as a Polars call (`enter_polars`), whose abort on Ctrl-C would reach the query.
    pub fn cancel(&self, py: Python) -> PyResult<()> {
        self.ipq.cancel();
        if !InProcessQuery::can_wait_on_this_thread() {
            return Ok(());
        }
        const WAIT_SLICE: Duration = Duration::from_millis(50);
        while !py.detach(|| self.ipq.wait_finished(Some(WAIT_SLICE))) {
            py.check_signals()?;
        }
        Ok(())
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
