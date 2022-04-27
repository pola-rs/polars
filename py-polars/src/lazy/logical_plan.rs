use std::io::BufWriter;
use pyo3::{PyObject, PyResult};
use pyo3::exceptions::PyValueError;
use polars::prelude::*;
use crate::file::get_file_like;
use crate::PyLazyFrame;
use pyo3::prelude::*;


#[pyclass]
#[repr(transparent)]
pub struct PyLogicalPlan {
    // option because we cannot get a self by value in pyo3
    pub lp: LogicalPlan,
}

impl From<LogicalPlan> for PyLogicalPlan {
    fn from(lp: LogicalPlan) -> Self {
        PyLogicalPlan{lp}
    }
}

#[pymethods]
impl PyLogicalPlan {
    pub fn to_lazy_frame(&self) -> PyLazyFrame {
        LazyFrame::from(self.lp.clone()).into()
    }

    pub fn to_json(&self,
        py_f: PyObject
    ) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        serde_json::to_writer(file, &self.lp).map_err(|err| PyValueError::new_err(format!("{:?}", err)))?;
        Ok(())
    }
}
