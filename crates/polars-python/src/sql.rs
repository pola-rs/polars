use parking_lot::RwLock;
use polars::sql::SQLContext;
use pyo3::prelude::*;

use crate::PyLazyFrame;
use crate::error::PyPolarsErr;

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PySQLContext {
    pub context: RwLock<SQLContext>,
}

impl Clone for PySQLContext {
    fn clone(&self) -> Self {
        Self {
            context: RwLock::new(self.context.read().clone()),
        }
    }
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PySQLContext {
    #[staticmethod]
    #[allow(clippy::new_without_default)]
    pub fn new() -> PySQLContext {
        PySQLContext {
            context: RwLock::new(SQLContext::new()),
        }
    }

    pub fn execute(&self, query: &str) -> PyResult<PyLazyFrame> {
        Ok(self
            .context
            .write()
            .execute(query)
            .map_err(PyPolarsErr::from)?
            .into())
    }

    pub fn get_tables(&self) -> PyResult<Vec<String>> {
        Ok(self.context.read().get_tables())
    }

    pub fn register(&self, name: &str, lf: PyLazyFrame) {
        self.context.write().register(name, lf.ldf.into_inner())
    }

    pub fn unregister(&self, name: &str) {
        self.context.write().unregister(name)
    }
}
