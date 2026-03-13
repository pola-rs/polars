use parking_lot::RwLock;
use polars::sql::{SQLContext, extract_table_identifiers};
use pyo3::prelude::*;

use crate::PyLazyFrame;
use crate::error::PyPolarsErr;

#[pyclass(frozen, skip_from_py_object)]
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

    /// Execute a SQL query in the current SQLContext.
    pub fn execute(&self, query: &str) -> PyResult<PyLazyFrame> {
        Ok(self
            .context
            .write()
            .execute(query)
            .map_err(PyPolarsErr::from)?
            .into())
    }

    /// Get a list of table names registered in the current SQLContext.
    pub fn get_tables(&self) -> PyResult<Vec<String>> {
        Ok(self.context.read().get_tables())
    }

    /// Register a table in the current SQLContext.
    pub fn register(&self, name: &str, lf: PyLazyFrame) {
        self.context.write().register(name, lf.ldf.into_inner())
    }

    /// Unregister a table from the current SQLContext.
    pub fn unregister(&self, name: &str) {
        self.context.write().unregister(name)
    }

    /// Extract table identifiers from a SQL query string.
    #[staticmethod]
    #[pyo3(signature = (query, include_schema=true, unique=false))]
    pub fn table_identifiers(
        query: &str,
        include_schema: bool,
        unique: bool,
    ) -> PyResult<Vec<String>> {
        extract_table_identifiers(query, include_schema, unique)
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }
}
