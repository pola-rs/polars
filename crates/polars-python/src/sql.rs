use parking_lot::RwLock;
use polars::sql::{CatalogDdl, SQLContext, extract_table_identifiers};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::PyLazyFrame;
use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt;

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
    pub fn execute(&self, py: Python<'_>, query: &str) -> PyResult<PyLazyFrame> {
        py.enter_polars(|| self.context.write().execute(query))
            .map(Into::into)
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

    /// Attach a catalog: table names resolve against it (see `SQLContext::register_catalog`).
    #[pyo3(signature = (name, default_namespace, default))]
    pub fn register_catalog(
        &self,
        name: &str,
        default_namespace: Option<Vec<String>>,
        default: bool,
    ) -> PyResult<()> {
        self.context
            .write()
            .register_catalog(name, default_namespace, default)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// The name of the default catalog, if one is set.
    pub fn default_catalog(&self) -> Option<String> {
        self.context.read().default_catalog().map(str::to_string)
    }

    /// Make an attached catalog the default.
    pub fn set_default_catalog(&self, name: &str) -> PyResult<()> {
        self.context
            .write()
            .set_default_catalog(name)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// Detach a catalog, returning the name it was attached under.
    pub fn unregister_catalog(&self, name: &str) -> Option<String> {
        self.context.write().unregister_catalog(name)
    }

    /// The catalog tables a query reads, as (catalog, identifier parts).
    pub fn catalog_tables(&self, query: &str) -> PyResult<Vec<(String, Vec<String>)>> {
        let tables = self
            .context
            .read()
            .catalog_tables(query)
            .map_err(PyPolarsErr::from)?;
        Ok(tables
            .into_iter()
            .map(|t| (t.catalog, t.identifier))
            .collect())
    }

    /// The catalog DDL of a query, as a dict keyed by `kind`, or None.
    pub fn catalog_ddl<'py>(
        &self,
        py: Python<'py>,
        query: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let ddl = self
            .context
            .read()
            .catalog_ddl(query)
            .map_err(PyPolarsErr::from)?;
        let Some(ddl) = ddl else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        match ddl {
            CatalogDdl::DropTable {
                name,
                table,
                if_exists,
                purge,
            } => {
                dict.set_item("kind", "drop_table")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("if_exists", if_exists)?;
                dict.set_item("purge", purge)?;
            },
        }
        Ok(Some(dict))
    }

    /// Execute a query with the scans of the catalog tables it reads, by full name.
    pub fn execute_with_catalog_tables(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
    ) -> PyResult<PyLazyFrame> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        py.enter_polars(|| {
            self.context
                .write()
                .execute_with_catalog_tables(query, tables)
        })
        .map(Into::into)
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
