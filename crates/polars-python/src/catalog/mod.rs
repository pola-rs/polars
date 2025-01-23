use polars::prelude::LazyFrame;
use polars_io::catalog::unity::client::{CatalogClient, CatalogClientBuilder};
use polars_io::catalog::unity::models::{CatalogInfo, ColumnInfo, SchemaInfo, TableInfo};
use polars_io::cloud::credential_provider::PlCredentialProvider;
use polars_io::pl_async;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use pyo3::{pyclass, pymethods, Bound, PyObject, PyResult, Python};

use crate::lazyframe::PyLazyFrame;
use crate::prelude::parse_cloud_options;
use crate::utils::to_py_err;

macro_rules! pydict_insert_keys {
    ($dict:expr, {$a:expr}) => {
        $dict.set_item(stringify!($a), $a).unwrap();
    };

    ($dict:expr, {$a:expr, $($args:expr),+}) => {
        pydict_insert_keys!($dict, { $a });
        pydict_insert_keys!($dict, { $($args),+ });
    };

    ($dict:expr, {$a:expr, $($args:expr),+,}) => {
        pydict_insert_keys!($dict, {$a, $($args),+});
    };
}

#[pyclass]
pub struct PyCatalogClient(CatalogClient);

#[pymethods]
impl PyCatalogClient {
    #[pyo3(signature = (workspace_url, bearer_token))]
    #[staticmethod]
    pub fn new(workspace_url: String, bearer_token: Option<String>) -> PyResult<Self> {
        let builder = CatalogClientBuilder::new().with_workspace_url(workspace_url);

        let builder = if let Some(bearer_token) = bearer_token {
            builder.with_bearer_token(bearer_token)
        } else {
            builder
        };

        builder.build().map(PyCatalogClient).map_err(to_py_err)
    }

    pub fn list_catalogs(&self, py: Python) -> PyResult<PyObject> {
        let v = py
            .allow_threads(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().list_catalogs())
            })
            .map_err(to_py_err)?;

        PyList::new(
            py,
            v.into_iter().map(|CatalogInfo { name, comment }| {
                let dict = PyDict::new(py);

                pydict_insert_keys!(dict, {
                    name,
                    comment,
                });

                dict
            }),
        )
        .map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name))]
    pub fn list_schemas(&self, py: Python, catalog_name: &str) -> PyResult<PyObject> {
        let v = py
            .allow_threads(|| {
                pl_async::get_runtime()
                    .block_on_potential_spawn(self.client().list_schemas(catalog_name))
            })
            .map_err(to_py_err)?;

        PyList::new(
            py,
            v.into_iter().map(|SchemaInfo { name, comment }| {
                let dict = PyDict::new(py);

                pydict_insert_keys!(dict, {
                    name,
                    comment,
                });

                dict
            }),
        )
        .map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name, schema_name))]
    pub fn list_tables(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
    ) -> PyResult<PyObject> {
        let v = py
            .allow_threads(|| {
                pl_async::get_runtime()
                    .block_on_potential_spawn(self.client().list_tables(catalog_name, schema_name))
            })
            .map_err(to_py_err)?;

        PyList::new(
            py,
            v.into_iter()
                .map(|table_entry| table_entry_to_pydict(py, table_entry)),
        )
        .map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name, schema_name, table_name))]
    pub fn get_table_info(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> PyResult<PyObject> {
        let table_entry = py
            .allow_threads(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().get_table_info(
                    catalog_name,
                    schema_name,
                    table_name,
                ))
            })
            .map_err(to_py_err)?;

        Ok(table_entry_to_pydict(py, table_entry).into())
    }

    #[pyo3(signature = (catalog_name, schema_name, table_name, cloud_options, credential_provider, retries))]
    pub fn scan_table(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
    ) -> PyResult<PyLazyFrame> {
        let table_info = py
            .allow_threads(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().get_table_info(
                    catalog_name,
                    schema_name,
                    table_name,
                ))
            })
            .map_err(to_py_err)?;

        let Some(storage_location) = table_info.storage_location.as_deref() else {
            return Err(PyValueError::new_err(
                "cannot scan catalog table: no storage_location found",
            ));
        };

        let cloud_options =
            parse_cloud_options(storage_location, cloud_options.unwrap_or_default())?
                .with_max_retries(retries)
                .with_credential_provider(
                    credential_provider.map(PlCredentialProvider::from_python_func_object),
                );

        Ok(
            LazyFrame::scan_catalog_table(&table_info, Some(cloud_options))
                .map_err(to_py_err)?
                .into(),
        )
    }
}

impl PyCatalogClient {
    fn client(&self) -> &CatalogClient {
        &self.0
    }
}

fn table_entry_to_pydict(py: Python, table_entry: TableInfo) -> Bound<'_, PyDict> {
    let TableInfo {
        name,
        comment,
        table_id,
        table_type,
        storage_location,
        data_source_format,
        columns,
    } = table_entry;

    let dict = PyDict::new(py);

    let columns = columns.map(|columns| {
        columns
            .into_iter()
            .map(
                |ColumnInfo {
                     name,
                     type_text,
                     type_interval_type,
                     position,
                     comment,
                     partition_index,
                 }| {
                    let dict = PyDict::new(py);

                    pydict_insert_keys!(dict, {
                        name,
                        type_text,
                        type_interval_type,
                        position,
                        comment,
                        partition_index,
                    });

                    dict
                },
            )
            .collect::<Vec<_>>()
    });

    let data_source_format = data_source_format.map(|x| x.to_string());
    let table_type = table_type.to_string();

    pydict_insert_keys!(dict, {
        name,
        comment,
        table_id,
        table_type,
        storage_location,
        data_source_format,
        columns,
    });

    dict
}
