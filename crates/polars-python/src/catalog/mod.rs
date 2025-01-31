use std::str::FromStr;

use polars::prelude::{LazyFrame, PlHashMap, PlSmallStr, Schema};
use polars_io::catalog::schema::parse_type_json_str;
use polars_io::catalog::unity::client::{CatalogClient, CatalogClientBuilder};
use polars_io::catalog::unity::models::{
    CatalogInfo, ColumnInfo, DataSourceFormat, NamespaceInfo, TableInfo, TableType,
};
use polars_io::cloud::credential_provider::PlCredentialProvider;
use polars_io::pl_async;
use pyo3::exceptions::PyValueError;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyNone, PyTuple};
use pyo3::{pyclass, pymethods, Bound, IntoPyObject, Py, PyAny, PyObject, PyResult, Python};

use crate::lazyframe::PyLazyFrame;
use crate::prelude::{parse_cloud_options, Wrap};
use crate::utils::{to_py_err, EnterPolarsExt};

macro_rules! pydict_insert_keys {
    ($dict:expr, {$a:expr}) => {
        $dict.set_item(stringify!($a), $a)?;
    };

    ($dict:expr, {$a:expr, $($args:expr),+}) => {
        pydict_insert_keys!($dict, { $a });
        pydict_insert_keys!($dict, { $($args),+ });
    };

    ($dict:expr, {$a:expr, $($args:expr),+,}) => {
        pydict_insert_keys!($dict, {$a, $($args),+});
    };
}

// Result dataclasses. These are initialized from Python by calling [`PyCatalogClient::init_classes`].

static CATALOG_INFO_CLS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static NAMESPACE_INFO_CLS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static TABLE_INFO_CLS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static COLUMN_INFO_CLS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

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
        let v = py.enter_polars(|| {
            pl_async::get_runtime().block_on_potential_spawn(self.client().list_catalogs())
        })?;

        let mut opt_err = None;

        let out = PyList::new(
            py,
            v.into_iter().map(|x| {
                let v = catalog_info_to_pyobject(py, x);
                if let Ok(v) = v {
                    Some(v)
                } else {
                    opt_err.replace(v);
                    None
                }
            }),
        )?;

        opt_err.transpose()?;

        Ok(out.into())
    }

    #[pyo3(signature = (catalog_name))]
    pub fn list_namespaces(&self, py: Python, catalog_name: &str) -> PyResult<PyObject> {
        let v = py.enter_polars(|| {
            pl_async::get_runtime()
                .block_on_potential_spawn(self.client().list_namespaces(catalog_name))
        })?;

        let mut opt_err = None;

        let out = PyList::new(
            py,
            v.into_iter().map(|x| {
                let v = namespace_info_to_pyobject(py, x);
                match v {
                    Ok(v) => Some(v),
                    Err(_) => {
                        opt_err.replace(v);
                        None
                    },
                }
            }),
        )?;

        opt_err.transpose()?;

        Ok(out.into())
    }

    #[pyo3(signature = (catalog_name, namespace))]
    pub fn list_tables(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
    ) -> PyResult<PyObject> {
        let v = py.enter_polars(|| {
            pl_async::get_runtime()
                .block_on_potential_spawn(self.client().list_tables(catalog_name, namespace))
        })?;

        let mut opt_err = None;

        let out = PyList::new(
            py,
            v.into_iter().map(|table_info| {
                let v = table_info_to_pyobject(py, table_info);

                if let Ok(v) = v {
                    Some(v)
                } else {
                    opt_err.replace(v);
                    None
                }
            }),
        )?
        .into();

        opt_err.transpose()?;

        Ok(out)
    }

    #[pyo3(signature = (table_name, catalog_name, namespace))]
    pub fn get_table_info(
        &self,
        py: Python,
        table_name: &str,
        catalog_name: &str,
        namespace: &str,
    ) -> PyResult<PyObject> {
        let table_info = py
            .enter_polars(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().get_table_info(
                    table_name,
                    catalog_name,
                    namespace,
                ))
            })
            .map_err(to_py_err)?;

        table_info_to_pyobject(py, table_info).map(|x| x.into())
    }

    #[pyo3(signature = (table_id, write))]
    pub fn get_table_credentials(
        &self,
        py: Python,
        table_id: &str,
        write: bool,
    ) -> PyResult<PyObject> {
        let table_credentials = py
            .enter_polars(|| {
                pl_async::get_runtime()
                    .block_on_potential_spawn(self.client().get_table_credentials(table_id, write))
            })
            .map_err(to_py_err)?;

        let expiry = table_credentials.expiration_time;

        let credentials = PyDict::new(py);
        // Keys in here are intended to be injected into `storage_options` from the Python side.
        // Note this currently really only exists for `aws_endpoint_url`.
        let storage_update_options = PyDict::new(py);

        {
            use polars_io::catalog::unity::models::{
                TableCredentialsAws, TableCredentialsAzure, TableCredentialsGcp,
                TableCredentialsVariants,
            };
            use TableCredentialsVariants::*;

            match table_credentials.into_enum() {
                Some(Aws(TableCredentialsAws {
                    access_key_id,
                    secret_access_key,
                    session_token,
                    access_point,
                })) => {
                    credentials.set_item("aws_access_key_id", access_key_id)?;
                    credentials.set_item("aws_secret_access_key", secret_access_key)?;

                    if let Some(session_token) = session_token {
                        credentials.set_item("aws_session_token", session_token)?;
                    }

                    if let Some(access_point) = access_point {
                        storage_update_options.set_item("aws_endpoint_url", access_point)?;
                    }
                },
                Some(Azure(TableCredentialsAzure { sas_token })) => {
                    credentials.set_item("sas_token", sas_token)?;
                },
                Some(Gcp(TableCredentialsGcp { oauth_token })) => {
                    credentials.set_item("bearer_token", oauth_token)?;
                },
                None => {},
            }
        }

        let credentials = if credentials.len()? > 0 {
            credentials.into_any()
        } else {
            PyNone::get(py).as_any().clone()
        };
        let storage_update_options = storage_update_options.into_any();
        let expiry = expiry.into_pyobject(py)?.into_any();

        Ok(PyTuple::new(py, [credentials, storage_update_options, expiry])?.into())
    }

    #[pyo3(signature = (catalog_name, namespace, table_name, cloud_options, credential_provider, retries))]
    pub fn scan_table(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
    ) -> PyResult<PyLazyFrame> {
        let table_info = py.enter_polars(|| {
            pl_async::get_runtime().block_on_potential_spawn(self.client().get_table_info(
                catalog_name,
                namespace,
                table_name,
            ))
        })?;

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

    #[pyo3(signature = (catalog_name, comment, storage_root))]
    pub fn create_catalog(
        &self,
        py: Python,
        catalog_name: &str,
        comment: Option<&str>,
        storage_root: Option<&str>,
    ) -> PyResult<PyObject> {
        let catalog_info = py
            .allow_threads(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().create_catalog(
                    catalog_name,
                    comment,
                    storage_root,
                ))
            })
            .map_err(to_py_err)?;

        catalog_info_to_pyobject(py, catalog_info).map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name, force))]
    pub fn delete_catalog(&self, py: Python, catalog_name: &str, force: bool) -> PyResult<()> {
        py.allow_threads(|| {
            pl_async::get_runtime()
                .block_on_potential_spawn(self.client().delete_catalog(catalog_name, force))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (catalog_name, namespace, comment, storage_root))]
    pub fn create_namespace(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
        comment: Option<&str>,
        storage_root: Option<&str>,
    ) -> PyResult<PyObject> {
        let namespace_info = py
            .allow_threads(|| {
                pl_async::get_runtime().block_on_potential_spawn(self.client().create_namespace(
                    catalog_name,
                    namespace,
                    comment,
                    storage_root,
                ))
            })
            .map_err(to_py_err)?;

        namespace_info_to_pyobject(py, namespace_info).map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name, namespace, force))]
    pub fn delete_namespace(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
        force: bool,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            pl_async::get_runtime().block_on_potential_spawn(self.client().delete_namespace(
                catalog_name,
                namespace,
                force,
            ))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (
        catalog_name, namespace, table_name, schema, table_type, data_source_format, comment,
        storage_root, properties
    ))]
    pub fn create_table(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
        schema: Option<Wrap<Schema>>,
        table_type: &str,
        data_source_format: Option<&str>,
        comment: Option<&str>,
        storage_root: Option<&str>,
        properties: Vec<(String, String)>,
    ) -> PyResult<PyObject> {
        let table_info = py.allow_threads(|| {
            pl_async::get_runtime()
                .block_on_potential_spawn(
                    self.client().create_table(
                        catalog_name,
                        namespace,
                        table_name,
                        schema.as_ref().map(|x| &x.0),
                        &TableType::from_str(table_type)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        data_source_format
                            .map(DataSourceFormat::from_str)
                            .transpose()
                            .map_err(|e| PyValueError::new_err(e.to_string()))?
                            .as_ref(),
                        comment,
                        storage_root,
                        &mut properties.iter().map(|(a, b)| (a.as_str(), b.as_str())),
                    ),
                )
                .map_err(to_py_err)
        })?;

        table_info_to_pyobject(py, table_info).map(|x| x.into())
    }

    #[pyo3(signature = (catalog_name, namespace, table_name))]
    pub fn delete_table(
        &self,
        py: Python,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            pl_async::get_runtime().block_on_potential_spawn(self.client().delete_table(
                catalog_name,
                namespace,
                table_name,
            ))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (type_json))]
    #[staticmethod]
    pub fn type_json_to_polars_type(py: Python, type_json: &str) -> PyResult<PyObject> {
        Ok(Wrap(parse_type_json_str(type_json).map_err(to_py_err)?)
            .into_pyobject(py)?
            .unbind())
    }

    #[pyo3(signature = (catalog_info_cls, namespace_info_cls, table_info_cls, column_info_cls))]
    #[staticmethod]
    pub fn init_classes(
        py: Python,
        catalog_info_cls: Py<PyAny>,
        namespace_info_cls: Py<PyAny>,
        table_info_cls: Py<PyAny>,
        column_info_cls: Py<PyAny>,
    ) {
        CATALOG_INFO_CLS.get_or_init(py, || catalog_info_cls);
        NAMESPACE_INFO_CLS.get_or_init(py, || namespace_info_cls);
        TABLE_INFO_CLS.get_or_init(py, || table_info_cls);
        COLUMN_INFO_CLS.get_or_init(py, || column_info_cls);
    }
}

impl PyCatalogClient {
    fn client(&self) -> &CatalogClient {
        &self.0
    }
}

fn catalog_info_to_pyobject(
    py: Python,
    CatalogInfo {
        name,
        comment,
        storage_location,
        properties,
        options,
        created_at,
        created_by,
        updated_at,
        updated_by,
    }: CatalogInfo,
) -> PyResult<Bound<'_, PyAny>> {
    let dict = PyDict::new(py);

    let properties = properties_to_pyobject(py, properties);
    let options = properties_to_pyobject(py, options);

    pydict_insert_keys!(dict, {
        name,
        comment,
        storage_location,
        properties,
        options,
        created_at,
        created_by,
        updated_at,
        updated_by
    });

    CATALOG_INFO_CLS
        .get(py)
        .unwrap()
        .bind(py)
        .call((), Some(&dict))
}

fn namespace_info_to_pyobject(
    py: Python,
    NamespaceInfo {
        name,
        comment,
        properties,
        storage_location,
        created_at,
        created_by,
        updated_at,
        updated_by,
    }: NamespaceInfo,
) -> PyResult<Bound<'_, PyAny>> {
    let dict = PyDict::new(py);

    let properties = properties_to_pyobject(py, properties);

    pydict_insert_keys!(dict, {
        name,
        comment,
        properties,
        storage_location,
        created_at,
        created_by,
        updated_at,
        updated_by
    });

    NAMESPACE_INFO_CLS
        .get(py)
        .unwrap()
        .bind(py)
        .call((), Some(&dict))
}

fn table_info_to_pyobject(py: Python, table_info: TableInfo) -> PyResult<Bound<'_, PyAny>> {
    let TableInfo {
        name,
        table_id,
        table_type,
        comment,
        storage_location,
        data_source_format,
        columns,
        properties,
        created_at,
        created_by,
        updated_at,
        updated_by,
    } = table_info;

    let column_info_cls = COLUMN_INFO_CLS.get(py).unwrap().bind(py);

    let columns = columns
        .map(|columns| {
            columns
                .into_iter()
                .map(
                    |ColumnInfo {
                         name,
                         type_name,
                         type_text,
                         type_json,
                         position,
                         comment,
                         partition_index,
                     }| {
                        let dict = PyDict::new(py);

                        let name = name.as_str();
                        let type_name = type_name.as_str();
                        let type_text = type_text.as_str();

                        pydict_insert_keys!(dict, {
                            name,
                            type_name,
                            type_text,
                            type_json,
                            position,
                            comment,
                            partition_index,
                        });

                        column_info_cls.call((), Some(&dict))
                    },
                )
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;

    let dict = PyDict::new(py);

    let data_source_format = data_source_format.map(|x| x.to_string());
    let table_type = table_type.to_string();
    let properties = properties_to_pyobject(py, properties);

    pydict_insert_keys!(dict, {
        name,
        comment,
        table_id,
        table_type,
        storage_location,
        data_source_format,
        columns,
        properties,
        created_at,
        created_by,
        updated_at,
        updated_by,
    });

    TABLE_INFO_CLS
        .get(py)
        .unwrap()
        .bind(py)
        .call((), Some(&dict))
}

fn properties_to_pyobject(
    py: Python,
    properties: PlHashMap<PlSmallStr, String>,
) -> Bound<'_, PyDict> {
    let dict = PyDict::new(py);

    for (key, value) in properties.into_iter() {
        dict.set_item(key.as_str(), value).unwrap();
    }

    dict
}
