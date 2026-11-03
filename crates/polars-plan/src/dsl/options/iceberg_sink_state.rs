use std::collections::BTreeMap;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;

use super::iceberg_properties::IcebergProperties;

/// `class IcebergSinkState` in Python
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct IcebergSinkState {
    /// Python class module. `None` (with the qualname) means `catalog_name` is resolved
    /// by the Iceberg catalog resolver wherever the sink runs.
    pub py_catalog_class_module: Option<PlSmallStr>,
    /// Python class qualname
    pub py_catalog_class_qualname: Option<PlSmallStr>,

    pub catalog_name: PlSmallStr,
    pub catalog_properties: IcebergProperties,

    pub table_name: PlSmallStr,
    pub mode: IcebergCommitMode,
    pub schema_mode: Option<IcebergSchemaMode>,
    #[cfg(feature = "python")]
    pub overwrite_filter: Option<PythonObject>, // pyiceberg.expressions.BooleanExpression
    pub dynamic_partition_overwrite: bool,
    pub create_if_not_exists: bool,
    pub create_namespace_if_not_exists: bool,
    pub snapshot_properties: BTreeMap<PlSmallStr, PlSmallStr>,
    pub iceberg_storage_properties: IcebergProperties,
    pub compression: PlSmallStr,
    pub compression_level: Option<i32>,
    pub row_group_size: Option<usize>,
    pub maintain_order: bool,
    pub sort_order_id: Option<i32>,
    pub partition_spec_id: Option<i32>,

    pub sink_uuid_str: String,

    #[cfg(feature = "python")]
    pub table_: Option<PythonObject>, // NoPickleOption[pyiceberg.table.Table]

    #[cfg(feature = "python")]
    pub catalog_: Option<PythonObject>, // NoPickleOption[pyiceberg.catalog.Catalog]

    #[cfg(feature = "python")]
    pub source_schema: Option<PythonObject>, // pyarrow.Schema

    #[cfg(feature = "python")]
    pub commit_result_df: Option<PythonObject>, // NoPickleOption[pl.DataFrame]

    #[cfg(feature = "python")]
    pub replace_files: Option<PythonObject>, // _ReplaceFiles: the data files a commit replaces
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergCommitMode {
    Append,
    Overwrite,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergSchemaMode {
    Merge,
    Overwrite,
}

#[cfg(feature = "python")]
mod _python_impl {
    use std::convert::Infallible;

    use pyo3::exceptions::PyValueError;
    use pyo3::pybacked::PyBackedStr;
    use pyo3::types::PyString;
    use pyo3::{Borrowed, Bound, FromPyObject, IntoPyObject, Py, PyAny, PyErr, PyResult, Python};

    use super::{IcebergCommitMode, IcebergSchemaMode, IcebergSinkState};

    impl IcebergSinkState {
        pub(crate) fn into_sink_state_obj(self) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                polars_utils::python_convert_registry::get_python_convert_registry()
                    .py_iceberg_sink_state_class(py)
                    .call(py, (), Some(&self.into_pyobject(py)?))
            })
        }
    }

    impl<'py> IntoPyObject<'py> for IcebergCommitMode {
        type Target = PyString;
        type Output = Bound<'py, Self::Target>;
        type Error = Infallible;

        fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Self::Append => "append",
                Self::Overwrite => "overwrite",
            }
            .into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for IcebergCommitMode {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(match &*ob.extract::<PyBackedStr>()? {
                "append" => Self::Append,
                "overwrite" => Self::Overwrite,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "invalid iceberg commit mode: '{v}'"
                    )));
                },
            })
        }
    }

    impl<'py> IntoPyObject<'py> for IcebergSchemaMode {
        type Target = PyString;
        type Output = Bound<'py, Self::Target>;
        type Error = Infallible;

        fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Self::Merge => "merge",
                Self::Overwrite => "overwrite",
            }
            .into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for IcebergSchemaMode {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(match &*ob.extract::<PyBackedStr>()? {
                "merge" => Self::Merge,
                "overwrite" => Self::Overwrite,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "invalid iceberg schema mode: '{v}'"
                    )));
                },
            })
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use std::collections::BTreeMap;

    use polars_utils::pl_serialize;
    use polars_utils::pl_str::PlSmallStr;

    use super::super::iceberg_properties::{IcebergProperties, JsonProperties};
    use super::{IcebergCommitMode, IcebergSinkState};

    fn properties<const N: usize>(entries: [(&str, &str); N]) -> IcebergProperties {
        IcebergProperties::Json(JsonProperties(Box::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.into()))
                .collect(),
        )))
    }

    /// The non-optional fields of `IcebergSinkState` before catalog names were accepted.
    #[derive(serde::Serialize)]
    struct OldIcebergSinkState {
        py_catalog_class_module: PlSmallStr,
        py_catalog_class_qualname: PlSmallStr,
        catalog_name: PlSmallStr,
        catalog_properties: IcebergProperties,
        table_name: PlSmallStr,
        mode: IcebergCommitMode,
        dynamic_partition_overwrite: bool,
        create_if_not_exists: bool,
        create_namespace_if_not_exists: bool,
        snapshot_properties: BTreeMap<PlSmallStr, PlSmallStr>,
        iceberg_storage_properties: IcebergProperties,
        compression: PlSmallStr,
        maintain_order: bool,
        sink_uuid_str: String,
    }

    #[test]
    fn test_deserialize_sink_state_from_older_client() {
        let old = OldIcebergSinkState {
            py_catalog_class_module: "pyiceberg.catalog.sql".into(),
            py_catalog_class_qualname: "SqlCatalog".into(),
            catalog_name: "sales".into(),
            catalog_properties: properties([("uri", "sqlite://")]),
            table_name: "ns.orders".into(),
            mode: IcebergCommitMode::Append,
            dynamic_partition_overwrite: false,
            create_if_not_exists: false,
            create_namespace_if_not_exists: false,
            snapshot_properties: BTreeMap::new(),
            iceberg_storage_properties: properties([]),
            compression: "zstd".into(),
            maintain_order: true,
            sink_uuid_str: "uuid".into(),
        };

        let mut bytes = vec![];
        pl_serialize::serialize_dsl(&mut bytes, &old).unwrap();
        let state: IcebergSinkState = pl_serialize::deserialize_dsl(&mut bytes.as_slice()).unwrap();

        assert_eq!(
            state.py_catalog_class_module.as_deref(),
            Some("pyiceberg.catalog.sql")
        );
        assert_eq!(
            state.py_catalog_class_qualname.as_deref(),
            Some("SqlCatalog")
        );
        assert_eq!(state.catalog_name, "sales");
        assert_eq!(state.catalog_properties, old.catalog_properties);
        assert_eq!(state.table_name, "ns.orders");
    }
}
