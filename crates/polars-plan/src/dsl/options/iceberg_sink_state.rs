//! Data models with from/to pyobject derives that match with dataclasses declared
//! on the Python side.
use std::collections::BTreeMap;

use polars_utils::pl_str::PlSmallStr;

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct IcebergSinkState {
    table: IcebergTableWrap,
    mode: IcebergCommitMode,
    sink_uuid_str: String,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct IcebergTableWrap {
    table_descriptor_: IcebergCatalogTableDescriptor,
    iceberg_storage_properties: Option<BTreeMap<PlSmallStr, PlSmallStr>>,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct IcebergCatalogTableDescriptor {
    table_identifier: PlSmallStr,
    catalog_config: PyIcebergCatalogConfig,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct PyIcebergCatalogConfig {
    class_module: PlSmallStr,
    class_qualname: PlSmallStr,
    /// Name of the catalog
    name: PlSmallStr,
    properties: BTreeMap<PlSmallStr, PlSmallStr>,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergCommitMode {
    Append,
    Overwrite,
}

#[cfg(feature = "python")]
mod _python_impl {
    use std::convert::Infallible;

    use pyo3::exceptions::PyValueError;
    use pyo3::pybacked::PyBackedStr;
    use pyo3::types::PyString;
    use pyo3::{
        Borrowed, Bound, FromPyObject, IntoPyObject, Py, PyAny, PyErr, PyResult, Python, intern,
    };

    use super::{IcebergCommitMode, IcebergSinkState};

    impl IcebergSinkState {
        pub(crate) fn get_py_iceberg_sink_state(
            self,
            orig_sink_state_obj: Option<&Py<PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let py_iceberg_sink_state =
                    polars_utils::python_convert_registry::get_python_convert_registry()
                        .py_iceberg_sink_state_class()
                        .call_method1(
                            py,
                            intern!(py, "_from_data_dict"),
                            (self.into_pyobject(py)?,),
                        )?;

                if let Some(orig_sink_state_obj) = orig_sink_state_obj {
                    let table_str = intern!(py, "table");

                    py_iceberg_sink_state
                        .getattr(py, table_str)?
                        .getattr(py, intern!(py, "table_"))?
                        .call_method1(
                            py,
                            intern!(py, "set"),
                            ({
                                orig_sink_state_obj
                                    .getattr(py, table_str)?
                                    .call_method0(py, intern!(py, "get"))?
                            },),
                        )?;
                }

                Ok(py_iceberg_sink_state)
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
}
