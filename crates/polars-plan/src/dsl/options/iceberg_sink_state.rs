use std::collections::BTreeMap;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;

/// `class IcebergSinkState` in Python
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct IcebergSinkState {
    /// Python class module
    pub py_catalog_class_module: PlSmallStr,
    /// Python class qualname
    pub py_catalog_class_qualname: PlSmallStr,

    pub catalog_name: PlSmallStr,
    pub catalog_properties: BTreeMap<PlSmallStr, PlSmallStr>,

    pub table_name: PlSmallStr,
    pub mode: IcebergCommitMode,
    pub iceberg_storage_properties: BTreeMap<PlSmallStr, PlSmallStr>,

    pub sink_uuid_str: String,

    #[cfg(feature = "python")]
    pub table_: Option<PythonObject>, // NoPickleOption[pyiceberg.table.Table]

    #[cfg(feature = "python")]
    pub commit_result_df: Option<PythonObject>, // NoPickleOption[pl.DataFrame]
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
    use pyo3::{Borrowed, Bound, FromPyObject, IntoPyObject, Py, PyAny, PyErr, PyResult, Python};

    use super::{IcebergCommitMode, IcebergSinkState};

    impl IcebergSinkState {
        pub(crate) fn into_sink_state_obj(self) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                polars_utils::python_convert_registry::get_python_convert_registry()
                    .py_iceberg_sink_state_class()
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
}
