use std::collections::BTreeMap;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;

/// `class DeltaSinkState` in Python
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "python", derive(pyo3::IntoPyObject, pyo3::FromPyObject))]
pub struct DeltaSinkState {
    pub table_uri: String,
    pub mode: DeltaCommitMode,
    pub storage_options: BTreeMap<PlSmallStr, PlSmallStr>,

    pub sink_uuid_str: String,

    /// Partition columns of the table when the sink was resolved.
    pub partition_columns: Option<Vec<PlSmallStr>>,

    #[cfg(feature = "python")]
    pub table_: Option<PythonObject>, // NoPickleOption[_LoadedTable]

    #[cfg(feature = "python")]
    pub source_schema: Option<PythonObject>, // str: Delta schema JSON

    #[cfg(feature = "python")]
    pub commit_result_df: Option<PythonObject>, // NoPickleOption[pl.DataFrame]
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DeltaCommitMode {
    ErrorIfExists,
    Append,
    Overwrite,
    Ignore,
}

#[cfg(feature = "python")]
mod _python_impl {
    use std::convert::Infallible;

    use pyo3::exceptions::PyValueError;
    use pyo3::pybacked::PyBackedStr;
    use pyo3::types::PyString;
    use pyo3::{Borrowed, Bound, FromPyObject, IntoPyObject, Py, PyAny, PyErr, PyResult, Python};

    use super::{DeltaCommitMode, DeltaSinkState};

    impl DeltaSinkState {
        pub(crate) fn into_sink_state_obj(self) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                polars_utils::python_convert_registry::get_python_convert_registry()
                    .py_delta_sink_state_class(py)
                    .call(py, (), Some(&self.into_pyobject(py)?))
            })
        }
    }

    impl<'py> IntoPyObject<'py> for DeltaCommitMode {
        type Target = PyString;
        type Output = Bound<'py, Self::Target>;
        type Error = Infallible;

        fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Self::ErrorIfExists => "error",
                Self::Append => "append",
                Self::Overwrite => "overwrite",
                Self::Ignore => "ignore",
            }
            .into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for DeltaCommitMode {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(match &*ob.extract::<PyBackedStr>()? {
                "error" => Self::ErrorIfExists,
                "append" => Self::Append,
                "overwrite" => Self::Overwrite,
                "ignore" => Self::Ignore,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "invalid delta commit mode: '{v}'"
                    )));
                },
            })
        }
    }
}
