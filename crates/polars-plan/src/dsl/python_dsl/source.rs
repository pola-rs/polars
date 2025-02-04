use polars_core::error::{polars_err, PolarsResult};
use polars_core::schema::SchemaRef;
use polars_utils::python_function::PythonFunction;
use pyo3::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PythonOptionsDsl {
    /// A function that returns a Python Generator.
    /// The generator should produce Polars DataFrame's.
    pub scan_fn: Option<PythonFunction>,
    /// Schema of the file.
    pub schema_fn: Option<PythonFunction>,
    pub python_source: PythonScanSource,
}

impl PythonOptionsDsl {
    pub(crate) fn get_schema(&self) -> PolarsResult<SchemaRef> {
        Python::with_gil(|py| {
            let schema = self
                .schema_fn
                .as_ref()
                .expect("should be set")
                .0
                .call0(py)
                .map_err(|e| polars_err!(ComputeError: "schema callable failed: {}", e))?;

            crate::plans::python::python_schema_to_rust(py, schema.into_bound(py))
        })
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PythonScanSource {
    Pyarrow,
    Cuda,
    #[default]
    IOPlugin,
}
