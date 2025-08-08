use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_err};
use polars_core::schema::SchemaRef;
use polars_utils::python_function::PythonFunction;
use pyo3::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::SpecialEq;

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct PythonOptionsDsl {
    /// A function that returns a Python Generator.
    /// The generator should produce Polars DataFrame's.
    pub scan_fn: Option<PythonFunction>,
    /// Either the schema fn or schema is set.
    pub schema_provider: Option<SpecialEq<Arc<SchemaProvider>>>,
    pub python_source: PythonScanSource,
    pub validate_schema: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SchemaProvider {
    Function(PythonFunction),
    Plain(SchemaRef),
}

impl SchemaProvider {
    pub fn get_schema(&self) -> PolarsResult<SchemaRef> {
        match self {
            Self::Function(func) => Python::with_gil(|py| {
                let schema = func
                    .0
                    .call0(py)
                    .map_err(|e| polars_err!(ComputeError: "schema callable failed: {}", e))?;
                crate::plans::python::python_schema_to_rust(py, schema.into_bound(py))
            }),

            Self::Plain(schema) => Ok(schema.clone()),
        }
    }
}

impl PythonOptionsDsl {
    pub fn get_schema(&self) -> PolarsResult<SchemaRef> {
        self.schema_provider
            .as_ref()
            .ok_or_else(|| polars_err!(ComputeError: "PythonOptionsDsl::schema_provider is None"))?
            .get_schema()
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum PythonScanSource {
    Pyarrow,
    Cuda,
    #[default]
    IOPlugin,
}
