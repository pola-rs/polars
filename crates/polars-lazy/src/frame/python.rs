use std::sync::Arc;

use polars_plan::dsl::python_dsl::SchemaProvider;
use pyo3::PyObject;

use self::python_dsl::{PythonOptionsDsl, PythonScanSource};
use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(
        schema: SchemaProvider,
        scan_fn: PyObject,
        pyarrow: bool,
        // Validate that the source gives the proper schema
        validate_schema: bool,
    ) -> Self {
        DslPlan::PythonScan {
            options: PythonOptionsDsl {
                // Should be a python function that returns a generator
                scan_fn: Some(scan_fn.into()),
                schema_provider: Some(SpecialEq::new(Arc::new(schema))),
                python_source: if pyarrow {
                    PythonScanSource::Pyarrow
                } else {
                    PythonScanSource::IOPlugin
                },
                validate_schema,
            },
        }
        .into()
    }
}
