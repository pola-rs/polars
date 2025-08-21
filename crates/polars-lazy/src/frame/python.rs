use std::sync::Arc;

use either::Either;
use polars_core::schema::SchemaRef;
use polars_utils::pl_str::PlSmallStr;
use pyo3::PyObject;

use self::python_dsl::{PythonOptionsDsl, PythonScanSource};
use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(
        schema: Either<PyObject, SchemaRef>,
        scan_fn: PyObject,
        pyarrow: bool,
        // Validate that the source gives the proper schema
        validate_schema: bool,
        explain_name: Option<PlSmallStr>,
        explain_detail: Option<PlSmallStr>,
        explain_subplan: Option<PlSmallStr>,
    ) -> Self {
        DslPlan::PythonScan {
            options: PythonOptionsDsl {
                // Should be a python function that returns a generator
                scan_fn: Some(scan_fn.into()),
                schema_fn: Some(SpecialEq::new(Arc::new(schema.map_left(|obj| obj.into())))),
                python_source: if pyarrow {
                    PythonScanSource::Pyarrow
                } else {
                    PythonScanSource::IOPlugin
                },
                validate_schema,
                explain_name,
                explain_detail,
                explain_subplan,
            },
        }
        .into()
    }
}
