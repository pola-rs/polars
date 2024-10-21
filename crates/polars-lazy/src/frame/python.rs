use polars_core::prelude::*;
use pyo3::PyObject;
use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(schema: Schema, scan_fn: PyObject, pyarrow: bool) -> Self {
        DslPlan::PythonScan {
            options: PythonOptions {
                // Should be a python function that returns a generator
                scan_fn: Some(scan_fn.into()),
                schema: PySchemaSource::SchemaRef(Arc::new(schema)),
                python_source: if pyarrow {
                    PythonScanSource::Pyarrow
                } else {
                    PythonScanSource::IOPlugin
                },
                ..Default::default()
            },
        }
        .into()
    }

    pub fn scan_from_python_functions(schema_fn: PyObject, scan_fn: PyObject) -> Self {
        DslPlan::PythonScan {
            options: PythonOptions {
                schema: PySchemaSource::PythonFunction(schema_fn.into()),
                scan_fn: Some(scan_fn.into()),
                python_source: PythonScanSource::IOPluginDeferredSchema,
                ..Default::default()
            },
        }
        .into()
    }
}
