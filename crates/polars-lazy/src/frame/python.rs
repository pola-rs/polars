use polars_core::prelude::*;
use pyo3::PyObject;

use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(schema: Schema, scan_fn: PyObject, pyarrow: bool) -> Self {
        DslPlan::PythonScan {
            options: PythonOptions {
                // Should be a python function that returns a generator
                scan_fn: Some(scan_fn.into()),
                schema: Arc::new(schema),
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
}
