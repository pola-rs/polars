use pyo3::PyObject;

use self::python_dsl::{PythonOptionsDsl, PythonScanSource};
use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(schema: PyObject, scan_fn: PyObject, pyarrow: bool) -> Self {
        DslPlan::PythonScan {
            options: PythonOptionsDsl {
                // Should be a python function that returns a generator
                scan_fn: Some(scan_fn.into()),
                schema_fn: Some(schema.into()),
                python_source: if pyarrow {
                    PythonScanSource::Pyarrow
                } else {
                    PythonScanSource::IOPlugin
                },
            },
        }
        .into()
    }
}
