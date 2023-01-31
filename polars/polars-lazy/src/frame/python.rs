use polars_core::prelude::*;

use crate::prelude::*;

impl LazyFrame {
    pub fn scan_from_python_function(schema: Schema, scan_fn: Vec<u8>, pyarrow: bool) -> Self {
        LogicalPlan::PythonScan {
            options: PythonOptions {
                scan_fn,
                schema: Arc::new(schema),
                pyarrow,
                ..Default::default()
            },
        }
        .into()
    }
}
