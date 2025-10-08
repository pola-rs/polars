use std::sync::{Arc, Mutex};

use polars_core::config;
use polars_plan::plans::{ExpandedPythonScan, python_df_to_rust};
use polars_utils::format_pl_smallstr;

use crate::execute::StreamingExecutionState;
use crate::nodes::io_sources::batch::GetBatchFn;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;

/// Note: Currently used for iceberg fallback.
pub fn python_dataset_scan_to_reader_builder(
    expanded_scan: &ExpandedPythonScan,
) -> Arc<dyn FileReaderBuilder> {
    use polars_plan::dsl::python_dsl::PythonScanSource as S;
    use pyo3::prelude::*;

    let (name, get_batch_fn) = match &expanded_scan.variant {
        S::Pyarrow => {
            // * Pyarrow is a oneshot function call.
            // * Arc / Mutex because because closure cannot be FnOnce
            let python_scan_function = Arc::new(Mutex::new(Some(expanded_scan.scan_fn.clone())));

            (
                format_pl_smallstr!("python[{} @ pyarrow]", &expanded_scan.name),
                Box::new(move |_state: &StreamingExecutionState| {
                    Python::with_gil(|py| {
                        let Some(python_scan_function) =
                            python_scan_function.lock().unwrap().take()
                        else {
                            return Ok(None);
                        };

                        // Note: to_dataset_scan() has already captured projection / limit.

                        let df = python_scan_function.call0(py)?;
                        let df = python_df_to_rust(py, df.bind(py).clone())?;

                        Ok(Some(df))
                    })
                }) as GetBatchFn,
            )
        },

        s => todo!("{:?}", s),
    };

    use crate::nodes::io_sources::batch::builder::BatchFnReaderBuilder;
    use crate::nodes::io_sources::batch::{BatchFnReader, GetBatchState};

    let reader = BatchFnReader {
        name: name.clone(),
        output_schema: None,
        get_batch_state: Some(GetBatchState::from(get_batch_fn)),
        execution_state: None,
        verbose: config::verbose(),
    };

    Arc::new(BatchFnReaderBuilder {
        name,
        reader: std::sync::Mutex::new(Some(reader)),
        execution_state: Default::default(),
    }) as Arc<dyn FileReaderBuilder>
}
