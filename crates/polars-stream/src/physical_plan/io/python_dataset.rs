use std::sync::{Arc, Mutex};

use polars_core::config;
use polars_plan::dsl::python_dsl::PythonScanSource;
use polars_plan::plans::python_df_to_rust;
use polars_utils::format_pl_smallstr;
use polars_utils::python_function::PythonObject;

use crate::execute::StreamingExecutionState;
use crate::nodes::io_sources::batch::GetBatchFn;
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;

/// Note: Currently used for iceberg fallback.
pub fn python_dataset_scan_to_reader_builder(
    reader_name: &str,
    scan_fn: PythonObject,
    python_source_type: &PythonScanSource,
) -> Arc<dyn FileReaderBuilder> {
    use polars_plan::dsl::python_dsl::PythonScanSource as S;
    use pyo3::prelude::*;

    let (name, get_batch_fn) = match python_source_type {
        S::Pyarrow => {
            // * Pyarrow is a oneshot function call.
            // * Arc / Mutex because because closure cannot be FnOnce
            let python_scan_function = Arc::new(Mutex::new(Some(scan_fn)));

            (
                format_pl_smallstr!("python[{} @ pyarrow]", reader_name),
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
        verbose: config::verbose(),
    };

    Arc::new(BatchFnReaderBuilder {
        name: name.clone(),
        reader: std::sync::Mutex::new(Some(reader)),
    }) as Arc<dyn FileReaderBuilder>
}
