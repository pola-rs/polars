use std::sync::Arc;

use polars_core::config;
use polars_plan::plans::{ExpandedPythonScan, python_df_to_rust};
use polars_utils::format_pl_smallstr;
use pyo3::exceptions::PyStopIteration;
use pyo3::{PyTypeInfo, intern};

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
            let generator = Python::attach(|py| {
                let generator = expanded_scan.scan_fn.call0(py).unwrap();

                generator.bind(py).get_item(0).unwrap().unbind()
            });

            (
                format_pl_smallstr!("python[{} @ pyarrow]", &expanded_scan.name),
                Box::new(move |_state: &StreamingExecutionState| {
                    Python::attach(|py| {
                        let generator = generator.bind(py);

                        match generator.call_method0(intern!(py, "__next__")) {
                            Ok(out) => python_df_to_rust(py, out).map(Some),
                            Err(err) if err.matches(py, PyStopIteration::type_object(py))? => {
                                Ok(None)
                            },
                            err => {
                                let _ = err?;
                                unreachable!()
                            },
                        }
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
