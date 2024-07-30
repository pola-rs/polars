use polars_core::error::to_compute_err;
use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{intern, PyTypeInfo};

use super::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) predicate_serialized: Option<Vec<u8>>,
}

fn python_df_to_rust(py: Python, df: Bound<PyAny>) -> PolarsResult<DataFrame> {
    let err = |_| polars_err!(ComputeError: "expected a polars.DataFrame; got {}", df);
    let pydf = df.getattr(intern!(py, "_df")).map_err(err)?;
    let raw_parts = pydf
        .call_method0(intern!(py, "into_raw_parts"))
        .map_err(err)?;
    let raw_parts = raw_parts.extract::<(usize, usize, usize)>().unwrap();

    let (ptr, len, cap) = raw_parts;
    unsafe {
        Ok(DataFrame::new_no_checks(Vec::from_raw_parts(
            ptr as *mut Series,
            len,
            cap,
        )))
    }
}

impl Executor for PythonScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run PythonScanExec")
            }
        }
        let with_columns = self.options.with_columns.take();
        let n_rows = self.options.n_rows.take();
        Python::with_gil(|py| {
            let pl = PyModule::import_bound(py, intern!(py, "polars")).unwrap();
            let utils = pl.getattr(intern!(py, "_utils")).unwrap();
            let callable = utils.getattr(intern!(py, "_execute_from_rust")).unwrap();

            let python_scan_function = self.options.scan_fn.take().unwrap().0;

            let with_columns = with_columns.map(|cols| cols.iter().cloned().collect::<Vec<_>>());

            let predicate = match &self.options.predicate {
                PythonPredicate::PyArrow(s) => s.into_py(py),
                PythonPredicate::None => None::<()>.into_py(py),
                PythonPredicate::Polars(_) => {
                    assert!(self.predicate.is_some(), "should be set");

                    match &self.predicate_serialized {
                        None => None::<()>.into_py(py),
                        Some(buf) => PyBytes::new_bound(py, buf).to_object(py),
                    }
                },
            };

            let generator_init = if matches!(
                self.options.python_source,
                PythonScanSource::Pyarrow | PythonScanSource::Cuda
            ) {
                let args = (python_scan_function, with_columns, predicate, n_rows);
                callable.call1(args).map_err(to_compute_err)
            } else {
                // If there are filters, take smaller chunks to ensure we can keep memory
                // pressure low.
                let batch_size = if self.predicate.is_some() {
                    Some(100_000usize)
                } else {
                    None
                };
                let args = (
                    python_scan_function,
                    with_columns,
                    predicate,
                    n_rows,
                    batch_size,
                );
                callable.call1(args).map_err(to_compute_err)
            }?;

            // This isn't a generator, but a `DataFrame`.
            // This is the pyarrow and the CuDF path.
            if generator_init.getattr(intern!(py, "_df")).is_ok() {
                let df = python_df_to_rust(py, generator_init)?;
                return if let Some(pred) = &self.predicate {
                    let mask = pred.evaluate(&df, state)?;
                    df.filter(mask.bool()?)
                } else {
                    Ok(df)
                };
            }

            // This is the IO plugin path.
            let generator = generator_init
                .get_item(0)
                .map_err(|_| polars_err!(ComputeError: "expected tuple got {}", generator_init))?;
            let can_parse_predicate = generator_init
                .get_item(1)
                .map_err(|_| polars_err!(ComputeError: "expected tuple got {}", generator))?;
            let can_parse_predicate = can_parse_predicate.extract::<bool>().map_err(
                |_| polars_err!(ComputeError: "expected bool got {}", can_parse_predicate),
            )?;

            let mut chunks = vec![];
            loop {
                match generator.call_method0(intern!(py, "__next__")) {
                    Ok(out) => {
                        let mut df = python_df_to_rust(py, out)?;
                        if let (Some(pred), false) = (&self.predicate, can_parse_predicate) {
                            let mask = pred.evaluate(&df, state)?;
                            df = df.filter(mask.bool()?)?;
                        }
                        chunks.push(df)
                    },
                    Err(err) if err.matches(py, PyStopIteration::type_object_bound(py)) => break,
                    Err(err) => {
                        polars_bail!(ComputeError: "caught exception during execution of a Python source, exception: {}", err)
                    },
                }
            }
            accumulate_dataframes_vertical(chunks)
        })
    }
}
