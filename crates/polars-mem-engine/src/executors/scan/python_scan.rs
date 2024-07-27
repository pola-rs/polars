use polars_core::error::to_compute_err;
use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::{intern, PyTypeInfo};

use super::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
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
                PythonPredicate::None => (None::<()>).into_py(py),
                PythonPredicate::Polars(_) => todo!(),
            };

            let generator = callable
                .call1((python_scan_function, with_columns, predicate, n_rows))
                .map_err(to_compute_err)?;

            // This isn't a generator, but a `DataFrame`.
            if generator.getattr(intern!(py, "_df")).is_ok() {
                return python_df_to_rust(py, generator);
            }

            let mut chunks = vec![];
            loop {
                match generator.call_method0(intern!(py, "__next__")) {
                    Ok(out) => {
                        let df = python_df_to_rust(py, out)?;
                        chunks.push(df)
                    },
                    Err(err) if err.matches(py, PyStopIteration::type_object_bound(py)) => break,
                    Err(err) => {
                        polars_bail!(ComputeError: "catched exception during execution of a Python source, exception: {}", err)
                    },
                }
            }
            accumulate_dataframes_vertical(chunks)
        })
    }
}
