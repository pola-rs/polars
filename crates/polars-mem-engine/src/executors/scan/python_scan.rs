use polars_core::error::to_compute_err;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_expr::state::node_timer::NodeTimer;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyNone, PyTuple};
use pyo3::{intern, IntoPyObjectExt, PyTypeInfo};

use self::python_dsl::PythonScanSource;
use super::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) predicate_serialized: Option<Vec<u8>>,
}

#[pyclass]
pub struct PyNodeTimer {
    timer: Option<NodeTimer>,
}

#[pymethods]
impl PyNodeTimer {
    pub fn record(
        &self,
        py: Python,
        name: &str,
        func: PyObject,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<PyObject> {
        match &self.timer {
            None => Ok(func.call1(py, args)?),
            Some(timer) => {
                let start = std::time::Instant::now();
                let result = func.call1(py, args)?;
                let end = std::time::Instant::now();
                timer.store(start, end, name.to_string());
                Ok(result)
            },
        }
    }
}

impl PyNodeTimer {
    pub fn new(timer: Option<NodeTimer>) -> Self {
        PyNodeTimer { timer }
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
            let pl = PyModule::import(py, intern!(py, "polars")).unwrap();
            let utils = pl.getattr(intern!(py, "_utils")).unwrap();
            let callable = utils.getattr(intern!(py, "_execute_from_rust")).unwrap();

            let python_scan_function = self.options.scan_fn.take().unwrap().0;

            let with_columns = with_columns.map(|cols| cols.iter().cloned().collect::<Vec<_>>());
            let mut could_serialize_predicate = true;

            let predicate = match &self.options.predicate {
                PythonPredicate::PyArrow(s) => s.into_bound_py_any(py).unwrap(),
                PythonPredicate::None => None::<()>.into_bound_py_any(py).unwrap(),
                PythonPredicate::Polars(_) => {
                    assert!(self.predicate.is_some(), "should be set");

                    match &self.predicate_serialized {
                        None => {
                            could_serialize_predicate = false;
                            PyNone::get(py).to_owned().into_any()
                        },
                        Some(buf) => PyBytes::new(py, buf).into_any(),
                    }
                },
            };

            let generator_init = if matches!(self.options.python_source, PythonScanSource::Cuda) {
                let args = (
                    python_scan_function,
                    with_columns.map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                    predicate,
                    n_rows,
                    true,
                );
                let result = callable.call1(args).map_err(to_compute_err).clone()?;

                let df = result
                    .get_item(0)
                    .map_err(|_| polars_err!(ComputeError: "expected tuple got {:?}", result))?;

                let timing_info = result
                    .get_item(1)
                    .map_err(|_| polars_err!(ComputeError: "expected tuple got {:?}", result))?;

                let name: String = timing_info
                .get_item(0)
                .map_err(|_| polars_err!(ComputeError: "Failed to extract name got {:?}", timing_info))?
                .extract::<String>()
                .map_err(|_| polars_err!(ComputeError: "Failed to convert name to String {:?}", timing_info))?;

                let start: u64 = timing_info
                    .get_item(1)
                    .map_err(|_| polars_err!(ComputeError: "Failed to extract start time"))?
                    .extract::<u64>()
                    .map_err(
                        |_| polars_err!(ComputeError: "Failed to convert start time to u64"),
                    )?;

                let end: u64 = timing_info
                    .get_item(2)
                    .map_err(|_| polars_err!(ComputeError: "Failed to extract end time"))?
                    .extract::<u64>()
                    .map_err(|_| polars_err!(ComputeError: "Failed to convert end time to u64"))?;
                if let Some(node_timer) = state.node_timer.as_ref() {
                    node_timer.store_raw(start, end, name);
                }
                Ok(df)
            } else if matches!(self.options.python_source, PythonScanSource::Pyarrow) {
                let args = (
                    python_scan_function,
                    with_columns.map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                    predicate,
                    n_rows,
                );
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
                    with_columns.map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
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
            )? && could_serialize_predicate;

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
                    Err(err) if err.matches(py, PyStopIteration::type_object(py))? => break,
                    Err(err) => {
                        polars_bail!(ComputeError: "caught exception during execution of a Python source, exception: {}", err)
                    },
                }
            }
            accumulate_dataframes_vertical(chunks)
        })
    }
}
