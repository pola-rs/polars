use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyNone};
use pyo3::{IntoPyObjectExt, PyTypeInfo, intern};

use self::python_dsl::PythonScanSource;
use super::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) predicate_serialized: Option<Vec<u8>>,
}

impl PythonScanExec {
    /// Get the output schema. E.g. the schema the plugins produce, not consume.
    fn get_schema(&self) -> &SchemaRef {
        self.options
            .output_schema
            .as_ref()
            .unwrap_or(&self.options.schema)
    }

    fn check_schema(&self, df: &DataFrame) -> PolarsResult<()> {
        if self.options.validate_schema {
            let output_schema = self.get_schema();
            polars_ensure!(df.schema() == output_schema, SchemaMismatch: "user provided schema: {:?} doesn't match the DataFrame schema: {:?}", output_schema, df.schema());
        }
        Ok(())
    }

    fn finish_df(
        &self,
        py: Python,
        df: Bound<'_, PyAny>,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        let df = python_df_to_rust(py, df)?;

        self.check_schema(&df)?;

        if let Some(pred) = &self.predicate {
            let mask = pred.evaluate(&df, state)?;
            df.filter(mask.bool()?)
        } else {
            Ok(df)
        }
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

            match self.options.python_source {
                PythonScanSource::Cuda => {
                    let args = (
                        python_scan_function,
                        with_columns
                            .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                        predicate,
                        n_rows,
                        // If this boolean is true, callback should return
                        // a dataframe and list of timings [(start, end,
                        // name)]
                        state.has_node_timer(),
                    );
                    let result = callable.call1(args)?;
                    let df = if state.has_node_timer() {
                        let df = result.get_item(0);
                        let timing_info: Vec<(u64, u64, String)> = result.get_item(1)?.extract()?;
                        state.record_raw_timings(&timing_info);
                        df?
                    } else {
                        result
                    };
                    self.finish_df(py, df, state)
                },
                PythonScanSource::Pyarrow => {
                    let args = (
                        python_scan_function,
                        with_columns
                            .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                        predicate,
                        n_rows,
                    );
                    let df = callable.call1(args)?;
                    self.finish_df(py, df, state)
                },
                PythonScanSource::IOPlugin => {
                    // If there are filters, take smaller chunks to ensure we can keep memory
                    // pressure low.
                    let batch_size = if self.predicate.is_some() {
                        Some(100_000usize)
                    } else {
                        None
                    };
                    let args = (
                        python_scan_function,
                        with_columns
                            .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                        predicate,
                        n_rows,
                        batch_size,
                    );

                    let generator_init = callable.call1(args)?;
                    let generator = generator_init.get_item(0).map_err(
                        |_| polars_err!(ComputeError: "expected tuple got {}", generator_init),
                    )?;
                    let can_parse_predicate = generator_init.get_item(1).map_err(
                        |_| polars_err!(ComputeError: "expected tuple got {}", generator),
                    )?;
                    let can_parse_predicate = can_parse_predicate.extract::<bool>().map_err(
                        |_| polars_err!(ComputeError: "expected bool got {}", can_parse_predicate),
                    )? && could_serialize_predicate;

                    let mut chunks = vec![];
                    loop {
                        match generator.call_method0(intern!(py, "__next__")) {
                            Ok(out) => {
                                let mut df = python_df_to_rust(py, out)?;
                                if let (Some(pred), false) = (&self.predicate, can_parse_predicate)
                                {
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
                    if chunks.is_empty() {
                        return Ok(DataFrame::empty_with_schema(self.get_schema().as_ref()));
                    }
                    let df = accumulate_dataframes_vertical(chunks)?;

                    self.check_schema(&df)?;
                    Ok(df)
                },
            }
        })
    }
}
