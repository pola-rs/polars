use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
}

impl Executor for PythonScanExec {
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if _state.verbose() {
                println!("run PythonScanExec")
            }
        }
        let with_columns = self.options.with_columns.take();
        let pyarrow_predicate = self.options.predicate.take();
        Python::with_gil(|py| {
            let pl = PyModule::import(py, "polars").unwrap();
            let pli = pl.getattr("internals").unwrap();
            let deser_and_exec = pli.getattr("_deser_and_exec").unwrap();

            let bytes = PyBytes::new(py, &self.options.scan_fn);

            let with_columns =
                with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols)));

            let out = deser_and_exec
                .call1((bytes, with_columns, pyarrow_predicate))
                .map_err(|err| PolarsError::ComputeError(format!("{err:?}").into()))?;
            let pydf = out.getattr("_df").unwrap();
            let raw_parts = pydf.call_method0("into_raw_parts").unwrap();
            let raw_parts = raw_parts.extract::<(usize, usize, usize)>().unwrap();

            let (ptr, len, cap) = raw_parts;
            unsafe {
                Ok(DataFrame::new_no_checks(Vec::from_raw_parts(
                    ptr as *mut Series,
                    len,
                    cap,
                )))
            }
        })
    }
}
