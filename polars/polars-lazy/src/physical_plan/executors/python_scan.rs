use polars_core::error::to_compute_err;
use pyo3::prelude::*;

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
        let n_rows = self.options.n_rows.take();
        Python::with_gil(|py| {
            let pl = PyModule::import(py, "polars").unwrap();
            let utils = pl.getattr("utils").unwrap();
            let callable = utils.getattr("_execute_from_rust").unwrap();

            let python_scan_function = self.options.scan_fn.take().unwrap().0;

            let with_columns =
                with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols)));

            let out = callable
                .call1((
                    python_scan_function,
                    with_columns,
                    pyarrow_predicate,
                    n_rows,
                ))
                .map_err(to_compute_err)?;
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
