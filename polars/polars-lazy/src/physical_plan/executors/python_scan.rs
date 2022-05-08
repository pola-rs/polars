use super::*;
use crate::prelude::*;
use pyo3::prelude::*;

pub(crate) struct PythonScanExec {
    pub(crate) options: PythonOptions,
}

impl Executor for PythonScanExec {
    fn execute(&mut self, cache: &ExecutionState) -> Result<DataFrame> {
        todo!()
    }
}
