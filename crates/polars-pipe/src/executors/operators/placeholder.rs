use std::sync::{Arc, Mutex};
use polars_core::error::PolarsResult;
use polars_utils::cell::SyncUnsafeCell;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Default, Clone)]
pub struct PlaceHolder {
    inner: Arc<Mutex<Option<Box<dyn Operator>>>>
}

impl PlaceHolder {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Default::default())
        }
    }

    pub fn replace(&self, op: Box<dyn Operator>) {
        let mut inner = self.inner.lock().unwrap();
        *inner = Some(op);
    }
}

impl Operator for PlaceHolder {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let mut inner = self.inner.lock().unwrap();
        let op = inner.as_mut().expect("placeholder should be replaced");
        op.execute(context, chunk)
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(Self {
            inner: self.inner.clone()
        })
    }

    fn fmt(&self) -> &str {
        "placeholder"
    }
}
