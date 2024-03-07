use std::sync::{Arc, Mutex};
use polars_core::error::PolarsResult;
use polars_utils::cell::SyncUnsafeCell;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

type Replace = Arc<Mutex<Option<Box<dyn Operator>>>>;

#[derive(Clone)]
struct CallBack {
    inner: Replace
}

impl CallBack {
    fn new() -> Self {
        Self {
            inner: Default::default()
        }
    }
    
    pub fn replace(&self, op: Box<dyn Operator>) {
        let mut inner = self.inner.lock().unwrap();
        *inner = Some(op);
    }
}

impl Operator for CallBack {
    fn execute(&mut self, context: &PExecutionContext, chunk: &DataChunk) -> PolarsResult<OperatorResult> {
        let mut inner = self.inner.lock().unwrap();
        let op = inner.as_mut().unwrap();
        op.execute(context, chunk)
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        panic!("should not be called")
    }

    fn fmt(&self) -> &str {
        "callback"
    }
}

#[derive(Default, Clone)]
pub struct PlaceHolder {
    inner: Arc<Mutex<Vec<(usize, CallBack)>>>
}

impl PlaceHolder {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Default::default())
        }
    }

    pub fn replace(&self, op: Box<dyn Operator>) {
        let mut inner = self.inner.lock().unwrap();
        for (thread_no, cb) in inner.iter() {
            cb.replace(op.split(*thread_no))
        }
    }
}

impl Operator for PlaceHolder {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        _chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        panic!("placeholder shouldbe replaced")
    }

    fn split(&self, thread_no: usize) -> Box<dyn Operator> {
        let cb = CallBack::new();
        let mut inner = self.inner.lock().unwrap();
        inner.push((thread_no, cb.clone()));
        Box::new(cb)
    }

    fn fmt(&self) -> &str {
        "placeholder"
    }
}
