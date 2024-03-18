use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
struct CallBack {
    inner: Arc<Mutex<Option<Box<dyn Operator>>>>,
}

impl CallBack {
    fn new() -> Self {
        Self {
            inner: Default::default(),
        }
    }

    fn replace(&self, op: Box<dyn Operator>) {
        let mut lock = self.inner.try_lock().expect("no-contention");
        *lock = Some(op);
    }
}

impl Operator for CallBack {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let mut lock = self.inner.try_lock().expect("no-contention");
        lock.as_mut().unwrap().execute(context, chunk)
    }

    fn flush(&mut self) -> PolarsResult<OperatorResult> {
        let mut lock = self.inner.try_lock().expect("no-contention");
        lock.as_mut().unwrap().flush()
    }

    fn must_flush(&self) -> bool {
        let lock = self.inner.try_lock().expect("no-contention");
        lock.as_ref().unwrap().must_flush()
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        panic!("should not be called")
    }

    fn fmt(&self) -> &str {
        "callback"
    }
}

#[derive(Clone, Default)]
pub struct PlaceHolder {
    inner: Arc<Mutex<Vec<(usize, CallBack)>>>,
}

impl PlaceHolder {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Default::default()),
        }
    }

    pub fn replace(&self, op: Box<dyn Operator>) {
        let inner = self.inner.lock().unwrap();
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
        panic!("placeholder should be replaced")
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
