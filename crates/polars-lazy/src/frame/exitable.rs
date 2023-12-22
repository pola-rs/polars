use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Mutex;

use polars_core::POOL;

use super::*;

impl LazyFrame {
    pub fn collect_concurrently(self) -> PolarsResult<InProcessQuery> {
        let (mut state, mut physical_plan, _) = self.prepare_collect(false)?;

        let (tx, rx) = channel();
        let token = state.cancel_token();
        POOL.spawn_fifo(move || {
            let result = physical_plan.execute(&mut state);
            tx.send(result).unwrap();
        });

        Ok(InProcessQuery {
            rx: Arc::new(Mutex::new(rx)),
            token,
        })
    }
}

#[derive(Clone)]
pub struct InProcessQuery {
    rx: Arc<Mutex<Receiver<PolarsResult<DataFrame>>>>,
    token: Arc<AtomicBool>,
}

impl InProcessQuery {
    /// Cancel the query at earliest convenience.
    pub fn cancel(&self) {
        self.token.store(true, Ordering::Relaxed)
    }

    /// Fetch the result.

    /// If it is ready, a materialized DataFrame is returned.
    /// If it is not ready it will return `None`.
    pub fn fetch(&self) -> Option<PolarsResult<DataFrame>> {
        let rx = self.rx.lock().unwrap();
        rx.try_recv().ok()
    }

    /// Await the result synchronously.
    pub fn fetch_blocking(&self) -> PolarsResult<DataFrame> {
        let rx = self.rx.lock().unwrap();
        rx.recv().unwrap()
    }
}

impl Drop for InProcessQuery {
    fn drop(&mut self) {
        self.token.store(true, Ordering::Relaxed);
    }
}
