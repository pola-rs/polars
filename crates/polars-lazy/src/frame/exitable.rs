use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Condvar, Mutex};

use polars_core::runtime::{RAYON, is_polars_thread};
use polars_utils::relaxed_cell::RelaxedCell;

use super::*;

impl LazyFrame {
    pub fn collect_concurrently(self) -> PolarsResult<InProcessQuery> {
        let (state, physical_plan, _) = self.prepare_collect(false)?;
        let physical_plan_is_cache_prefiller = physical_plan.is_cache_prefiller();

        let (tx, rx) = channel();
        let token = state.cancel_token();
        let finished = Arc::new(Finished::default());
        let run = {
            let finished = finished.clone();
            move || run_query(finished, physical_plan, state, tx)
        };

        if physical_plan_is_cache_prefiller {
            #[cfg(feature = "async")]
            {
                polars_core::runtime::ASYNC.spawn_blocking(run);
            }
            #[cfg(not(feature = "async"))]
            {
                std::thread::spawn(run);
            }
        } else {
            RAYON.spawn_fifo(run);
        }

        Ok(InProcessQuery {
            rx: Arc::new(Mutex::new(rx)),
            token,
            finished,
        })
    }
}

/// Executes the query and sends its result, then drops what is left of it (the physical plan,
/// with its sources, and the state) before marking it finished.
fn run_query(
    finished: Arc<Finished>,
    mut physical_plan: Box<dyn Executor>,
    mut state: ExecutionState,
    tx: Sender<PolarsResult<DataFrame>>,
) {
    QUERY_THREAD.set(true);
    // Declared first, dropped last: the query is marked finished even if it panics.
    let _finished = SetFinishedOnDrop(finished);
    let result = physical_plan.execute(&mut state);
    tx.send(result).unwrap();
    drop(physical_plan);
    drop(state);
}

thread_local! {
    /// Set on a thread that runs (or ran) an in-process query: not always one of Polars' pools.
    static QUERY_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the query's thread has finished: its plan and state dropped.
#[derive(Default)]
struct Finished {
    done: Mutex<bool>,
    cond: Condvar,
}

impl Finished {
    fn wait(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = self.cond.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

struct SetFinishedOnDrop(Arc<Finished>);

impl Drop for SetFinishedOnDrop {
    fn drop(&mut self) {
        *self.0.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.0.cond.notify_all();
    }
}

#[derive(Clone)]
pub struct InProcessQuery {
    rx: Arc<Mutex<Receiver<PolarsResult<DataFrame>>>>,
    token: Arc<RelaxedCell<bool>>,
    finished: Arc<Finished>,
}

impl InProcessQuery {
    /// Cancel the query, and wait until it has stopped.
    ///
    /// When this returns, nothing of the query runs any more: its tasks have ended and its
    /// sources and sinks have been dropped (closing, for example, the connections a database
    /// source opened). This blocks until the running work reaches a point where it can stop,
    /// such as the end of a batch a source is reading. A query that has already finished
    /// returns at once.
    ///
    /// Called from one of Polars' own threads (in a callback of a running query, say), this
    /// only requests the cancellation and returns without waiting: the query may be waiting
    /// for that very thread.
    pub fn cancel(&self) {
        self.token.store(true);
        if !(is_polars_thread() || QUERY_THREAD.get()) {
            self.finished.wait();
        }
    }

    /// Fetch the result.
    ///
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
        self.token.store(true);
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

    /// Records when it is dropped: when the query's plan has been torn down.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn cancel_waits_until_the_query_is_dropped() -> PolarsResult<()> {
        let started = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));

        // The plan holds `flag` until it is dropped.
        let flag = DropFlag(dropped.clone());
        let started2 = started.clone();
        let q = df![ "a" => [1, 2, 3] ]?
            .lazy()
            .map(
                move |df| {
                    let _ = &flag;
                    started2.store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(300));
                    Ok(df)
                },
                AllowedOptimizations::default(),
                None,
                None,
            )
            .collect_concurrently()?;

        while !started.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        q.cancel();
        assert!(dropped.load(Ordering::SeqCst));
        assert!(q.fetch().is_some());

        // Cancelling again, after the end, returns at once.
        q.cancel();
        Ok(())
    }
}
