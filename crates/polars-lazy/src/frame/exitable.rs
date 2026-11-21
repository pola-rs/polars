use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use polars_core::error::abort::{QueryAborted, decode_polars_abort};
use polars_core::runtime::{RAYON, is_polars_thread};
use polars_utils::relaxed_cell::RelaxedCell;

use super::*;

impl LazyFrame {
    pub fn collect_concurrently(self) -> PolarsResult<InProcessQuery> {
        let (mut state, physical_plan, _) = self.prepare_collect(false)?;
        let physical_plan_is_cache_prefiller = physical_plan.is_cache_prefiller();

        let (tx, rx) = channel();
        let token = state.cancellable_token();
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

/// Executes the query, drops what is left of it (the physical plan, with its sources, and the
/// state), then sends its result and marks it finished. It never panics: it runs as a job of a
/// thread pool, where a panic aborts the process.
fn run_query(
    finished: Arc<Finished>,
    physical_plan: Box<dyn Executor>,
    state: ExecutionState,
    tx: Sender<PolarsResult<DataFrame>>,
) {
    QUERY_THREAD.set(true);
    // Declared first, dropped last: the query is marked finished whatever happens.
    let _finished = SetFinishedOnDrop(finished);
    let result = catch_unwind(AssertUnwindSafe(move || {
        let mut physical_plan = physical_plan;
        let mut state = state;
        let result = physical_plan.execute(&mut state);
        drop(physical_plan);
        drop(state);
        result
    }))
    .unwrap_or_else(|payload| Err(panic_to_error(payload)));
    // The handle may have been dropped: nobody wants the result then.
    let _ = tx.send(result);
}

/// The error a background query fails with when it panics (a Polars abort, such as a keyboard
/// interrupt raised in another thread's Polars call, is one).
fn panic_to_error(payload: Box<dyn Any + Send>) -> PolarsError {
    match decode_polars_abort(&*payload) {
        Some(QueryAborted::KeyboardInterrupt) => {
            polars_err!(ComputeError: "query interrupted (keyboard interrupt)")
        },
        Some(QueryAborted::OocOutOfDisk) => polars_err!(
            ComputeError: "query aborted: spilled data exceeded the out-of-core disk budget"
        ),
        None => {
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            polars_err!(ComputeError: "query panicked: {msg}")
        },
    }
}

thread_local! {
    /// Set on a thread that runs (or ran) an in-process query: not always one of Polars' pools.
    static QUERY_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the query's thread has finished: its plan and state dropped, its result sent.
#[derive(Default)]
struct Finished {
    done: Mutex<bool>,
    cond: Condvar,
}

impl Finished {
    /// Waits until finished, at most `timeout` (`None`: no limit). Returns whether it finished.
    fn wait(&self, timeout: Option<Duration>) -> bool {
        let done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        let done = match timeout {
            None => self
                .cond
                .wait_while(done, |done| !*done)
                .unwrap_or_else(|e| e.into_inner()),
            Some(timeout) => {
                self.cond
                    .wait_timeout_while(done, timeout, |done| !*done)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            },
        };
        *done
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
    /// Request the query to stop, and return at once.
    ///
    /// To wait until it has stopped, use `wait_finished` (or `cancel_blocking`).
    pub fn cancel(&self) {
        self.token.store(true);
    }

    /// Cancel the query, and wait until it has stopped (see `wait_finished`), unless
    /// `can_wait_on_this_thread` is false: then this only requests the cancellation.
    pub fn cancel_blocking(&self) {
        self.cancel();
        if Self::can_wait_on_this_thread() {
            self.wait_finished(None);
        }
    }

    /// Wait until the query has finished (`timeout`: at most that long), and return whether it
    /// has.
    ///
    /// Once it has, nothing of the query runs any more: its tasks have ended and its sources and
    /// sinks have been dropped (closing, for example, the connections a database source opened).
    /// After a `cancel`, this waits until the running work reaches a point where it can stop,
    /// such as the end of a batch a source is reading.
    ///
    /// Must not be called on a thread the query may wait for (see `can_wait_on_this_thread`).
    pub fn wait_finished(&self, timeout: Option<Duration>) -> bool {
        self.finished.wait(timeout)
    }

    /// Whether this thread may wait for a query to finish: false on one of Polars' own threads
    /// (in a callback of a running query, say), as the query may be waiting for that very thread.
    pub fn can_wait_on_this_thread() -> bool {
        !(is_polars_thread() || QUERY_THREAD.get())
    }

    /// Fetch the result.
    ///
    /// If it is ready, a materialized DataFrame is returned: by then nothing of the query runs
    /// any more. If it is not ready it will return `None`.
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
        q.cancel_blocking();
        assert!(dropped.load(Ordering::SeqCst));
        assert!(q.fetch().is_some());

        // Cancelling again, after the end, returns at once.
        q.cancel_blocking();
        Ok(())
    }

    /// Records when it is dropped, after a while.
    struct SlowDropFlag(Arc<AtomicBool>);

    impl Drop for SlowDropFlag {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(300));
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn fetch_blocking_returns_after_the_query_is_dropped() -> PolarsResult<()> {
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = SlowDropFlag(dropped.clone());
        let q = df![ "a" => [1, 2, 3] ]?
            .lazy()
            .map(
                move |df| {
                    let _ = &flag;
                    Ok(df)
                },
                AllowedOptimizations::default(),
                None,
                None,
            )
            .collect_concurrently()?;
        q.fetch_blocking()?;
        assert!(dropped.load(Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn a_panicking_query_fails_instead_of_aborting() -> PolarsResult<()> {
        let q = df![ "a" => [1, 2, 3] ]?
            .lazy()
            .map(
                |_| panic!("boom"),
                AllowedOptimizations::default(),
                None,
                None,
            )
            .collect_concurrently()?;
        let err = q.fetch_blocking().unwrap_err();
        assert!(err.to_string().contains("query panicked: boom"), "{err}");
        assert!(q.wait_finished(Some(Duration::from_secs(10))));
        Ok(())
    }

    #[test]
    fn dropping_the_handle_while_the_query_runs() -> PolarsResult<()> {
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = DropFlag(dropped.clone());
        let q = df![ "a" => [1, 2, 3] ]?
            .lazy()
            .map(
                move |df| {
                    let _ = &flag;
                    std::thread::sleep(Duration::from_millis(200));
                    Ok(df)
                },
                AllowedOptimizations::default(),
                None,
                None,
            )
            .collect_concurrently()?;
        let finished = q.finished.clone();
        drop(q);
        assert!(finished.wait(Some(Duration::from_secs(10))));
        assert!(dropped.load(Ordering::SeqCst));
        Ok(())
    }
}
