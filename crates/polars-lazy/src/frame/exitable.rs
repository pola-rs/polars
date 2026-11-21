use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use polars_core::error::ErrString;
use polars_core::error::abort::{QueryAborted, decode_polars_abort, raise_polars_abort};
use polars_core::runtime::is_polars_thread;
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
            // A thread of its own rather than a Rayon job, as a foreground query runs on the
            // caller's thread: queued behind busy workers, the query wouldn't start (and a
            // cancel would wait for a worker), and a worker waiting inside another query could
            // run it within that query's frame (a cancel of that one would wait for this one).
            // Its parallel work still runs on Rayon.
            std::thread::Builder::new()
                .name("polars-background-query".into())
                .spawn(run)
                .map_err(
                    |e| polars_err!(ComputeError: "could not start the query's thread: {e}"),
                )?;
        }

        Ok(InProcessQuery {
            rx: Arc::new(Mutex::new(rx)),
            token,
            finished,
        })
    }
}

/// Executes the query, drops what is left of it (the physical plan, with its sources, and the
/// state), then sends its result and marks it finished. It never panics, which would end its
/// thread without a result: a panic becomes the query's error, and a Polars abort (a keyboard
/// interrupt) is raised again where the result is fetched.
fn run_query(
    finished: Arc<Finished>,
    physical_plan: Box<dyn Executor>,
    state: ExecutionState,
    tx: Sender<QueryOutcome>,
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
    .map_or_else(panic_outcome, Ok);
    // The handle may have been dropped: nobody wants the result then.
    let _ = tx.send(result);
}

/// A background query's result, or the Polars abort that stopped it (raised again by `fetch`).
type QueryOutcome = Result<PolarsResult<DataFrame>, QueryAborted>;

/// The outcome of a background query that panicked: a Polars abort (such as a keyboard interrupt
/// raised in another thread's Polars call) stays one; another panic becomes an error. The error
/// is built without `POLARS_PANIC_ON_ERR`'s panic, which would escape the query's thread.
fn panic_outcome(payload: Box<dyn Any + Send>) -> QueryOutcome {
    match decode_polars_abort(&*payload) {
        Some(reason) => Err(reason),
        None => {
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            Ok(Err(PolarsError::ComputeError(ErrString::new_owned(
                format!("query panicked: {msg}"),
            ))))
        },
    }
}

/// The query's result; a Polars abort that stopped it is raised again here, on the fetching
/// thread, to be caught as the caller's own (`catch_polars_abort`).
fn into_result(outcome: QueryOutcome) -> PolarsResult<DataFrame> {
    outcome.unwrap_or_else(|reason| raise_polars_abort(reason))
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
    rx: Arc<Mutex<Receiver<QueryOutcome>>>,
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
    /// any more. If it is not ready it will return `None`. A Polars abort (a keyboard interrupt)
    /// that stopped the query is raised again here.
    pub fn fetch(&self) -> Option<PolarsResult<DataFrame>> {
        // Received before raising an abort, so as not to poison the lock.
        let outcome = self.rx.lock().unwrap().try_recv().ok();
        outcome.map(into_result)
    }

    /// Await the result synchronously.
    ///
    /// Like `fetch`, this raises a Polars abort (`catch_polars_abort`) that stopped the query
    /// again: a keyboard interrupt while it ran.
    pub fn fetch_blocking(&self) -> PolarsResult<DataFrame> {
        let outcome = self.rx.lock().unwrap().recv().unwrap();
        into_result(outcome)
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
    fn an_abort_is_raised_again_where_the_result_is_fetched() -> PolarsResult<()> {
        use polars_core::error::abort::catch_polars_abort;

        let q = df![ "a" => [1, 2, 3] ]?
            .lazy()
            .map(
                |_| raise_polars_abort(QueryAborted::KeyboardInterrupt),
                AllowedOptimizations::default(),
                None,
                None,
            )
            .collect_concurrently()?;
        assert!(q.wait_finished(Some(Duration::from_secs(10))));
        let fetched = catch_polars_abort(AssertUnwindSafe(|| q.fetch_blocking()));
        assert!(matches!(fetched, Err(QueryAborted::KeyboardInterrupt)));
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
