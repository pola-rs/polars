#![allow(unsafe_op_in_unsafe_fn)]
use std::any::Any;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use atomic_waker::AtomicWaker;
use parking_lot::Mutex;
use polars_error::abort::try_raise_polars_abort;

/// The state of the task. Can't be part of the TaskData enum as it needs to be
/// atomically updateable, even when we hold the lock on the data.
#[derive(Default)]
struct TaskState {
    state: AtomicU8,
}

impl TaskState {
    /// Default state, not running, not scheduled.
    const IDLE: u8 = 0;

    /// Task is scheduled, that is (task.schedule)(task) was called.
    const SCHEDULED: u8 = 1;

    /// Task is currently running.
    const RUNNING: u8 = 2;

    /// Task notified while running.
    const NOTIFIED_WHILE_RUNNING: u8 = 3;

    /// Wake this task. Returns true if task.schedule should be called.
    fn wake(&self) -> bool {
        // We always write, even if the state doesn't change. If we only loaded
        // SCHEDULED / NOTIFIED_WHILE_RUNNING, nothing would order our caller's
        // prior writes with the upcoming poll, which could then miss them
        // while we skip rescheduling, losing the wakeup. As a read-modify-write
        // we either precede start_running (which then acquires our writes) or
        // observe RUNNING and reschedule.
        self.state
            .try_update(Ordering::Release, Ordering::Relaxed, |state| match state {
                Self::SCHEDULED | Self::NOTIFIED_WHILE_RUNNING => Some(state),
                Self::RUNNING => Some(Self::NOTIFIED_WHILE_RUNNING),
                Self::IDLE => Some(Self::SCHEDULED),
                _ => unreachable!("invalid TaskState"),
            })
            .is_ok_and(|state| state == Self::IDLE)
    }

    /// Start running this task.
    fn start_running(&self) {
        // Must be a read-modify-write, see wake().
        let old_state = self.state.swap(Self::RUNNING, Ordering::Acquire);
        assert_eq!(old_state, Self::SCHEDULED);
    }

    /// Done running this task. Returns true if task.schedule should be called.
    fn reschedule_after_running(&self) -> bool {
        self.state
            .try_update(Ordering::Release, Ordering::Relaxed, |state| match state {
                Self::RUNNING => Some(Self::IDLE),
                Self::NOTIFIED_WHILE_RUNNING => Some(Self::SCHEDULED),
                _ => panic!("TaskState::reschedule_after_running() called on invalid state"),
            })
            .map(|old_state| old_state == Self::NOTIFIED_WHILE_RUNNING)
            .unwrap_or(false)
    }
}

enum TaskData<F: Future> {
    Empty,
    Polling(F, Waker),
    Ready(F::Output),
    Panic(Box<dyn Any + Send + 'static>),
    Cancelled,
    Joined,
}

struct Task<F: Future, S, M> {
    state: TaskState,
    data: Mutex<TaskData<F>>,
    join_waker: AtomicWaker,
    schedule: S,
    metadata: M,
}

impl<'a, F, S, M> Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    /// # Safety
    /// It is the responsibility of the caller that before lifetime 'a ends the
    /// task is either polled to completion or cancelled.
    unsafe fn spawn(future: F, schedule: S, metadata: M) -> Arc<Self> {
        let task = Arc::new(Self {
            state: TaskState::default(),
            data: Mutex::new(TaskData::Empty),
            join_waker: AtomicWaker::new(),
            schedule,
            metadata,
        });

        let waker = unsafe { Waker::from_raw(std_shim::raw_waker(task.clone())) };
        *task.data.try_lock().unwrap() = TaskData::Polling(future, waker);
        task
    }

    fn into_dyn(self: Arc<Self>) -> Arc<dyn DynTask<F::Output, M>> {
        let arc: Arc<dyn DynTask<F::Output, M> + 'a> = self;
        let arc: Arc<dyn DynTask<F::Output, M>> = unsafe { std::mem::transmute(arc) };
        arc
    }
}

impl<F, S, M> Wake for Task<F, S, M>
where
    F: Future + Send,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    fn wake(self: Arc<Self>) {
        if self.state.wake() {
            let schedule = self.schedule;
            (schedule)(self.into_dyn());
        }
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.clone().wake()
    }
}

/// Partially type-erased task: no future.
pub trait DynTask<T, M>: Send + Sync + Runnable<M> + Joinable<T> + Cancellable {}

impl<F, S, M> DynTask<F::Output, M> for Task<F, S, M>
where
    F: Future + Send,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
}

/// Partially type-erased task: no future or return type.
pub trait Runnable<M>: Send + Sync {
    /// Gives the metadata for this task.
    fn metadata(&self) -> &M;

    /// Runs a task, and returns true if the task is done.
    fn run(self: Arc<Self>) -> bool;

    /// Schedules this task.
    fn schedule(self: Arc<Self>);
}

impl<F, S, M> Runnable<M> for Task<F, S, M>
where
    F: Future + Send,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    #[inline]
    fn metadata(&self) -> &M {
        &self.metadata
    }

    fn run(self: Arc<Self>) -> bool {
        let mut data = self.data.lock();

        let poll_result = match &mut *data {
            TaskData::Polling(future, waker) => {
                self.state.start_running();
                // SAFETY: we always store a Task in an Arc and never move it.
                let fut = unsafe { Pin::new_unchecked(future) };
                let mut ctx = Context::from_waker(waker);
                catch_unwind(AssertUnwindSafe(|| {
                    try_raise_polars_abort();
                    fut.poll(&mut ctx)
                }))
            },
            TaskData::Cancelled => return true,
            _ => unreachable!("invalid TaskData when polling"),
        };

        *data = match poll_result {
            Err(error) => TaskData::Panic(error),
            Ok(Poll::Ready(output)) => TaskData::Ready(output),
            Ok(Poll::Pending) => {
                drop(data);
                if self.state.reschedule_after_running() {
                    let schedule = self.schedule;
                    (schedule)(self.into_dyn());
                }
                return false;
            },
        };

        drop(data);
        self.join_waker.wake();
        true
    }

    fn schedule(self: Arc<Self>) {
        if self.state.wake() {
            (self.schedule)(self.clone().into_dyn());
        }
    }
}

/// Partially type-erased task: no future or metadata.
pub trait Joinable<T>: Send + Sync + Cancellable {
    fn poll_join(&self, ctx: &mut Context<'_>) -> Poll<T>;
}

impl<F, S, M> Joinable<F::Output> for Task<F, S, M>
where
    F: Future + Send,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    fn poll_join(&self, cx: &mut Context<'_>) -> Poll<F::Output> {
        self.join_waker.register(cx.waker());
        if let Some(mut data) = self.data.try_lock() {
            if matches!(*data, TaskData::Empty | TaskData::Polling(..)) {
                return Poll::Pending;
            }

            match core::mem::replace(&mut *data, TaskData::Joined) {
                TaskData::Ready(output) => Poll::Ready(output),
                TaskData::Panic(error) => resume_unwind(error),
                TaskData::Cancelled => panic!("joined on cancelled task"),
                _ => unreachable!("invalid TaskData when joining"),
            }
        } else {
            Poll::Pending
        }
    }
}

/// Task metadata that is notified when its task is cancelled.
pub trait OnCancel {
    fn on_cancel(&self);
}

/// Fully type-erased task.
pub trait Cancellable: Send + Sync {
    fn cancel(&self);
}

impl<F, S, M> Cancellable for Task<F, S, M>
where
    F: Future + Send,
    F::Output: Send + 'static,
    S: Send + Sync + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    fn cancel(&self) {
        let mut data = self.data.lock();
        match *data {
            // Already done.
            TaskData::Panic(_) | TaskData::Joined => {},

            // Still in-progress, cancel.
            _ => {
                *data = TaskData::Cancelled;
                self.metadata.on_cancel();
                if let Some(join_waker) = self.join_waker.take() {
                    join_waker.wake();
                }
            },
        }
    }
}

pub fn spawn<F, S, M>(future: F, schedule: S, metadata: M) -> Arc<dyn DynTask<F::Output, M>>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    unsafe { Task::spawn(future, schedule, metadata) }.into_dyn()
}

/// Takes a future and turns it into a runnable task with associated metadata.
///
/// When the task is pending its waker will be set to call schedule
/// with the runnable.
pub unsafe fn spawn_with_lifetime<'a, F, S, M>(
    future: F,
    schedule: S,
    metadata: M,
) -> Arc<dyn DynTask<F::Output, M>>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Arc<dyn Runnable<M>>) + Send + Sync + Copy + 'static,
    M: OnCancel + Send + Sync + 'static,
{
    Task::spawn(future, schedule, metadata).into_dyn()
}

// Copied from the standard library, except without the 'static bound.
mod std_shim {
    use std::mem::ManuallyDrop;
    use std::sync::Arc;
    use std::task::{RawWaker, RawWakerVTable, Wake};

    #[inline(always)]
    pub unsafe fn raw_waker<'a, W: Wake + Send + Sync + 'a>(waker: Arc<W>) -> RawWaker {
        // Increment the reference count of the arc to clone it.
        //
        // The #[inline(always)] is to ensure that raw_waker and clone_waker are
        // always generated in the same code generation unit as one another, and
        // therefore that the structurally identical const-promoted RawWakerVTable
        // within both functions is deduplicated at LLVM IR code generation time.
        // This allows optimizing Waker::will_wake to a single pointer comparison of
        // the vtable pointers, rather than comparing all four function pointers
        // within the vtables.
        #[inline(always)]
        unsafe fn clone_waker<W: Wake + Send + Sync>(waker: *const ()) -> RawWaker {
            unsafe { Arc::increment_strong_count(waker as *const W) };
            RawWaker::new(
                waker,
                &RawWakerVTable::new(
                    clone_waker::<W>,
                    wake::<W>,
                    wake_by_ref::<W>,
                    drop_waker::<W>,
                ),
            )
        }

        // Wake by value, moving the Arc into the Wake::wake function
        unsafe fn wake<W: Wake + Send + Sync>(waker: *const ()) {
            let waker = unsafe { Arc::from_raw(waker as *const W) };
            <W as Wake>::wake(waker);
        }

        // Wake by reference, wrap the waker in ManuallyDrop to avoid dropping it
        unsafe fn wake_by_ref<W: Wake + Send + Sync>(waker: *const ()) {
            let waker = unsafe { ManuallyDrop::new(Arc::from_raw(waker as *const W)) };
            <W as Wake>::wake_by_ref(&waker);
        }

        // Decrement the reference count of the Arc on drop
        #[inline]
        unsafe fn drop_waker<W: Wake + Send + Sync>(waker: *const ()) {
            unsafe { Arc::decrement_strong_count(waker as *const W) };
        }

        RawWaker::new(
            Arc::into_raw(waker) as *const (),
            &RawWakerVTable::new(
                clone_waker::<W>,
                wake::<W>,
                wake_by_ref::<W>,
                drop_waker::<W>,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::hint::spin_loop;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::task::{Poll, Waker};
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;

    use super::{OnCancel, Runnable, spawn};

    struct Meta;

    impl OnCancel for Meta {
        fn on_cancel(&self) {}
    }

    static QUEUE: Mutex<Vec<Arc<dyn Runnable<Meta>>>> = Mutex::new(Vec::new());
    static WAKER: Mutex<Option<Waker>> = Mutex::new(None);
    static VALUE: AtomicU64 = AtomicU64::new(0);
    static OBSERVED: AtomicU64 = AtomicU64::new(0);

    fn schedule(r: Arc<dyn Runnable<Meta>>) {
        QUEUE.lock().push(r);
    }

    fn pop() -> Option<Arc<dyn Runnable<Meta>>> {
        let mut q = QUEUE.lock();
        assert!(q.len() <= 1);
        q.pop()
    }

    fn jitter(n: u64) {
        for _ in 0..n {
            spin_loop();
        }
    }

    /// A wake of an already-scheduled task must not be lost: the pending run
    /// has to observe everything the waker wrote before waking. Races a waker
    /// (publish value, then wake) against a run of the already-scheduled task
    /// (read value, return Pending), and checks that either the run saw the
    /// value or the task got rescheduled.
    ///
    /// Only reproduces the race reliably with optimizations, run with
    /// `cargo test -p polars-async --lib --release`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "only effective in release builds")]
    fn test_wake_while_scheduled_not_lost() {
        let fut = std::future::poll_fn(|cx| {
            WAKER.lock().get_or_insert_with(|| cx.waker().clone());
            OBSERVED.store(VALUE.load(Ordering::Acquire), Ordering::Relaxed);
            Poll::<()>::Pending
        });
        spawn(fut, schedule, Meta).schedule();
        // Initial run to register the waker; the task is now idle.
        assert!(!pop().unwrap().run());
        let waker = WAKER.lock().clone().unwrap();

        let go = AtomicU64::new(0);
        let done = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        let to_run: Mutex<Option<Arc<dyn Runnable<Meta>>>> = Mutex::new(None);
        let deadline = Instant::now() + Duration::from_secs(1);

        let wait_go = |round: u64| {
            while go.load(Ordering::Acquire) < round {
                spin_loop();
            }
        };

        std::thread::scope(|s| {
            // Runner.
            s.spawn(|| {
                for round in 1.. {
                    wait_go(round);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let r = to_run.lock().take().unwrap();
                    jitter(round % 7);
                    assert!(!r.run());
                    done.fetch_add(1, Ordering::Release);
                }
            });

            // Waker. Wakes by value, as the by-ref path clones the Arc, which
            // is a full fence on x86 and hides the race there.
            s.spawn(|| {
                for round in 1.. {
                    let w = waker.clone();
                    wait_go(round);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    jitter((round / 7) % 11);
                    VALUE.store(round, Ordering::Release);
                    w.wake();
                    done.fetch_add(1, Ordering::Release);
                }
            });

            let mut round = 0;
            let mut lost = false;
            while Instant::now() < deadline {
                round += 1;
                // Make sure the task is scheduled before the race starts.
                let r = pop().unwrap_or_else(|| {
                    waker.wake_by_ref();
                    pop().unwrap()
                });
                *to_run.lock() = Some(r);
                go.store(round, Ordering::Release);
                while done.load(Ordering::Acquire) < 2 * round {
                    spin_loop();
                }

                if OBSERVED.load(Ordering::Relaxed) < round && QUEUE.lock().is_empty() {
                    lost = true;
                    break;
                }
            }
            stop.store(true, Ordering::Relaxed);
            go.store(round + 1, Ordering::Release);

            assert!(!lost, "lost wakeup in round {round}");
            eprintln!("{round} rounds without lost wakeup");
        });

        WAKER.lock().take();
        QUEUE.lock().clear();
    }
}
