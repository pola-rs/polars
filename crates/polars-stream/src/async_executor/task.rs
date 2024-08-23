use std::any::Any;
use std::future::Future;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, Wake, Waker};

use parking_lot::Mutex;

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
        self.state
            .fetch_update(Ordering::Release, Ordering::Relaxed, |state| match state {
                Self::SCHEDULED | Self::NOTIFIED_WHILE_RUNNING => None,
                Self::RUNNING => Some(Self::NOTIFIED_WHILE_RUNNING),
                Self::IDLE => Some(Self::SCHEDULED),
                _ => unreachable!("invalid TaskState"),
            })
            .map(|state| state == Self::IDLE)
            .unwrap_or(false)
    }

    /// Start running this task.
    fn start_running(&self) {
        assert_eq!(self.state.load(Ordering::Acquire), Self::SCHEDULED);
        self.state.store(Self::RUNNING, Ordering::Relaxed);
    }

    /// Done running this task. Returns true if task.schedule should be called.
    fn reschedule_after_running(&self) -> bool {
        self.state
            .fetch_update(Ordering::Release, Ordering::Relaxed, |state| match state {
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
    data: Mutex<(TaskData<F>, Option<Waker>)>,
    schedule: S,
    metadata: M,
}

impl<'a, F, S, M> Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    /// # Safety
    /// It is the responsibility of the caller that before lifetime 'a ends the
    /// task is either polled to completion or cancelled.
    unsafe fn spawn(future: F, schedule: S, metadata: M) -> Arc<Self> {
        let task = Arc::new(Self {
            state: TaskState::default(),
            data: Mutex::new((TaskData::Empty, None)),
            schedule,
            metadata,
        });

        let waker = unsafe { Waker::from_raw(std_shim::raw_waker(task.clone())) };
        task.data.try_lock().unwrap().0 = TaskData::Polling(future, waker);
        task
    }

    fn into_runnable(self: Arc<Self>) -> Runnable<M> {
        let arc: Arc<dyn DynTask<M> + 'a> = self;
        let arc: Arc<dyn DynTask<M>> = unsafe { std::mem::transmute(arc) };
        Runnable(arc)
    }

    fn into_join_handle(self: Arc<Self>) -> JoinHandle<F::Output> {
        let arc: Arc<dyn Joinable<F::Output> + 'a> = self;
        let arc: Arc<dyn Joinable<F::Output>> = unsafe { std::mem::transmute(arc) };
        JoinHandle(Some(arc))
    }

    fn into_cancel_handle(self: Arc<Self>) -> CancelHandle {
        let arc: Arc<dyn Cancellable + 'a> = self;
        let arc: Arc<dyn Cancellable> = unsafe { std::mem::transmute(arc) };
        CancelHandle(Arc::downgrade(&arc))
    }
}

impl<'a, F, S, M> Wake for Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    fn wake(self: Arc<Self>) {
        if self.state.wake() {
            let schedule = self.schedule;
            (schedule)(self.into_runnable());
        }
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.clone().wake()
    }
}

pub trait DynTask<M>: Send + Sync {
    fn metadata(&self) -> &M;
    fn run(self: Arc<Self>) -> bool;
    fn schedule(self: Arc<Self>);
}

impl<'a, F, S, M> DynTask<M> for Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    fn metadata(&self) -> &M {
        &self.metadata
    }

    fn run(self: Arc<Self>) -> bool {
        let mut data = self.data.lock();

        let poll_result = match &mut data.0 {
            TaskData::Polling(future, waker) => {
                self.state.start_running();
                // SAFETY: we always store a Task in an Arc and never move it.
                let fut = unsafe { Pin::new_unchecked(future) };
                let mut ctx = Context::from_waker(waker);
                catch_unwind(AssertUnwindSafe(|| fut.poll(&mut ctx)))
            },
            TaskData::Cancelled => return true,
            _ => unreachable!("invalid TaskData when polling"),
        };

        data.0 = match poll_result {
            Err(error) => TaskData::Panic(error),
            Ok(Poll::Ready(output)) => TaskData::Ready(output),
            Ok(Poll::Pending) => {
                drop(data);
                if self.state.reschedule_after_running() {
                    let schedule = self.schedule;
                    (schedule)(self.into_runnable());
                }
                return false;
            },
        };

        let join_waker = data.1.take();
        drop(data);
        if let Some(w) = join_waker {
            w.wake();
        }
        true
    }

    fn schedule(self: Arc<Self>) {
        if self.state.wake() {
            (self.schedule)(self.clone().into_runnable());
        }
    }
}

trait Joinable<T>: Send + Sync {
    fn cancel_handle(self: Arc<Self>) -> CancelHandle;
    fn poll_join(&self, ctx: &mut Context<'_>) -> Poll<T>;
}

impl<'a, F, S, M> Joinable<F::Output> for Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    fn cancel_handle(self: Arc<Self>) -> CancelHandle {
        self.into_cancel_handle()
    }

    fn poll_join(&self, cx: &mut Context<'_>) -> Poll<F::Output> {
        let mut data = self.data.lock();
        if matches!(data.0, TaskData::Empty | TaskData::Polling(..)) {
            data.1 = Some(cx.waker().clone());
            return Poll::Pending;
        }

        match core::mem::replace(&mut data.0, TaskData::Joined) {
            TaskData::Ready(output) => Poll::Ready(output),
            TaskData::Panic(error) => resume_unwind(error),
            TaskData::Cancelled => panic!("joined on cancelled task"),
            _ => unreachable!("invalid TaskData when joining"),
        }
    }
}

trait Cancellable: Send + Sync {
    fn cancel(&self);
}

impl<'a, F, S, M> Cancellable for Task<F, S, M>
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Send + Sync + 'static,
    M: Send + Sync + 'static,
{
    fn cancel(&self) {
        let mut data = self.data.lock();
        match data.0 {
            // Already done.
            TaskData::Panic(_) | TaskData::Joined => {},

            // Still in-progress, cancel.
            _ => {
                data.0 = TaskData::Cancelled;
                if let Some(join_waker) = data.1.take() {
                    join_waker.wake();
                }
            },
        }
    }
}

pub struct Runnable<M>(Arc<dyn DynTask<M>>);

impl<M> Runnable<M> {
    /// Gives the metadata for this task.
    pub fn metadata(&self) -> &M {
        self.0.metadata()
    }

    /// Runs a task, and returns true if the task is done.
    pub fn run(self) -> bool {
        self.0.run()
    }

    /// Schedules this task.
    pub fn schedule(self) {
        self.0.schedule()
    }
}

pub struct JoinHandle<T>(Option<Arc<dyn Joinable<T>>>);
pub struct CancelHandle(Weak<dyn Cancellable>);
pub struct AbortOnDropHandle<T> {
    join_handle: JoinHandle<T>,
    cancel_handle: CancelHandle,
}

impl<T> JoinHandle<T> {
    pub fn cancel_handle(&self) -> CancelHandle {
        let arc = self
            .0
            .as_ref()
            .expect("called cancel_handle on joined JoinHandle");
        Arc::clone(arc).cancel_handle()
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let joinable = self.0.take().expect("JoinHandle polled after completion");

        if let Poll::Ready(output) = joinable.poll_join(ctx) {
            return Poll::Ready(output);
        }

        self.0 = Some(joinable);
        Poll::Pending
    }
}

impl CancelHandle {
    pub fn cancel(&self) {
        if let Some(t) = self.0.upgrade() {
            t.cancel();
        }
    }
}

impl<T> AbortOnDropHandle<T> {
    pub fn new(join_handle: JoinHandle<T>) -> Self {
        let cancel_handle = join_handle.cancel_handle();
        Self {
            join_handle,
            cancel_handle,
        }
    }
}

impl<T> Future for AbortOnDropHandle<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.join_handle).poll(cx)
    }
}

impl<T> Drop for AbortOnDropHandle<T> {
    fn drop(&mut self) {
        self.cancel_handle.cancel();
    }
}

pub fn spawn<F, S, M>(future: F, schedule: S, metadata: M) -> (Runnable<M>, JoinHandle<F::Output>)
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    let task = unsafe { Task::spawn(future, schedule, metadata) };
    (task.clone().into_runnable(), task.into_join_handle())
}

/// Takes a future and turns it into a runnable task with associated metadata.
///
/// When the task is pending its waker will be set to call schedule
/// with the runnable.
pub unsafe fn spawn_with_lifetime<'a, F, S, M>(
    future: F,
    schedule: S,
    metadata: M,
) -> (Runnable<M>, JoinHandle<F::Output>)
where
    F: Future + Send + 'a,
    F::Output: Send + 'static,
    S: Fn(Runnable<M>) + Send + Sync + Copy + 'static,
    M: Send + Sync + 'static,
{
    let task = Task::spawn(future, schedule, metadata);
    (task.clone().into_runnable(), task.into_join_handle())
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
