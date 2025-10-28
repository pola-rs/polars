#![allow(clippy::disallowed_types)]

mod park_group;
mod task;

use std::cell::{Cell, UnsafeCell};
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, Location};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::task::{Context, Poll};
use std::time::Instant;

use crossbeam_deque::{Injector, Steal, Stealer, Worker as WorkQueue};
use crossbeam_utils::CachePadded;
use park_group::ParkGroup;
use parking_lot::Mutex;
use polars_utils::relaxed_cell::RelaxedCell;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use slotmap::SlotMap;
use task::{Cancellable, DynTask, Runnable};

static NUM_EXECUTOR_THREADS: RelaxedCell<usize> = RelaxedCell::new_usize(0);
pub fn set_num_threads(t: usize) {
    NUM_EXECUTOR_THREADS.store(t);
}

static TRACK_METRICS: RelaxedCell<bool> = RelaxedCell::new_bool(false);

pub fn track_task_metrics(should_track: bool) {
    TRACK_METRICS.store(should_track);
}

static GLOBAL_SCHEDULER: OnceLock<Executor> = OnceLock::new();

thread_local!(
    /// Used to store which executor thread this is.
    static TLS_THREAD_ID: Cell<usize> = const { Cell::new(usize::MAX) };
);

slotmap::new_key_type! {
    struct TaskKey;
}

/// High priority tasks are scheduled preferentially over low priority tasks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    High,
}

/// Metadata associated with a task to help schedule it and clean it up.
struct ScopedTaskMetadata {
    task_key: TaskKey,
    completed_tasks: Weak<Mutex<Vec<TaskKey>>>,
}

#[derive(Default)]
#[repr(align(128))]
pub struct TaskMetrics {
    pub total_polls: RelaxedCell<u64>,
    pub total_stolen_polls: RelaxedCell<u64>,
    pub total_poll_time_ns: RelaxedCell<u64>,
    pub max_poll_time_ns: RelaxedCell<u64>,
}

struct TaskMetadata {
    spawn_location: &'static Location<'static>,
    priority: TaskPriority,
    freshly_spawned: AtomicBool,
    scoped: Option<ScopedTaskMetadata>,
    metrics: Option<Arc<TaskMetrics>>,
}

impl Drop for TaskMetadata {
    fn drop(&mut self) {
        if let Some(scoped) = &self.scoped {
            if let Some(completed_tasks) = scoped.completed_tasks.upgrade() {
                completed_tasks.lock().push(scoped.task_key);
            }
        }
    }
}

pub struct JoinHandle<T>(Arc<dyn DynTask<T, TaskMetadata>>);
pub struct CancelHandle(Weak<dyn Cancellable>);

impl<T> JoinHandle<T> {
    pub fn metrics(&self) -> Option<&Arc<TaskMetrics>> {
        self.0.metadata().metrics.as_ref()
    }

    #[allow(unused)]
    pub fn spawn_location(&self) -> &'static Location<'static> {
        self.0.metadata().spawn_location
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        let coerce: Weak<dyn DynTask<T, TaskMetadata>> = Arc::downgrade(&self.0);
        CancelHandle(coerce)
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_join(ctx)
    }
}

impl CancelHandle {
    pub fn cancel(&self) {
        if let Some(t) = self.0.upgrade() {
            t.cancel();
        }
    }
}

pub struct AbortOnDropHandle<T> {
    join_handle: JoinHandle<T>,
    cancel_handle: CancelHandle,
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

/// A task ready to run.
type ReadyTask = Arc<dyn Runnable<TaskMetadata>>;

/// A per-thread task list.
struct ThreadLocalTaskList {
    // May be used from any thread.
    high_prio_tasks_stealer: Stealer<ReadyTask>,

    // SAFETY: these may only be used on the thread this task list belongs to.
    high_prio_tasks: WorkQueue<ReadyTask>,
    local_slot: UnsafeCell<Option<ReadyTask>>,
}

unsafe impl Sync for ThreadLocalTaskList {}

struct Executor {
    park_group: ParkGroup,
    thread_task_lists: Vec<CachePadded<ThreadLocalTaskList>>,
    global_high_prio_task_queue: Injector<ReadyTask>,
    global_low_prio_task_queue: Injector<ReadyTask>,
}

impl Executor {
    fn schedule_task(&self, task: ReadyTask) {
        let thread = TLS_THREAD_ID.get();
        let meta = task.metadata();
        let opt_ttl = self.thread_task_lists.get(thread);

        let mut use_global_queue = opt_ttl.is_none();
        if meta.freshly_spawned.load(Ordering::Relaxed) {
            use_global_queue = true;
            meta.freshly_spawned.store(false, Ordering::Relaxed);
        }

        if use_global_queue {
            // Scheduled from an unknown thread, add to global queue.
            if meta.priority == TaskPriority::High {
                self.global_high_prio_task_queue.push(task);
            } else {
                self.global_low_prio_task_queue.push(task);
            }
            self.park_group.unpark_one();
        } else {
            let ttl = opt_ttl.unwrap();
            // SAFETY: this slot may only be accessed from the local thread, which we are.
            let slot = unsafe { &mut *ttl.local_slot.get() };

            if meta.priority == TaskPriority::High {
                // Insert new task into thread local slot, taking out the old task.
                let Some(task) = slot.replace(task) else {
                    // We pushed a task into our local slot which was empty. Since
                    // we are already awake, no need to notify anyone.
                    return;
                };

                ttl.high_prio_tasks.push(task);
                self.park_group.unpark_one();
            } else {
                // Optimization: while this is a low priority task we have no
                // high priority tasks on this thread so we'll execute this one.
                if ttl.high_prio_tasks.is_empty() && slot.is_none() {
                    *slot = Some(task);
                } else {
                    self.global_low_prio_task_queue.push(task);
                    self.park_group.unpark_one();
                }
            }
        }
    }

    fn try_steal_task<R: Rng>(&self, thread: usize, rng: &mut R) -> Option<ReadyTask> {
        // Try to get a global task.
        loop {
            match self.global_high_prio_task_queue.steal() {
                Steal::Empty => break,
                Steal::Success(task) => return Some(task),
                Steal::Retry => std::hint::spin_loop(),
            }
        }

        loop {
            match self.global_low_prio_task_queue.steal() {
                Steal::Empty => break,
                Steal::Success(task) => return Some(task),
                Steal::Retry => std::hint::spin_loop(),
            }
        }

        // Try to steal tasks.
        let ttl = &self.thread_task_lists[thread];
        for _ in 0..4 {
            let mut retry = true;
            while retry {
                retry = false;

                for idx in random_permutation(self.thread_task_lists.len() as u32, rng) {
                    let foreign_ttl = &self.thread_task_lists[idx as usize];
                    match foreign_ttl
                        .high_prio_tasks_stealer
                        .steal_batch_and_pop(&ttl.high_prio_tasks)
                    {
                        Steal::Empty => {},
                        Steal::Success(task) => return Some(task),
                        Steal::Retry => retry = true,
                    }
                }

                std::hint::spin_loop()
            }
        }

        None
    }

    fn runner(&self, thread: usize) {
        TLS_THREAD_ID.set(thread);

        let mut rng = SmallRng::from_rng(&mut rand::rng());
        let mut worker = self.park_group.new_worker();

        loop {
            let ttl = &self.thread_task_lists[thread];
            let mut local = true;
            let task = (|| {
                // Try to get a task from LIFO slot.
                if let Some(task) = unsafe { (*ttl.local_slot.get()).take() } {
                    return Some(task);
                }

                // Try to get a local high-priority task.
                if let Some(task) = ttl.high_prio_tasks.pop() {
                    return Some(task);
                }

                // Try to steal a task.
                local = false;
                if let Some(task) = self.try_steal_task(thread, &mut rng) {
                    return Some(task);
                }

                // Prepare to park, then try one more steal attempt.
                let park = worker.prepare_park();
                if let Some(task) = self.try_steal_task(thread, &mut rng) {
                    return Some(task);
                }

                park.park();
                None
            })();

            if let Some(task) = task {
                worker.recruit_next();
                if let Some(metrics) = task.metadata().metrics.clone() {
                    let start = Instant::now();
                    task.run();
                    let elapsed_ns = start.elapsed().as_nanos() as u64;
                    metrics.total_polls.fetch_add(1);
                    if !local {
                        metrics.total_stolen_polls.fetch_add(1);
                    }
                    metrics.total_poll_time_ns.fetch_add(elapsed_ns);
                    metrics.max_poll_time_ns.fetch_max(elapsed_ns);
                } else {
                    task.run();
                }
            }
        }
    }

    fn global() -> &'static Executor {
        GLOBAL_SCHEDULER.get_or_init(|| {
            let mut n_threads = NUM_EXECUTOR_THREADS.load();
            if n_threads == 0 {
                n_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
            }

            let thread_task_lists = (0..n_threads)
                .map(|t| {
                    std::thread::Builder::new()
                        .name(format!("async-executor-{t}"))
                        .spawn(move || Self::global().runner(t))
                        .unwrap();

                    let high_prio_tasks = WorkQueue::new_lifo();
                    CachePadded::new(ThreadLocalTaskList {
                        high_prio_tasks_stealer: high_prio_tasks.stealer(),
                        high_prio_tasks,
                        local_slot: UnsafeCell::new(None),
                    })
                })
                .collect();
            Self {
                park_group: ParkGroup::new(),
                thread_task_lists,
                global_high_prio_task_queue: Injector::new(),
                global_low_prio_task_queue: Injector::new(),
            }
        })
    }
}

pub struct TaskScope<'scope, 'env: 'scope> {
    // Keep track of in-progress tasks so we can forcibly cancel them
    // when the scope ends, to ensure the lifetimes are respected.
    // Tasks add their own key to completed_tasks when done so we can
    // reclaim the memory used by the cancel_handles.
    cancel_handles: Mutex<SlotMap<TaskKey, CancelHandle>>,
    completed_tasks: Arc<Mutex<Vec<TaskKey>>>,

    // Copied from std::thread::scope. Necessary to prevent unsoundness.
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

impl<'scope> TaskScope<'scope, '_> {
    // Not Drop because that extends lifetimes.
    fn destroy(&self) {
        // Make sure all tasks are cancelled.
        for (_, t) in self.cancel_handles.lock().drain() {
            t.cancel();
        }
    }

    fn clear_completed_tasks(&self) {
        let mut cancel_handles = self.cancel_handles.lock();
        for t in self.completed_tasks.lock().drain(..) {
            cancel_handles.remove(t);
        }
    }

    #[track_caller]
    pub fn spawn_task<F: Future + Send + 'scope>(
        &self,
        priority: TaskPriority,
        fut: F,
    ) -> JoinHandle<F::Output>
    where
        <F as Future>::Output: Send + 'static,
    {
        let spawn_location = Location::caller();
        self.clear_completed_tasks();

        let mut runnable = None;
        let mut join_handle = None;
        self.cancel_handles.lock().insert_with_key(|task_key| {
            let metrics = TRACK_METRICS.load().then(Arc::default);
            let dyn_task = unsafe {
                // SAFETY: we make sure to cancel this task before 'scope ends.
                let executor = Executor::global();
                let on_wake = move |task| executor.schedule_task(task);
                task::spawn_with_lifetime(
                    fut,
                    on_wake,
                    TaskMetadata {
                        spawn_location,
                        priority,
                        freshly_spawned: AtomicBool::new(true),
                        scoped: Some(ScopedTaskMetadata {
                            task_key,
                            completed_tasks: Arc::downgrade(&self.completed_tasks),
                        }),
                        metrics,
                    },
                )
            };
            runnable = Some(Arc::clone(&dyn_task));
            let jh = JoinHandle(dyn_task);
            let cancel_handle = jh.cancel_handle();
            join_handle = Some(jh);
            cancel_handle
        });
        runnable.unwrap().schedule();
        join_handle.unwrap()
    }
}

pub fn task_scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope TaskScope<'scope, 'env>) -> T,
{
    // By having this local variable inaccessible to anyone we guarantee
    // that either abort is called killing the entire process, or that this
    // executor is properly destroyed.
    let scope = TaskScope {
        cancel_handles: Mutex::default(),
        completed_tasks: Arc::new(Mutex::default()),
        scope: PhantomData,
        env: PhantomData,
    };

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| f(&scope)));

    // Make sure all tasks are properly destroyed.
    scope.destroy();

    match result {
        Err(e) => std::panic::resume_unwind(e),
        Ok(result) => result,
    }
}

#[track_caller]
pub fn spawn<F: Future + Send + 'static>(priority: TaskPriority, fut: F) -> JoinHandle<F::Output>
where
    <F as Future>::Output: Send + 'static,
{
    let spawn_location = Location::caller();
    let executor = Executor::global();
    let on_wake = move |task| executor.schedule_task(task);
    let metrics = TRACK_METRICS.load().then(Arc::default);
    let dyn_task = task::spawn(
        fut,
        on_wake,
        TaskMetadata {
            spawn_location,
            priority,
            freshly_spawned: AtomicBool::new(true),
            scoped: None,
            metrics,
        },
    );
    Arc::clone(&dyn_task).schedule();
    JoinHandle(dyn_task)
}

fn random_permutation<R: Rng>(len: u32, rng: &mut R) -> impl Iterator<Item = u32> {
    let modulus = len.next_power_of_two();
    let halfwidth = modulus.trailing_zeros() / 2;
    let mask = modulus - 1;
    let displace_zero = rng.random::<u32>();
    let odd1 = rng.random::<u32>() | 1;
    let odd2 = rng.random::<u32>() | 1;
    let uniform_first = ((rng.random::<u32>() as u64 * len as u64) >> 32) as u32;

    (0..modulus)
        .map(move |mut i| {
            // Invertible permutation on [0, modulus).
            i = i.wrapping_add(displace_zero);
            i = i.wrapping_mul(odd1);
            i ^= (i & mask) >> halfwidth;
            i = i.wrapping_mul(odd2);
            i & mask
        })
        .filter(move |i| *i < len)
        .map(move |mut i| {
            i += uniform_first;
            if i >= len {
                i -= len;
            }
            i
        })
}
