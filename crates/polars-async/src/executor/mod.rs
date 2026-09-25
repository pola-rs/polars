#![allow(clippy::disallowed_types)]

#[cfg(feature = "numa")]
mod numa;

#[cfg(not(feature = "numa"))]
#[path = "numa/dummy.rs"]
mod numa;

mod park_group;
mod task;

use std::cell::{Cell, UnsafeCell};
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, Location};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use crossbeam_deque::{Injector, Steal, Stealer, Worker as WorkQueue};
use crossbeam_utils::CachePadded;
use numa::{NumaRegionId, cpu_idx_to_numa_region, num_numa_regions, pin_thread_to_numa_region};
use park_group::ParkGroup;
use parking_lot::Mutex;
use polars_utils::cpu_time::thread_cpu_ns;
use polars_utils::live_timer::LiveTimerSession;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::with_drop::WithDrop;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use slotmap::SlotMap;
use task::{Cancellable, DynTask, Runnable};

thread_local! {
    pub static ALLOW_RAYON_THREADS: Cell<bool> = const { Cell::new(true) };
    pub static THREAD_SPAWNED_BY_POLARS_EXECUTOR: Cell<bool> = const { Cell::new(false) };

    /// Attribution inherited by tasks spawned during the current poll.
    static TLS_ATTRIBUTION: Cell<Option<Arc<dyn TaskAttribution>>> = const { Cell::new(None) };

    /// Used to store which executor thread this is.
    static TLS_THREAD_ID: Cell<usize> = const { Cell::new(usize::MAX) };
    /// In which NUMA region is this executor thread supposed to run.
    static TLS_NUMA_REGION: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Returns whether this thread is actively used for scheduling tasks.
pub fn is_scheduling_polars_executor_thread() -> bool {
    TLS_THREAD_ID.get() != usize::MAX
}

static TRACK_METRICS: RelaxedCell<bool> = RelaxedCell::new_bool(false);

pub fn track_task_metrics(should_track: bool) {
    TRACK_METRICS.store(should_track);
}

static TRACK_POLL_CPU: RelaxedCell<bool> = RelaxedCell::new_bool(false);

/// Measure thread CPU time per poll, excluding time spent descheduled.
///
/// Requires task metrics. Disabled by default due to clock-read overhead;
/// unavailable on Windows, where the thread clock is too coarse.
pub fn track_poll_cpu_time(should_track: bool) {
    TRACK_POLL_CPU.store(should_track);
}

/// Receives the metrics of every task spawned on this attribution's behalf.
///
/// Implementations must distinguish queries: the executor is process-wide.
pub trait TaskAttribution: Send + Sync + 'static {
    /// Called once per spawn, before the task is first polled.
    ///
    /// Retain the `Arc` to read metrics as the task runs.
    fn task_spawned(&self, metrics: &Arc<TaskMetrics>);

    /// Opens a poll session to measure wall time with concurrent polls counted once.
    fn poll_session(&self) -> Option<LiveTimerSession> {
        None
    }
}

/// Executor wall time split into polling, scheduling and parking.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerStateTimes {
    /// Running task futures.
    pub poll_ns: u64,
    /// Looking for a task: the local slot, the local queue, stealing, unparking.
    pub overhead_ns: u64,
    /// Asleep with nothing to do.
    pub parked_ns: u64,
    /// Thread CPU across those polls, when [`track_poll_cpu_time`] is on.
    pub poll_cpu_ns: u64,
    /// Part of `poll_ns` spent on tasks without attribution.
    pub unattributed_poll_ns: u64,
}

impl WorkerStateTimes {
    /// Total wall time; `unattributed_poll_ns` is already included in `poll_ns`.
    pub fn total_ns(&self) -> u64 {
        self.poll_ns + self.overhead_ns + self.parked_ns
    }

    /// Counter changes since `earlier`.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            poll_ns: self.poll_ns.saturating_sub(earlier.poll_ns),
            overhead_ns: self.overhead_ns.saturating_sub(earlier.overhead_ns),
            parked_ns: self.parked_ns.saturating_sub(earlier.parked_ns),
            poll_cpu_ns: self.poll_cpu_ns.saturating_sub(earlier.poll_cpu_ns),
            unattributed_poll_ns: self
                .unattributed_poll_ns
                .saturating_sub(earlier.unattributed_poll_ns),
        }
    }
}

/// Counters with one writer per cache line. Relaxed reads may be stale.
#[derive(Default)]
#[repr(align(128))]
struct WorkerStateCounters {
    poll_ns: RelaxedCell<u64>,
    overhead_ns: RelaxedCell<u64>,
    parked_ns: RelaxedCell<u64>,
    poll_cpu_ns: RelaxedCell<u64>,
    unattributed_poll_ns: RelaxedCell<u64>,
}

impl WorkerStateCounters {
    fn read(&self) -> WorkerStateTimes {
        WorkerStateTimes {
            poll_ns: self.poll_ns.load(),
            overhead_ns: self.overhead_ns.load(),
            parked_ns: self.parked_ns.load(),
            poll_cpu_ns: self.poll_cpu_ns.load(),
            unattributed_poll_ns: self.unattributed_poll_ns.load(),
        }
    }
}

/// Retains worker totals even after runners exit or give up their identity.
static WORKER_STATES: LazyLock<Mutex<Vec<Arc<WorkerStateCounters>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

thread_local! {
    static TLS_WORKER_STATE: Arc<WorkerStateCounters> = {
        let counters = Arc::<WorkerStateCounters>::default();
        WORKER_STATES.lock().push(Arc::clone(&counters));
        counters
    };
}

/// Sums every executor thread's state times.
///
/// Use [`WorkerStateTimes::since`] for deltas. Concurrent queries share these totals.
pub fn worker_state_times() -> WorkerStateTimes {
    let mut total = WorkerStateTimes::default();
    for counters in WORKER_STATES.lock().iter() {
        let t = counters.read();
        total.poll_ns += t.poll_ns;
        total.overhead_ns += t.overhead_ns;
        total.parked_ns += t.parked_ns;
        total.poll_cpu_ns += t.poll_cpu_ns;
        total.unattributed_poll_ns += t.unattributed_poll_ns;
    }
    total
}

/// Captures the current attribution and restores it on each poll, including
/// when the future runs on another runtime.
pub fn with_current_attribution<F: Future>(fut: F) -> AttributedFuture<F> {
    AttributedFuture {
        attribution: current_task_attribution(),
        fut,
    }
}

pin_project_lite::pin_project! {
    pub struct AttributedFuture<F> {
        attribution: Option<Arc<dyn TaskAttribution>>,
        #[pin]
        fut: F,
    }
}

impl<F: Future> Future for AttributedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let _guard = scoped_task_attribution(this.attribution.clone());
        this.fut.poll(cx)
    }
}

fn new_task_metrics(attribution: Option<&Arc<dyn TaskAttribution>>) -> Option<Arc<TaskMetrics>> {
    let metrics: Option<Arc<TaskMetrics>> = TRACK_METRICS.load().then(Arc::default);
    if let Some((attribution, metrics)) = attribution.zip(metrics.as_ref()) {
        attribution.task_spawned(metrics);
    }
    metrics
}

/// Restores the previous attribution when dropped.
#[must_use = "dropping the guard immediately restores the previous attribution"]
pub struct AttributionGuard(Option<Arc<dyn TaskAttribution>>);

impl Drop for AttributionGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        TLS_ATTRIBUTION.with(|slot| slot.set(previous));
    }
}

/// Credits tasks spawned from this thread to `attribution` until the guard drops.
pub fn scoped_task_attribution(attribution: Option<Arc<dyn TaskAttribution>>) -> AttributionGuard {
    AttributionGuard(TLS_ATTRIBUTION.with(|slot| slot.replace(attribution)))
}

fn current_task_attribution() -> Option<Arc<dyn TaskAttribution>> {
    // `Cell` cannot lend a reference, so take and put back.
    TLS_ATTRIBUTION.with(|slot| {
        let current = slot.take();
        slot.set(current.clone());
        current
    })
}

static GLOBAL_SCHEDULER: OnceLock<Executor> = OnceLock::new();

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
    /// Thread CPU time with [`track_poll_cpu_time`] enabled; zero if unavailable.
    pub total_poll_cpu_time_ns: RelaxedCell<u64>,
    pub done: RelaxedCell<bool>,
}

struct TaskMetadata {
    spawn_location: &'static Location<'static>,
    priority: TaskPriority,
    freshly_spawned: AtomicBool,
    scoped: Option<ScopedTaskMetadata>,
    metrics: Option<Arc<TaskMetrics>>,
    /// Inherited at spawn and restored during each poll.
    attribution: Option<Arc<dyn TaskAttribution>>,
}

impl Drop for TaskMetadata {
    fn drop(&mut self) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.done.store(true);
        }

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

    #[inline]
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
    thread_numa_regions: Vec<NumaRegionId>,
    thread_task_lists: Vec<CachePadded<ThreadLocalTaskList>>,
    global_high_prio_task_queue: Injector<ReadyTask>,
    global_low_prio_task_queue: Injector<ReadyTask>,
    thread_name_idx: AtomicUsize,

    // These three are tracked per NUMA region.
    park_groups: Vec<ParkGroup>,
    thread_id_send: Vec<Sender<Arc<AtomicUsize>>>,
    thread_id_recv: Vec<Receiver<Arc<AtomicUsize>>>,
    num_runners_without_identity: Vec<AtomicUsize>,
}

impl Executor {
    fn unpark_one_worker_random_numa_region(&self) {
        if self.park_groups.len() == 1 {
            self.park_groups[0].unpark_one();
        } else {
            let mut rng = rand::rng();
            for index in random_permutation(self.park_groups.len() as u32, &mut rng) {
                if self.park_groups[index as usize].unpark_one() {
                    break;
                }
            }
        }
    }

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
            // Scheduled from an unknown thread, add to global queue and wake
            // a worker in a random NUMA region.
            if meta.priority == TaskPriority::High {
                self.global_high_prio_task_queue.push(task);
            } else {
                self.global_low_prio_task_queue.push(task);
            }

            self.unpark_one_worker_random_numa_region();
        } else {
            let ttl = opt_ttl.unwrap();
            let numa = self.thread_numa_regions[thread];
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
                if !self.park_groups[numa.0].unpark_one() && self.park_groups.len() > 1 {
                    self.unpark_one_worker_random_numa_region();
                }
            } else {
                // Optimization: while this is a low priority task we have no
                // high priority tasks on this thread so we'll execute this one.
                if ttl.high_prio_tasks.is_empty() && slot.is_none() {
                    *slot = Some(task);
                } else {
                    self.global_low_prio_task_queue.push(task);
                    if !self.park_groups[numa.0].unpark_one() && self.park_groups.len() > 1 {
                        self.unpark_one_worker_random_numa_region();
                    }
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
        let steal_iters = 4;
        for steal_idx in 0..steal_iters {
            // For the first few steal attempts try to limit ourselves to our
            // own NUMA region, then on the last attempt try globally.
            let limit_to_numa_region = steal_idx != steal_iters - 1;

            let mut retry = true;
            while retry {
                retry = false;

                for idx in random_permutation(self.thread_task_lists.len() as u32, rng) {
                    if limit_to_numa_region
                        && self.thread_numa_regions[thread]
                            != self.thread_numa_regions[idx as usize]
                    {
                        continue;
                    }

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

    fn runner(&self, initial_thread_id: Option<usize>, numa_region: NumaRegionId) {
        TLS_THREAD_ID.set(initial_thread_id.unwrap_or(usize::MAX));
        TLS_NUMA_REGION.set(numa_region.0);
        ALLOW_RAYON_THREADS.set(false);
        THREAD_SPAWNED_BY_POLARS_EXECUTOR.set(true);

        pin_thread_to_numa_region(numa_region);

        let mut rng = SmallRng::from_rng(&mut rand::rng());
        let mut worker = self.park_groups[numa_region.0].new_worker();
        // Register lazily: `block_in_place` creates runners, and the registry
        // is never pruned. Metrics-disabled queries must not grow it.
        let mut worker_state: Option<Arc<WorkerStateCounters>> = None;

        loop {
            // If we're a runner without an assigned thread id, get one.
            let mut thread_id = TLS_THREAD_ID.get();
            if thread_id == usize::MAX {
                if let Some(tid) = self.acquire_thread_identity(numa_region) {
                    TLS_THREAD_ID.set(tid);
                    thread_id = tid;
                } else {
                    return;
                }
            }

            let ttl = &self.thread_task_lists[thread_id];
            let mut local = true;
            // Separate time spent searching for work from time parked.
            let track_states = TRACK_METRICS.load();
            let worker_state: Option<&WorkerStateCounters> = track_states
                .then(|| &**worker_state.get_or_insert_with(|| TLS_WORKER_STATE.with(Arc::clone)));
            let fetch_start = track_states.then(Instant::now);
            let mut parked_ns = 0u64;
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
                if let Some(task) = self.try_steal_task(thread_id, &mut rng) {
                    return Some(task);
                }

                // Prepare to park, then try one more steal attempt.
                let park = worker.prepare_park();
                if let Some(task) = self.try_steal_task(thread_id, &mut rng) {
                    return Some(task);
                }

                let park_start = track_states.then(Instant::now);
                park.park();
                if let Some(park_start) = park_start {
                    parked_ns = park_start.elapsed().as_nanos() as u64;
                }
                None
            })();

            if let Some((fetch_start, worker_state)) = fetch_start.zip(worker_state) {
                let elapsed_ns = fetch_start.elapsed().as_nanos() as u64;
                worker_state
                    .overhead_ns
                    .fetch_add(elapsed_ns.saturating_sub(parked_ns));
                if parked_ns != 0 {
                    worker_state.parked_ns.fetch_add(parked_ns);
                }
            }

            if let Some(task) = task {
                // Try to recruit another worker, and if there's no idle workers
                // left in this NUMA region, one from another.
                if worker.recruit_next() == Some(false) && self.park_groups.len() > 1 {
                    self.unpark_one_worker_random_numa_region();
                }

                if track_states {
                    // Read before running: `run` consumes the task.
                    let metrics = task.metadata().metrics.clone();
                    let attribution = task.metadata().attribution.clone();
                    let had_attribution = attribution.is_some();

                    // Count overlapping polls once toward node occupancy.
                    let _occupancy = attribution.as_ref().and_then(|a| a.poll_session());
                    let _attribution = scoped_task_attribution(attribution);

                    let cpu_start = TRACK_POLL_CPU.load().then(thread_cpu_ns).flatten();
                    let start = Instant::now();
                    task.run();
                    let elapsed_ns = start.elapsed().as_nanos() as u64;
                    // Avoid reading the clock when CPU tracking is disabled.
                    let cpu_ns = cpu_start.map_or(0, |before| {
                        thread_cpu_ns().map_or(0, |after| after.saturating_sub(before))
                    });

                    // Include tasks without their own metrics in worker totals.
                    if let Some(worker_state) = worker_state {
                        worker_state.poll_ns.fetch_add(elapsed_ns);
                        if cpu_ns != 0 {
                            worker_state.poll_cpu_ns.fetch_add(cpu_ns);
                        }
                        if !had_attribution {
                            worker_state.unattributed_poll_ns.fetch_add(elapsed_ns);
                        }
                    }

                    if let Some(metrics) = metrics {
                        metrics.total_polls.fetch_add(1);
                        if !local {
                            metrics.total_stolen_polls.fetch_add(1);
                        }
                        metrics.total_poll_time_ns.fetch_add(elapsed_ns);
                        metrics.total_poll_cpu_time_ns.fetch_add(cpu_ns);
                        metrics.max_poll_time_ns.fetch_max(elapsed_ns);
                    }
                } else {
                    task.run();
                }
            }
        }
    }

    fn spawn_runner_without_identity(&self, numa_region: NumaRegionId) {
        self.num_runners_without_identity[numa_region.0].fetch_add(1, Ordering::AcqRel);
        let t = self.thread_name_idx.fetch_add(1, Ordering::Relaxed);
        std::thread::Builder::new()
            .name(format!("async-executor-{t}"))
            .spawn(move || Self::global().runner(None, numa_region))
            .unwrap();
    }

    fn acquire_thread_identity(&self, numa_region: NumaRegionId) -> Option<usize> {
        loop {
            match self.thread_id_recv[numa_region.0].recv_timeout(Duration::from_secs(10)) {
                Ok(tid_msg) => {
                    let thread_id = tid_msg.swap(usize::MAX, Ordering::AcqRel);
                    if thread_id != usize::MAX {
                        // Important: we check queue again after reducing count.
                        let num_left = self.num_runners_without_identity[numa_region.0]
                            .fetch_sub(1, Ordering::AcqRel)
                            - 1;
                        if num_left == 0 && !self.thread_id_recv[numa_region.0].is_empty() {
                            self.spawn_runner_without_identity(numa_region);
                        }
                        return Some(thread_id);
                    }
                },
                Err(_) => {
                    // Important: we check queue again after reducing count.
                    self.num_runners_without_identity[numa_region.0].fetch_sub(1, Ordering::AcqRel);
                    if self.thread_id_recv[numa_region.0].is_empty() {
                        return None;
                    }
                    self.num_runners_without_identity[numa_region.0].fetch_add(1, Ordering::AcqRel);
                },
            }
        }
    }

    fn ensure_runner_without_identity_exists(&self, numa_region: NumaRegionId) {
        if self.num_runners_without_identity[numa_region.0].fetch_add(0, Ordering::AcqRel) == 0 {
            self.spawn_runner_without_identity(numa_region);
        }
    }

    fn global() -> &'static Executor {
        GLOBAL_SCHEDULER.get_or_init(|| {
            let n_threads = polars_config::config().max_threads();
            let thread_numa_regions: Vec<_> = (0..n_threads).map(cpu_idx_to_numa_region).collect();
            let thread_task_lists = (0..n_threads)
                .map(|t| {
                    let numa_region = thread_numa_regions[t];
                    std::thread::Builder::new()
                        .name(format!("async-executor-{t}"))
                        .spawn(move || Self::global().runner(Some(t), numa_region))
                        .unwrap();

                    let high_prio_tasks = WorkQueue::new_lifo();
                    CachePadded::new(ThreadLocalTaskList {
                        high_prio_tasks_stealer: high_prio_tasks.stealer(),
                        high_prio_tasks,
                        local_slot: UnsafeCell::new(None),
                    })
                })
                .collect();
            let (thread_id_send, thread_id_recv) = (0..num_numa_regions())
                .map(|_| crossbeam_channel::unbounded())
                .unzip();
            let park_groups = (0..num_numa_regions()).map(|_| ParkGroup::new()).collect();
            Self {
                park_groups,
                thread_numa_regions,
                thread_task_lists,
                global_high_prio_task_queue: Injector::new(),
                global_low_prio_task_queue: Injector::new(),
                thread_id_send,
                thread_id_recv,
                thread_name_idx: AtomicUsize::new(n_threads),
                num_runners_without_identity: (0..num_numa_regions())
                    .map(|_| AtomicUsize::new(0))
                    .collect(),
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
        // `task_spawned` takes the graph metrics lock; call it before locking
        // `cancel_handles` to avoid lock inversion.
        let attribution = current_task_attribution();
        let metrics = new_task_metrics(attribution.as_ref());

        self.cancel_handles.lock().insert_with_key(|task_key| {
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
                        attribution,
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
    let attribution = current_task_attribution();
    let metrics = new_task_metrics(attribution.as_ref());
    let dyn_task = task::spawn(
        fut,
        on_wake,
        TaskMetadata {
            spawn_location,
            priority,
            freshly_spawned: AtomicBool::new(true),
            scoped: None,
            metrics,
            attribution,
        },
    );
    Arc::clone(&dyn_task).schedule();
    JoinHandle(dyn_task)
}

/// Runs the given function on this thread while allowing another thread to take
/// over this thread's task execution duties.
///
/// Simply directly calls f() if this thread is not an async executor thread.
pub fn block_in_place<R, F: FnOnce() -> R>(f: F) -> R {
    let thread_id = TLS_THREAD_ID.replace(usize::MAX);
    if thread_id == usize::MAX {
        return f();
    }
    let numa_region = TLS_NUMA_REGION.get();

    // Send off our thread id to another runner, we just become an ordinary thread.
    let executor = Executor::global();
    let msg = Arc::new(AtomicUsize::new(thread_id));
    executor.thread_id_send[numa_region]
        .send(msg.clone())
        .unwrap();
    executor.ensure_runner_without_identity_exists(NumaRegionId(numa_region)); // Important: *after* sending in channel.

    // Try to steal our thread id back afterwards, even if f panics. If we can't
    // steal our thread id back we become a runner without identity.
    let _restore_identity = WithDrop::new(msg, |msg| {
        let thread_id = msg.swap(usize::MAX, Ordering::AcqRel);
        if thread_id != usize::MAX {
            TLS_THREAD_ID.set(thread_id);
        } else {
            executor.num_runners_without_identity[numa_region].fetch_add(1, Ordering::AcqRel);
        }
    });

    f()
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
