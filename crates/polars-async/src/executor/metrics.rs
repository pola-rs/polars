use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Default)]
#[repr(align(128))]
pub(super) struct TaskMetrics {
    total_polls: RelaxedCell<u64>,
    total_stolen_polls: RelaxedCell<u64>,
    total_poll_time_ns: RelaxedCell<u64>,
    max_poll_time_ns: RelaxedCell<u64>,
    done: AtomicBool,
}

impl TaskMetrics {
    /// Called by the runner after each poll, `done` is published last.
    pub(super) fn record_poll(&self, elapsed_ns: u64, stolen: bool, done: bool) {
        self.total_polls.fetch_add(1);
        if stolen {
            self.total_stolen_polls.fetch_add(1);
        }
        self.total_poll_time_ns.fetch_add(elapsed_ns);
        self.max_poll_time_ns.fetch_max(elapsed_ns);
        if done {
            self.mark_done();
        }
    }

    pub(super) fn mark_done(&self) {
        self.done.store(true, Ordering::Release);
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct TaskMetricsSnapshot {
    pub total_polls: u64,
    pub total_stolen_polls: u64,
    pub total_poll_time_ns: u64,
    pub max_poll_time_ns: u64,
    pub num_running_tasks: u32,
}

impl TaskMetricsSnapshot {
    fn add_task(&mut self, m: &TaskMetrics) {
        self.total_polls += m.total_polls.load();
        self.total_stolen_polls += m.total_stolen_polls.load();
        self.total_poll_time_ns += m.total_poll_time_ns.load();
        self.max_poll_time_ns = self.max_poll_time_ns.max(m.max_poll_time_ns.load());
    }
}

/// Aggregates the metrics of all tasks spawned with it.
#[derive(Default)]
pub struct TaskMetricAggregator {
    state: Mutex<AggregatorState>,
}

#[derive(Default)]
struct AggregatorState {
    finished: TaskMetricsSnapshot,
    live: Vec<Arc<TaskMetrics>>,
    retain_amort: usize,
}

impl AggregatorState {
    fn compact(&mut self) {
        self.retain_amort = 0;
        let finished = &mut self.finished;
        self.live.retain_mut(|m| {
            // A task freed without finishing a poll (e.g. pending with all
            // its wakers dropped) leaves us as the only owner.
            let is_finished = m.done.load(Ordering::Acquire) || Arc::get_mut(m).is_some();
            if is_finished {
                finished.add_task(m);
            }
            !is_finished
        });
    }
}

impl TaskMetricAggregator {
    pub(super) fn new_task_metrics(&self) -> Arc<TaskMetrics> {
        let metrics = Arc::<TaskMetrics>::default();
        let mut state = self.state.lock();
        state.retain_amort += 2; // Grows twice as fast as push.
        if state.retain_amort >= state.live.len() {
            state.compact();
        }
        state.live.push(metrics.clone());
        metrics
    }

    /// Cumulative over all tasks ever spawned with this aggregator.
    pub fn snapshot(&self) -> TaskMetricsSnapshot {
        let mut state = self.state.lock();
        state.compact();
        let mut out = state.finished;
        for m in &state.live {
            out.add_task(m);
        }
        out.num_running_tasks = state.live.len() as u32;
        out
    }
}
