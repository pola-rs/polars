use std::sync::Arc;
use std::time::Duration;

pub use polars_io::metrics::{IOMetrics, OptIOMetrics};
use slotmap::{SecondaryMap, SlotMap};

use crate::LogicalPipe;
use crate::async_executor::TaskMetrics;
use crate::graph::{GraphNodeKey, LogicalPipeKey};
use crate::pipe::PipeMetrics;

#[derive(Default, Clone)]
pub struct NodeMetrics {
    pub total_polls: u64,
    pub total_stolen_polls: u64,
    pub total_poll_time_ns: u64,
    pub max_poll_time_ns: u64,

    pub total_state_updates: u64,
    pub total_state_update_time_ns: u64,
    pub max_state_update_time_ns: u64,

    pub morsels_sent: u64,
    pub rows_sent: u64,
    pub largest_morsel_sent: u64,
    pub morsels_received: u64,
    pub rows_received: u64,
    pub largest_morsel_received: u64,

    pub io_total_active_ns: u64,
    pub io_total_bytes_requested: u64,
    pub io_total_bytes_received: u64,
    pub io_total_bytes_sent: u64,

    pub state_update_in_progress: bool,
    pub num_running_tasks: u32,
    pub done: bool,
}

impl NodeMetrics {
    fn add_task(&mut self, task_metrics: &TaskMetrics) {
        self.total_polls += task_metrics.total_polls.load();
        self.total_stolen_polls += task_metrics.total_stolen_polls.load();
        self.total_poll_time_ns += task_metrics.total_poll_time_ns.load();
        self.max_poll_time_ns = self
            .max_poll_time_ns
            .max(task_metrics.max_poll_time_ns.load());
        self.num_running_tasks += (!task_metrics.done.load()) as u32;
    }

    fn add_io(&mut self, io_metrics: &IOMetrics) {
        self.io_total_active_ns += io_metrics.io_timer.total_time_live_ns();
        self.io_total_bytes_requested += io_metrics.bytes_requested.load();
        self.io_total_bytes_received += io_metrics.bytes_received.load();
        self.io_total_bytes_sent += io_metrics.bytes_sent.load();
    }

    fn start_state_update(&mut self) {
        self.state_update_in_progress = true;
    }

    fn stop_state_update(&mut self, time: Duration, is_done: bool) {
        let time_ns = time.as_nanos() as u64;
        self.total_state_updates += 1;
        self.total_state_update_time_ns += time_ns;
        self.max_state_update_time_ns = self.max_state_update_time_ns.max(time_ns);
        self.state_update_in_progress = false;
        self.done = is_done;
    }

    fn add_send_metrics(&mut self, pipe_metrics: &PipeMetrics) {
        self.morsels_sent += pipe_metrics.morsels_sent.load();
        self.rows_sent += pipe_metrics.rows_sent.load();
        self.largest_morsel_sent = self
            .largest_morsel_sent
            .max(pipe_metrics.largest_morsel_sent.load());
    }

    fn add_recv_metrics(&mut self, pipe_metrics: &PipeMetrics) {
        self.morsels_received += pipe_metrics.morsels_received.load();
        self.rows_received += pipe_metrics.rows_received.load();
        self.largest_morsel_received = self
            .largest_morsel_received
            .max(pipe_metrics.largest_morsel_received.load());
    }
}

#[derive(Default, Clone)]
pub struct GraphMetrics {
    node_metrics: SecondaryMap<GraphNodeKey, NodeMetrics>,
    in_progress_io_metrics: SecondaryMap<GraphNodeKey, Vec<Arc<IOMetrics>>>,
    in_progress_task_metrics: SecondaryMap<GraphNodeKey, Vec<Arc<TaskMetrics>>>,
    in_progress_pipe_metrics: SecondaryMap<LogicalPipeKey, Vec<Arc<PipeMetrics>>>,
}

impl GraphMetrics {
    pub fn add_task(&mut self, key: GraphNodeKey, task_metrics: Arc<TaskMetrics>) {
        self.in_progress_task_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .push(task_metrics);
    }

    pub fn add_pipe(&mut self, key: LogicalPipeKey, pipe_metrics: Arc<PipeMetrics>) {
        self.in_progress_pipe_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .push(pipe_metrics);
    }

    pub fn start_state_update(&mut self, key: GraphNodeKey) {
        self.node_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .start_state_update();
    }

    pub fn stop_state_update(&mut self, key: GraphNodeKey, time: Duration, is_done: bool) {
        self.node_metrics[key].stop_state_update(time, is_done);
    }

    pub fn flush(&mut self, pipes: &SlotMap<LogicalPipeKey, LogicalPipe>) {
        for (key, in_progress_task_metrics) in self.in_progress_task_metrics.iter_mut() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.num_running_tasks = 0;
            for task_metrics in in_progress_task_metrics.drain(..) {
                this_node_metrics.add_task(&task_metrics);
            }
        }

        for (key, in_progress_io_metrics) in self.in_progress_io_metrics.iter_mut() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.num_running_tasks = 0;
            for io_metrics in in_progress_io_metrics.drain(..) {
                this_node_metrics.add_io(&io_metrics);
            }
        }

        for (key, in_progress_pipe_metrics) in self.in_progress_pipe_metrics.iter_mut() {
            for pipe_metrics in in_progress_pipe_metrics.drain(..) {
                let pipe = &pipes[key];
                self.node_metrics
                    .entry(pipe.receiver)
                    .unwrap()
                    .or_default()
                    .add_recv_metrics(&pipe_metrics);
                self.node_metrics
                    .entry(pipe.sender)
                    .unwrap()
                    .or_default()
                    .add_send_metrics(&pipe_metrics);
            }
        }
    }

    pub fn get(&self, key: GraphNodeKey) -> Option<&NodeMetrics> {
        self.node_metrics.get(key)
    }

    pub fn iter(&self) -> slotmap::secondary::Iter<'_, GraphNodeKey, NodeMetrics> {
        self.node_metrics.iter()
    }
}

pub struct MetricsBuilder {
    pub graph_key: GraphNodeKey,
    pub graph_metrics: Arc<parking_lot::Mutex<GraphMetrics>>,
}

impl MetricsBuilder {
    pub fn new_io_metrics(&self) -> Arc<IOMetrics> {
        let io_metrics: Arc<IOMetrics> = Default::default();

        self.graph_metrics
            .lock()
            .in_progress_io_metrics
            .entry(self.graph_key)
            .unwrap()
            .or_default()
            .push(Arc::clone(&io_metrics));

        io_metrics
    }
}
