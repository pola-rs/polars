use std::sync::Arc;
use std::time::Duration;

use slotmap::SecondaryMap;

use crate::async_executor::TaskMetrics;
use crate::graph::{Graph, GraphNodeKey, LogicalPipeKey};
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
}

impl NodeMetrics {
    fn add_task(&mut self, task_metrics: &TaskMetrics) {
        self.total_polls += task_metrics.total_polls.load();
        self.total_stolen_polls += task_metrics.total_stolen_polls.load();
        self.total_poll_time_ns += task_metrics.total_poll_time_ns.load();
        self.max_poll_time_ns = self
            .max_poll_time_ns
            .max(task_metrics.max_poll_time_ns.load());
    }

    fn add_state_update(&mut self, time: Duration) {
        let time_ns = time.as_nanos() as u64;
        self.total_state_updates += 1;
        self.total_state_update_time_ns += time_ns;
        self.max_state_update_time_ns = self.max_state_update_time_ns.max(time_ns);
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

#[derive(Default)]
pub struct GraphMetrics {
    node_metrics: SecondaryMap<GraphNodeKey, NodeMetrics>,
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

    pub fn add_state_update(&mut self, key: GraphNodeKey, time: Duration) {
        self.node_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .add_state_update(time);
    }

    pub fn flush(&mut self, graph: &Graph) {
        for (key, in_progress_task_metrics) in self.in_progress_task_metrics.iter_mut() {
            for task_metrics in in_progress_task_metrics.drain(..) {
                self.node_metrics
                    .entry(key)
                    .unwrap()
                    .or_default()
                    .add_task(&task_metrics);
            }
        }

        for (key, in_progress_pipe_metrics) in self.in_progress_pipe_metrics.iter_mut() {
            for pipe_metrics in in_progress_pipe_metrics.drain(..) {
                let pipe = &graph.pipes[key];
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
}
