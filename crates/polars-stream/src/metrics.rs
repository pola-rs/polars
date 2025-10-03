use std::sync::Arc;
use std::time::Duration;

use slotmap::SecondaryMap;

use crate::async_executor::TaskMetrics;
use crate::graph::GraphNodeKey;

#[derive(Default, Clone)]
pub struct NodeMetrics {
    pub total_polls: u64,
    pub total_stolen_polls: u64,
    pub total_poll_time_ns: u64,
    pub max_poll_time_ns: u64,
    
    pub total_state_updates: u64,
    pub total_state_update_time_ns: u64,
    pub max_state_update_time_ns: u64,
}

impl NodeMetrics {
    fn add_task(&mut self, task_metrics: &TaskMetrics) {
        self.total_polls += task_metrics.total_polls.load();
        self.total_stolen_polls += task_metrics.total_stolen_polls.load();
        self.total_poll_time_ns += task_metrics.total_poll_time_ns.load();
        self.max_poll_time_ns = self.max_poll_time_ns.max(task_metrics.max_poll_time_ns.load());
    }
    
    fn add_state_update(&mut self, time: Duration) {
        let time_ns = time.as_nanos() as u64;
        self.total_state_updates += 1;
        self.total_state_update_time_ns += time_ns;
        self.max_state_update_time_ns = self.max_state_update_time_ns.max(time_ns);
    }
}


#[derive(Default)]
pub struct GraphMetrics {
    node_metrics: SecondaryMap<GraphNodeKey, NodeMetrics>,
    in_progress_task_metrics: SecondaryMap<GraphNodeKey, Vec<Arc<TaskMetrics>>>,
}

impl GraphMetrics {
    pub fn add_task(&mut self, key: GraphNodeKey, task_metrics: Arc<TaskMetrics>) {
        self.in_progress_task_metrics.entry(key).unwrap().or_default().push(task_metrics);
    }
    
    pub fn add_state_update(&mut self, key: GraphNodeKey, time: Duration) {
        self.node_metrics.entry(key).unwrap().or_default().add_state_update(time);
    }
    
    pub fn flush_tasks(&mut self) {
        for (key, in_progress_task_metrics) in self.in_progress_task_metrics.iter_mut() {
            for task_metrics in in_progress_task_metrics.drain(..) {
                self.node_metrics.entry(key).unwrap().or_default().add_task(&task_metrics);
            }
        }
    }
    
    pub fn get(&self, key: GraphNodeKey) -> Option<&NodeMetrics> {
        self.node_metrics.get(key)
    }
}

