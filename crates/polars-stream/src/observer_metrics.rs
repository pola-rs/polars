use std::sync::Arc;

use parking_lot::Mutex;
use polars_descriptions::NodeMetricsDescription;
use polars_observer::QueryMetrics;
use slotmap::{Key, SecondaryMap, SlotMap};

use crate::graph::GraphNodeKey;
use crate::metrics::{GraphMetrics, NodeMetrics};
use crate::physical_plan::PhysNodeKey;
use crate::skeleton::StreamingQuery;
use crate::{LogicalPipe, LogicalPipeKey};

pub struct StreamingQueryMetrics {
    pub metrics: Arc<Mutex<GraphMetrics>>,
    pub pipes: SlotMap<LogicalPipeKey, LogicalPipe>,
    pub phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
}

impl StreamingQueryMetrics {
    pub fn from_query(query: &StreamingQuery) -> Option<Box<dyn QueryMetrics>> {
        let metrics = query.metrics.clone()?;
        Some(Box::new(Self {
            metrics,
            pipes: query.graph.pipes.clone(),
            phys_to_graph: query.phys_to_graph.clone(),
        }))
    }
}

impl QueryMetrics for StreamingQueryMetrics {
    fn snapshot(&self) -> Vec<NodeMetricsDescription> {
        let mut metrics = { self.metrics.lock().clone() };
        metrics.flush(&self.pipes);

        self.phys_to_graph
            .iter()
            .map(|(phys_key, graph_key)| {
                let node = metrics.get(*graph_key).cloned().unwrap_or_default();
                metrics_row(phys_key.data().as_ffi(), &node)
            })
            .collect()
    }
}

fn metrics_row(phys_node_key: u64, m: &NodeMetrics) -> NodeMetricsDescription {
    NodeMetricsDescription {
        phys_node_key,
        total_polls: m.total_polls,
        total_stolen_polls: m.total_stolen_polls,
        total_poll_time_ns: m.total_poll_time_ns,
        max_poll_time_ns: m.max_poll_time_ns,
        total_state_updates: m.total_state_updates,
        total_state_update_time_ns: m.total_state_update_time_ns,
        max_state_update_time_ns: m.max_state_update_time_ns,
        morsels_sent: m.morsels_sent,
        rows_sent: m.rows_sent,
        largest_morsel_sent: m.largest_morsel_sent,
        morsels_received: m.morsels_received,
        rows_received: m.rows_received,
        largest_morsel_received: m.largest_morsel_received,
        io_total_active_ns: m.io_total_active_ns,
        io_total_bytes_requested: m.io_total_bytes_requested,
        io_total_bytes_received: m.io_total_bytes_received,
        io_total_bytes_sent: m.io_total_bytes_sent,
        total_time_ns: m.total_poll_time_ns + m.total_state_update_time_ns,
        done: m.done,
    }
}
