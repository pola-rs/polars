use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeMetricsDescription {
    pub phys_node_key: u64,
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
    pub total_time_ns: u64,
    pub done: bool,
    pub custom: Vec<CustomMetricDescription>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MetricUnit {
    #[default]
    #[serde(rename = "1")]
    Unit,
    #[serde(rename = "By")]
    Bytes,
    #[serde(rename = "ns")]
    DurationNs,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomMetricDescription {
    pub key: String,
    pub unit: MetricUnit,
    /// The value is only [`None`] when it was never set/updated.
    pub value: Option<i64>,
}
