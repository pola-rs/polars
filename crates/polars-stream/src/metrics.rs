use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use polars_async::executor::TaskMetrics;
pub use polars_descriptions::MetricUnit;
pub use polars_io::metrics::{IOMetrics, OptIOMetrics};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;
use slotmap::{SecondaryMap, SlotMap};

use crate::LogicalPipe;
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

    pub custom: Vec<CustomMetric>,
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

    fn reset_io_metrics(&mut self) {
        self.io_total_active_ns = 0;
        self.io_total_bytes_requested = 0;
        self.io_total_bytes_received = 0;
        self.io_total_bytes_sent = 0;
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
    in_progress_io_metrics: SecondaryMap<GraphNodeKey, Arc<IOMetrics>>,
    in_progress_custom_metrics: SecondaryMap<GraphNodeKey, Arc<CustomMetrics>>,
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

        for (key, io_metrics) in self.in_progress_io_metrics.iter_mut() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.reset_io_metrics();
            this_node_metrics.add_io(io_metrics);
        }

        for (key, custom_metrics) in self.in_progress_custom_metrics.iter() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.custom = custom_metrics.snapshot_and_compact();
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

pub struct NodeMetricsRegistry {
    pub graph_key: GraphNodeKey,
    pub graph_metrics: Option<Arc<parking_lot::Mutex<GraphMetrics>>>,
}

impl NodeMetricsRegistry {
    pub fn is_some(&self) -> bool {
        self.graph_metrics.is_some()
    }

    /// Registers this node's IO metrics.
    ///
    /// Nodes call this once per phase with the same [`IOMetrics`] each time, so
    /// repeat calls are expected and do nothing.
    ///
    /// # Panics
    /// If called with a different [`IOMetrics`] than this node registered before.
    pub fn register_io_metrics(&self, io_metrics: Arc<IOMetrics>) {
        let Some(registry) = &self.graph_metrics else {
            return;
        };

        let mut guard = registry.lock();

        use slotmap::secondary::Entry;

        match guard.in_progress_io_metrics.entry(self.graph_key).unwrap() {
            Entry::Occupied(e) => {
                // Each node should only have 1 set of metrics, identified by the Arc address.
                // But the registration can be called multiple times (per phase).
                assert!(Arc::ptr_eq(&io_metrics, e.get()));
            },
            Entry::Vacant(e) => {
                e.insert(io_metrics);
            },
        };
    }

    /// Registers a metric of the given [`MetricKind`].
    ///
    /// # Panics
    /// If `key` was already registered with a different unit or aggregation.
    pub fn register_custom_metric<K: MetricKind>(
        &self,
        key: &'static str,
        unit: MetricUnit,
    ) -> Metric<K> {
        let Some(registry) = &self.graph_metrics else {
            return Metric::default();
        };

        let metrics = registry
            .lock()
            .in_progress_custom_metrics
            .entry(self.graph_key)
            .unwrap()
            .or_default()
            .clone();

        let metric_idx = metrics.register(Spec::new(PlSmallStr::from_static(key), unit, K::AGG));

        Metric::new(MetricRef {
            metrics: Some(metrics),
            metric_idx,
        })
    }

    /// Registers a UpDownCounter that combines by summing
    pub fn new_counter(&self, key: &'static str, unit: MetricUnit) -> Metric<kind::Sum> {
        self.register_custom_metric(key, unit)
    }

    /// Registers a counter that combines by taking the highest share
    pub fn new_max(&self, key: &'static str, unit: MetricUnit) -> Metric<kind::Max> {
        self.register_custom_metric(key, unit)
    }

    /// Registers a gauge that combines values by taking the latest reading
    pub fn new_gauge(&self, key: &'static str, unit: MetricUnit) -> Metric<kind::Gauge> {
        self.register_custom_metric(key, unit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggMode {
    Sum,
    Max,
    Latest,
}

impl AggMode {
    #[inline]
    pub const fn identity(self) -> i64 {
        match self {
            Self::Sum => 0,
            Self::Max => i64::MIN,
            Self::Latest => 0,
        }
    }

    #[inline]
    fn fold(self, compacted: Compacted, cell: &Cell) -> Compacted {
        let (value, timestamp) = (cell.value.load(), cell.timestamp.load());

        match self {
            Self::Sum => Compacted {
                value: compacted.value.wrapping_add(value),
                timestamp: compacted.timestamp.max(timestamp),
            },
            Self::Max => Compacted {
                value: compacted.value.max(value),
                timestamp: compacted.timestamp.max(timestamp),
            },
            Self::Latest => {
                if compacted.timestamp > timestamp {
                    compacted
                } else {
                    Compacted { value, timestamp }
                }
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Spec {
    key: PlSmallStr,
    unit: MetricUnit,
    agg: AggMode,
}

impl Spec {
    const fn new(key: PlSmallStr, unit: MetricUnit, agg: AggMode) -> Self {
        Self { key, unit, agg }
    }
}

#[derive(Debug, Clone)]
pub struct CustomMetric {
    pub key: PlSmallStr,
    pub unit: MetricUnit,
    pub agg: AggMode,
    /// `None` when the counter never received a reading
    pub value: Option<i64>,
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

#[repr(align(64))]
struct Cell {
    value: RelaxedCell<i64>,
    /// `0` until the first write.
    timestamp: RelaxedCell<u64>,
}

impl Cell {
    pub fn new(value: i64) -> Self {
        Self {
            value: RelaxedCell::from(value),
            timestamp: RelaxedCell::from(0),
        }
    }

    /// Nanoseconds since [`EPOCH`], offset by 1 so that `0` means never written.
    #[inline]
    fn stamp_now(&self) {
        self.timestamp.store(EPOCH.elapsed().as_nanos() as u64 + 1);
    }
}

#[derive(Clone, Copy)]
struct Compacted {
    value: i64,
    timestamp: u64,
}

impl Compacted {
    pub fn new(value: i64) -> Self {
        Self {
            value,
            timestamp: 0,
        }
    }

    pub fn value(&self) -> Option<i64> {
        (self.timestamp != 0).then_some(self.value)
    }
}

struct MetricState {
    spec: Spec,
    compacted: Compacted,
    live: Vec<Arc<Cell>>,
}

#[derive(Default)]
pub(crate) struct CustomMetrics {
    state: parking_lot::Mutex<Vec<MetricState>>,
}

impl CustomMetrics {
    /// Registers a metric, returning its index.
    ///
    /// # Panics
    /// If the same key is registered with a different unit or aggregation method.
    fn register(&self, spec: Spec) -> usize {
        let mut state = self.state.lock();

        if let Some(metric_idx) = state.iter().position(|m| m.spec.key == spec.key) {
            assert_eq!(
                state[metric_idx].spec, spec,
                "metric `{}` was already registered differently",
                spec.key,
            );
            return metric_idx;
        }

        state.push(MetricState {
            compacted: Compacted::new(spec.agg.identity()),
            spec,
            live: Vec::new(),
        });

        state.len() - 1
    }

    /// Takes a cell of this metric for one task.
    fn new_cell(&self, metric_idx: usize) -> Arc<Cell> {
        let mut state = self.state.lock();
        let state = &mut state[metric_idx];

        let cell = Arc::new(Cell::new(state.spec.agg.identity()));
        state.live.push(cell.clone());

        cell
    }

    /// Reads every metric in registration order, and compacts the cells
    /// whose task has dropped.
    pub fn snapshot_and_compact(&self) -> Vec<CustomMetric> {
        let mut state = self.state.lock();

        state
            .iter_mut()
            .map(
                |MetricState {
                     spec,
                     compacted,
                     live,
                 }| {
                    live.retain_mut(|cell| {
                        if Arc::get_mut(cell).is_none() {
                            return true;
                        }

                        *compacted = spec.agg.fold(*compacted, cell);
                        false
                    });

                    let folded = live
                        .iter()
                        .fold(*compacted, |acc, cell| spec.agg.fold(acc, cell));

                    CustomMetric {
                        key: spec.key.clone(),
                        unit: spec.unit,
                        agg: spec.agg,
                        value: folded.value(),
                    }
                },
            )
            .collect()
    }
}

#[derive(Default, Clone)]
struct MetricRef {
    metrics: Option<Arc<CustomMetrics>>,
    metric_idx: usize,
}

impl MetricRef {
    fn take_cell(&self) -> Option<Arc<Cell>> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.new_cell(self.metric_idx))
    }
}

pub trait MetricKind: Default + Clone {
    const AGG: AggMode;
}

pub mod kind {
    use super::{AggMode, MetricKind};

    /// Combines by summing
    #[derive(Default, Clone, Copy)]
    pub struct Sum;

    /// Combines by taking the highest value
    #[derive(Default, Clone, Copy)]
    pub struct Max;

    /// Combines by taking latest value
    #[derive(Default, Clone, Copy)]
    pub struct Gauge;

    impl MetricKind for Sum {
        const AGG: AggMode = AggMode::Sum;
    }

    impl MetricKind for Max {
        const AGG: AggMode = AggMode::Max;
    }

    impl MetricKind for Gauge {
        const AGG: AggMode = AggMode::Latest;
    }
}

#[derive(Default, Clone)]
pub struct Metric<K: MetricKind>(MetricRef, K);

impl<K: MetricKind> Metric<K> {
    fn new(metric: MetricRef) -> Self {
        Self(metric, K::default())
    }

    pub fn reporter(&self) -> MetricReporter<K> {
        MetricReporter(self.0.take_cell(), K::default())
    }
}

#[must_use]
#[derive(Default, Clone)]
pub struct MetricReporter<K: MetricKind>(Option<Arc<Cell>>, K);

/// Represents an UpDownCounter
impl MetricReporter<kind::Sum> {
    #[inline]
    pub fn add(&self, delta: i64) {
        if let Some(cell) = &self.0 {
            cell.value.fetch_add(delta);
            cell.stamp_now();
        }
    }

    #[inline]
    pub fn sub(&self, delta: i64) {
        if let Some(cell) = &self.0 {
            cell.value.fetch_sub(delta);
            cell.stamp_now();
        }
    }
}

/// Represents kinda a counter (since it can only go up)
impl MetricReporter<kind::Max> {
    #[inline]
    pub fn record(&self, value: i64) {
        if let Some(cell) = &self.0 {
            cell.value.fetch_max(value);
            cell.stamp_now();
        }
    }
}

impl MetricReporter<kind::Gauge> {
    #[inline]
    pub fn set(&self, value: i64) {
        if let Some(cell) = &self.0 {
            cell.value.store(value);
            cell.stamp_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use slotmap::SlotMap;

    use super::*;
    use crate::graph::GraphNodeKey;
    use crate::metrics::GraphMetrics;

    const ROWS: &str = "test.rows";
    const PEAK: &str = "test.peak";
    const HELD: &str = "test.held";

    // Registration order, for reaching into the state directly.
    const ROWS_IDX: usize = 0;
    const PEAK_IDX: usize = 1;
    const HELD_IDX: usize = 2;

    struct Metrics {
        rows: Metric<kind::Sum>,
        peak: Metric<kind::Max>,
        held: Metric<kind::Gauge>,
    }

    fn registry() -> NodeMetricsRegistry {
        let mut nodes: SlotMap<GraphNodeKey, ()> = SlotMap::with_key();

        NodeMetricsRegistry {
            graph_key: nodes.insert(()),
            graph_metrics: Some(Arc::new(parking_lot::Mutex::new(GraphMetrics::default()))),
        }
    }

    fn register(registry: &NodeMetricsRegistry) -> Metrics {
        Metrics {
            rows: registry.new_counter(ROWS, MetricUnit::Unit),
            peak: registry.new_max(PEAK, MetricUnit::Bytes),
            held: registry.new_gauge(HELD, MetricUnit::Bytes),
        }
    }

    /// A registry with the three metrics above already registered.
    fn node() -> (NodeMetricsRegistry, Metrics) {
        let registry = registry();
        let metrics = register(&registry);
        (registry, metrics)
    }

    impl NodeMetricsRegistry {
        fn counters(&self) -> Arc<CustomMetrics> {
            self.graph_metrics
                .as_ref()
                .unwrap()
                .lock()
                .in_progress_custom_metrics[self.graph_key]
                .clone()
        }

        /// `None` only for a metric that never recorded.
        fn reading(&self, key: &str) -> Option<i64> {
            self.counters()
                .snapshot_and_compact()
                .into_iter()
                .find(|metric| metric.key == key)
                .unwrap()
                .value
        }

        fn value(&self, key: &str) -> i64 {
            self.reading(key).unwrap()
        }

        fn live_cells(&self, metric_idx: usize) -> usize {
            self.counters().state.lock()[metric_idx].live.len()
        }

        fn result(&self, metric_idx: usize) -> i64 {
            self.counters().state.lock()[metric_idx].compacted.value
        }
    }

    #[test]
    fn snapshot_reports_metrics_in_registration_order() {
        let (registry, _metrics) = node();

        let keys: Vec<_> = registry
            .counters()
            .snapshot_and_compact()
            .into_iter()
            .map(|metric| metric.key)
            .collect();

        assert_eq!(keys, [ROWS, PEAK, HELD]);
    }

    #[test]
    fn reporters_net_out_across_tasks() {
        let (registry, metrics) = node();

        let opener = metrics.rows.reporter();
        let closer = metrics.rows.reporter();

        opener.add(3);
        // One task's own cell goes negative; the total is still the net.
        closer.sub(2);
        assert_eq!(registry.value(ROWS), 1);

        // And it survives the negative cell being compacted.
        drop(closer);
        assert_eq!(registry.value(ROWS), 1);
        assert_eq!(registry.result(ROWS_IDX), -2);
    }

    #[test]
    fn a_gauge_reports_the_latest_set_across_tasks() {
        let (registry, metrics) = node();

        let first = metrics.held.reporter();
        let second = metrics.held.reporter();

        first.set(2);
        second.set(3);
        assert_eq!(registry.value(HELD), 3);

        // Registration order does not matter, only which write came last.
        first.set(10);
        assert_eq!(registry.value(HELD), 10);

        // A lower value still replaces a higher one.
        second.set(1);
        assert_eq!(registry.value(HELD), 1);
    }

    #[test]
    fn a_gauge_keeps_the_latest_set_through_compaction() {
        let (registry, metrics) = node();

        let stale = metrics.held.reporter();
        let fresh = metrics.held.reporter();

        stale.set(7);
        fresh.set(4);

        // Compacting the older cell must not let it override the live one.
        drop(stale);
        assert_eq!(registry.value(HELD), 4);

        // And a finished task's last value is what remains once all are gone.
        drop(fresh);
        assert_eq!(registry.value(HELD), 4);
        assert_eq!(registry.live_cells(HELD_IDX), 0);

        // A task that never sets does not reset the gauge.
        let idle = metrics.held.reporter();
        assert_eq!(registry.value(HELD), 4);

        // A later set wins over the compacted value.
        idle.set(9);
        assert_eq!(registry.value(HELD), 9);
    }

    #[test]
    fn max_reports_the_highest_share_not_their_sum() {
        let (registry, metrics) = node();

        let reporters: Vec<_> = (0..3).map(|_| metrics.peak.reporter()).collect();
        for (reporter, peak) in reporters.iter().zip([-5, -9, -7]) {
            reporter.record(peak);
        }

        // Every reading is negative, so a mark seeded at `0` would be wrong.
        assert_eq!(registry.value(PEAK), -5);

        // A lower reading does not pull the mark back down, and a task that
        // never records must not drag it up to `0`.
        reporters[0].record(-11);
        let idle = metrics.peak.reporter();
        assert_eq!(registry.value(PEAK), -5);

        // Nor does compacting the cell that set it, or one recorded after.
        drop(idle);
        drop(reporters);
        assert_eq!(registry.value(PEAK), -5);
        metrics.peak.reporter().record(-8);
        assert_eq!(registry.value(PEAK), -5);
    }

    #[test]
    fn a_max_that_records_its_identity_still_reports_it() {
        let (registry, metrics) = node();

        // Absence is tracked by timestamp, not by the `i64::MIN` seed.
        metrics.peak.reporter().record(i64::MIN);
        assert_eq!(registry.reading(PEAK), Some(i64::MIN));
    }

    #[test]
    fn a_metric_with_no_readings_reports_nothing() {
        let (registry, metrics) = node();

        assert_eq!(registry.reading(ROWS), None);
        assert_eq!(registry.reading(PEAK), None);
        assert_eq!(registry.reading(HELD), None);

        // Taking a reporter is not a reading.
        let _idle = metrics.rows.reporter();
        assert_eq!(registry.reading(ROWS), None);

        // But a write that nets to zero is.
        metrics.rows.reporter().add(0);
        assert_eq!(registry.reading(ROWS), Some(0));
    }

    #[test]
    fn registering_the_same_key_twice_shares_one_counter() {
        let registry = registry();

        register(&registry).rows.reporter().add(4);
        let registered = registry.counters().state.lock().len();

        register(&registry).rows.reporter().add(6);

        // The second registration found every key and added nothing.
        assert_eq!(registry.counters().state.lock().len(), registered);
        assert_eq!(registry.value(ROWS), 10);
    }

    #[test]
    #[should_panic(expected = "already registered differently")]
    fn reregistering_a_metric_with_a_different_aggregation_is_rejected() {
        let registry = registry();

        registry.new_counter(ROWS, MetricUnit::Unit);
        registry.new_max(ROWS, MetricUnit::Unit);
    }

    #[test]
    fn compacting_a_cell_moves_its_value_rather_than_losing_it() {
        let (registry, metrics) = node();

        let live = metrics.rows.reporter();
        let finished = metrics.rows.reporter();
        live.add(7);
        finished.add(11);

        // Counted while both tasks are still running.
        assert_eq!(registry.value(ROWS), 18);
        assert_eq!(registry.live_cells(ROWS_IDX), 2);

        drop(finished);
        assert_eq!(registry.value(ROWS), 18);
        assert_eq!(registry.live_cells(ROWS_IDX), 1);
        assert_eq!(registry.result(ROWS_IDX), 11);

        // A still-running task keeps accruing after its sibling was compacted.
        live.add(1);
        assert_eq!(registry.value(ROWS), 19);

        drop(live);
        assert_eq!(registry.value(ROWS), 19);
        assert_eq!(registry.live_cells(ROWS_IDX), 0);
    }

    #[test]
    fn a_clone_keeps_the_cell_it_shares_alive() {
        let (registry, metrics) = node();

        let reporter = metrics.rows.reporter();
        let clone = reporter.clone();
        reporter.add(3);

        drop(reporter);
        assert_eq!(registry.value(ROWS), 3);
        assert_eq!(registry.live_cells(ROWS_IDX), 1);

        // The clone is still a writer, so the cell must not have been compacted.
        clone.add(4);
        assert_eq!(registry.value(ROWS), 7);

        drop(clone);
        assert_eq!(registry.value(ROWS), 7);
        assert_eq!(registry.live_cells(ROWS_IDX), 0);
    }

    #[test]
    fn a_task_only_takes_reporters_for_the_metrics_it_writes() {
        let (registry, metrics) = node();

        // One serial task counting rows, alongside parallel tasks tracking the
        // high-water mark. Neither allocates for the other's metric.
        let serial = metrics.rows.reporter();
        let parallel: Vec<_> = (0..3).map(|_| metrics.peak.reporter()).collect();

        serial.add(5);
        for (reporter, peak) in parallel.iter().zip([2, 8, 4]) {
            reporter.record(peak);
        }

        assert_eq!(registry.live_cells(ROWS_IDX), 1);
        assert_eq!(registry.live_cells(PEAK_IDX), 3);
        assert_eq!(registry.value(ROWS), 5);
        assert_eq!(registry.value(PEAK), 8);
    }

    #[test]
    fn a_reading_taken_while_tasks_run_and_retire_never_goes_backwards() {
        use std::sync::atomic::{AtomicBool, Ordering};

        const TASKS: i64 = 8;
        const PER_TASK: i64 = 20_000;

        let (registry, metrics) = node();
        let shared = &registry;
        let done = AtomicBool::new(false);

        std::thread::scope(|scope| {
            // Compaction happens under the observer's own call, so a reading
            // that dropped or double-counted a retiring cell would have to
            // move the total the wrong way to do it.
            let observer = scope.spawn(|| {
                let mut previous = None;
                while !done.load(Ordering::Relaxed) {
                    let seen = shared.reading(ROWS);
                    assert!(seen >= previous, "{seen:?} follows {previous:?}");
                    assert!(seen <= Some(TASKS * PER_TASK), "{seen:?} exceeds the total");
                    previous = seen;
                }
            });

            let metrics = &metrics;
            let tasks: Vec<_> = (0..TASKS)
                .map(|task| {
                    scope.spawn(move || {
                        let rows = metrics.rows.reporter();
                        let peak = metrics.peak.reporter();
                        for _ in 0..PER_TASK {
                            rows.add(1);
                            peak.record(task);
                        }
                    })
                })
                .collect();

            for task in tasks {
                task.join().unwrap();
            }
            done.store(true, Ordering::Relaxed);
            observer.join().unwrap();
        });

        assert_eq!(registry.value(ROWS), TASKS * PER_TASK);
        assert_eq!(registry.value(PEAK), TASKS - 1);
        assert_eq!(registry.live_cells(ROWS_IDX), 0);
    }
}
