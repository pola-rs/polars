use std::time::Instant;

use polars_buffer::Buffer;
use polars_mem_engine::column_to_mask;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::metrics::{Metric, MetricReporter, MetricUnit, NodeMetricsRegistry, kind};

#[derive(Default)]
struct FilterMetrics {
    rows_dropped: Metric<kind::UpDownCounter>,
    morsels_received: Metric<kind::UpDownCounter>,
    largest_morsel_received: Metric<kind::Max>,
    eval_wall_ns: Metric<kind::UpDownCounter>,
}

struct FilterReporter {
    rows_dropped: MetricReporter<kind::UpDownCounter>,
    morsels_received: MetricReporter<kind::UpDownCounter>,
    largest_morsel_received: MetricReporter<kind::Max>,
    eval_wall_ns: MetricReporter<kind::UpDownCounter>,
}

impl FilterMetrics {
    fn register(registry: &NodeMetricsRegistry) -> Self {
        Self {
            rows_dropped: registry.new_counter("filter.rows_dropped", MetricUnit::Unit),
            morsels_received: registry.new_counter("filter.morsels_received", MetricUnit::Unit),
            largest_morsel_received: registry
                .new_max("filter.largest_morsel_received", MetricUnit::Unit),
            eval_wall_ns: registry.new_counter("filter.eval_wall_ns", MetricUnit::DurationNs),
        }
    }

    fn reporters(&self) -> FilterReporter {
        FilterReporter {
            rows_dropped: self.rows_dropped.reporter(),
            morsels_received: self.morsels_received.reporter(),
            largest_morsel_received: self.largest_morsel_received.reporter(),
            eval_wall_ns: self.eval_wall_ns.reporter(),
        }
    }
}

pub struct FilterNode {
    predicate: StreamExpr,
    projection: Option<Buffer<usize>>,
    metrics: FilterMetrics,
}

impl FilterNode {
    pub fn new(
        predicate: StreamExpr,
        projection: Option<Buffer<usize>>,
        metrics_registry: NodeMetricsRegistry,
    ) -> Self {
        Self {
            predicate,
            projection,
            metrics: FilterMetrics::register(&metrics_registry),
        }
    }
}

impl ComputeNode for FilterNode {
    fn name(&self) -> &str {
        "filter"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let receivers = recv_ports[0].take().unwrap().parallel();
        let senders = send_ports[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;

            let metrics = slf.metrics.reporters();

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let height_in = morsel.height() as i64;
                    let started = Instant::now();

                    let morsel = morsel
                        .async_try_map(|mut df| async move {
                            let mask = slf
                                .predicate
                                .evaluate(&df, &state.in_memory_exec_state)
                                .await?;
                            let mask = column_to_mask(&mask, df.height())?;

                            if let Some(projection) = slf.projection.as_deref() {
                                df = unsafe {
                                    DataFrame::new_unchecked(
                                        df.height(),
                                        projection
                                            .iter()
                                            .map(|&i| df.columns()[i].clone())
                                            .collect(),
                                    )
                                }
                            }

                            // We already parallelize, call the sequential filter.
                            df.filter_seq(mask.as_ref())
                        })
                        .await?;

                    metrics.rows_dropped.add(height_in - morsel.height() as i64);
                    metrics.morsels_received.add(1);
                    metrics.largest_morsel_received.record(height_in);
                    metrics
                        .eval_wall_ns
                        .add(started.elapsed().as_nanos() as i64);

                    if morsel.height() == 0 {
                        continue;
                    }

                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
