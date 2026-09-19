use std::sync::Arc;

use polars_buffer::Buffer;
use polars_mem_engine::column_to_mask;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::metrics::{CustomMetric, CustomMetrics, MetricKind, NodeMetricsRegistrator};

pub struct FilterNode {
    predicate: StreamExpr,
    projection: Option<Buffer<usize>>,
    metrics: Option<FilterMetrics>,
}

struct FilterMetrics {
    handle: Arc<CustomMetrics>,
    rows_filtered: Arc<CustomMetric>,
}

impl FilterNode {
    pub fn new(predicate: StreamExpr, projection: Option<Buffer<usize>>) -> Self {
        Self {
            predicate,
            projection,
            metrics: None,
        }
    }
}

impl ComputeNode for FilterNode {
    fn name(&self) -> &str {
        "filter"
    }

    fn set_phase_metrics_registrator(&mut self, metrics_registrator: NodeMetricsRegistrator) {
        let metrics = self.metrics.get_or_insert_with(|| {
            let handle = Arc::<CustomMetrics>::default();
            let rows_filtered = handle.slot(
                PlSmallStr::from_static("filter.rows_filtered"),
                MetricKind::Literal,
            );

            FilterMetrics {
                handle,
                rows_filtered,
            }
        });

        metrics_registrator.register_custom_metrics(metrics.handle.clone());
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
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let height_in = morsel.height() as u64;
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

                    if let Some(metrics) = &slf.metrics {
                        metrics
                            .rows_filtered
                            .add(height_in - morsel.height() as u64);
                    }

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
