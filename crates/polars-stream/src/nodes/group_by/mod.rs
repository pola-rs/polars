//! The streaming group-by.
//!
//! While receiving input, each pipeline aggregates rows with frequent keys in small hot tables
//! and stores the other (cold) rows with their hashed keys in spillable frames. Groups evicted
//! from a hot table, and the hot tables at the end of the input, become pre-aggregates. Once all
//! input is received, the cold rows and pre-aggregates are split into hash partitions, which are
//! aggregated independently. With at most one partition per pipeline, all partitions are built
//! before sending. Otherwise the cold rows are first stored per partition, and the partitions are
//! built a few at a time while the results are sent.
mod flush;
mod pre_agg;
mod sink;
mod table;
mod tuning;

use std::sync::Arc;

use futures::FutureExt;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::parallel::for_each_parallel;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, PlRandomState};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::groups::Grouper;
use polars_expr::reduce::GroupedReduction;
use polars_expr::state::ExecutionState;
use polars_ooc::{MostRecentSpillContext, memory_manager};
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use tokio::sync::mpsc::channel;

use self::flush::Flush;
use self::pre_agg::SplitPreAggs;
use self::sink::GroupBySink;
use self::table::{GroupByParams, PartitionTable};
use self::tuning::GroupByTuning;
use super::{ComputeNode, NodeMemoryUsage};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::hash_shuffle::batches::{LoadedFrame, PrepareFrame, build_from_batches, make_batches};
use crate::hash_shuffle::frame_with_keys::SpillFrameWithKeys;
use crate::hash_shuffle::store::StoredPartition;
use crate::metrics::{Metric, MetricUnit, NodeMetricsRegistry, kind};
use crate::pipe::{RecvPort, SendPort};

/// Which columns of an input's morsels are kept (and spilled), and how the reductions of
/// that input read them.
pub struct InputPayload {
    /// The input columns which must be kept.
    pub stored_cols: Vec<PlSmallStr>,
    /// The subset of `stored_cols` needed to evaluate `fused_selectors` and the
    /// `fused_reductions`.
    pub gather_cols: Vec<PlSmallStr>,
    /// Elementwise expressions evaluated inside the node, so that their output is never stored
    /// with the cold rows.
    pub fused_selectors: Vec<StreamExpr>,

    /// Reductions whose input columns are all in `stored_cols`.
    pub direct_reductions: Vec<usize>,
    /// Reductions with at least one input column produced by `fused_selectors`. A single
    /// subset is shared by all input columns of one update call, so such a reduction reads
    /// all of its inputs from the materialized frame.
    pub fused_reductions: Vec<usize>,
}

impl InputPayload {
    /// Appends the fused columns, evaluated on all rows of `df`.
    async fn push_fused(
        &self,
        df: &mut DataFrame,
        exec_state: &ExecutionState,
    ) -> PolarsResult<()> {
        for selector in &self.fused_selectors {
            let c = selector
                .evaluate_preserve_len_broadcast(df, exec_state)
                .await?;
            unsafe { df.push_column_unchecked(c.rechunk()) };
        }
        Ok(())
    }

    /// Materializes the fused columns for rows `idxs` of `df`.
    async fn materialize_fused<'a>(
        &self,
        df: &DataFrame,
        idxs: &'a [IdxSize],
        identity_idxs: &'a mut Vec<IdxSize>,
        exec_state: &ExecutionState,
    ) -> PolarsResult<Option<(DataFrame, &'a [IdxSize])>> {
        if self.fused_reductions.is_empty() || idxs.is_empty() {
            return Ok(None);
        }

        let mut eval_df = unsafe { df.select_unchecked(&self.gather_cols) }?;
        // 75% or more of the rows, don't gather.
        let subset = if idxs.len() as u64 >= df.height() as u64 * 3 / 4 {
            idxs
        } else {
            eval_df = unsafe { eval_df.take_slice_unchecked_impl(idxs, false) };
            identity_idxs.extend(identity_idxs.len() as IdxSize..idxs.len() as IdxSize);
            &identity_idxs[..idxs.len()]
        };

        self.push_fused(&mut eval_df, exec_state).await?;
        Ok(Some((eval_df, subset)))
    }

    /// Feeds rows `idxs` of `df` to every reduction of this input. The direct reductions
    /// read `df` itself, the fused ones a frame materialized for those rows only.
    #[allow(clippy::too_many_arguments)]
    async fn update_reductions<'a>(
        &self,
        df: &DataFrame,
        idxs: &'a [IdxSize],
        identity_idxs: &'a mut Vec<IdxSize>,
        grouped_reduction_cols: &[Vec<PlSmallStr>],
        reductions: &mut [Box<dyn GroupedReduction>],
        exec_state: &ExecutionState,
        mut update: impl FnMut(&mut dyn GroupedReduction, &[&Column], &[IdxSize]) -> PolarsResult<()>,
    ) -> PolarsResult<()> {
        let fused_frame = self
            .materialize_fused(df, idxs, identity_idxs, exec_state)
            .await?;
        let direct = (&self.direct_reductions, df, idxs);
        let fused = fused_frame
            .as_ref()
            .map(|(fused_df, subset)| (&self.fused_reductions, fused_df, *subset));

        for (red_idxs, src_df, subset) in std::iter::once(direct).chain(fused) {
            let mut in_cols = Vec::new();
            for red_idx in red_idxs {
                in_cols.clear();
                in_cols.extend(
                    grouped_reduction_cols[*red_idx]
                        .iter()
                        .map(|col| src_df.column(col).unwrap()),
                );
                update(&mut *reductions[*red_idx], &in_cols, subset)?;
            }
        }
        Ok(())
    }
}

enum GroupByState {
    Sink(GroupBySink),
    Flush(Box<Flush>),
    Done,
}

pub struct GroupByNode {
    state: GroupByState,
    params: Arc<GroupByParams>,
    num_inputs: usize,
    num_pipelines: usize,
    tuning: GroupByTuning,
    spill_ctx: MostRecentSpillContext,
    estimated_groups: Metric<kind::Sum>,
    actual_groups: Metric<kind::Sum>,
}

impl GroupByNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_schema: Arc<Schema>,
        // Input stream i selects keys with key_selectors_per_input[i].
        key_selectors_per_input: Vec<Vec<StreamExpr>>,
        // Input stream i feeds grouped_reductions[k] for each k in reductions_per_input[i].
        reductions_per_input: Vec<Vec<usize>>,
        grouper: Box<dyn Grouper>,
        // grouped_reductions[k] is passed input cols grouped_reduction_cols[k].
        grouped_reduction_cols: Vec<Vec<PlSmallStr>>,
        payload_per_input: Vec<InputPayload>,
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
        output_schema: Arc<Schema>,
        random_state: PlRandomState,
        num_pipelines: usize,
        has_order_sensitive_agg: bool,
        metrics_registry: NodeMetricsRegistry,
    ) -> Self {
        let config = polars_config::config();
        let hot_table_size = (config.hot_table_size() as usize)
            .next_power_of_two()
            .max(2);
        let max_hot_table_size = (config.max_hot_table_size() as usize)
            .next_power_of_two()
            .max(hot_table_size);
        let num_inputs = key_selectors_per_input.len();
        let sink = GroupBySink::new(
            key_selectors_per_input,
            random_state,
            hot_table_size,
            max_hot_table_size,
            num_pipelines,
        );
        let params = GroupByParams {
            key_schema,
            output_schema,
            reductions_per_input,
            grouped_reduction_cols,
            payload_per_input,
            grouper,
            grouped_reductions,
            has_order_sensitive_agg,
        };
        Self {
            state: GroupByState::Sink(sink),
            params: Arc::new(params),
            num_inputs,
            num_pipelines,
            tuning: GroupByTuning::from_env(),
            spill_ctx: MostRecentSpillContext::new(
                "group-by".into(),
                metrics_registry.task_metrics(),
            ),
            estimated_groups: metrics_registry
                .new_counter("group_by.estimated_groups", MetricUnit::Unit),
            actual_groups: metrics_registry.new_counter("group_by.actual_groups", MetricUnit::Unit),
        }
    }

    /// Builds the results once all input is received. With at most one partition per pipeline
    /// all partitions are built here; otherwise the cold rows are stored per partition, and the
    /// partitions are built while the results are sent.
    fn start_flush(
        &self,
        sink: GroupBySink,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Flush> {
        let (stats, frames, pre_aggs) = sink.finish_input(&self.params, &self.spill_ctx, state)?;
        let num_partitions = self.tuning.num_partitions(&stats, self.num_pipelines);
        let partitioner = HashPartitioner::new(num_partitions, 0);
        let pre_aggs = SplitPreAggs::new(pre_aggs, &partitioner, state)?;
        let num_frames = frames.len();
        let batches = make_batches(frames, self.tuning.batch_bytes());
        let reserve_groups = (stats.estimated_groups * 5 / 4 / num_partitions as u64) as usize;
        let build_at_once = num_partitions <= self.num_pipelines;
        if polars_config::config().verbose() {
            let how = if build_at_once {
                "built at once"
            } else {
                "stored"
            };
            eprintln!(
                "[group-by]: {num_partitions} partitions {how}, {num_frames} frames in {} batches, {} pre-aggregates, {} estimated groups",
                batches.len(),
                pre_aggs.len(),
                stats.estimated_groups
            );
        }
        self.estimated_groups
            .reporter()
            .add(stats.estimated_groups as i64);
        let actual_groups = self.actual_groups.reporter();

        let (outputs, stored, pre_aggs) = if build_at_once {
            let outputs =
                self.build_partitions(pre_aggs, batches, &partitioner, reserve_groups, state)?;
            actual_groups.add(outputs.iter().map(|out| out.height() as i64).sum());
            (outputs, Vec::new(), SplitPreAggs::default())
        } else {
            let stored = self.store_partitions(batches, &partitioner, state)?;
            (Vec::new(), stored, pre_aggs)
        };
        Ok(Flush::new(
            outputs,
            stored,
            pre_aggs,
            self.num_pipelines,
            self.params.clone(),
            reserve_groups,
            actual_groups,
        ))
    }

    /// Builds every partition: first from the pre-aggregates, then from the cold frames one batch
    /// at a time. Returns the non-empty results, or one empty result if all are empty.
    fn build_partitions(
        &self,
        pre_aggs: SplitPreAggs,
        batches: Vec<Vec<(usize, SpillFrameWithKeys)>>,
        partitioner: &HashPartitioner,
        reserve_groups: usize,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Vec<DataFrame>> {
        // The partition index, its table, and the result once finished.
        type Partition = (usize, PartitionTable, DataFrame);

        let num_partitions = partitioner.num_partitions();
        let mut partitions: Vec<Partition> = (0..num_partitions)
            .map(|p| {
                let table = PartitionTable::new(self.params.clone(), reserve_groups);
                (p, table, DataFrame::empty())
            })
            .collect();
        {
            let pre_aggs = &pre_aggs;
            for_each_parallel(
                partitions.iter_mut(),
                state.task_metrics(),
                &|(p, table, _)| async move {
                    for (pre_agg, groups) in pre_aggs.in_partition(*p) {
                        table.add_pre_agg(pre_agg, groups)?;
                    }
                    memory_manager().spill().await;
                    Ok(())
                },
            )?;
        }
        pre_aggs.free(state)?;

        let params = &*self.params;
        let prepare: &PrepareFrame<'_> =
            &move |input, frame| params.prepare_frame(input, frame, state).boxed();
        let add_batch =
            |(p, table, _): &mut Partition, batch: &[LoadedFrame]| -> PolarsResult<()> {
                // Partitions start at different frames, so that tasks don't read the same frame at
                // the same time.
                let start = *p * batch.len() / num_partitions;
                for i in 0..batch.len() {
                    let frame = &batch[(start + i) % batch.len()];
                    let rows = frame.partition_rows(*p);
                    if rows.is_some_and(|rows| rows.is_empty()) {
                        continue;
                    }
                    table.add_rows(frame.input, &frame.frame, rows)?;
                }
                Ok(())
            };
        build_from_batches(
            batches,
            partitioner,
            &mut partitions,
            Some(prepare),
            &add_batch,
            state,
        )?;

        for_each_parallel(partitions.iter_mut(), state.task_metrics(), &|(
            _,
            table,
            out,
        )| async move {
            *out = table.finish()?;
            memory_manager().spill().await;
            Ok(())
        })?;
        let mut outputs: Vec<DataFrame> = partitions.into_iter().map(|(_, _, out)| out).collect();
        // An empty result is still sent, as one empty morsel.
        if outputs.iter().any(|out| out.height() > 0) {
            outputs.retain(|out| out.height() > 0);
        } else {
            outputs.truncate(1);
        }
        Ok(outputs)
    }

    /// Stores the cold rows per partition, as one frame per batch and input, so that the
    /// partitions can be built a few at a time while the results are sent.
    fn store_partitions(
        &self,
        batches: Vec<Vec<(usize, SpillFrameWithKeys)>>,
        partitioner: &HashPartitioner,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Vec<StoredPartition>> {
        let spill_ctx =
            MostRecentSpillContext::new("group-by-partition".into(), state.task_metrics.clone());
        let mut stored: Vec<StoredPartition> = (0..partitioner.num_partitions())
            .map(|p| StoredPartition::new(p, self.num_inputs, spill_ctx.clone()))
            .collect();
        build_from_batches(
            batches,
            partitioner,
            &mut stored,
            None,
            &|stored: &mut StoredPartition, batch: &[LoadedFrame]| {
                stored.write_batch(batch);
                Ok(())
            },
            state,
        )?;
        Ok(stored)
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "group-by"
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            GroupByState::Sink(_) => NodeMemoryUsage::Accumulating,
            GroupByState::Flush(_) => NodeMemoryUsage::Draining,
            GroupByState::Done => NodeMemoryUsage::Bounded,
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == self.num_inputs && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = GroupByState::Done;
            },
            // All inputs are done, build the results and transition to sending them.
            GroupByState::Sink(_) if recv.iter().all(|r| matches!(r, PortState::Done)) => {
                let GroupByState::Sink(sink) =
                    core::mem::replace(&mut self.state, GroupByState::Done)
                else {
                    unreachable!()
                };
                self.state = GroupByState::Flush(Box::new(self.start_flush(sink, state)?));
            },
            GroupByState::Flush(flush) if flush.is_finished() => {
                self.state = GroupByState::Done;
            },
            // Nothing to change.
            GroupByState::Done | GroupByState::Sink(_) | GroupByState::Flush(_) => {},
        }

        // Communicate our state.
        match &self.state {
            GroupByState::Sink(_) => {
                recv.fill(PortState::Ready);
                send[0] = PortState::Blocked;
            },
            GroupByState::Flush(_) => {
                recv.fill(PortState::Done);
                send[0] = PortState::Ready;
            },
            GroupByState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
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
        assert!(send_ports.len() == 1 && recv_ports.len() == self.num_inputs);
        match &mut self.state {
            GroupByState::Sink(sink) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports.iter().any(|r| r.is_some()));

                // If we have multiple input streams merge them into one (still identifying which
                // input stream it came from).
                let (senders, receivers): (Vec<_>, Vec<_>) =
                    (0..self.num_pipelines).map(|_| channel(1)).unzip();
                for (i, recv_port) in recv_ports.iter_mut().enumerate() {
                    if let Some(recv_port) = recv_port.take() {
                        for (mut r, s) in recv_port
                            .parallel()
                            .into_iter()
                            .zip(senders.iter().cloned())
                        {
                            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                                while let Ok(morsel) = r.recv().await {
                                    if s.send((i, morsel)).await.is_err() {
                                        break;
                                    }
                                }

                                Ok(())
                            }));
                        }
                    }
                }
                sink.spawn(
                    scope,
                    receivers,
                    state,
                    join_handles,
                    &self.params,
                    &self.spill_ctx,
                )
            },
            GroupByState::Flush(flush) => {
                assert!(recv_ports.iter().all(|r| r.is_none()));
                let senders = send_ports[0].take().unwrap().parallel();
                flush.spawn(scope, senders, state, join_handles);
            },
            GroupByState::Done => unreachable!(),
        }
    }
}
