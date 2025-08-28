use std::sync::Arc;

use polars_core::POOL;
use polars_core::prelude::{IntoColumn, PlHashSet, PlRandomState};
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::groups::Grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::hot_groups::{HotGrouper, new_hash_hot_grouper};
use polars_expr::reduce::GroupedReduction;
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::sparse_init_vec::SparseInitVec;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use crate::async_executor;
use crate::async_primitives::connector::Receiver;
use crate::expression::StreamExpr;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::in_memory_source::InMemorySourceNode;

#[cfg(debug_assertions)]
const DEFAULT_HOT_TABLE_SIZE: usize = 4;
#[cfg(not(debug_assertions))]
const DEFAULT_HOT_TABLE_SIZE: usize = 4096;

struct LocalGroupBySinkState {
    hot_grouper: Box<dyn HotGrouper>,
    hot_grouped_reductions: Vec<Box<dyn GroupedReduction>>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // morsel_idxs_values_per_p[p][start..stop] contains the offsets into cold_morsels[i]
    // for partition p, where start, stop are:
    // let start = morsel_idxs_offsets[i * num_partitions + p];
    // let stop = morsel_idxs_offsets[(i + 1) * num_partitions + p];
    cold_morsels: Vec<(u64, HashKeys, DataFrame)>,
    morsel_idxs_values_per_p: Vec<Vec<IdxSize>>,
    morsel_idxs_offsets_per_p: Vec<usize>,

    // Similar to the above, but for (evicted) pre-aggregates.
    pre_aggs: Vec<(HashKeys, Vec<Box<dyn GroupedReduction>>)>,
    pre_agg_idxs_values_per_p: Vec<Vec<IdxSize>>,
    pre_agg_idxs_offsets_per_p: Vec<usize>,
}

impl LocalGroupBySinkState {
    fn new(
        key_schema: Arc<Schema>,
        reductions: Vec<Box<dyn GroupedReduction>>,
        hot_table_size: usize,
        num_partitions: usize,
    ) -> Self {
        let hot_grouper = new_hash_hot_grouper(key_schema, hot_table_size);
        Self {
            hot_grouper,
            hot_grouped_reductions: reductions,

            sketch_per_p: vec![CardinalitySketch::new(); num_partitions],

            cold_morsels: Vec::new(),
            morsel_idxs_values_per_p: vec![Vec::new(); num_partitions],
            morsel_idxs_offsets_per_p: vec![0; num_partitions],

            pre_aggs: Vec::new(),
            pre_agg_idxs_values_per_p: vec![Vec::new(); num_partitions],
            pre_agg_idxs_offsets_per_p: vec![0; num_partitions],
        }
    }

    fn flush_evictions(&mut self, partitioner: &HashPartitioner) {
        let hash_keys = self.hot_grouper.take_evicted_keys();
        let reductions = self
            .hot_grouped_reductions
            .iter_mut()
            .map(|hgr| hgr.take_evictions())
            .collect_vec();
        self.add_pre_agg(hash_keys, reductions, partitioner);
    }

    fn add_pre_agg(
        &mut self,
        hash_keys: HashKeys,
        reductions: Vec<Box<dyn GroupedReduction>>,
        partitioner: &HashPartitioner,
    ) {
        hash_keys.gen_idxs_per_partition(
            partitioner,
            &mut self.pre_agg_idxs_values_per_p,
            &mut self.sketch_per_p,
            true,
        );
        self.pre_agg_idxs_offsets_per_p
            .extend(self.pre_agg_idxs_values_per_p.iter().map(|vp| vp.len()));
        self.pre_aggs.push((hash_keys, reductions));
    }
}

struct GroupBySinkState {
    key_selectors: Vec<StreamExpr>,
    grouper: Box<dyn Grouper>,
    uniq_grouped_reduction_cols: Vec<PlSmallStr>,
    grouped_reduction_cols: Vec<PlSmallStr>,
    grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    locals: Vec<LocalGroupBySinkState>,
    random_state: PlRandomState,
    partitioner: HashPartitioner,
}

impl GroupBySinkState {
    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        receivers: Vec<Receiver<Morsel>>,
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        for (mut recv, local) in receivers.into_iter().zip(&mut self.locals) {
            let key_selectors = &self.key_selectors;
            let uniq_grouped_reduction_cols = &self.uniq_grouped_reduction_cols;
            let grouped_reduction_cols = &self.grouped_reduction_cols;
            let random_state = &self.random_state;
            let partitioner = self.partitioner.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut hot_idxs = Vec::new();
                let mut hot_group_idxs = Vec::new();
                let mut cold_idxs = Vec::new();
                while let Ok(morsel) = recv.recv().await {
                    // Compute hot group indices from key.
                    let seq = morsel.seq().to_u64();
                    let mut df = morsel.into_df();
                    let mut key_columns = Vec::new();
                    for selector in key_selectors {
                        let s = selector.evaluate(&df, &state.in_memory_exec_state).await?;
                        key_columns.push(s.into_column());
                    }
                    let keys = DataFrame::new_with_broadcast_len(key_columns, df.height())?;
                    let hash_keys = HashKeys::from_df(&keys, *random_state, true, false);

                    hot_idxs.clear();
                    hot_group_idxs.clear();
                    cold_idxs.clear();
                    local.hot_grouper.insert_keys(
                        &hash_keys,
                        &mut hot_idxs,
                        &mut hot_group_idxs,
                        &mut cold_idxs,
                    );

                    // Drop columns not used for reductions (key-only columns).
                    if uniq_grouped_reduction_cols.len() < grouped_reduction_cols.len() {
                        df = df._select_impl(uniq_grouped_reduction_cols).unwrap();
                    }
                    df.rechunk_mut(); // For gathers.

                    // Update hot reductions.
                    for (col, reduction) in grouped_reduction_cols
                        .iter()
                        .zip(&mut local.hot_grouped_reductions)
                    {
                        unsafe {
                            // SAFETY: we resize the reduction to the number of groups beforehand.
                            reduction.resize(local.hot_grouper.num_groups());
                            reduction.update_groups_while_evicting(
                                df.column(col).unwrap(),
                                &hot_idxs,
                                &hot_group_idxs,
                                seq,
                            )?;
                        }
                    }

                    // Store cold keys.
                    // TODO: don't always gather, if majority cold simply store all and remember offsets into it.
                    if !cold_idxs.is_empty() {
                        unsafe {
                            let cold_keys = hash_keys.gather_unchecked(&cold_idxs);
                            let cold_df = df.take_slice_unchecked_impl(&cold_idxs, false);

                            cold_keys.gen_idxs_per_partition(
                                &partitioner,
                                &mut local.morsel_idxs_values_per_p,
                                &mut local.sketch_per_p,
                                true,
                            );
                            local
                                .morsel_idxs_offsets_per_p
                                .extend(local.morsel_idxs_values_per_p.iter().map(|vp| vp.len()));
                            local.cold_morsels.push((seq, cold_keys, cold_df));
                        }
                    }

                    // If we have too many evicted rows, flush them.
                    if local.hot_grouper.num_evictions() >= get_ideal_morsel_size() {
                        local.flush_evictions(&partitioner);
                    }
                }
                Ok(())
            }));
        }
    }

    fn combine_locals(&mut self) -> PolarsResult<Vec<GroupByPartition>> {
        // Finalize pre-aggregations.
        POOL.install(|| {
            self.locals
                .as_mut_slice()
                .into_par_iter()
                .with_max_len(1)
                .for_each(|l| {
                    if l.hot_grouper.num_evictions() > 0 {
                        l.flush_evictions(&self.partitioner);
                    }
                    let hot_keys = l.hot_grouper.keys();
                    let hot_reductions = core::mem::take(&mut l.hot_grouped_reductions);
                    l.add_pre_agg(hot_keys, hot_reductions, &self.partitioner);
                });
        });

        // To reduce maximum memory usage we want to drop the morsels
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let morsels_per_local = self
            .locals
            .iter_mut()
            .map(|l| Arc::new(core::mem::take(&mut l.cold_morsels)))
            .collect_vec();
        let pre_aggs_per_local = self
            .locals
            .iter_mut()
            .map(|l| Arc::new(core::mem::take(&mut l.pre_aggs)))
            .collect_vec();
        enum ToDrop<A, B> {
            A(A),
            B(B),
        }
        let (drop_q_send, drop_q_recv) = async_channel::bounded(self.locals.len());
        let num_partitions = self.locals[0].sketch_per_p.len();
        let output_per_partition: SparseInitVec<GroupByPartition> =
            SparseInitVec::with_capacity(num_partitions);
        let locals = &self.locals;
        let grouper_template = &self.grouper;
        let grouped_reductions_template = &self.grouped_reductions;
        let grouped_reduction_cols = &self.grouped_reduction_cols;

        async_executor::task_scope(|s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_morsels_per_local = Arc::new(morsels_per_local);
            let arc_pre_aggs_per_local = Arc::new(pre_aggs_per_local);
            let mut join_handles = Vec::new();
            for p in 0..num_partitions {
                let arc_morsels_per_local = Arc::clone(&arc_morsels_per_local);
                let arc_pre_aggs_per_local = Arc::clone(&arc_pre_aggs_per_local);
                let drop_q_send = drop_q_send.clone();
                let drop_q_recv = drop_q_recv.clone();
                let output_per_partition = &output_per_partition;
                join_handles.push(s.spawn_task(TaskPriority::High, async move {
                    // Extract from outer arc and drop outer arc.
                    let morsels_per_local = Arc::unwrap_or_clone(arc_morsels_per_local);
                    let pre_aggs_per_local = Arc::unwrap_or_clone(arc_pre_aggs_per_local);

                    // Compute cardinality estimate and total amount of
                    // payload for this partition.
                    let mut sketch = CardinalitySketch::new();
                    for l in locals {
                        sketch.combine(&l.sketch_per_p[p]);
                    }

                    // Allocate grouper and reductions.
                    let est_num_groups = sketch.estimate() * 5 / 4;
                    let mut p_grouper = grouper_template.new_empty();
                    let mut p_reductions = grouped_reductions_template
                        .iter()
                        .map(|gr| gr.new_empty())
                        .collect_vec();
                    p_grouper.reserve(est_num_groups);
                    for r in &mut p_reductions {
                        r.reserve(est_num_groups);
                    }

                    // Insert morsels.
                    let mut skip_drop_attempt = false;
                    let mut group_idxs = Vec::new();
                    for (l, l_morsels) in locals.iter().zip(morsels_per_local) {
                        // Try to help with dropping.
                        if !skip_drop_attempt {
                            drop(drop_q_recv.try_recv());
                        }

                        for (i, morsel) in l_morsels.iter().enumerate() {
                            let (seq_id, keys, cols) = morsel;
                            unsafe {
                                let p_morsel_idxs_start =
                                    l.morsel_idxs_offsets_per_p[i * num_partitions + p];
                                let p_morsel_idxs_stop =
                                    l.morsel_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                    [p_morsel_idxs_start..p_morsel_idxs_stop];

                                group_idxs.clear();
                                p_grouper.insert_keys_subset(
                                    keys,
                                    p_morsel_idxs,
                                    Some(&mut group_idxs),
                                );
                                for (c, r) in grouped_reduction_cols.iter().zip(&mut p_reductions) {
                                    let values = cols.column(c.as_str()).unwrap();
                                    r.resize(p_grouper.num_groups());
                                    r.update_groups_subset(
                                        values,
                                        p_morsel_idxs,
                                        &group_idxs,
                                        *seq_id,
                                    )?;
                                }
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_morsels) {
                            // If we're the last thread to process this set of morsels we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(drop_q_send.try_send(ToDrop::A(l)));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // Insert pre-aggregates.
                    for (l, l_pre_aggs) in locals.iter().zip(pre_aggs_per_local) {
                        // Try to help with dropping.
                        if !skip_drop_attempt {
                            drop(drop_q_recv.try_recv());
                        }

                        for (i, key_pre_aggs) in l_pre_aggs.iter().enumerate() {
                            let (keys, pre_aggs) = key_pre_aggs;
                            unsafe {
                                let p_pre_agg_idxs_start =
                                    l.pre_agg_idxs_offsets_per_p[i * num_partitions + p];
                                let p_pre_agg_idxs_stop =
                                    l.pre_agg_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_pre_agg_idxs = &l.pre_agg_idxs_values_per_p[p]
                                    [p_pre_agg_idxs_start..p_pre_agg_idxs_stop];

                                group_idxs.clear();
                                p_grouper.insert_keys_subset(
                                    keys,
                                    p_pre_agg_idxs,
                                    Some(&mut group_idxs),
                                );
                                for (pre_agg, r) in pre_aggs.iter().zip(&mut p_reductions) {
                                    r.resize(p_grouper.num_groups());
                                    r.combine_subset(&**pre_agg, p_pre_agg_idxs, &group_idxs)?;
                                }
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_pre_aggs) {
                            // If we're the last thread to process this set of morsels we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(drop_q_send.try_send(ToDrop::B(l)));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // We're done, help others out by doing drops.
                    drop(drop_q_send); // So we don't deadlock trying to receive from ourselves.
                    while let Ok(to_drop) = drop_q_recv.recv().await {
                        drop(to_drop);
                    }

                    output_per_partition
                        .try_set(
                            p,
                            GroupByPartition {
                                grouper: p_grouper,
                                grouped_reductions: p_reductions,
                            },
                        )
                        .ok()
                        .unwrap();

                    PolarsResult::Ok(())
                }));
            }

            // Drop outer arc after spawning each thread so the inner arcs
            // can get dropped as soon as they're processed. We also have to
            // drop the drop queue sender so we don't deadlock waiting for it
            // to end.
            drop(arc_morsels_per_local);
            drop(arc_pre_aggs_per_local);
            drop(drop_q_send);

            polars_io::pl_async::get_runtime().block_on(async move {
                for handle in join_handles {
                    handle.await?;
                }
                PolarsResult::Ok(())
            })?;
            PolarsResult::Ok(())
        })?;

        // Drop remaining local state in parallel.
        POOL.install(|| {
            core::mem::take(&mut self.locals)
                .into_par_iter()
                .with_max_len(1)
                .for_each(drop);
        });

        Ok(output_per_partition.try_assume_init().ok().unwrap())
    }
}

struct GroupByPartition {
    grouper: Box<dyn Grouper>,
    grouped_reductions: Vec<Box<dyn GroupedReduction>>,
}

impl GroupByPartition {
    fn into_df(self, key_schema: &Schema, output_schema: &Schema) -> PolarsResult<DataFrame> {
        let mut out = self.grouper.get_keys_in_group_order(key_schema);
        let out_names = output_schema.iter_names().skip(out.width());
        for (mut r, name) in self.grouped_reductions.into_iter().zip(out_names) {
            unsafe {
                out.with_column_unchecked(r.finalize()?.with_name(name.clone()).into_column());
            }
        }
        Ok(out)
    }
}

enum GroupByState {
    Sink(GroupBySinkState),
    Source(InMemorySourceNode),
    Done,
}

pub struct GroupByNode {
    state: GroupByState,
    key_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
}

impl GroupByNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_schema: Arc<Schema>,
        key_selectors: Vec<StreamExpr>,
        grouper: Box<dyn Grouper>,
        grouped_reduction_cols: Vec<PlSmallStr>,
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
        output_schema: Arc<Schema>,
        random_state: PlRandomState,
        num_pipelines: usize,
    ) -> Self {
        let hot_table_size = std::env::var("POLARS_HOT_TABLE_SIZE")
            .map(|sz| sz.parse::<usize>().unwrap())
            .unwrap_or(DEFAULT_HOT_TABLE_SIZE);
        let num_partitions = num_pipelines;
        let uniq_grouped_reduction_cols = grouped_reduction_cols
            .iter()
            .cloned()
            .collect::<PlHashSet<_>>()
            .into_iter()
            .collect_vec();
        let locals = (0..num_pipelines)
            .map(|_| {
                let reductions = grouped_reductions.iter().map(|gr| gr.new_empty()).collect();
                LocalGroupBySinkState::new(
                    key_schema.clone(),
                    reductions,
                    hot_table_size,
                    num_partitions,
                )
            })
            .collect();
        let partitioner = HashPartitioner::new(num_partitions, 0);
        Self {
            state: GroupByState::Sink(GroupBySinkState {
                key_selectors,
                grouped_reductions,
                grouper,
                random_state,
                uniq_grouped_reduction_cols,
                grouped_reduction_cols,
                locals,
                partitioner,
            }),
            key_schema,
            output_schema,
        }
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "group-by"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = GroupByState::Done;
            },
            // Input is done, transition to being a source.
            GroupByState::Sink(_) if matches!(recv[0], PortState::Done) => {
                let GroupByState::Sink(mut sink) =
                    core::mem::replace(&mut self.state, GroupByState::Done)
                else {
                    unreachable!()
                };
                let partitions = sink.combine_locals()?;
                let dfs = POOL.install(|| {
                    partitions
                        .into_par_iter()
                        .map(|p| p.into_df(&self.key_schema, &self.output_schema))
                        .collect::<Result<Vec<_>, _>>()
                })?;

                let df = accumulate_dataframes_vertical_unchecked(dfs);
                let source = InMemorySourceNode::new(Arc::new(df), MorselSeq::new(0));
                self.state = GroupByState::Source(source);
            },
            // Defer to source node implementation.
            GroupByState::Source(src) => {
                src.update_state(&mut [], send, state)?;
                if send[0] == PortState::Done {
                    self.state = GroupByState::Done;
                }
            },
            // Nothing to change.
            GroupByState::Done | GroupByState::Sink(_) => {},
        }

        // Communicate our state.
        match &self.state {
            GroupByState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            GroupByState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            GroupByState::Done => {
                recv[0] = PortState::Done;
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
        assert!(send_ports.len() == 1 && recv_ports.len() == 1);
        match &mut self.state {
            GroupByState::Sink(sink) => {
                assert!(send_ports[0].is_none());
                sink.spawn(
                    scope,
                    recv_ports[0].take().unwrap().parallel(),
                    state,
                    join_handles,
                )
            },
            GroupByState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            GroupByState::Done => unreachable!(),
        }
    }
}
