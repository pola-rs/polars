use std::sync::Arc;

use polars_async::executor;
use polars_core::prelude::{Column, IntoColumn, PlRandomState};
use polars_core::runtime::{ASYNC, RAYON};
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::EvictIdx;
use polars_expr::groups::Grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::hot_groups::{HotGrouper, new_hash_hot_grouper};
use polars_expr::reduce::GroupedReduction;
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::f2_sketch::F2Sketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::sparse_init_vec::SparseInitVec;
use polars_utils::{IdxSize, UnitVec};
use rayon::prelude::*;
use tokio::sync::mpsc::{Receiver, channel};

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::metrics::{Metric, MetricUnit, NodeMetricsRegistry, kind};
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::in_memory_source::InMemorySourceNode;

#[cfg(debug_assertions)]
const KEY_SLICE_SIZE: usize = 64;
#[cfg(not(debug_assertions))]
const KEY_SLICE_SIZE: usize = 4096;

/// The number of hot groups up to which the reductions of a sliced morsel are updated
/// per slice.
const MAX_SLICE_REDUCTION_GROUPS: usize = 64;

/// The hot tables only grow if the frequently missed keys were missed at least this
/// many times each.
const HOT_TABLE_GROW_MIN_REPEAT: f64 = 8.0;

/// The hot tables only grow if the frequently missed keys make up at least this
/// fraction of the missed rows.
const HOT_TABLE_GROW_MIN_HEAVY_SHARE: f64 = 0.5;

/// The hot tables grow to hold this many slots per key they should hold, if the
/// maximum size allows.
const HOT_TABLE_GROW_SLOTS_PER_KEY: f64 = 1.5;

/// The hot tables only grow if the keys they should hold fill at most this fraction
/// of the slots after growing.
const HOT_TABLE_GROW_MAX_LOAD: f64 = 0.75;

struct PreAgg {
    keys: HashKeys,
    reduction_idxs: UnitVec<usize>,
    reductions: Vec<Box<dyn GroupedReduction>>,
}

struct LocalGroupBySinkState {
    hot_grouper_per_input: Vec<Box<dyn HotGrouper>>,
    hot_grouped_reductions: Vec<Box<dyn GroupedReduction>>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // The number of slots of each hot grouper, and the rows missed by them.
    hot_table_size: usize,
    miss_f2: F2Sketch,

    // morsel_idxs_values_per_p[p][start..stop] contains the offsets into cold_morsels[i]
    // for partition p, where start, stop are:
    // let start = morsel_idxs_offsets[i * num_partitions + p];
    // let stop = morsel_idxs_offsets[(i + 1) * num_partitions + p];
    cold_morsels: Vec<(usize, u64, HashKeys, SpillFrame)>,
    morsel_idxs_values_per_p: Vec<Vec<IdxSize>>,
    morsel_idxs_offsets_per_p: Vec<usize>,

    // Similar to the above, but for (evicted) pre-aggregates.
    // The UnitVec contains the indices of the grouped reductions.
    pre_aggs: Vec<PreAgg>,
    pre_agg_idxs_values_per_p: Vec<Vec<IdxSize>>,
    pre_agg_idxs_offsets_per_p: Vec<usize>,
}

impl LocalGroupBySinkState {
    fn new(
        key_schema: Arc<Schema>,
        reductions: Vec<Box<dyn GroupedReduction>>,
        hot_table_size: usize,
        num_partitions: usize,
        num_inputs: usize,
    ) -> Self {
        let hot_grouper_per_input = (0..num_inputs)
            .map(|_| new_hash_hot_grouper(key_schema.clone(), hot_table_size))
            .collect();
        Self {
            hot_grouper_per_input,
            hot_grouped_reductions: reductions,

            sketch_per_p: vec![CardinalitySketch::new(); num_partitions],

            hot_table_size,
            miss_f2: F2Sketch::new(),

            cold_morsels: Vec::new(),
            morsel_idxs_values_per_p: vec![Vec::new(); num_partitions],
            morsel_idxs_offsets_per_p: vec![0; num_partitions],

            pre_aggs: Vec::new(),
            pre_agg_idxs_values_per_p: vec![Vec::new(); num_partitions],
            pre_agg_idxs_offsets_per_p: vec![0; num_partitions],
        }
    }

    fn flush_evictions(
        &mut self,
        input_idx: usize,
        reduction_idxs: &[usize],
        partitioner: &HashPartitioner,
    ) {
        let hash_keys = self.hot_grouper_per_input[input_idx].take_evicted_keys();
        let reductions = reduction_idxs
            .iter()
            .map(|r| self.hot_grouped_reductions[*r].take_evictions())
            .collect_vec();
        self.add_pre_agg(hash_keys, reduction_idxs, reductions, partitioner, true);
    }

    /// Adds a pre-aggregate over `hash_keys`, which count as missed by the hot table
    /// if `is_miss`.
    fn add_pre_agg(
        &mut self,
        hash_keys: HashKeys,
        reduction_idxs: &[usize],
        reductions: Vec<Box<dyn GroupedReduction>>,
        partitioner: &HashPartitioner,
        is_miss: bool,
    ) {
        hash_keys.gen_idxs_per_partition(
            partitioner,
            &mut self.pre_agg_idxs_values_per_p,
            &mut self.sketch_per_p,
            is_miss.then_some(&mut self.miss_f2),
            true,
        );
        self.pre_agg_idxs_offsets_per_p
            .extend(self.pre_agg_idxs_values_per_p.iter().map(|vp| vp.len()));
        let pre_agg = PreAgg {
            keys: hash_keys,
            reduction_idxs: UnitVec::from_slice(reduction_idxs),
            reductions,
        };
        self.pre_aggs.push(pre_agg);
    }

    /// Grows the hot tables at once to a size that holds the frequently missed keys,
    /// if there are few enough of them and they make up enough of the missed rows.
    fn maybe_grow_hot_tables(&mut self, max_size: usize) {
        let size = self.hot_table_size;
        if size >= max_size {
            return;
        }

        // Cheap early exit before estimating the number of distinct missed keys.
        let misses = self.miss_f2.num_inserts() as f64;
        if misses < HOT_TABLE_GROW_MIN_REPEAT * size as f64 {
            return;
        }

        // Model the missed rows as `heavy` keys missed `repeat` times each plus keys
        // missed once. The row count, the distinct key count and F2 then determine
        // both.
        let f2 = self.miss_f2.estimate();
        let distinct: f64 = self.sketch_per_p.iter().map(|s| s.estimate() as f64).sum();
        let distinct = distinct.min(misses);
        let excess = misses - distinct;
        let denom = f2 - 2.0 * misses + distinct;
        if excess <= 0.0 || denom <= 0.0 {
            return;
        }
        let repeat = (f2 - misses) / excess;
        let heavy = excess * excess / denom;
        let heavy_share = heavy * repeat / misses;
        if repeat < HOT_TABLE_GROW_MIN_REPEAT || heavy_share < HOT_TABLE_GROW_MIN_HEAVY_SHARE {
            return;
        }

        let num_hot_keys = self
            .hot_grouper_per_input
            .iter()
            .map(|g| g.num_groups() as usize)
            .max()
            .unwrap_or(0);
        let want = num_hot_keys as f64 + heavy;
        let new_size = ((HOT_TABLE_GROW_SLOTS_PER_KEY * want) as usize)
            .next_power_of_two()
            .min(max_size);
        if new_size <= size || want > HOT_TABLE_GROW_MAX_LOAD * new_size as f64 {
            return;
        }

        for hot_grouper in &mut self.hot_grouper_per_input {
            while hot_grouper.num_slots() < new_size {
                hot_grouper.double();
            }
        }
        self.hot_table_size = new_size;
        if polars_config::config().verbose() {
            eprintln!(
                "[group-by]: hot table {size} -> {new_size} slots (missed rows: {misses}, distinct: {distinct:.0}, heavy keys: {heavy:.0}, repeat: {repeat:.1}, heavy share: {heavy_share:.2})"
            );
        }
    }
}

/// Which columns of an input's morsels are kept (and spilled), and how the reductions of
/// that input read them.
pub struct InputPayload {
    /// The input columns which must be kept.
    pub stored_cols: Vec<PlSmallStr>,
    /// The subset of `stored_cols` needed to evaluate `fused_selectors` and the
    /// `fused_reductions`.
    pub gather_cols: Vec<PlSmallStr>,
    /// Elementwise expressions evaluated inside the node on only the rows being reduced,
    /// so that their output never enters the spilled cold morsels.
    pub fused_selectors: Vec<StreamExpr>,

    /// Reductions whose input columns are all in `stored_cols`.
    pub direct_reductions: Vec<usize>,
    /// Reductions with at least one input column produced by `fused_selectors`. A single
    /// subset is shared by all input columns of one update call, so such a reduction reads
    /// all of its inputs from the materialized frame.
    pub fused_reductions: Vec<usize>,
}

impl InputPayload {
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

        for selector in &self.fused_selectors {
            let c = selector
                .evaluate_preserve_len_broadcast(&eval_df, exec_state)
                .await?;
            unsafe { eval_df.push_column_unchecked(c.rechunk()) };
        }
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

struct GroupBySinkState {
    key_selectors_per_input: Vec<Vec<StreamExpr>>,
    reductions_per_input: Vec<Vec<usize>>,
    payload_per_input: Vec<InputPayload>,
    grouper: Box<dyn Grouper>,
    grouped_reduction_cols: Vec<Vec<PlSmallStr>>,
    grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    locals: Vec<LocalGroupBySinkState>,
    random_state: PlRandomState,
    partitioner: HashPartitioner,
    has_order_sensitive_agg: bool,
    max_hot_table_size: usize,

    estimated_groups: Metric<kind::Sum>,
    actual_groups: Metric<kind::Sum>,
}

impl GroupBySinkState {
    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        receivers: Vec<Receiver<(usize, Morsel)>>,
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        spill_ctx: &'env MostRecentSpillContext,
    ) {
        for (mut recv, local) in receivers.into_iter().zip(&mut self.locals) {
            let key_selectors_per_input = &self.key_selectors_per_input;
            let reductions_per_input = &self.reductions_per_input;
            let payload_per_input = &self.payload_per_input;
            let grouped_reduction_cols = &self.grouped_reduction_cols;
            let random_state = &self.random_state;
            let partitioner = self.partitioner.clone();
            let has_order_sensitive_agg = self.has_order_sensitive_agg;
            let max_hot_table_size = self.max_hot_table_size;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut hot_idxs = Vec::new();
                let mut hot_group_idxs = Vec::new();
                let mut cold_idxs = Vec::new();
                let mut identity_idxs: Vec<IdxSize> = Vec::new();
                let mut all_hot_per_input = vec![true; key_selectors_per_input.len()];
                while let Some((input_idx, morsel)) = recv.recv().await {
                    let seq = morsel.seq().to_u64();
                    let mut df = morsel.into_df().await;
                    let mut key_columns = Vec::new();
                    for selector in &key_selectors_per_input[input_idx] {
                        let s = selector.evaluate(&df, &state.in_memory_exec_state).await?;
                        key_columns.push(s.into_column());
                    }
                    let keys = unsafe {
                        DataFrame::new_unchecked_with_broadcast(df.height(), key_columns)?
                    };

                    // Drop columns which are neither reduction inputs nor fused sources.
                    let payload = &payload_per_input[input_idx];
                    if payload.stored_cols.len() < df.width() {
                        df = unsafe { df.select_unchecked(&payload.stored_cols) }.unwrap();
                    }
                    df.rechunk_mut(); // For gathers.

                    let slice_size = if all_hot_per_input[input_idx] {
                        KEY_SLICE_SIZE
                    } else {
                        df.height().max(1)
                    };
                    let reduce_per_slice = slice_size < df.height()
                        && local.hot_grouper_per_input[input_idx].num_groups() as usize
                            <= MAX_SLICE_REDUCTION_GROUPS;
                    all_hot_per_input[input_idx] = true;
                    let mut evictions_before = 0;
                    for offset in (0..df.height()).step_by(slice_size) {
                        // Compute hot group indices from key.
                        let hot_grouper = &mut local.hot_grouper_per_input[input_idx];
                        if hot_idxs.is_empty() {
                            evictions_before = hot_grouper.num_evictions();
                        }
                        let slice_keys = keys.slice(offset as i64, slice_size);
                        let hash_keys =
                            HashKeys::from_df(&slice_keys, random_state.clone(), true, false);
                        let hot_start = hot_idxs.len();
                        cold_idxs.clear();
                        hot_grouper.insert_keys(
                            &hash_keys,
                            &mut hot_idxs,
                            &mut hot_group_idxs,
                            &mut cold_idxs,
                            has_order_sensitive_agg,
                        );
                        if !reduce_per_slice {
                            for idx in &mut hot_idxs[hot_start..] {
                                *idx += offset as IdxSize;
                            }
                        }

                        // Store cold keys.
                        if !cold_idxs.is_empty() {
                            all_hot_per_input[input_idx] = false;
                            let mut cold_keys = hash_keys;
                            let mut cold_df = df.slice(offset as i64, slice_size);

                            // 75% or more cold, don't gather.
                            if cold_idxs.len() as u64 >= cold_df.height() as u64 * 3 / 4 {
                                unsafe {
                                    cold_keys.gen_idxs_per_partition_subset(
                                        &cold_idxs,
                                        &partitioner,
                                        &mut local.morsel_idxs_values_per_p,
                                        &mut local.sketch_per_p,
                                        Some(&mut local.miss_f2),
                                        true,
                                    );
                                }
                            } else {
                                unsafe {
                                    cold_keys = cold_keys.gather_unchecked(&cold_idxs);
                                    cold_df = cold_df.take_slice_unchecked_impl(&cold_idxs, false);
                                }

                                cold_keys.gen_idxs_per_partition(
                                    &partitioner,
                                    &mut local.morsel_idxs_values_per_p,
                                    &mut local.sketch_per_p,
                                    Some(&mut local.miss_f2),
                                    true,
                                );
                            }

                            local
                                .morsel_idxs_offsets_per_p
                                .extend(local.morsel_idxs_values_per_p.iter().map(|vp| vp.len()));
                            let sf = SpillFrame::new(cold_df, spill_ctx).await;
                            local.cold_morsels.push((input_idx, seq, cold_keys, sf));
                        }

                        if reduce_per_slice || offset + slice_size >= df.height() {
                            let reduce_df = if reduce_per_slice {
                                &df.slice(offset as i64, slice_size)
                            } else {
                                &df
                            };
                            let has_evictions = local.hot_grouper_per_input[input_idx]
                                .num_evictions()
                                != evictions_before;
                            let num_groups = local.hot_grouper_per_input[input_idx].num_groups();
                            for red_idx in &reductions_per_input[input_idx] {
                                local.hot_grouped_reductions[*red_idx].resize(num_groups);
                            }
                            payload
                                .update_reductions(
                                    reduce_df,
                                    &hot_idxs,
                                    &mut identity_idxs,
                                    grouped_reduction_cols,
                                    &mut local.hot_grouped_reductions,
                                    &state.in_memory_exec_state,
                                    |reduction, in_cols, subset| unsafe {
                                        if has_evictions {
                                            reduction.update_groups_while_evicting(
                                                in_cols,
                                                subset,
                                                &hot_group_idxs,
                                                seq,
                                            )
                                        } else {
                                            let group_idxs =
                                                EvictIdx::cast_to_idxs(&hot_group_idxs);
                                            reduction.update_groups_subset(
                                                in_cols, subset, group_idxs, seq,
                                            )
                                        }
                                    },
                                )
                                .await?;
                            hot_idxs.clear();
                            hot_group_idxs.clear();
                        }
                    }
                    let hot_grouper = &local.hot_grouper_per_input[input_idx];

                    // If we have too many evicted rows, flush them.
                    if hot_grouper.num_evictions() >= get_ideal_morsel_size() {
                        local.flush_evictions(
                            input_idx,
                            &reductions_per_input[input_idx],
                            &partitioner,
                        );
                    }

                    local.maybe_grow_hot_tables(max_hot_table_size);
                }
                Ok(())
            }));
        }
    }

    fn combine_locals(
        &mut self,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Vec<GroupByPartition>> {
        let exec_state = &state.in_memory_exec_state;
        // Finalize pre-aggregations.
        RAYON.install(|| {
            self.locals
                .as_mut_slice()
                .into_par_iter()
                .with_max_len(1)
                .for_each(|l| {
                    for (input_idx, r_idxs) in self.reductions_per_input.iter().enumerate() {
                        let hot_grouper = &mut l.hot_grouper_per_input[input_idx];
                        if hot_grouper.num_evictions() > 0 {
                            l.flush_evictions(input_idx, r_idxs, &self.partitioner);
                        }
                    }

                    let mut opt_hot_reductions =
                        l.hot_grouped_reductions.drain(..).map(Some).collect_vec();
                    for (input_idx, r_idxs) in self.reductions_per_input.iter().enumerate() {
                        let hot_grouper = &mut l.hot_grouper_per_input[input_idx];
                        let hot_keys = hot_grouper.keys();
                        let hot_reductions = r_idxs
                            .iter()
                            .map(|r| opt_hot_reductions[*r].take().unwrap())
                            .collect_vec();
                        l.add_pre_agg(hot_keys, r_idxs, hot_reductions, &self.partitioner, false);
                    }
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
        let reductions_per_input = &self.reductions_per_input;
        let payload_per_input = &self.payload_per_input;
        let grouped_reductions_template = &self.grouped_reductions;
        let grouped_reduction_cols = &self.grouped_reduction_cols;

        let estimated_groups_metric = &self.estimated_groups.reporter();
        let actual_groups_metric = &self.actual_groups.reporter();

        executor::task_scope(state.task_metrics(), |s| {
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

                    let sketch_estimate = sketch.estimate();
                    estimated_groups_metric.add(sketch_estimate as i64);

                    // Allocate grouper and reductions.
                    let est_num_groups = sketch_estimate * 5 / 4;
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
                    let mut identity_idxs: Vec<IdxSize> = Vec::new();
                    for (l, l_morsels) in locals.iter().zip(morsels_per_local) {
                        // Try to help with dropping.
                        if !skip_drop_attempt {
                            drop(drop_q_recv.try_recv());
                        }

                        for (i, morsel) in l_morsels.iter().enumerate() {
                            let (input_idx, seq_id, keys, sf) = morsel;
                            let morsel_df = sf.get().await;
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

                                for red_idx in &reductions_per_input[*input_idx] {
                                    p_reductions[*red_idx].resize(p_grouper.num_groups());
                                }

                                payload_per_input[*input_idx]
                                    .update_reductions(
                                        &morsel_df,
                                        p_morsel_idxs,
                                        &mut identity_idxs,
                                        grouped_reduction_cols,
                                        &mut p_reductions,
                                        exec_state,
                                        |reduction, in_cols, subset| {
                                            reduction.update_groups_subset(
                                                in_cols,
                                                subset,
                                                &group_idxs,
                                                *seq_id,
                                            )
                                        },
                                    )
                                    .await?;
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
                            let PreAgg {
                                keys,
                                reduction_idxs: r_idxs,
                                reductions: pre_aggs,
                            } = key_pre_aggs;
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
                                for (pre_agg, r_idx) in pre_aggs.iter().zip(r_idxs.iter()) {
                                    let r = &mut p_reductions[*r_idx];
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

                    // Each input only resizes its own reductions, so ensure all have the right length.
                    for r in &mut p_reductions {
                        r.resize(p_grouper.num_groups());
                    }

                    actual_groups_metric.add(p_grouper.num_groups() as i64);

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

            ASYNC.block_in_place_on(async move {
                for handle in join_handles {
                    handle.await?;
                }
                PolarsResult::Ok(())
            })?;
            PolarsResult::Ok(())
        })?;

        // Drop remaining local state in parallel.
        RAYON.install(|| {
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
                out.push_column_unchecked(r.finalize()?.with_name(name.clone()).into_column());
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
    num_inputs: usize,
    num_pipelines: usize,
    output_schema: Arc<Schema>,
    spill_ctx: MostRecentSpillContext,
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
        let num_partitions = num_pipelines;
        let locals = (0..num_pipelines)
            .map(|_| {
                let reductions = grouped_reductions.iter().map(|gr| gr.new_empty()).collect();
                LocalGroupBySinkState::new(
                    key_schema.clone(),
                    reductions,
                    hot_table_size,
                    num_partitions,
                    num_inputs,
                )
            })
            .collect();
        let partitioner = HashPartitioner::new(num_partitions, 0);
        Self {
            state: GroupByState::Sink(GroupBySinkState {
                key_selectors_per_input,
                reductions_per_input,
                payload_per_input,
                grouped_reductions,
                grouper,
                random_state,
                grouped_reduction_cols,
                locals,
                partitioner,
                has_order_sensitive_agg,
                max_hot_table_size,
                estimated_groups: metrics_registry
                    .new_counter("group_by.estimated_groups", MetricUnit::Unit),
                actual_groups: metrics_registry
                    .new_counter("group_by.actual_groups", MetricUnit::Unit),
            }),
            key_schema,
            num_inputs,
            num_pipelines,
            output_schema,
            spill_ctx: MostRecentSpillContext::new(
                "group-by".into(),
                metrics_registry.task_metrics(),
            ),
        }
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "group-by"
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            GroupByState::Sink(_) => NodeMemoryUsage::Accumulating,
            GroupByState::Source(src) => src.memory_usage(),
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
            // All inputs is done, transition to being a source.
            GroupByState::Sink(_) if recv.iter().all(|r| matches!(r, PortState::Done)) => {
                let GroupByState::Sink(mut sink) =
                    core::mem::replace(&mut self.state, GroupByState::Done)
                else {
                    unreachable!()
                };
                let partitions = sink.combine_locals(state)?;
                let dfs = RAYON.install(|| {
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
                recv.fill(PortState::Ready);
                send[0] = PortState::Blocked;
            },
            GroupByState::Source(..) => {
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
                sink.spawn(scope, receivers, state, join_handles, &self.spill_ctx)
            },
            GroupByState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            GroupByState::Done => unreachable!(),
        }
    }
}
