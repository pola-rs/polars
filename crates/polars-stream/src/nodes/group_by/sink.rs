use polars_arrow::bitmap::MutableBitmap;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::parallel::for_each_parallel;
use polars_core::frame::DataFrame;
use polars_core::prelude::{IntoColumn, PlRandomState};
use polars_error::PolarsResult;
use polars_expr::EvictIdx;
use polars_expr::hash_keys::HashKeys;
use polars_expr::hot_groups::{HotGrouper, new_hash_hot_grouper};
use polars_expr::reduce::GroupedReduction;
use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext, memory_manager};
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::f2_sketch::F2Sketch;
use polars_utils::itertools::Itertools;
use polars_utils::{IdxSize, UnitVec};
use tokio::sync::mpsc::Receiver;

use super::pre_agg::PreAgg;
use super::table::GroupByParams;
use super::tuning::{InputStats, budget};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::hash_shuffle::frame_with_keys::{
    FrameWithKeys, FrameWithKeysBuilder, SpillFrameWithKeys,
};
use crate::morsel::{Morsel, get_ideal_morsel_size};

#[cfg(debug_assertions)]
const KEY_SLICE_SIZE: usize = 64;
#[cfg(not(debug_assertions))]
const KEY_SLICE_SIZE: usize = 4096;

/// The number of hot groups up to which the reductions of a sliced morsel are updated
/// per slice.
const MAX_SLICE_REDUCTION_GROUPS: usize = 64;

/// The fewest rows a mostly cold morsel needs to be stored as it is, while the cold rows likely
/// fit in memory.
const MIN_ROWS_STORED_WHOLE: usize = 4096;

/// The hot tables only grow if the frequently missed keys were missed at least this
/// many times each.
const HOT_TABLE_GROW_MIN_REPEAT: f64 = 4.0;

/// The hot tables only grow if the frequently missed keys make up at least this
/// fraction of the missed rows.
const HOT_TABLE_GROW_MIN_HEAVY_SHARE: f64 = 0.5;

/// The hot tables grow to hold this many slots per key they should hold, if the
/// maximum size allows.
const HOT_TABLE_GROW_SLOTS_PER_KEY: f64 = 1.5;

/// The hot tables only grow if the keys they should hold fill at most this fraction
/// of the slots after growing.
const HOT_TABLE_GROW_MAX_LOAD: f64 = 0.75;

/// The state of one pipeline of the sink.
struct LocalSink {
    // Created when the input first sends this pipeline a non-empty morsel.
    hot_grouper_per_input: Vec<Option<Box<dyn HotGrouper>>>,
    hot_grouped_reductions: Vec<Box<dyn GroupedReduction>>,

    // The number of slots of each hot grouper, and the rows missed by them.
    hot_table_size: usize,
    miss_f2: F2Sketch,

    // The keys of the cold rows and the pre-aggregates.
    key_sketch: CardinalitySketch,

    // Per input, the cold rows not yet in a frame.
    cold_row_builders: Vec<Option<FrameWithKeysBuilder>>,
    // Each frame with the input it came from.
    cold_frames: Vec<(usize, SpillFrameWithKeys)>,
    // The estimated bytes of `cold_frames`.
    cold_bytes: u64,
    // In creation order.
    pre_aggs: Vec<PreAgg>,
}

impl LocalSink {
    fn new(params: &GroupByParams, hot_table_size: usize) -> Self {
        let num_inputs = params.reductions_per_input.len();
        Self {
            hot_grouper_per_input: (0..num_inputs).map(|_| None).collect(),
            hot_grouped_reductions: params
                .grouped_reductions
                .iter()
                .map(|r| r.new_empty())
                .collect(),
            hot_table_size,
            miss_f2: F2Sketch::new(),
            key_sketch: CardinalitySketch::new(),
            cold_row_builders: (0..num_inputs).map(|_| None).collect(),
            cold_frames: Vec::new(),
            cold_bytes: 0,
            pre_aggs: Vec::new(),
        }
    }

    /// The hot table of an input that sent this pipeline rows.
    fn hot_grouper(&mut self, input_idx: usize) -> &mut dyn HotGrouper {
        self.hot_grouper_per_input[input_idx]
            .as_deref_mut()
            .unwrap()
    }

    fn flush_evictions(&mut self, input_idx: usize, reduction_idxs: &[usize]) {
        let keys = self.hot_grouper(input_idx).take_evicted_keys();
        let reductions = reduction_idxs
            .iter()
            .map(|r| self.hot_grouped_reductions[*r].take_evictions())
            .collect();
        self.add_pre_agg(keys, reduction_idxs, reductions, true);
    }

    /// Adds a pre-aggregate over `keys`, which count as missed by the hot table if `is_miss`.
    fn add_pre_agg(
        &mut self,
        keys: HashKeys,
        reduction_idxs: &[usize],
        reductions: Vec<Box<dyn GroupedReduction>>,
        is_miss: bool,
    ) {
        keys.for_each_hash(|_, opt_h| {
            if let Some(h) = opt_h {
                self.key_sketch.insert(h);
                if is_miss {
                    self.miss_f2.insert(h);
                }
            }
        });
        self.pre_aggs.push(PreAgg {
            keys,
            reduction_idxs: UnitVec::from_slice(reduction_idxs),
            reductions,
            split: None,
        });
    }

    /// Stores the cold rows `cold_idxs` of a slice `df` of a morsel of `input_idx`, and adds
    /// their hashes to the sketches. A whole morsel that may be stored as it is and has mostly
    /// cold rows becomes a frame with a mask of its cold rows; otherwise the cold rows are
    /// gathered into the builder of the input, which becomes a frame once it holds a morsel's
    /// worth of rows.
    async fn store_cold_rows(
        &mut self,
        input_idx: usize,
        df: DataFrame,
        keys: HashKeys,
        cold_idxs: &[IdxSize],
        may_store_whole: bool,
        spill_ctx: &MostRecentSpillContext,
    ) {
        unsafe {
            keys.for_each_hash_subset(cold_idxs, |_, opt_h| {
                if let Some(h) = opt_h {
                    self.key_sketch.insert(h);
                    self.miss_f2.insert(h);
                }
            });
        }

        // 75% or more cold, don't gather.
        let frame = if may_store_whole && cold_idxs.len() as u64 >= df.height() as u64 * 3 / 4 {
            let mut mask = MutableBitmap::from_len_zeroed(df.height());
            for idx in cold_idxs {
                mask.set(*idx as usize, true);
            }
            FrameWithKeys {
                frame: df,
                keys,
                mask: Some(mask.freeze()),
            }
        } else {
            let frame_rows = get_ideal_morsel_size();
            let builder = self.cold_row_builders[input_idx].get_or_insert_with(|| {
                let mut builder = FrameWithKeysBuilder::new(df.schema().clone(), &keys);
                builder.reserve(frame_rows);
                builder
            });
            unsafe { builder.gather_extend(&df, &keys, cold_idxs) };
            if builder.len() < frame_rows {
                return;
            }
            let frame = builder.freeze_reset();
            builder.reserve(frame_rows);
            frame
        };

        let frame = SpillFrameWithKeys::new(frame, spill_ctx).await;
        self.cold_bytes += frame.estimated_size() as u64;
        self.cold_frames.push((input_idx, frame));
    }

    /// The fewest rows a morsel needs to be stored as it is. Once the cold rows of all pipelines
    /// likely exceed 3/4 of the memory budget this is larger, as every stored frame adds work
    /// when the partitions are written out.
    fn min_rows_stored_whole(&self, num_pipelines: usize) -> usize {
        let cold_bytes = self.cold_bytes.saturating_mul(num_pipelines as u64);
        if cold_bytes < budget() / 4 * 3 {
            MIN_ROWS_STORED_WHOLE
        } else {
            (get_ideal_morsel_size() / 4).max(MIN_ROWS_STORED_WHOLE)
        }
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
        let distinct = (self.key_sketch.estimate() as f64).min(misses);
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
            .flatten()
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

        for hot_grouper in self.hot_grouper_per_input.iter_mut().flatten() {
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

    /// Ends the input of this pipeline: stores the cold rows left in the builders, flushes the
    /// evictions and adds the hot tables as the last pre-aggregates.
    fn finish_input(
        &mut self,
        reductions_per_input: &[Vec<usize>],
        spill_ctx: &MostRecentSpillContext,
    ) {
        for (input_idx, builder) in self.cold_row_builders.iter_mut().enumerate() {
            if let Some(mut builder) = builder.take()
                && !builder.is_empty()
            {
                let frame = SpillFrameWithKeys::new_unregistered(builder.freeze_reset());
                spill_ctx.register_no_spill_check(&frame);
                self.cold_frames.push((input_idx, frame));
            }
        }

        for (input_idx, r_idxs) in reductions_per_input.iter().enumerate() {
            let hot_grouper = &self.hot_grouper_per_input[input_idx];
            if hot_grouper.as_ref().is_some_and(|g| g.num_evictions() > 0) {
                self.flush_evictions(input_idx, r_idxs);
            }
        }

        let mut hot_reductions = self
            .hot_grouped_reductions
            .drain(..)
            .map(Some)
            .collect_vec();
        for (input_idx, r_idxs) in reductions_per_input.iter().enumerate() {
            let Some(hot_grouper) = self.hot_grouper_per_input[input_idx].take() else {
                continue;
            };
            let reductions = r_idxs
                .iter()
                .map(|r| hot_reductions[*r].take().unwrap())
                .collect();
            self.add_pre_agg(hot_grouper.keys(), r_idxs, reductions, false);
        }
    }
}

/// The node state while it receives its input.
pub struct GroupBySink {
    // Input stream i selects keys with key_selectors_per_input[i].
    key_selectors_per_input: Vec<Vec<StreamExpr>>,
    random_state: PlRandomState,
    hot_table_size: usize,
    max_hot_table_size: usize,
    // Per pipeline, created when it first receives a non-empty morsel.
    locals: Vec<Option<LocalSink>>,
}

impl GroupBySink {
    pub fn new(
        key_selectors_per_input: Vec<Vec<StreamExpr>>,
        random_state: PlRandomState,
        hot_table_size: usize,
        max_hot_table_size: usize,
        num_pipelines: usize,
    ) -> Self {
        Self {
            key_selectors_per_input,
            random_state,
            hot_table_size,
            max_hot_table_size,
            locals: (0..num_pipelines).map(|_| None).collect(),
        }
    }

    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        receivers: Vec<Receiver<(usize, Morsel)>>,
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        params: &'env GroupByParams,
        spill_ctx: &'env MostRecentSpillContext,
    ) {
        let num_pipelines = self.locals.len();
        for (mut recv, opt_local) in receivers.into_iter().zip(&mut self.locals) {
            let key_selectors_per_input = &self.key_selectors_per_input;
            let random_state = &self.random_state;
            let hot_table_size = self.hot_table_size;
            let max_hot_table_size = self.max_hot_table_size;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut hot_idxs = Vec::new();
                let mut hot_group_idxs = Vec::new();
                let mut cold_idxs = Vec::new();
                let mut identity_idxs: Vec<IdxSize> = Vec::new();
                let mut all_hot_per_input = vec![true; key_selectors_per_input.len()];
                while let Some((input_idx, morsel)) = recv.recv().await {
                    if morsel.height() == 0 {
                        continue;
                    }
                    let local =
                        opt_local.get_or_insert_with(|| LocalSink::new(params, hot_table_size));
                    if local.hot_grouper_per_input[input_idx].is_none() {
                        local.hot_grouper_per_input[input_idx] = Some(new_hash_hot_grouper(
                            params.key_schema.clone(),
                            local.hot_table_size,
                        ));
                    }

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
                    let payload = &params.payload_per_input[input_idx];
                    if payload.stored_cols.len() < df.width() {
                        df = unsafe { df.select_unchecked(&payload.stored_cols) }.unwrap();
                    }
                    df.rechunk_mut(); // For gathers.

                    let slice_size = if all_hot_per_input[input_idx] {
                        KEY_SLICE_SIZE
                    } else {
                        df.height()
                    };
                    // A large enough morsel processed as one slice may be stored as it is.
                    let may_store_whole = slice_size >= df.height()
                        && df.height() >= local.min_rows_stored_whole(num_pipelines);
                    let reduce_per_slice = slice_size < df.height()
                        && local.hot_grouper(input_idx).num_groups() as usize
                            <= MAX_SLICE_REDUCTION_GROUPS;
                    all_hot_per_input[input_idx] = true;
                    let mut evictions_before = 0;
                    for offset in (0..df.height()).step_by(slice_size) {
                        // Compute hot group indices from key.
                        let hot_grouper = local.hot_grouper(input_idx);
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
                            params.has_order_sensitive_agg,
                        );
                        if !reduce_per_slice {
                            for idx in &mut hot_idxs[hot_start..] {
                                *idx += offset as IdxSize;
                            }
                        }

                        if !cold_idxs.is_empty() {
                            all_hot_per_input[input_idx] = false;
                            local
                                .store_cold_rows(
                                    input_idx,
                                    df.slice(offset as i64, slice_size),
                                    hash_keys,
                                    &cold_idxs,
                                    may_store_whole,
                                    spill_ctx,
                                )
                                .await;
                        }

                        if reduce_per_slice || offset + slice_size >= df.height() {
                            let reduce_df = if reduce_per_slice {
                                &df.slice(offset as i64, slice_size)
                            } else {
                                &df
                            };
                            let has_evictions =
                                local.hot_grouper(input_idx).num_evictions() != evictions_before;
                            let num_groups = local.hot_grouper(input_idx).num_groups();
                            for red_idx in &params.reductions_per_input[input_idx] {
                                local.hot_grouped_reductions[*red_idx].resize(num_groups);
                            }
                            payload
                                .update_reductions(
                                    reduce_df,
                                    &hot_idxs,
                                    &mut identity_idxs,
                                    &params.grouped_reduction_cols,
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

                    // If we have too many evicted rows, flush them.
                    if local.hot_grouper(input_idx).num_evictions() >= get_ideal_morsel_size() {
                        local.flush_evictions(input_idx, &params.reductions_per_input[input_idx]);
                    }

                    local.maybe_grow_hot_tables(max_hot_table_size);
                }
                Ok(())
            }));
        }
    }

    /// Ends the input: every pipeline that received rows, in parallel, stores the cold rows left
    /// in its builders and turns its evictions and hot tables into pre-aggregates. Returns the
    /// statistics, the cold frames, and the pre-aggregates pipeline by pipeline in creation
    /// order.
    #[allow(clippy::type_complexity)]
    pub fn finish_input(
        self,
        params: &GroupByParams,
        spill_ctx: &MostRecentSpillContext,
        state: &StreamingExecutionState,
    ) -> PolarsResult<(InputStats, Vec<(usize, SpillFrameWithKeys)>, Vec<PreAgg>)> {
        let mut locals: Vec<LocalSink> = self.locals.into_iter().flatten().collect();
        for_each_parallel(
            locals.iter_mut(),
            state.task_metrics(),
            &|local| async move {
                local.finish_input(&params.reductions_per_input, spill_ctx);
                memory_manager().spill().await;
                Ok(())
            },
        )?;

        let mut stats = InputStats {
            num_reductions: params.grouped_reductions.len() as u64,
            ..Default::default()
        };
        let mut sketch = CardinalitySketch::new();
        let mut frames = Vec::new();
        let mut pre_aggs = Vec::new();
        for local in locals {
            sketch.combine(&local.key_sketch);
            for (_, frame) in &local.cold_frames {
                stats.cold_rows += frame.num_rows() as u64;
                stats.cold_bytes += frame.estimated_size() as u64;
            }
            for pre_agg in &local.pre_aggs {
                stats.pre_agg_groups += pre_agg.keys.len() as u64;
                stats.pre_agg_key_bytes += pre_agg.keys.estimated_size() as u64;
            }
            frames.extend(local.cold_frames);
            pre_aggs.extend(local.pre_aggs);
        }
        stats.estimated_groups = sketch.estimate() as u64;
        Ok((stats, frames, pre_aggs))
    }
}
