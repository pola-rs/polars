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

/// An input stops using its hot table if its growing states hold more than this fraction of
/// the values they received.
const GROWING_STATES_MAX_KEPT_FRACTION: f64 = 0.5;

/// An input stops using its hot table if its growing states, times the number of pipelines,
/// use more than the memory budget divided by this.
const GROWING_STATES_BUDGET_DIVISOR: u64 = 8;

/// Whether growing states shrink the data is checked once an input has at least this many hot
/// rows per hot group.
const GROWING_STATES_MIN_ROWS_PER_GROUP: u64 = 8;

/// Below this many groups per pipeline an input keeps its hot table even if its growing states
/// do not shrink.
const GROWING_STATES_MIN_GROUPS_PER_PIPELINE: usize = 2;

/// An input stops using its hot table if the cold rows plus this many times the evictions make
/// up at least `MIN_MISS_SCORE` of the rows in a window, and the table does not grow. An
/// evicted group comes back as a pre-aggregate row, so the rows it took in saved little.
const EVICTION_WEIGHT: f64 = 3.0;
const MIN_MISS_SCORE: f64 = 0.4;
/// Hot rows with fixed-width keys are cheap, so these inputs need more misses, and an evicted
/// group counts as one missed row.
const MIN_MISS_SCORE_FIXED_WIDTH: f64 = 0.65;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrowOutcome {
    Grew,
    /// Too few missed rows to decide.
    NeedMoreData,
    Declined,
    AtLimit,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum HotTableMode {
    #[default]
    Hot,
    /// The hot table missed too many rows and does not grow.
    Bypass,
    /// The growing states of the hot table do not pay off.
    ColdGrowing,
}

/// The windows in which the misses of a hot table are counted double in length up to this many
/// morsels.
const MAX_WINDOW_MORSELS: u64 = 16;

/// Whether an input uses its hot table, and how often a full table missed its rows. The
/// misses are counted in windows of rows that double in length.
#[derive(Clone)]
struct HotTableStats {
    mode: HotTableMode,
    window_rows: u64,
    window_cold_rows: u64,
    window_evictions: u64,
    window_target: u64,
}

impl HotTableStats {
    fn new() -> Self {
        Self {
            mode: HotTableMode::Hot,
            window_rows: 0,
            window_cold_rows: 0,
            window_evictions: 0,
            window_target: 2 * get_ideal_morsel_size() as u64,
        }
    }

    fn reset_window(&mut self, target: u64) {
        self.window_rows = 0;
        self.window_cold_rows = 0;
        self.window_evictions = 0;
        self.window_target = target;
    }
}

/// Whether an input whose hot states can grow uses its hot table, and what it received so far.
/// The size of the states is checked after every morsel. Whether they pay off is first checked
/// after a morsel of hot rows, then each time the number of hot rows doubles.
#[derive(Clone, Default)]
struct GrowingStates {
    /// Some reduction of the input has states that grow with the values of a group.
    may_go_cold: bool,
    /// The number of reductions of the input with growing states.
    num_growing: u64,
    hot_rows: u64,
    next_check_rows: u64,
    /// The values and bytes held by the pre-aggregates of the input.
    pre_agg_values: u64,
    pre_agg_bytes: u64,
}

/// The state of one pipeline of the sink.
struct LocalSink {
    // Created when the input first sends this pipeline a non-empty morsel.
    hot_grouper_per_input: Vec<Option<Box<dyn HotGrouper>>>,
    hot_grouped_reductions: Vec<Box<dyn GroupedReduction>>,

    // The number of slots of each hot grouper, and the rows missed by them.
    hot_table_size: usize,
    miss_f2: F2Sketch,
    miss_sketch: CardinalitySketch,
    hot_table_stats_per_input: Vec<HotTableStats>,
    eviction_weight: f64,
    min_miss_score: f64,

    // The keys of the cold rows and the pre-aggregates.
    key_sketch: CardinalitySketch,

    // Per input, the cold rows not yet in a frame.
    cold_row_builders: Vec<Option<FrameWithKeysBuilder>>,
    // 0..n, to gather all rows of a morsel.
    all_row_idxs: Vec<IdxSize>,
    // Each frame with the input it came from.
    cold_frames: Vec<(usize, SpillFrameWithKeys)>,
    // The estimated bytes of `cold_frames`.
    cold_bytes: u64,
    // In creation order.
    pre_aggs: Vec<PreAgg>,
    // See `InputStats`.
    pre_agg_work: u64,
    pre_agg_state_bytes: u64,
    growing_states_per_input: Vec<GrowingStates>,
}

impl LocalSink {
    fn new(params: &GroupByParams, hot_table_size: usize) -> Self {
        let num_inputs = params.reductions_per_input.len();
        let fixed_width_keys = params.key_schema.iter_values().all(|dt| {
            let dt = dt.to_physical();
            dt.is_primitive_numeric() || dt.is_bool()
        });
        Self {
            eviction_weight: if fixed_width_keys {
                1.0
            } else {
                EVICTION_WEIGHT
            },
            min_miss_score: if fixed_width_keys {
                MIN_MISS_SCORE_FIXED_WIDTH
            } else {
                MIN_MISS_SCORE
            },
            hot_grouper_per_input: (0..num_inputs).map(|_| None).collect(),
            hot_grouped_reductions: params
                .grouped_reductions
                .iter()
                .map(|r| r.new_empty())
                .collect(),
            hot_table_size,
            miss_f2: F2Sketch::new(),
            miss_sketch: CardinalitySketch::new(),
            hot_table_stats_per_input: vec![HotTableStats::new(); num_inputs],
            key_sketch: CardinalitySketch::new(),
            cold_row_builders: (0..num_inputs).map(|_| None).collect(),
            all_row_idxs: Vec::new(),
            cold_frames: Vec::new(),
            cold_bytes: 0,
            pre_aggs: Vec::new(),
            pre_agg_work: 0,
            pre_agg_state_bytes: 0,
            growing_states_per_input: params
                .reductions_per_input
                .iter()
                .map(|r_idxs| {
                    let num_growing = r_idxs
                        .iter()
                        .filter(|r| params.grouped_reductions[**r].state_size().is_some())
                        .count();
                    // Cold rows only exist without order-sensitive reductions.
                    let may_go_cold = num_growing > 0 && !params.has_order_sensitive_agg;
                    GrowingStates {
                        may_go_cold,
                        num_growing: num_growing as u64,
                        next_check_rows: get_ideal_morsel_size() as u64,
                        ..Default::default()
                    }
                })
                .collect(),
        }
    }

    /// The hot table of an input that sent this pipeline rows.
    fn hot_grouper(&mut self, input_idx: usize) -> &mut dyn HotGrouper {
        self.hot_grouper_per_input[input_idx]
            .as_deref_mut()
            .unwrap()
    }

    fn flush_evictions(&mut self, input_idx: usize, reduction_idxs: &[usize]) {
        if self.hot_grouper(input_idx).num_evictions() == 0 {
            return;
        }
        let keys = self.hot_grouper(input_idx).take_evicted_keys();
        let reductions = reduction_idxs
            .iter()
            .map(|r| self.hot_grouped_reductions[*r].take_evictions())
            .collect();
        self.add_pre_agg(input_idx, keys, reduction_idxs, reductions, true);
    }

    /// Adds a pre-aggregate over `keys`, which count as missed by the hot table if `is_miss`.
    fn add_pre_agg(
        &mut self,
        input_idx: usize,
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
                    self.miss_sketch.insert(h);
                }
            }
        });
        let mut num_values = 0;
        let mut bytes = 0;
        for size in reductions.iter().filter_map(|r| r.state_size()) {
            num_values += size.num_values as u64;
            bytes += size.bytes as u64;
        }
        let growing = &mut self.growing_states_per_input[input_idx];
        growing.pre_agg_values += num_values;
        growing.pre_agg_bytes += bytes;
        self.pre_agg_work += (keys.len() as u64).max(num_values);
        self.pre_agg_state_bytes += bytes;
        self.pre_aggs.push(PreAgg {
            keys,
            reduction_idxs: UnitVec::from_slice(reduction_idxs),
            reductions,
            split: None,
        });
    }

    /// Stops using the hot table of `input_idx` if its growing states keep too many of the
    /// values they received, or use too much memory. Its hot groups become a pre-aggregate.
    fn maybe_go_cold(
        &mut self,
        input_idx: usize,
        reductions_per_input: &[Vec<usize>],
        num_pipelines: usize,
    ) {
        let growing = &self.growing_states_per_input[input_idx];
        if !growing.may_go_cold
            || self.hot_table_stats_per_input[input_idx].mode != HotTableMode::Hot
        {
            return;
        }
        let hot_rows = growing.hot_rows;
        let next_check_rows = growing.next_check_rows;
        let num_growing = growing.num_growing;

        let r_idxs = &reductions_per_input[input_idx];
        let mut values = growing.pre_agg_values;
        let mut bytes = growing.pre_agg_bytes;
        for size in r_idxs
            .iter()
            .filter_map(|r| self.hot_grouped_reductions[*r].state_size())
        {
            values += size.num_values as u64;
            bytes += size.bytes as u64;
        }
        let too_large =
            bytes.saturating_mul(GROWING_STATES_BUDGET_DIVISOR * num_pipelines as u64) > budget();

        #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u32.
        let num_hot_groups = self.hot_grouper(input_idx).num_groups() as u64;
        let check_shrink = hot_rows >= next_check_rows
            && hot_rows >= GROWING_STATES_MIN_ROWS_PER_GROUP * num_hot_groups;
        let keeps_too_much = check_shrink && {
            let received = hot_rows * num_growing;
            let num_groups = num_hot_groups as usize + self.key_sketch.estimate();
            values as f64 > GROWING_STATES_MAX_KEPT_FRACTION * received as f64
                && num_groups >= GROWING_STATES_MIN_GROUPS_PER_PIPELINE * num_pipelines
        };

        let growing = &mut self.growing_states_per_input[input_idx];
        if check_shrink {
            growing.next_check_rows = 2 * hot_rows;
        }
        if !keeps_too_much && !too_large {
            return;
        }
        if polars_config::config().verbose() {
            eprintln!(
                "[group-by]: input {input_idx} stops using its hot table (hot rows: {}, values kept: {values}, state bytes: {bytes})",
                growing.hot_rows
            );
        }
        self.stop_hot_table(input_idx, r_idxs, HotTableMode::ColdGrowing);
    }

    /// Stores every later row of `input_idx` as a cold row. Its evictions and hot groups become
    /// pre-aggregates.
    fn stop_hot_table(&mut self, input_idx: usize, r_idxs: &[usize], mode: HotTableMode) {
        self.hot_table_stats_per_input[input_idx].mode = mode;
        self.flush_evictions(input_idx, r_idxs);
        let hot_grouper = self.hot_grouper_per_input[input_idx].take().unwrap();
        let reductions = r_idxs
            .iter()
            .map(|r| {
                let template = self.hot_grouped_reductions[*r].new_empty();
                std::mem::replace(&mut self.hot_grouped_reductions[*r], template)
            })
            .collect();
        self.add_pre_agg(input_idx, hot_grouper.keys(), r_idxs, reductions, false);
    }

    /// Adds the hashes of the rows `cold_idxs` of `keys`, all rows if `None`, to the sketches,
    /// which count them as missed by the hot table if `is_miss`.
    fn insert_cold_hashes(
        &mut self,
        keys: &HashKeys,
        cold_idxs: Option<&[IdxSize]>,
        is_miss: bool,
    ) {
        let insert_hash = |_, opt_h: Option<u64>| {
            if let Some(h) = opt_h {
                self.key_sketch.insert(h);
                if is_miss {
                    self.miss_f2.insert(h);
                    self.miss_sketch.insert(h);
                }
            }
        };
        match cold_idxs {
            Some(idxs) => unsafe { keys.for_each_hash_subset(idxs, insert_hash) },
            None => keys.for_each_hash(insert_hash),
        }
    }

    /// Stores the cold rows `cold_idxs` of a slice `df` of a morsel of `input_idx`, all rows if
    /// `None`, and adds their hashes to the sketches, see `insert_cold_hashes`. A whole morsel that may be stored as it is and has mostly
    /// cold rows becomes a frame with a mask of its cold rows; otherwise the cold rows are
    /// gathered into the builder of the input, which becomes a frame once it holds a morsel's
    /// worth of rows.
    #[allow(clippy::too_many_arguments)]
    async fn store_cold_rows(
        &mut self,
        input_idx: usize,
        df: DataFrame,
        keys: HashKeys,
        cold_idxs: Option<&[IdxSize]>,
        is_miss: bool,
        may_store_whole: bool,
        spill_ctx: &MostRecentSpillContext,
    ) {
        self.insert_cold_hashes(&keys, cold_idxs, is_miss);

        // 75% or more cold, don't gather.
        let num_cold = cold_idxs.map_or(df.height(), |idxs| idxs.len());
        let frame = if may_store_whole && num_cold as u64 >= df.height() as u64 * 3 / 4 {
            let mask = cold_idxs
                .filter(|idxs| idxs.len() < df.height())
                .map(|idxs| {
                    let mut mask = MutableBitmap::from_len_zeroed(df.height());
                    for idx in idxs {
                        mask.set(*idx as usize, true);
                    }
                    mask.freeze()
                });
            FrameWithKeys {
                frame: df,
                keys,
                mask,
            }
        } else {
            let frame_rows = get_ideal_morsel_size();
            let builder = self.cold_row_builders[input_idx].get_or_insert_with(|| {
                let mut builder = FrameWithKeysBuilder::new(df.schema().clone(), &keys);
                builder.reserve(frame_rows);
                builder
            });
            let idxs = match cold_idxs {
                Some(idxs) => idxs,
                None => {
                    let n = self.all_row_idxs.len();
                    if n < df.height() {
                        self.all_row_idxs
                            .extend(n as IdxSize..df.height() as IdxSize);
                    }
                    &self.all_row_idxs[..df.height()]
                },
            };
            unsafe { builder.gather_extend(&df, &keys, idxs) };
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
    fn maybe_grow_hot_tables(&mut self, max_size: usize) -> GrowOutcome {
        let size = self.hot_table_size;
        if size >= max_size {
            return GrowOutcome::AtLimit;
        }

        // Cheap early exit before estimating the number of distinct missed keys.
        let misses = self.miss_f2.num_inserts() as f64;
        if misses < HOT_TABLE_GROW_MIN_REPEAT * size as f64 {
            return GrowOutcome::NeedMoreData;
        }

        // Model the missed rows as `heavy` keys missed `repeat` times each plus keys
        // missed once. The row count, the distinct key count and F2 then determine
        // both.
        let f2 = self.miss_f2.estimate();
        let distinct = (self.miss_sketch.estimate() as f64).min(misses);
        let excess = misses - distinct;
        let denom = f2 - 2.0 * misses + distinct;
        if excess <= 0.0 || denom <= 0.0 {
            return GrowOutcome::Declined;
        }
        let repeat = (f2 - misses) / excess;
        let heavy = excess * excess / denom;
        let heavy_share = heavy * repeat / misses;
        if repeat < HOT_TABLE_GROW_MIN_REPEAT || heavy_share < HOT_TABLE_GROW_MIN_HEAVY_SHARE {
            return GrowOutcome::Declined;
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
            return GrowOutcome::Declined;
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
        for stats in &mut self.hot_table_stats_per_input {
            if stats.mode == HotTableMode::Hot {
                stats.reset_window(2 * get_ideal_morsel_size() as u64);
            }
        }
        GrowOutcome::Grew
    }

    /// Grows the hot tables if that pays off. Otherwise, once a window of rows of `input_idx` is
    /// complete, stops using its hot table if enough of the rows missed the full table.
    fn after_hot_morsel(
        &mut self,
        input_idx: usize,
        reductions_per_input: &[Vec<usize>],
        has_order_sensitive_agg: bool,
        max_size: usize,
    ) {
        if self.maybe_grow_hot_tables(max_size) == GrowOutcome::Grew {
            return;
        }
        let stats = &mut self.hot_table_stats_per_input[input_idx];
        if stats.mode != HotTableMode::Hot || stats.window_rows < stats.window_target {
            return;
        }
        // Cold rows drop the order of the rows.
        if has_order_sensitive_agg {
            return;
        }
        let window_rows = stats.window_rows;
        let miss_score = (stats.window_cold_rows as f64
            + self.eviction_weight * stats.window_evictions as f64)
            / window_rows as f64;
        if miss_score < self.min_miss_score || memory_manager().has_activity() {
            let max_target = MAX_WINDOW_MORSELS * get_ideal_morsel_size() as u64;
            stats.reset_window((2 * stats.window_target).min(max_target));
            return;
        }

        let mut outcome = GrowOutcome::AtLimit;
        if self.hot_table_size < max_size {
            // Until the missed keys were missed well over the times growth needs, more rows may
            // show that they repeat enough to grow, if they fit in the largest table.
            let num_hot_keys = self
                .hot_grouper_per_input
                .iter()
                .flatten()
                .map(|g| g.num_groups() as usize)
                .max()
                .unwrap_or(0);
            let may_grow_later = |max_missed_keys: usize, min_missed_keys: usize, misses: u64| {
                (num_hot_keys + max_missed_keys) as f64 <= HOT_TABLE_GROW_MAX_LOAD * max_size as f64
                    && (misses as f64) < 2.0 * HOT_TABLE_GROW_MIN_REPEAT * min_missed_keys as f64
            };
            // Growth statistics only see evictions once flushed.
            let pending: usize = self
                .hot_grouper_per_input
                .iter()
                .flatten()
                .map(|g| g.num_evictions())
                .sum();
            let misses = self.miss_f2.num_inserts() + pending as u64;
            let missed_keys = self.miss_sketch.estimate();
            if (misses as f64) < HOT_TABLE_GROW_MIN_REPEAT * self.hot_table_size as f64
                || may_grow_later(missed_keys + pending, missed_keys, misses)
            {
                return;
            }
            for (i, r_idxs) in reductions_per_input.iter().enumerate() {
                if self.hot_grouper_per_input[i].is_some() {
                    self.flush_evictions(i, r_idxs);
                }
            }
            outcome = self.maybe_grow_hot_tables(max_size);
            if matches!(outcome, GrowOutcome::Grew | GrowOutcome::NeedMoreData) || {
                let missed_keys = self.miss_sketch.estimate();
                may_grow_later(missed_keys, missed_keys, self.miss_f2.num_inserts())
            } {
                return;
            }
        }

        if polars_config::config().verbose() {
            eprintln!(
                "[group-by]: input {input_idx} bypasses its hot table (window rows: {window_rows}, miss score: {miss_score:.2}, hot table: {} slots, growth: {outcome:?})",
                self.hot_table_size
            );
        }
        self.stop_hot_table(
            input_idx,
            &reductions_per_input[input_idx],
            HotTableMode::Bypass,
        );
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
            if self.hot_grouper_per_input[input_idx].is_some() {
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
            self.add_pre_agg(input_idx, hot_grouper.keys(), r_idxs, reductions, false);
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
                    let cold_only =
                        local.hot_table_stats_per_input[input_idx].mode != HotTableMode::Hot;
                    if !cold_only && local.hot_grouper_per_input[input_idx].is_none() {
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

                    if cold_only {
                        let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
                        let may_store_whole =
                            df.height() >= local.min_rows_stored_whole(num_pipelines);
                        local
                            .store_cold_rows(
                                input_idx,
                                df,
                                hash_keys,
                                None,
                                false,
                                may_store_whole,
                                spill_ctx,
                            )
                            .await;
                        continue;
                    }

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
                    let evictions_at_start = local.hot_grouper(input_idx).num_evictions();
                    let mut num_cold = 0;
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
                        local.growing_states_per_input[input_idx].hot_rows +=
                            (hot_idxs.len() - hot_start) as u64;

                        if !cold_idxs.is_empty() {
                            all_hot_per_input[input_idx] = false;
                            num_cold += cold_idxs.len();
                            local
                                .store_cold_rows(
                                    input_idx,
                                    df.slice(offset as i64, slice_size),
                                    hash_keys,
                                    Some(&cold_idxs),
                                    true,
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

                    let num_evictions = local.hot_grouper(input_idx).num_evictions();
                    let stats = &mut local.hot_table_stats_per_input[input_idx];
                    stats.window_rows += df.height() as u64;
                    stats.window_cold_rows += num_cold as u64;
                    stats.window_evictions += (num_evictions - evictions_at_start) as u64;

                    // If we have too many evicted rows, flush them.
                    if num_evictions >= get_ideal_morsel_size() {
                        local.flush_evictions(input_idx, &params.reductions_per_input[input_idx]);
                    }

                    local.maybe_go_cold(input_idx, &params.reductions_per_input, num_pipelines);
                    local.after_hot_morsel(
                        input_idx,
                        &params.reductions_per_input,
                        params.has_order_sensitive_agg,
                        max_hot_table_size,
                    );
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
            has_growing_states: params
                .grouped_reductions
                .iter()
                .any(|r| r.state_size().is_some()),
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
            stats.pre_agg_work += local.pre_agg_work;
            stats.pre_agg_state_bytes += local.pre_agg_state_bytes;
            frames.extend(local.cold_frames);
            pre_aggs.extend(local.pre_aggs);
        }
        stats.estimated_groups = sketch.estimate() as u64;
        Ok((stats, frames, pre_aggs))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use polars_core::prelude::{Column, DataType, Field, Schema};
    use polars_expr::groups::new_hash_grouper;
    use polars_expr::reduce::new_max_reduction;

    use super::*;
    use crate::nodes::group_by::InputPayload;

    fn params(num_inputs: usize) -> GroupByParams {
        let key_schema = Arc::new(Schema::from_iter([Field::new("k".into(), DataType::Int64)]));
        GroupByParams {
            key_schema: key_schema.clone(),
            output_schema: key_schema.clone(),
            reductions_per_input: (0..num_inputs).map(|i| vec![i]).collect(),
            grouped_reduction_cols: (0..num_inputs).map(|_| vec!["v".into()]).collect(),
            payload_per_input: (0..num_inputs)
                .map(|_| InputPayload {
                    stored_cols: Vec::new(),
                    gather_cols: Vec::new(),
                    fused_selectors: Vec::new(),
                    direct_reductions: Vec::new(),
                    fused_reductions: Vec::new(),
                })
                .collect(),
            grouper: new_hash_grouper(key_schema),
            grouped_reductions: (0..num_inputs)
                .map(|_| new_max_reduction(DataType::Int64, false).unwrap())
                .collect(),
            has_order_sensitive_agg: false,
        }
    }

    fn local_sink(params: &GroupByParams, table_size: usize) -> LocalSink {
        let mut local = LocalSink::new(params, table_size);
        for grouper in &mut local.hot_grouper_per_input {
            *grouper = Some(new_hash_hot_grouper(params.key_schema.clone(), table_size));
        }
        local
    }

    /// Sends `keys` through the hot table of `input_idx` and counts its misses, as the sink does.
    fn insert(local: &mut LocalSink, input_idx: usize, keys: &[i64]) {
        let df = DataFrame::new_infer_height(vec![Column::new("k".into(), keys)]).unwrap();
        let keys = HashKeys::from_df(&df, PlRandomState::default(), true, false);
        let (mut hot_idxs, mut hot_group_idxs, mut cold_idxs) =
            (Vec::new(), Vec::new(), Vec::new());
        let grouper = local.hot_grouper(input_idx);
        let evictions_before = grouper.num_evictions();
        grouper.insert_keys(
            &keys,
            &mut hot_idxs,
            &mut hot_group_idxs,
            &mut cold_idxs,
            false,
        );
        let evictions = grouper.num_evictions() - evictions_before;
        local.insert_cold_hashes(&keys, Some(&cold_idxs), true);
        let stats = &mut local.hot_table_stats_per_input[input_idx];
        stats.window_rows += df.height() as u64;
        stats.window_cold_rows += cold_idxs.len() as u64;
        stats.window_evictions += evictions as u64;
    }

    fn unique_keys(n: i64) -> Vec<i64> {
        (1_000_000..1_000_000 + n).collect()
    }

    /// Input 1 holds unflushed evictions of frequent keys while input 0 only misses unique keys.
    fn decide_with_other_input(evictions: bool) -> LocalSink {
        let params = params(2);
        let mut local = local_sink(&params, 4);
        if evictions {
            // Runs of equal keys evict the key in their slot.
            let runs: Vec<i64> = (0..200)
                .flat_map(|_| (0..30).flat_map(|k| [k, k]))
                .collect();
            insert(&mut local, 1, &runs);
            assert!(local.hot_grouper(1).num_evictions() > 1000);
        }
        insert(&mut local, 0, &unique_keys(1000));
        local.hot_table_stats_per_input[0].window_target = 1000;
        local.after_hot_morsel(0, &params.reductions_per_input, false, 1024);
        local
    }

    #[test]
    fn test_bypass_unique_keys() {
        let local = decide_with_other_input(false);
        assert!(local.hot_table_stats_per_input[0].mode == HotTableMode::Bypass);
        assert_eq!(local.hot_table_size, 4);
    }

    #[test]
    fn test_pending_evictions_of_other_input_grow() {
        let local = decide_with_other_input(true);
        assert!(local.hot_table_size > 4);
        for stats in &local.hot_table_stats_per_input {
            assert!(stats.mode == HotTableMode::Hot);
            assert_eq!(stats.window_rows, 0);
        }
    }

    #[test]
    fn test_wait_while_missed_keys_fit() {
        let params = params(1);
        let mut local = local_sink(&params, 4);
        let max_size = 64;
        // Fill the largest table to more keys than the slots per key allow, but at most its
        // maximum load.
        let keys = unique_keys(100);
        let mut num_inserted = 0;
        let num_keys = |local: &mut LocalSink| {
            local.hot_grouper(0).num_groups() as usize + local.miss_sketch.estimate()
        };
        while num_keys(&mut local) as f64 <= max_size as f64 / HOT_TABLE_GROW_SLOTS_PER_KEY {
            insert(&mut local, 0, &keys[num_inserted..num_inserted + 1]);
            num_inserted += 1;
        }
        assert!(num_keys(&mut local) as f64 <= HOT_TABLE_GROW_MAX_LOAD * max_size as f64);
        local.hot_table_stats_per_input[0].window_target = num_inserted as u64;
        local.after_hot_morsel(0, &params.reductions_per_input, false, max_size);
        assert!(local.hot_table_stats_per_input[0].mode == HotTableMode::Hot);

        insert(&mut local, 0, &keys[num_inserted..]);
        local.after_hot_morsel(0, &params.reductions_per_input, false, max_size);
        assert!(local.hot_table_stats_per_input[0].mode == HotTableMode::Bypass);
    }
}
