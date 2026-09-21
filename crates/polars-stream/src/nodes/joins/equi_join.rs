use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_arrow::array::builder::ShareStrategy;
use polars_async::executor;
use polars_core::config;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::{ASYNC, RAYON};
use polars_core::schema::{Schema, SchemaExt};
use polars_defs::join::{JoinArgs, JoinBuildSide, JoinType, MaintainOrderJoin};
use polars_expr::hash_keys::HashKeys;
use polars_expr::idx_table::{IdxTable, new_idx_table};
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_ops::series::coalesce_columns;
use polars_plan::plans::options::RuntimeFilter;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::sparse_init_vec::SparseInitVec;
use polars_utils::{IdxSize, format_pl_smallstr};
use rayon::prelude::*;

use super::runtime_filter::{KeyFilterBuilder, RuntimeFilters};
use super::{
    BufferedStream, LOPSIDED_SAMPLE_FACTOR, emit_morsel_size, fold_sample, sample_sink, send_frames,
};
use crate::expression::StreamExpr;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_source::InMemorySourceNode;

/// Where one of the fused predicate's input columns is found at probe time.
struct FusedColumn {
    is_left: bool,
    /// Index into that side's payload.
    index: usize,
}

/// A match condition on top of the equi keys, applied to candidate pairs.
struct FusedPredicate {
    expr: StreamExpr,
    /// The columns the predicate reads, in the order it was compiled against.
    columns: Vec<FusedColumn>,
}

impl FusedPredicate {
    fn new(
        expr: StreamExpr,
        schema: &Schema,
        left_payload_schema: &Schema,
        right_payload_schema: &Schema,
    ) -> PolarsResult<Self> {
        let columns = schema
            .iter_names()
            .map(|name| {
                // Payload schemas are keyed by output name.
                let column = if let Some(index) = left_payload_schema.index_of(name) {
                    FusedColumn {
                        is_left: true,
                        index,
                    }
                } else if let Some(index) = right_payload_schema.index_of(name) {
                    FusedColumn {
                        is_left: false,
                        index,
                    }
                } else {
                    polars_bail!(
                        ColumnNotFound:
                        "fused predicate reads '{name}', which the join does not output"
                    )
                };
                Ok(column)
            })
            .try_collect_vec()?;

        Ok(Self { expr, columns })
    }

    /// Compacts the candidate pairs the predicate accepts to the front of both slices.
    ///
    /// The two match slices describe the same pairs positionally. Returns how many
    /// survived, which the caller truncates its own buffers to.
    async fn retain_matches(
        &self,
        left_is_build: bool,
        build_payload: &DataFrame,
        build_match: &mut [IdxSize],
        probe_payload: &DataFrame,
        probe_match: &mut [IdxSize],
        state: &ExecutionState,
    ) -> PolarsResult<usize> {
        let n = build_match.len();
        assert_eq!(n, probe_match.len());

        // `probe_subset` can hand back far more candidates than the morsel limit, so the
        // gathered predicate inputs are bounded here instead.
        let batch_size = get_ideal_morsel_size();
        let mut kept = 0;
        let mut start = 0;

        while start < n {
            let end = (start + batch_size).min(n);
            let len = end - start;

            let columns = self
                .columns
                .iter()
                .map(|column| {
                    let (payload, idxs) = if column.is_left == left_is_build {
                        (build_payload, &build_match[start..end])
                    } else {
                        (probe_payload, &probe_match[start..end])
                    };
                    // Payload columns already carry their output name.
                    unsafe { payload.columns()[column.index].take_slice_unchecked(idxs) }
                })
                .collect();
            let df = unsafe { DataFrame::new_unchecked(len, columns) };

            let mask = self
                .expr
                .evaluate_preserve_len_broadcast(&df, state)
                .await?;
            let mask = mask.as_materialized_series().bool()?.rechunk();
            let mask = mask.downcast_as_array();

            // A null is not a match.
            let keep = match mask.validity() {
                Some(validity) => mask.values() & validity,
                None => mask.values().clone(),
            };

            match keep.set_bits() {
                0 => {},
                // The batch survives whole, so it moves as one block.
                set if set == len => {
                    if kept != start {
                        build_match.copy_within(start..end, kept);
                        probe_match.copy_within(start..end, kept);
                    }
                    kept += len;
                },
                // SAFETY: We are in bounds
                _ => {
                    for i in keep.true_idx_iter() {
                        debug_assert!(start + i < end);
                        debug_assert!(kept <= start + i);
                        unsafe {
                            let build = *build_match.get_unchecked(start + i);
                            let probe = *probe_match.get_unchecked(start + i);
                            *build_match.get_unchecked_mut(kept) = build;
                            *probe_match.get_unchecked_mut(kept) = probe;
                        }
                        kept += 1;
                    }
                },
            }

            start = end;
        }

        Ok(kept)
    }
}

/// Rechunks `payload` the first time rows are gathered from it.
fn rechunk_once(payload: &mut DataFrame, rechunked: &mut bool) {
    if !*rechunked {
        payload.rechunk_mut();
        *rechunked = true;
    }
}

struct EquiJoinParams {
    left_is_build: Option<bool>,
    preserve_order_build: bool,
    preserve_order_probe: bool,
    left_key_schema: Arc<Schema>,
    left_key_selectors: Vec<StreamExpr>,
    right_key_selectors: Vec<StreamExpr>,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    left_payload_schema: Arc<Schema>,
    right_payload_schema: Arc<Schema>,
    args: JoinArgs,
    fused_predicate: Option<FusedPredicate>,
    // Key filters for the scans below the planned probe side, set once from a
    // complete sample of the planned build side or from its build.
    runtime_filters: RuntimeFilters,
    random_state: PlRandomState,
    sample_limit: usize,
}

/// The side a plan's build side names, if any.
fn build_side_left(side: Option<&JoinBuildSide>) -> Option<bool> {
    match side {
        Some(JoinBuildSide::ForceLeft | JoinBuildSide::PreferLeft) => Some(true),
        Some(JoinBuildSide::ForceRight | JoinBuildSide::PreferRight) => Some(false),
        None => None,
    }
}

impl EquiJoinParams {
    /// The side the plan asked to build from, if any.
    fn planned_build_left(&self) -> Option<bool> {
        build_side_left(self.args.build_side.as_ref())
    }

    /// Whether the build keys go to the runtime filters: the filters exist,
    /// were not set from a sample, and describe the side being built.
    fn publishes_runtime_filters(&self) -> bool {
        !self.runtime_filters.is_empty()
            && !self.runtime_filters.is_set()
            && self.left_is_build == self.planned_build_left()
    }

    /// Should we emit unmatched rows from the build side?
    fn emit_unmatched_build(&self) -> bool {
        if self.left_is_build.unwrap() {
            self.args.how == JoinType::Left || self.args.how == JoinType::Full
        } else {
            self.args.how == JoinType::Right || self.args.how == JoinType::Full
        }
    }

    /// Should we emit unmatched rows from the probe side?
    fn emit_unmatched_probe(&self) -> bool {
        if self.left_is_build.unwrap() {
            self.args.how == JoinType::Right || self.args.how == JoinType::Full
        } else {
            self.args.how == JoinType::Left || self.args.how == JoinType::Full
        }
    }
}

/// A payload selector contains for each column whether that column should be
/// included in the payload, and if yes with what name.
fn compute_payload_selector(
    this: &Schema,
    other: &Schema,
    this_key_schema: &Schema,
    other_key_schema: &Schema,
    output_schema: &Schema,
    is_left: bool,
    args: &JoinArgs,
) -> PolarsResult<Vec<Option<PlSmallStr>>> {
    let should_coalesce = args.should_coalesce();

    this.iter_names()
        .map(|c| {
            'create_and_return_selector: {
                let selector = if args.how == JoinType::Right {
                    if is_left {
                        if should_coalesce && this_key_schema.contains(c) {
                            // Coalesced to RHS output key.
                            None
                        } else {
                            Some(c.clone())
                        }
                    } else if !other.contains(c)
                        || (should_coalesce && other_key_schema.contains(c))
                    {
                        Some(c.clone())
                    } else {
                        break 'create_and_return_selector;
                    }
                } else if should_coalesce && this_key_schema.contains(c) {
                    if is_left {
                        Some(c.clone())
                    } else if args.how == JoinType::Full {
                        // We must keep the right-hand side keycols around for
                        // coalescing. Note that this bypasses the filter below because of the
                        // early return.
                        let key_idx = this_key_schema.index_of(c).unwrap();
                        let name = format_pl_smallstr!("__POLARS_COALESCE_KEYCOL_{key_idx}");
                        return Ok(Some(name));
                    } else {
                        None
                    }
                } else if !other.contains(c) || is_left {
                    Some(c.clone())
                } else {
                    break 'create_and_return_selector;
                };

                let selector = selector.filter(|name| output_schema.contains(name.as_str()));

                return Ok(selector);
            }

            let suffixed = format_pl_smallstr!("{}{}", c, args.suffix());
            if other.contains(&suffixed) {
                polars_bail!(
                    Duplicate:
                    "column with name '{suffixed}' already exists\n\n\
                    You may want to try:\n\
                    - renaming the column prior to joining\n\
                    - using the `suffix` parameter to specify \
                    a suffix different to the default one ('_right')"
                )
            }

            Ok(Some(suffixed))
        })
        .collect()
}

/// Fixes names and does coalescing of columns post-join.
fn postprocess_join(df: DataFrame, params: &EquiJoinParams) -> DataFrame {
    if params.args.how == JoinType::Full && params.args.should_coalesce() {
        // TODO: don't do string-based column lookups for each dataframe, pre-compute coalesce indices.
        let new_cols = df
            .columns()
            .iter()
            .filter_map(|c| {
                if let Some(key_idx) = params.left_key_schema.index_of(c.name()) {
                    let other = df
                        .column(&format_pl_smallstr!("__POLARS_COALESCE_KEYCOL_{key_idx}"))
                        .unwrap();
                    return Some(coalesce_columns(&[c.clone(), other.clone()]).unwrap());
                }

                if c.name().starts_with("__POLARS_COALESCE_KEYCOL") {
                    return None;
                }

                Some(c.clone())
            })
            .collect();

        unsafe { DataFrame::new_unchecked(df.height(), new_cols) }
    } else {
        df
    }
}

fn select_schema(schema: &Schema, selector: &[Option<PlSmallStr>]) -> Schema {
    schema
        .iter_fields()
        .zip(selector)
        .filter_map(|(f, name)| Some(f.with_name(name.clone()?)))
        .collect()
}

async fn select_keys(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<HashKeys> {
    Ok(select_keys_with_columns(df, key_selectors, params, state)
        .await?
        .0)
}

async fn select_key_columns(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let mut key_columns = Vec::new();
    for selector in key_selectors {
        key_columns.push(selector.evaluate(df, state).await?.into_column());
    }
    unsafe { DataFrame::new_unchecked_with_broadcast(df.height(), key_columns) }
}

async fn select_keys_with_columns(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<(HashKeys, DataFrame)> {
    let keys = select_key_columns(df, key_selectors, state).await?;
    let hash_keys = HashKeys::from_df(
        &keys,
        params.random_state.clone(),
        params.args.nulls_equal,
        false,
    );
    Ok((hash_keys, keys))
}

fn select_payload(df: DataFrame, selector: &[Option<PlSmallStr>]) -> DataFrame {
    let height = df.height();
    let new_cols = df
        .into_columns()
        .into_iter()
        .zip(selector)
        .filter_map(|(c, name)| Some(c.with_name(name.clone()?)))
        .collect();

    unsafe { DataFrame::new_unchecked(height, new_cols) }
}

fn estimate_cardinality(
    morsels: &[Morsel],
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<f64> {
    if morsels.is_empty() || params.sample_limit == 0 {
        return Ok(0.0);
    }
    let (sketch, rows) = fold_sample(
        morsels,
        params.sample_limit,
        CardinalitySketch::new,
        |mut sketch, df| {
            let hash_keys = ASYNC.block_on(select_keys(df, key_selectors, params, state))?;
            hash_keys.sketch_cardinality(&mut sketch);
            Ok(sketch)
        },
        |mut a, b| {
            a.combine(&b);
            a
        },
    )?;
    Ok(sketch.estimate() as f64 / rows as f64)
}

fn estimate_size_per_row(morsels: &[Morsel]) -> f64 {
    let mut total_size = 0;
    let mut total_height = 0;
    for m in morsels {
        total_size += m.df_blocking().estimated_size();
        total_height += m.height();
    }
    total_size as f64 / total_height as f64
}

#[derive(Default)]
struct SampleState {
    left: Vec<Morsel>,
    left_len: usize,
    right: Vec<Morsel>,
    right_len: usize,
    /// The only side being read: the preferred build side of a join with runtime
    /// filters, until it ends or reaches the sample limit. A side that ends is
    /// complete, so its key ranges are published before the other side is read.
    only_side: Option<bool>,
}

impl SampleState {
    fn len(&self, left: bool) -> usize {
        if left { self.left_len } else { self.right_len }
    }

    /// Whether a side is being read.
    fn is_open(&self, left: bool) -> bool {
        self.only_side.is_none_or(|only| only == left)
    }

    fn try_transition_to_build(
        &mut self,
        recv: &[PortState],
        params: &mut EquiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<Option<BuildState>> {
        if let Some(left) = self.only_side {
            let idx = if left { 0 } else { 1 };
            let len = self.len(left);
            if len >= params.sample_limit {
                if config::verbose() {
                    eprintln!("preferred build side reached the sample limit, sampling both sides");
                }
            } else if recv[idx] == PortState::Done {
                if config::verbose() {
                    eprintln!("preferred build side done with {len} rows, publishing its ranges");
                }
                self.publish_runtime_filters(left, params, state)?;
                // Nothing can match an empty side; the other side is never read.
                if len == 0 {
                    return Ok(Some(self.start_build(left, params, state, spill_ctx)?));
                }
            } else {
                return Ok(None);
            }
            self.only_side = None;
        }

        let left_saturated = self.left_len >= params.sample_limit;
        let right_saturated = self.right_len >= params.sample_limit;
        let left_done = recv[0] == PortState::Done || left_saturated;
        let right_done = recv[1] == PortState::Done || right_saturated;
        #[expect(clippy::nonminimal_bool)]
        let stop_sampling = (left_done && right_done)
            || (left_done && self.right_len >= LOPSIDED_SAMPLE_FACTOR * self.left_len)
            || (right_done && self.left_len >= LOPSIDED_SAMPLE_FACTOR * self.right_len);
        if !stop_sampling {
            return Ok(None);
        }

        if config::verbose() {
            eprintln!(
                "choosing build side, sample lengths are: {} vs. {}",
                self.left_len, self.right_len
            );
        }

        let estimate_cardinalities = || {
            let left_cardinality = estimate_cardinality(
                &self.left,
                &params.left_key_selectors,
                params,
                &state.in_memory_exec_state,
            )?;
            let right_cardinality = estimate_cardinality(
                &self.right,
                &params.right_key_selectors,
                params,
                &state.in_memory_exec_state,
            )?;
            if config::verbose() {
                eprintln!(
                    "estimated cardinalities are: {left_cardinality} vs. {right_cardinality}"
                );
            }
            PolarsResult::Ok((left_cardinality, right_cardinality))
        };

        let left_is_build = match (left_saturated, right_saturated) {
            // Don't bother estimating cardinality, just choose smaller side as
            // we have everything in-memory anyway.
            (false, false) => self.left_len < self.right_len,

            // Choose the unsaturated side, the saturated side could be
            // arbitrarily big.
            (false, true) => true,
            (true, false) => false,

            (true, true) => {
                // A preference with runtime filters does not decide; the sample does.
                match params.args.build_side {
                    Some(JoinBuildSide::PreferLeft) if params.runtime_filters.is_empty() => true,
                    Some(JoinBuildSide::PreferRight) if params.runtime_filters.is_empty() => false,
                    Some(JoinBuildSide::ForceLeft | JoinBuildSide::ForceRight) => unreachable!(),
                    _ => {
                        // Estimate cardinality and choose smaller, minimizing expected memory usage.
                        let (lc, rc) = estimate_cardinalities()?;
                        let ls = estimate_size_per_row(&self.left);
                        let rs = estimate_size_per_row(&self.right);
                        lc * ls < rc * rs
                    },
                }
            },
        };

        if config::verbose() {
            eprintln!(
                "build side chosen: {}",
                if left_is_build { "left" } else { "right" }
            );
        }

        Ok(Some(self.start_build(
            left_is_build,
            params,
            state,
            spill_ctx,
        )?))
    }

    /// Hand the keys of a completely sampled side to the runtime filters. The
    /// filter holds whichever side is built later, as no key of the other side
    /// outside it can match.
    fn publish_runtime_filters(
        &self,
        left: bool,
        params: &mut EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let (morsels, key_selectors) = if left {
            (&self.left, &params.left_key_selectors)
        } else {
            (&self.right, &params.right_key_selectors)
        };
        let new_builders = || params.runtime_filters.new_builders();
        let builders = RAYON.install(|| {
            morsels
                .par_iter()
                .try_fold(new_builders, |mut builders, morsel| {
                    let df = morsel.df_blocking();
                    let keys = ASYNC.block_on(select_key_columns(
                        &df,
                        key_selectors,
                        &state.in_memory_exec_state,
                    ))?;
                    params.runtime_filters.extend(&keys, &mut builders)?;
                    PolarsResult::Ok(builders)
                })
                .try_reduce(new_builders, |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        a.merge(b);
                    }
                    Ok(a)
                })
        })?;
        params.runtime_filters.publish(builders);
        Ok(())
    }

    /// Start building from `left_is_build`, feeding it the morsels sampled from
    /// that side; the other side's samples are probed first later.
    fn start_build(
        &mut self,
        left_is_build: bool,
        params: &mut EquiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<BuildState> {
        params.left_is_build = Some(left_is_build);
        let mut sampled_build_morsels = BufferedStream::new(
            "equi-join-left-sample".into(),
            core::mem::take(&mut self.left),
            MorselSeq::default(),
        );
        let mut sampled_probe_morsels = BufferedStream::new(
            "equi-join-right-sample".into(),
            core::mem::take(&mut self.right),
            MorselSeq::default(),
        );
        if !left_is_build {
            core::mem::swap(&mut sampled_build_morsels, &mut sampled_probe_morsels);
        }

        let partitioner = HashPartitioner::new(state.num_pipelines, 0);
        let mut build_state = BuildState::new(
            state.num_pipelines,
            state.num_pipelines,
            params,
            sampled_probe_morsels,
        );

        // Simulate the sample build morsels flowing into the build side.
        if !sampled_build_morsels.is_empty() {
            executor::task_scope(|scope| {
                let mut join_handles = Vec::new();
                let receivers = sampled_build_morsels
                    .reinsert(state.num_pipelines, None, scope, &mut join_handles)
                    .unwrap();

                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::partition_and_sink(
                            recv,
                            local_builder,
                            partitioner.clone(),
                            params,
                            state,
                            spill_ctx,
                        ),
                    ));
                }

                ASYNC.block_in_place_on(async move {
                    for handle in join_handles {
                        handle.await?;
                    }
                    PolarsResult::Ok(())
                })
            })?;
        }

        Ok(build_state)
    }
}

#[derive(Default)]
struct LocalBuilder {
    // The complete list of morsels and their computed hashes seen by this builder.
    morsels: Vec<(MorselSeq, SpillFrame, HashKeys)>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // The key of each runtime filter seen by this builder.
    key_filters: Vec<KeyFilterBuilder>,

    // morsel_idxs_values_per_p[p][start..stop] contains the offsets into morsels[i]
    // for partition p, where start, stop are:
    // let start = morsel_idxs_offsets[i * num_partitions + p];
    // let stop = morsel_idxs_offsets[(i + 1) * num_partitions + p];
    morsel_idxs_values_per_p: Vec<Vec<IdxSize>>,
    morsel_idxs_offsets_per_p: Vec<usize>,
}

struct BuildState {
    local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    fn new(
        num_pipelines: usize,
        num_partitions: usize,
        params: &EquiJoinParams,
        sampled_probe_morsels: BufferedStream,
    ) -> Self {
        let local_builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                morsels: Vec::new(),
                sketch_per_p: vec![CardinalitySketch::default(); num_partitions],
                key_filters: if params.publishes_runtime_filters() {
                    params.runtime_filters.new_builders()
                } else {
                    Vec::new()
                },
                morsel_idxs_values_per_p: vec![Vec::new(); num_partitions],
                morsel_idxs_offsets_per_p: vec![0; num_partitions],
            })
            .collect();
        Self {
            local_builders,
            sampled_probe_morsels,
        }
    }

    async fn partition_and_sink(
        mut recv: PortReceiver,
        local: &mut LocalBuilder,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<()> {
        let track_unmatchable = params.emit_unmatched_build();
        let (key_selectors, payload_selector);
        if params.left_is_build.unwrap() {
            payload_selector = &params.left_payload_select;
            key_selectors = &params.left_key_selectors;
        } else {
            payload_selector = &params.right_payload_select;
            key_selectors = &params.right_key_selectors;
        };

        let publishes_runtime_filters = params.publishes_runtime_filters();
        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload. We must rechunk the payload for
            // later gathers.
            let df = morsel.df().await;
            let (hash_keys, keys) =
                select_keys_with_columns(&df, key_selectors, params, &state.in_memory_exec_state)
                    .await?;
            if publishes_runtime_filters {
                params
                    .runtime_filters
                    .extend(&keys, &mut local.key_filters)?;
            }
            let mut payload = select_payload(df.clone(), payload_selector);
            payload.rechunk_mut();

            hash_keys.gen_idxs_per_partition(
                &partitioner,
                &mut local.morsel_idxs_values_per_p,
                &mut local.sketch_per_p,
                track_unmatchable,
            );

            local
                .morsel_idxs_offsets_per_p
                .extend(local.morsel_idxs_values_per_p.iter().map(|vp| vp.len()));
            let sf = SpillFrame::new(payload, spill_ctx).await;
            local.morsels.push((morsel.seq(), sf, hash_keys));
        }
        Ok(())
    }

    /// Whether no row was built, sampled morsels included.
    fn is_empty(&self) -> bool {
        self.local_builders
            .iter()
            .all(|b| b.morsels.iter().all(|(_, _, keys)| keys.is_empty()))
    }

    /// Hand every build key to the runtime filters. Filters set from a sample
    /// keep their value, whichever side is built; filters of a side that was
    /// not built get a predicate that skips nothing.
    fn publish_runtime_filters(&mut self, params: &mut EquiJoinParams) {
        if !params.publishes_runtime_filters() {
            params.runtime_filters.publish_nothing();
            return;
        }
        let locals = self
            .local_builders
            .iter_mut()
            .map(|l| std::mem::take(&mut l.key_filters));
        params.runtime_filters.publish_merged(locals);
    }

    fn finalize_ordered(&mut self, params: &EquiJoinParams, table: &dyn IdxTable) -> ProbeState {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let probe_tables: SparseInitVec<ProbeTable> = SparseInitVec::with_capacity(num_partitions);

        RAYON.scope(|s| {
            for p in 0..num_partitions {
                let probe_tables = &probe_tables;
                s.spawn(move |_| {
                    // TODO: every thread does an identical linearize, we can do a single parallel one.
                    let mut kmerge = BinaryHeap::with_capacity(local_builders.len());
                    let mut cur_idx_per_loc = vec![0; local_builders.len()];

                    // Compute cardinality estimate and total amount of
                    // payload for this partition, and initialize k-way merge.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for (l_idx, l) in local_builders.iter().enumerate() {
                        let Some((seq, _, _)) = l.morsels.first() else {
                            continue;
                        };
                        kmerge.push(Priority(Reverse(seq), l_idx));

                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.morsel_idxs_offsets_per_p.len();
                        payload_rows +=
                            l.morsel_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_table = table.new_empty();
                    p_table.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(payload_schema.clone());
                    p_payload.reserve(payload_rows);

                    let mut p_row_positions = Vec::new();
                    if track_unmatchable {
                        p_row_positions.reserve(payload_rows);
                    }

                    // Linearize and build.
                    unsafe {
                        // Row offset of the current morsel in the full build input.
                        let mut morsel_row_offset = 0u64;
                        while let Some(Priority(Reverse(_seq), l_idx)) = kmerge.pop() {
                            let l = local_builders.get_unchecked(l_idx);
                            let idx_in_l = *cur_idx_per_loc.get_unchecked(l_idx);
                            *cur_idx_per_loc.get_unchecked_mut(l_idx) += 1;
                            if let Some((next_seq, _, _)) = l.morsels.get(idx_in_l + 1) {
                                kmerge.push(Priority(Reverse(next_seq), l_idx));
                            }

                            let (_mseq, sf, keys) = l.morsels.get_unchecked(idx_in_l);
                            let payload = sf.get_blocking();
                            let p_morsel_idxs_start =
                                l.morsel_idxs_offsets_per_p[idx_in_l * num_partitions + p];
                            let p_morsel_idxs_stop =
                                l.morsel_idxs_offsets_per_p[(idx_in_l + 1) * num_partitions + p];
                            let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                [p_morsel_idxs_start..p_morsel_idxs_stop];
                            p_table.insert_keys_subset(keys, p_morsel_idxs, track_unmatchable);
                            p_payload.gather_extend(&payload, p_morsel_idxs, ShareStrategy::Never);

                            if track_unmatchable {
                                #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
                                p_row_positions.extend(
                                    p_morsel_idxs.iter().map(|i| morsel_row_offset + *i as u64),
                                );
                            }
                            morsel_row_offset += payload.height() as u64;
                        }
                    }

                    probe_tables
                        .try_set(
                            p,
                            ProbeTable {
                                hash_table: p_table,
                                payload: p_payload.freeze(),
                                row_positions: p_row_positions,
                            },
                        )
                        .ok()
                        .unwrap();
                });
            }
        });

        ProbeState {
            table_per_partition: probe_tables.try_assume_init().ok().unwrap(),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
            unordered_morsel_seq: AtomicU64::new(0),
        }
    }

    fn finalize_unordered(&mut self, params: &EquiJoinParams, table: &dyn IdxTable) -> ProbeState {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        // To reduce maximum memory usage we want to drop the morsels
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let morsels_per_local_builder = self
            .local_builders
            .iter_mut()
            .map(|b| Arc::new(core::mem::take(&mut b.morsels)))
            .collect_vec();
        let (morsel_drop_q_send, morsel_drop_q_recv) =
            async_channel::bounded(morsels_per_local_builder.len());
        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let probe_tables: SparseInitVec<ProbeTable> = SparseInitVec::with_capacity(num_partitions);

        executor::task_scope(|s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_morsels_per_local_builder = Arc::new(morsels_per_local_builder);
            let mut join_handles = Vec::new();
            for p in 0..num_partitions {
                let arc_morsels_per_local_builder = Arc::clone(&arc_morsels_per_local_builder);
                let morsel_drop_q_send = morsel_drop_q_send.clone();
                let morsel_drop_q_recv = morsel_drop_q_recv.clone();
                let probe_tables = &probe_tables;
                join_handles.push(s.spawn_task(TaskPriority::High, async move {
                    // Extract from outer arc and drop outer arc.
                    let morsels_per_local_builder =
                        Arc::unwrap_or_clone(arc_morsels_per_local_builder);

                    // Compute cardinality estimate and total amount of
                    // payload for this partition.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for l in local_builders {
                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.morsel_idxs_offsets_per_p.len();
                        payload_rows +=
                            l.morsel_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_table = table.new_empty();
                    p_table.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(payload_schema.clone());
                    p_payload.reserve(payload_rows);

                    // Build.
                    let mut skip_drop_attempt = false;
                    for (l, l_morsels) in local_builders.iter().zip(morsels_per_local_builder) {
                        // Try to help with dropping the processed morsels.
                        if !skip_drop_attempt {
                            drop(morsel_drop_q_recv.try_recv());
                        }

                        for (i, morsel) in l_morsels.iter().enumerate() {
                            let (_mseq, sf, keys) = morsel;
                            let payload = sf.get().await;
                            unsafe {
                                let p_morsel_idxs_start =
                                    l.morsel_idxs_offsets_per_p[i * num_partitions + p];
                                let p_morsel_idxs_stop =
                                    l.morsel_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                    [p_morsel_idxs_start..p_morsel_idxs_stop];
                                p_table.insert_keys_subset(keys, p_morsel_idxs, track_unmatchable);
                                p_payload.gather_extend(
                                    &payload,
                                    p_morsel_idxs,
                                    ShareStrategy::Never,
                                );
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_morsels) {
                            // If we're the last thread to process this set of morsels we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(morsel_drop_q_send.try_send(l));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // We're done, help others out by doing drops.
                    drop(morsel_drop_q_send); // So we don't deadlock trying to receive from ourselves.
                    while let Ok(l_morsels) = morsel_drop_q_recv.recv().await {
                        drop(l_morsels);
                    }

                    probe_tables
                        .try_set(
                            p,
                            ProbeTable {
                                hash_table: p_table,
                                payload: p_payload.freeze(),
                                row_positions: Vec::new(),
                            },
                        )
                        .ok()
                        .unwrap();
                }));
            }

            // Drop outer arc after spawning each thread so the inner arcs
            // can get dropped as soon as they're processed. We also have to
            // drop the drop queue sender so we don't deadlock waiting for it
            // to end.
            drop(arc_morsels_per_local_builder);
            drop(morsel_drop_q_send);

            ASYNC.block_in_place_on(async move {
                for handle in join_handles {
                    handle.await;
                }
            });
        });

        ProbeState {
            table_per_partition: probe_tables.try_assume_init().ok().unwrap(),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
            unordered_morsel_seq: AtomicU64::new(0),
        }
    }
}

struct ProbeTable {
    hash_table: Box<dyn IdxTable>,
    payload: DataFrame,
    // Position of each payload row in the full build input. Only filled for
    // ordered joins that emit unmatched build rows.
    row_positions: Vec<u64>,
}

struct ProbeState {
    table_per_partition: Vec<ProbeTable>,
    max_seq_sent: MorselSeq,
    sampled_probe_morsels: BufferedStream,

    // For unordered joins we relabel output morsels to speed up the linearizer.
    unordered_morsel_seq: AtomicU64,
}

impl ProbeState {
    /// Returns the max morsel sequence sent.
    async fn partition_and_probe(
        mut recv: PortReceiver,
        mut send: PortSender,
        partitions: &[ProbeTable],
        unordered_morsel_seq: &AtomicU64,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<MorselSeq> {
        // TODO: shuffle after partitioning and keep probe tables thread-local.
        let mut partition_idxs = vec![Vec::new(); partitioner.num_partitions()];
        let mut probe_partitions = Vec::new();
        let mut materialized_idxsize_range = Vec::new();
        let mut table_match = Vec::new();
        let mut probe_match = Vec::new();
        let mut max_seq = MorselSeq::default();

        let probe_limit = get_ideal_morsel_size() as IdxSize;
        let mark_matches = params.emit_unmatched_build();
        let emit_unmatched = params.emit_unmatched_probe();
        assert!(params.fused_predicate.is_none() || (!mark_matches && !emit_unmatched));

        let (key_selectors, payload_selector, build_payload_schema, probe_payload_schema);
        if params.left_is_build.unwrap() {
            key_selectors = &params.right_key_selectors;
            payload_selector = &params.right_payload_select;
            build_payload_schema = &params.left_payload_schema;
            probe_payload_schema = &params.right_payload_schema;
        } else {
            key_selectors = &params.left_key_selectors;
            payload_selector = &params.left_payload_select;
            build_payload_schema = &params.right_payload_schema;
            probe_payload_schema = &params.left_payload_schema;
        };

        let mut build_out = DataFrameBuilder::new(build_payload_schema.clone());
        let mut probe_out = DataFrameBuilder::new(probe_payload_schema.clone());

        // A simple estimate used to size reserves.
        let mut selectivity_estimate = 1.0;
        let mut selectivity_estimate_confidence = 0.0;

        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload.
            let (sf, in_seq, src_token, wait_token) = morsel.into_inner();
            let df = sf.into_df().await;

            let df_height = df.height();
            if df_height == 0 {
                continue;
            }

            let hash_keys =
                select_keys(&df, key_selectors, params, &state.in_memory_exec_state).await?;
            let mut payload = select_payload(df, payload_selector);
            let mut payload_rechunked = false;
            let mut total_matches = 0;

            // Use selectivity estimate to reserve for morsel builders.
            let max_match_per_key_est = (selectivity_estimate * 1.2) as usize + 16;
            let out_est_size = ((selectivity_estimate * 1.2 * df_height as f64) as usize)
                .min(probe_limit as usize);
            build_out.reserve(out_est_size + max_match_per_key_est);

            unsafe {
                let mut new_morsel =
                    |build: &mut DataFrameBuilder, probe: &mut DataFrameBuilder| {
                        let mut build_df = build.freeze_reset();
                        let mut probe_df = probe.freeze_reset();
                        let out_df = if params.left_is_build.unwrap() {
                            build_df.hstack_mut_unchecked(probe_df.columns());
                            build_df
                        } else {
                            probe_df.hstack_mut_unchecked(build_df.columns());
                            probe_df
                        };
                        let out_df = postprocess_join(out_df, params);
                        let out_seq = if params.preserve_order_probe {
                            in_seq
                        } else {
                            MorselSeq::new(unordered_morsel_seq.fetch_add(1, Ordering::Relaxed))
                        };
                        max_seq = out_seq;
                        Morsel::new_unregistered(out_df, out_seq, src_token.clone())
                    };

                if params.preserve_order_probe {
                    // To preserve the order we can't do bulk probes per partition and must follow
                    // the order of the probe morsel. We can still group probes that are
                    // consecutively on the same partition.
                    probe_partitions.clear();
                    hash_keys.gen_partitions(&partitioner, &mut probe_partitions, emit_unmatched);

                    let mut probe_group_start = 0;
                    while probe_group_start < probe_partitions.len() {
                        let p_idx = probe_partitions[probe_group_start];
                        let mut probe_group_end = probe_group_start + 1;
                        while probe_partitions.get(probe_group_end) == Some(&p_idx) {
                            probe_group_end += 1;
                        }
                        let Some(p) = partitions.get(p_idx as usize) else {
                            probe_group_start = probe_group_end;
                            continue;
                        };

                        materialized_idxsize_range.extend(
                            materialized_idxsize_range.len() as IdxSize..probe_group_end as IdxSize,
                        );

                        while probe_group_start < probe_group_end {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            table_match.clear();
                            probe_group_start += p.hash_table.probe_subset(
                                &hash_keys,
                                &materialized_idxsize_range[probe_group_start..probe_group_end],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize
                                || probe_group_start == probe_partitions.len()
                            {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                let out_len = probe_match.len();
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                if probe_group_end != probe_partitions.len() {
                                    // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                    // matches and just do a large reserve.
                                    let old_est = probe_limit as usize + max_match_per_key_est;
                                    build_out.reserve(old_est.max(out_len + 16));
                                }
                            }
                        }
                    }
                } else {
                    // Partition and probe the tables.
                    for p in partition_idxs.iter_mut() {
                        p.clear();
                    }
                    hash_keys.gen_idxs_per_partition(
                        &partitioner,
                        &mut partition_idxs,
                        &mut [],
                        emit_unmatched,
                    );

                    for (p, idxs_in_p) in partitions.iter().zip(&partition_idxs) {
                        let mut offset = 0;
                        while offset < idxs_in_p.len() {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            let probe_start = probe_match.len();
                            table_match.clear();
                            offset += p.hash_table.probe_subset(
                                &hash_keys,
                                &idxs_in_p[offset..],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if let Some(fused_predicate) = &params.fused_predicate
                                && !table_match.is_empty()
                            {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                let kept = fused_predicate
                                    .retain_matches(
                                        params.left_is_build.unwrap(),
                                        &p.payload,
                                        &mut table_match,
                                        &payload,
                                        &mut probe_match[probe_start..],
                                        &state.in_memory_exec_state,
                                    )
                                    .await?;
                                table_match.truncate(kept);
                                probe_match.truncate(probe_start + kept);
                            }

                            if table_match.is_empty() {
                                continue;
                            }
                            total_matches += table_match.len();

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                let out_len = probe_match.len();
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                // matches and just do a large reserve.
                                let old_est = probe_limit as usize + max_match_per_key_est;
                                build_out.reserve(old_est.max(out_len + 16));
                            }
                        }
                    }
                }

                if !probe_match.is_empty() {
                    rechunk_once(&mut payload, &mut payload_rechunked);
                    probe_out.gather_extend(&payload, &probe_match, ShareStrategy::Always);
                    probe_match.clear();
                    let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                    if send.send(out_morsel).await.is_err() {
                        return Ok(max_seq);
                    }
                }
            }

            drop(wait_token);

            // Move selectivity estimate a bit towards latest value. Allows rapid changes at first.
            // TODO: implement something more re-usable and robust.
            selectivity_estimate = selectivity_estimate_confidence * selectivity_estimate
                + (1.0 - selectivity_estimate_confidence)
                    * (total_matches as f64 / df_height as f64);
            selectivity_estimate_confidence = (selectivity_estimate_confidence + 0.1).min(0.8);
        }

        Ok(max_seq)
    }

    fn ordered_unmatched(&mut self, params: &EquiJoinParams) -> DataFrame {
        // TODO: parallelize this operator.

        let build_payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        let mut unmarked_idxs = Vec::new();
        let mut row_positions = Vec::new();

        unsafe {
            let mut build_out = DataFrameBuilder::new(build_payload_schema.clone());
            for p in &self.table_per_partition {
                p.hash_table
                    .unmarked_keys(&mut unmarked_idxs, 0, IdxSize::MAX);
                row_positions.extend(
                    unmarked_idxs
                        .iter()
                        .map(|i| *p.row_positions.get_unchecked(*i as usize)),
                );
                build_out.gather_extend(
                    &p.payload,
                    &unmarked_idxs,
                    ShareStrategy::Never, // Don't keep entire table alive for unmatched indices.
                );
            }

            let mut perm: Vec<IdxSize> = (0..row_positions.len() as IdxSize).collect();
            perm.sort_unstable_by_key(|i| *row_positions.get_unchecked(*i as usize));

            let mut build_df = build_out.freeze().take_slice_unchecked(&perm);
            let out_df = if params.left_is_build.unwrap() {
                let probe_df =
                    DataFrame::full_null(&params.right_payload_schema, build_df.height());
                build_df.hstack_mut_unchecked(probe_df.columns());
                build_df
            } else {
                let mut probe_df =
                    DataFrame::full_null(&params.left_payload_schema, build_df.height());
                probe_df.hstack_mut_unchecked(build_df.columns());
                probe_df
            };
            postprocess_join(out_df, params)
        }
    }
}

impl Drop for ProbeState {
    fn drop(&mut self) {
        RAYON.install(|| {
            // Parallel drop as the state might be quite big.
            self.table_per_partition.par_drain(..).for_each(drop);
        })
    }
}

struct EmitUnmatchedState {
    partitions: Vec<ProbeTable>,
    active_partition_idx: usize,
    offset_in_active_p: usize,
    morsel_seq: MorselSeq,
}

impl EmitUnmatchedState {
    async fn emit_unmatched(
        &mut self,
        send: PortSender,
        params: &EquiJoinParams,
        num_pipelines: usize,
    ) -> PolarsResult<()> {
        let total_len: usize = self
            .partitions
            .iter()
            .map(|p| p.hash_table.num_keys() as usize)
            .sum();
        let morsel_size = emit_morsel_size(total_len, num_pipelines);

        let mut unmarked_idxs = Vec::new();
        let partitions = &self.partitions;
        let active_partition_idx = &mut self.active_partition_idx;
        let offset_in_active_p = &mut self.offset_in_active_p;
        send_frames(send, &mut self.morsel_seq, || {
            while let Some(p) = partitions.get(*active_partition_idx) {
                // Generate a chunk of unmarked key indices.
                *offset_in_active_p += p.hash_table.unmarked_keys(
                    &mut unmarked_idxs,
                    *offset_in_active_p as IdxSize,
                    morsel_size as IdxSize,
                ) as usize;
                if unmarked_idxs.is_empty() {
                    *active_partition_idx += 1;
                    *offset_in_active_p = 0;
                    continue;
                }

                // Gather and create full-null counterpart.
                let out_df = unsafe {
                    let mut build_df = p.payload.take_slice_unchecked_impl(&unmarked_idxs, false);
                    let len = build_df.height();
                    if params.left_is_build.unwrap() {
                        let probe_df = DataFrame::full_null(&params.right_payload_schema, len);
                        build_df.hstack_mut_unchecked(probe_df.columns());
                        build_df
                    } else {
                        let mut probe_df = DataFrame::full_null(&params.left_payload_schema, len);
                        probe_df.hstack_mut_unchecked(build_df.columns());
                        probe_df
                    }
                };
                return Some(postprocess_join(out_df, params));
            }
            None
        })
        .await
    }
}

enum EquiJoinState {
    Sample(SampleState),
    Build(BuildState),
    Probe(ProbeState),
    EmitUnmatchedBuild(EmitUnmatchedState),
    EmitUnmatchedBuildInOrder(InMemorySourceNode),
    Done,
}

pub struct EquiJoinNode {
    state: EquiJoinState,
    params: EquiJoinParams,
    table: Box<dyn IdxTable>,
    spill_ctx: MostRecentSpillContext,
}

impl EquiJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        left_key_schema: Arc<Schema>,
        right_key_schema: Arc<Schema>,
        unique_key_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        fused_predicate: Option<(StreamExpr, Arc<Schema>)>,
        runtime_filters: Vec<RuntimeFilter>,
        args: JoinArgs,
        num_pipelines: usize,
    ) -> PolarsResult<Self> {
        let sample_limit: usize = polars_config::config()
            .join_sample_limit()
            .try_into()
            .unwrap();
        let left_is_build = match args.maintain_order {
            MaintainOrderJoin::None => match args.build_side {
                Some(JoinBuildSide::ForceLeft) => Some(true),
                Some(JoinBuildSide::ForceRight) => Some(false),
                Some(JoinBuildSide::PreferLeft) | Some(JoinBuildSide::PreferRight) | None => {
                    if sample_limit == 0 {
                        Some(args.build_side != Some(JoinBuildSide::PreferRight))
                    } else {
                        None
                    }
                },
            },
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => {
                if args.build_side == Some(JoinBuildSide::ForceLeft) {
                    polars_warn!("can't force left build-side with left-maintaining cross-join");
                }
                Some(false)
            },
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => {
                if args.build_side == Some(JoinBuildSide::ForceRight) {
                    polars_warn!("can't force right build-side with right-maintaining cross-join");
                }
                Some(true)
            },
        };

        let preserve_order_probe = args.maintain_order != MaintainOrderJoin::None;
        let preserve_order_build = matches!(
            args.maintain_order,
            MaintainOrderJoin::LeftRight | MaintainOrderJoin::RightLeft
        );

        let left_payload_select = compute_payload_selector(
            &left_input_schema,
            &right_input_schema,
            &left_key_schema,
            &right_key_schema,
            &output_schema,
            true,
            &args,
        )?;
        let right_payload_select = compute_payload_selector(
            &right_input_schema,
            &left_input_schema,
            &right_key_schema,
            &left_key_schema,
            &output_schema,
            false,
            &args,
        )?;

        // A filter is only published for the side the plan named.
        debug_assert!(runtime_filters.is_empty() || args.build_side.is_some());
        let runtime_filters = RuntimeFilters::new(runtime_filters, &unique_key_schema);

        let left_payload_schema = Arc::new(select_schema(&left_input_schema, &left_payload_select));
        let right_payload_schema =
            Arc::new(select_schema(&right_input_schema, &right_payload_select));

        // Unmatched-row bookkeeping would count a candidate before the fused predicate runs, and
        // the ordered probe would evaluate it a partition group at a time.
        assert!(
            fused_predicate.is_none()
                || (matches!(args.how, JoinType::Inner)
                    && args.maintain_order == MaintainOrderJoin::None)
        );
        let fused_predicate = fused_predicate
            .map(|(expr, schema)| {
                FusedPredicate::new(expr, &schema, &left_payload_schema, &right_payload_schema)
            })
            .transpose()?;

        let params = EquiJoinParams {
            left_is_build,
            preserve_order_build,
            preserve_order_probe,
            left_key_schema,
            left_key_selectors,
            right_key_selectors,
            left_payload_select,
            right_payload_select,
            left_payload_schema,
            right_payload_schema,
            args,
            fused_predicate,
            runtime_filters,
            random_state: PlRandomState::default(),
            sample_limit,
        };

        let state = if left_is_build.is_some() {
            EquiJoinState::Build(BuildState::new(
                num_pipelines,
                num_pipelines,
                &params,
                BufferedStream::default(),
            ))
        } else {
            // A forced side never samples, so this names a preferred one.
            let only_side = if params.runtime_filters.is_empty() {
                None
            } else {
                params.planned_build_left()
            };
            EquiJoinState::Sample(SampleState {
                only_side,
                ..Default::default()
            })
        };

        Ok(Self {
            state,
            params,
            table: new_idx_table(unique_key_schema),
            spill_ctx: MostRecentSpillContext::new("equi-join".into()),
        })
    }
}

impl ComputeNode for EquiJoinNode {
    fn name(&self) -> &str {
        "equi-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            self.state = EquiJoinState::Done;
        }

        // If we are sampling and both sides are done/filled, transition to building.
        if let EquiJoinState::Sample(sample_state) = &mut self.state {
            if let Some(build_state) = sample_state.try_transition_to_build(
                recv,
                &mut self.params,
                state,
                &self.spill_ctx,
            )? {
                self.state = EquiJoinState::Build(build_state);
            }
        }

        let build_idx = if self.params.left_is_build == Some(true) {
            0
        } else {
            1
        };
        let probe_idx = 1 - build_idx;

        // If we are building and the build input is done, transition to probing.
        // An inner join with nothing to probe against is done without reading
        // the probe side.
        if let EquiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                build_state.publish_runtime_filters(&mut self.params);
                self.state = if self.params.args.how == JoinType::Inner && build_state.is_empty() {
                    EquiJoinState::Done
                } else if self.params.preserve_order_build {
                    EquiJoinState::Probe(build_state.finalize_ordered(&self.params, &*self.table))
                } else {
                    EquiJoinState::Probe(build_state.finalize_unordered(&self.params, &*self.table))
                };
            }
        }

        // If we are probing and the probe input is done, emit unmatched if
        // necessary, otherwise we're done.
        if let EquiJoinState::Probe(probe_state) = &mut self.state {
            let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
            if samples_consumed && recv[probe_idx] == PortState::Done {
                if self.params.emit_unmatched_build() {
                    if self.params.preserve_order_build {
                        let unmatched = probe_state.ordered_unmatched(&self.params);
                        let src = InMemorySourceNode::new(
                            Arc::new(unmatched),
                            probe_state.max_seq_sent.successor(),
                        );
                        self.state = EquiJoinState::EmitUnmatchedBuildInOrder(src);
                    } else {
                        self.state = EquiJoinState::EmitUnmatchedBuild(EmitUnmatchedState {
                            partitions: core::mem::take(&mut probe_state.table_per_partition),
                            active_partition_idx: 0,
                            offset_in_active_p: 0,
                            morsel_seq: probe_state.max_seq_sent.successor(),
                        });
                    }
                } else {
                    self.state = EquiJoinState::Done;
                }
            }
        }

        // Finally, check if we are done emitting unmatched keys.
        if let EquiJoinState::EmitUnmatchedBuild(emit_state) = &mut self.state {
            if emit_state.active_partition_idx >= emit_state.partitions.len() {
                self.state = EquiJoinState::Done;
            }
        }

        match &mut self.state {
            EquiJoinState::Sample(sample_state) => {
                send[0] = PortState::Blocked;
                for (idx, left) in [(0, true), (1, false)] {
                    if recv[idx] == PortState::Done {
                        continue;
                    }
                    let open = sample_state.is_open(left);
                    recv[idx] = if open && sample_state.len(left) < self.params.sample_limit {
                        PortState::Ready
                    } else {
                        PortState::Blocked
                    };
                }
            },
            EquiJoinState::Build(_) => {
                send[0] = PortState::Blocked;
                if recv[build_idx] != PortState::Done {
                    recv[build_idx] = PortState::Ready;
                }
                if recv[probe_idx] != PortState::Done {
                    recv[probe_idx] = PortState::Blocked;
                }
            },
            EquiJoinState::Probe(probe_state) => {
                if recv[probe_idx] != PortState::Done {
                    core::mem::swap(&mut send[0], &mut recv[probe_idx]);
                } else {
                    let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
                    send[0] = if samples_consumed {
                        PortState::Done
                    } else {
                        PortState::Ready
                    };
                }
                recv[build_idx] = PortState::Done;
            },
            EquiJoinState::EmitUnmatchedBuild(_) => {
                send[0] = PortState::Ready;
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
            },
            EquiJoinState::EmitUnmatchedBuildInOrder(src_node) => {
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
                src_node.update_state(&mut [], &mut send[0..1], state)?;
                if send[0] == PortState::Done {
                    self.state = EquiJoinState::Done;
                }
            },
            EquiJoinState::Done => {
                send[0] = PortState::Done;
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(
            self.state,
            EquiJoinState::Sample { .. } | EquiJoinState::Build { .. }
        )
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);

        let build_idx = if self.params.left_is_build == Some(true) {
            0
        } else {
            1
        };
        let probe_idx = 1 - build_idx;

        match &mut self.state {
            EquiJoinState::Sample(sample_state) => {
                assert!(send_ports[0].is_none());
                // A side without a port is done, unless it is not being read.
                let final_len = |left: bool| {
                    let idx = if left { 0 } else { 1 };
                    let known = recv_ports[idx].is_none() && sample_state.is_open(left);
                    let len = if known {
                        sample_state.len(left)
                    } else {
                        usize::MAX
                    };
                    Arc::new(RelaxedCell::from(len))
                };
                let left_final_len = final_len(true);
                let right_final_len = final_len(false);

                if let Some(left_recv) = recv_ports[0].take() {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        sample_sink(
                            left_recv.serial(),
                            &mut sample_state.left,
                            &mut sample_state.left_len,
                            left_final_len.clone(),
                            right_final_len.clone(),
                            self.params.sample_limit,
                        ),
                    ));
                }
                if let Some(right_recv) = recv_ports[1].take() {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        sample_sink(
                            right_recv.serial(),
                            &mut sample_state.right,
                            &mut sample_state.right_len,
                            right_final_len,
                            left_final_len,
                            self.params.sample_limit,
                        ),
                    ));
                }
            },
            EquiJoinState::Build(build_state) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let receivers = recv_ports[build_idx].take().unwrap().parallel();

                let partitioner = HashPartitioner::new(state.num_pipelines, 0);
                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::partition_and_sink(
                            recv,
                            local_builder,
                            partitioner.clone(),
                            &self.params,
                            state,
                            &self.spill_ctx,
                        ),
                    ));
                }
            },
            EquiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                let receivers = probe_state
                    .sampled_probe_morsels
                    .reinsert(
                        state.num_pipelines,
                        recv_ports[probe_idx].take(),
                        scope,
                        join_handles,
                    )
                    .unwrap();

                let partitioner = HashPartitioner::new(state.num_pipelines, 0);
                let probe_tasks = receivers
                    .into_iter()
                    .zip(senders)
                    .map(|(recv, send)| {
                        scope.spawn_task(
                            TaskPriority::High,
                            ProbeState::partition_and_probe(
                                recv,
                                send,
                                &probe_state.table_per_partition,
                                &probe_state.unordered_morsel_seq,
                                partitioner.clone(),
                                &self.params,
                                state,
                            ),
                        )
                    })
                    .collect_vec();

                let max_seq_sent = &mut probe_state.max_seq_sent;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    for probe_task in probe_tasks {
                        *max_seq_sent = (*max_seq_sent).max(probe_task.await?);
                    }
                    Ok(())
                }));
            },
            EquiJoinState::EmitUnmatchedBuild(emit_state) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let send = send_ports[0].take().unwrap().serial();
                join_handles.push(scope.spawn_task(
                    TaskPriority::Low,
                    emit_state.emit_unmatched(send, &self.params, state.num_pipelines),
                ));
            },
            EquiJoinState::EmitUnmatchedBuildInOrder(src_node) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                src_node.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            EquiJoinState::Done => unreachable!(),
        }
    }
}
