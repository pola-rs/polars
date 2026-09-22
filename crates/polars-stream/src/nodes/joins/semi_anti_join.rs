use std::sync::Arc;

use polars_arrow::array::BooleanArray;
use polars_arrow::array::builder::ShareStrategy;
use polars_arrow::bitmap::{BitmapBuilder, MutableBitmap};
use polars_async::executor;
use polars_core::config;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::{ASYNC, RAYON};
use polars_core::schema::Schema;
use polars_defs::join::{JoinArgs, JoinBuildSide, JoinType, MaintainOrderJoin};
use polars_expr::groups::{Grouper, new_hash_grouper};
use polars_expr::hash_keys::HashKeys;
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_plan::plans::options::RuntimeFilter;
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::sparse_init_vec::SparseInitVec;
use rayon::prelude::*;

use super::runtime_filter::{KeyFilterBuilder, RuntimeFilters};
use super::{
    BufferedStream, LOPSIDED_SAMPLE_FACTOR, emit_morsel_size, fold_sample, sample_sink, send_frames,
};
use crate::expression::StreamExpr;
use crate::nodes::compute_node_prelude::*;

/// Bytes a hash table takes per key on top of the key itself.
const KEY_SLOT_OVERHEAD: f64 = 16.0;

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

fn hash_keys(keys: &DataFrame, params: &SemiAntiJoinParams, null_is_valid: bool) -> HashKeys {
    HashKeys::from_df(keys, params.random_state.clone(), null_is_valid, false)
}

async fn select_keys(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &SemiAntiJoinParams,
    null_is_valid: bool,
    state: &ExecutionState,
) -> PolarsResult<HashKeys> {
    let keys = select_key_columns(df, key_selectors, state).await?;
    Ok(hash_keys(&keys, params, null_is_valid))
}

struct SemiAntiJoinParams {
    left_is_build: Option<bool>,
    left_key_selectors: Vec<StreamExpr>,
    right_key_selectors: Vec<StreamExpr>,
    output_schema: Arc<Schema>,
    nulls_equal: bool,
    is_anti: bool,
    return_bool: bool,
    build_side: Option<JoinBuildSide>,
    // Key filters for the scans below the planned probe side, set once from
    // the build of the planned build side.
    runtime_filters: RuntimeFilters,
    random_state: PlRandomState,
    sample_limit: usize,
}

impl SemiAntiJoinParams {
    fn left_is_build(&self) -> bool {
        self.left_is_build.unwrap()
    }

    /// Whether the build keys go to the runtime filters: the filters exist and
    /// describe the side being built.
    fn publishes_runtime_filters(&self) -> bool {
        let planned_left = matches!(
            self.build_side,
            Some(JoinBuildSide::ForceLeft | JoinBuildSide::PreferLeft)
        );
        !self.runtime_filters.is_empty()
            && !self.runtime_filters.is_set()
            && self.left_is_build == Some(planned_left)
    }

    /// Whether the built rows that no probe row matches are the output.
    fn emits_unmatched_build(&self) -> bool {
        self.is_anti && self.left_is_build()
    }

    /// Whether null keys get a group when the given side is built. The rows
    /// of an anti join's left side must stay, whatever their keys.
    fn null_is_valid_when_built(&self, left: bool) -> bool {
        self.nulls_equal || (self.is_anti && left)
    }

    /// Whether the join outputs nothing when no build rows came in.
    fn empty_build_gives_empty_output(&self) -> bool {
        !self.return_bool && (!self.is_anti || self.left_is_build())
    }
}

pub struct SemiAntiJoinNode {
    state: SemiAntiJoinState,
    params: SemiAntiJoinParams,
    grouper: Box<dyn Grouper>,
    spill_ctx: MostRecentSpillContext,
}

impl SemiAntiJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        unique_key_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        runtime_filters: Vec<RuntimeFilter>,
        args: JoinArgs,
        return_bool: bool,
        num_pipelines: usize,
    ) -> PolarsResult<Self> {
        let sample_limit: usize = polars_config::config()
            .join_sample_limit()
            .try_into()
            .unwrap();
        let is_anti = args.how == JoinType::Anti;

        // Only an unordered join that outputs rows can be built from the left side.
        let left_is_build = if return_bool || args.maintain_order != MaintainOrderJoin::None {
            Some(false)
        } else {
            match args.build_side {
                Some(JoinBuildSide::ForceLeft) => Some(true),
                Some(JoinBuildSide::ForceRight) => Some(false),
                Some(JoinBuildSide::PreferLeft | JoinBuildSide::PreferRight) | None => {
                    if sample_limit == 0 {
                        Some(args.build_side == Some(JoinBuildSide::PreferLeft))
                    } else {
                        None
                    }
                },
            }
        };

        // A filter is only published for the side the plan named. The keys of
        // both sides have the same dtypes.
        debug_assert!(runtime_filters.is_empty() || args.build_side.is_some());
        let params = SemiAntiJoinParams {
            left_is_build,
            left_key_selectors,
            right_key_selectors,
            output_schema,
            random_state: PlRandomState::default(),
            nulls_equal: args.nulls_equal,
            return_bool,
            is_anti,
            build_side: args.build_side,
            runtime_filters: RuntimeFilters::new(runtime_filters, &unique_key_schema),
            sample_limit,
        };
        let state = if left_is_build.is_some() {
            SemiAntiJoinState::Build(BuildState::new(
                num_pipelines,
                num_pipelines,
                &params,
                BufferedStream::default(),
            ))
        } else {
            SemiAntiJoinState::Sample(SampleState::default())
        };

        Ok(Self {
            state,
            params,
            grouper: new_hash_grouper(unique_key_schema),
            spill_ctx: MostRecentSpillContext::new("semi-anti-join".into()),
        })
    }
}

enum SemiAntiJoinState {
    Sample(SampleState),
    Build(BuildState),
    Probe(ProbeState),
    EmitBuild(EmitBuildState),
    Done,
}

/// Estimated bytes per row the build of one side keeps, from its sample.
struct RetainedBytes {
    /// Distinct keys, as a fraction of the rows.
    key_ratio: f64,
    key_width: f64,
    row_width: f64,
}

impl RetainedBytes {
    fn from_sample(
        morsels: &[Morsel],
        key_selectors: &[StreamExpr],
        null_is_valid: bool,
        params: &SemiAntiJoinParams,
        state: &ExecutionState,
    ) -> PolarsResult<Self> {
        if morsels.is_empty() || params.sample_limit == 0 {
            return Ok(Self {
                key_ratio: 0.0,
                key_width: 0.0,
                row_width: 0.0,
            });
        }
        let ((sketch, key_bytes, row_bytes), rows) = fold_sample(
            morsels,
            params.sample_limit,
            || (CardinalitySketch::new(), 0usize, 0usize),
            |(mut sketch, key_bytes, row_bytes), df| {
                let keys = ASYNC.block_on(select_key_columns(df, key_selectors, state))?;
                hash_keys(&keys, params, null_is_valid).sketch_cardinality(&mut sketch);
                Ok((
                    sketch,
                    key_bytes + keys.estimated_size(),
                    row_bytes + df.estimated_size(),
                ))
            },
            |(mut a, ak, ar), (b, bk, br)| {
                a.combine(&b);
                (a, ak + bk, ar + br)
            },
        )?;
        let rows = rows as f64;
        Ok(Self {
            key_ratio: (sketch.estimate() as f64 / rows).min(1.0),
            key_width: key_bytes as f64 / rows,
            row_width: row_bytes as f64 / rows,
        })
    }

    /// Bytes retained when building a table of the distinct keys of `rows` rows.
    fn keys(&self, rows: usize) -> f64 {
        rows as f64 * self.key_ratio * (self.key_width + KEY_SLOT_OVERHEAD)
    }

    /// Bytes retained when building the distinct keys of `rows` rows and
    /// keeping every row.
    fn keys_and_rows(&self, rows: usize) -> f64 {
        self.keys(rows) + rows as f64 * (self.row_width + size_of::<IdxSize>() as f64)
    }
}

#[derive(Default)]
struct SampleState {
    left: Vec<Morsel>,
    left_len: usize,
    right: Vec<Morsel>,
    right_len: usize,
}

impl SampleState {
    fn len(&self, left: bool) -> usize {
        if left { self.left_len } else { self.right_len }
    }

    fn try_transition_to_build(
        &mut self,
        recv: &[PortState],
        params: &mut SemiAntiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<Option<BuildState>> {
        let left_saturated = self.left_len >= params.sample_limit;
        let right_saturated = self.right_len >= params.sample_limit;
        let left_done = recv[0] == PortState::Done || left_saturated;
        let right_done = recv[1] == PortState::Done || right_saturated;
        // A left side that reached the sample limit is only built when it is
        // preferred. Otherwise the right side is built, however big it is.
        let prefer_left = params.build_side == Some(JoinBuildSide::PreferLeft);
        let stop_sampling = (left_done && right_done)
            || (left_saturated && !prefer_left)
            || (left_done && self.right_len >= LOPSIDED_SAMPLE_FACTOR * self.left_len)
            || (right_done && self.left_len >= LOPSIDED_SAMPLE_FACTOR * self.right_len);
        if !stop_sampling {
            return Ok(None);
        }

        if config::verbose() {
            eprintln!(
                "choosing semi/anti-join build side, sample lengths are: {} vs. {}",
                self.left_len, self.right_len
            );
        }

        // Building the left side also keeps all of its rows, so it is only
        // chosen when it was read completely and its estimated bytes are no
        // more than the key bytes of the sampled right rows.
        let left_complete = recv[0] == PortState::Done;
        let left_is_build = match (left_complete, right_saturated) {
            (false, true) => left_saturated && prefer_left,
            (false, false) => false,
            (true, _) => {
                let left = RetainedBytes::from_sample(
                    &self.left,
                    &params.left_key_selectors,
                    params.null_is_valid_when_built(true),
                    params,
                    &state.in_memory_exec_state,
                )?;
                let right = RetainedBytes::from_sample(
                    &self.right,
                    &params.right_key_selectors,
                    params.null_is_valid_when_built(false),
                    params,
                    &state.in_memory_exec_state,
                )?;
                let left_bytes = left.keys_and_rows(self.left_len);
                let right_bytes = right.keys(self.right_len);
                if config::verbose() {
                    eprintln!(
                        "estimated retained bytes are: {left_bytes:.0} (left, keys and rows) vs. {right_bytes:.0} (right, keys)"
                    );
                }
                left_bytes <= right_bytes
            },
        };

        if config::verbose() {
            eprintln!(
                "semi/anti-join build side chosen: {}",
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

    /// Start building from `left_is_build`, feeding it the morsels sampled from
    /// that side; the other side's samples are probed first later.
    fn start_build(
        &mut self,
        left_is_build: bool,
        params: &mut SemiAntiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<BuildState> {
        params.left_is_build = Some(left_is_build);
        let mut sampled_build_morsels = BufferedStream::new(
            "semi-anti-join-left-sample".into(),
            core::mem::take(&mut self.left),
            MorselSeq::default(),
        );
        let mut sampled_probe_morsels = BufferedStream::new(
            "semi-anti-join-right-sample".into(),
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
    // The complete list of keys as seen by this builder.
    keys: Vec<HashKeys>,

    // The output rows of each morsel, when the left side is built.
    payload: Vec<SpillFrame>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // key_idxs_values_per_p[p][start..stop] contains the offsets into keys[i]
    // for partition p, where start, stop are:
    // let start = key_idxs_offsets[i * num_partitions + p];
    // let stop = key_idxs_offsets[(i + 1) * num_partitions + p];
    key_idxs_values_per_p: Vec<Vec<IdxSize>>,
    key_idxs_offsets_per_p: Vec<usize>,
    // The key of each runtime filter seen by this builder.
    key_filters: Vec<KeyFilterBuilder>,
}

struct BuildState {
    local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    fn new(
        num_pipelines: usize,
        num_partitions: usize,
        params: &SemiAntiJoinParams,
        sampled_probe_morsels: BufferedStream,
    ) -> Self {
        let local_builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                keys: Vec::new(),
                payload: Vec::new(),
                sketch_per_p: vec![CardinalitySketch::default(); num_partitions],
                key_idxs_values_per_p: vec![Vec::new(); num_partitions],
                key_idxs_offsets_per_p: vec![0; num_partitions],
                key_filters: if params.publishes_runtime_filters() {
                    params.runtime_filters.new_builders()
                } else {
                    Vec::new()
                },
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
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<()> {
        let key_selectors = if params.left_is_build() {
            &params.left_key_selectors
        } else {
            &params.right_key_selectors
        };

        let publishes_runtime_filters = params.publishes_runtime_filters();
        while let Ok(morsel) = recv.recv().await {
            let df = morsel.df().await;
            let keys = select_key_columns(&df, key_selectors, &state.in_memory_exec_state).await?;
            if publishes_runtime_filters {
                params
                    .runtime_filters
                    .extend(&keys, &mut local.key_filters)?;
            }
            let hash_keys = hash_keys(
                &keys,
                params,
                params.null_is_valid_when_built(params.left_is_build()),
            );

            hash_keys.gen_idxs_per_partition(
                &partitioner,
                &mut local.key_idxs_values_per_p,
                &mut local.sketch_per_p,
                false,
            );

            local
                .key_idxs_offsets_per_p
                .extend(local.key_idxs_values_per_p.iter().map(|vp| vp.len()));
            local.keys.push(hash_keys);

            if params.left_is_build() {
                // The payload is rechunked for the gathers in the finalize.
                let mut payload = df.select(params.output_schema.iter_names())?;
                payload.rechunk_mut();
                local
                    .payload
                    .push(SpillFrame::new(payload, spill_ctx).await);
            }
        }
        Ok(())
    }

    /// Whether no key was built, sampled morsels included.
    fn is_empty(&self) -> bool {
        self.local_builders
            .iter()
            .all(|b| b.keys.iter().all(|keys| keys.is_empty()))
    }

    /// Hand every build key to the runtime filters; filters of a side that was
    /// not built get a predicate that skips nothing.
    fn publish_runtime_filters(&mut self, params: &SemiAntiJoinParams) {
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

    fn finalize(&mut self, params: &SemiAntiJoinParams, grouper: &dyn Grouper) -> ProbeState {
        let left_is_build = params.left_is_build();

        // To reduce maximum memory usage we want to drop the original keys
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let keys_per_local_builder = self
            .local_builders
            .iter_mut()
            .map(|b| {
                Arc::new((
                    core::mem::take(&mut b.keys),
                    core::mem::take(&mut b.payload),
                ))
            })
            .collect_vec();
        let (key_drop_q_send, key_drop_q_recv) =
            async_channel::bounded(keys_per_local_builder.len());
        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let groupers: SparseInitVec<Box<dyn Grouper>> =
            SparseInitVec::with_capacity(num_partitions);
        let payloads: SparseInitVec<BuildPayload> = SparseInitVec::with_capacity(num_partitions);

        executor::task_scope(|s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_keys_per_local_builder = Arc::new(keys_per_local_builder);
            let mut join_handles = Vec::new();
            for p in 0..num_partitions {
                let arc_keys_per_local_builder = Arc::clone(&arc_keys_per_local_builder);
                let key_drop_q_send = key_drop_q_send.clone();
                let key_drop_q_recv = key_drop_q_recv.clone();
                let groupers = &groupers;
                let payloads = &payloads;
                join_handles.push(s.spawn_task(TaskPriority::High, async move {
                    // Extract from outer arc and drop outer arc.
                    let keys_per_local_builder = Arc::unwrap_or_clone(arc_keys_per_local_builder);

                    // Compute cardinality estimate and total amount of
                    // payload for this partition.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for l in local_builders {
                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.key_idxs_offsets_per_p.len();
                        payload_rows += l.key_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_grouper = grouper.new_empty();
                    p_grouper.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(params.output_schema.clone());
                    let mut p_group_idxs = Vec::new();
                    if left_is_build {
                        p_payload.reserve(payload_rows);
                        p_group_idxs.reserve(payload_rows);
                    }

                    // Build.
                    let mut skip_drop_attempt = false;
                    for (l, l_keys) in local_builders.iter().zip(keys_per_local_builder) {
                        // Try to help with dropping the processed keys.
                        if !skip_drop_attempt {
                            drop(key_drop_q_recv.try_recv());
                        }

                        for (i, keys) in l_keys.0.iter().enumerate() {
                            unsafe {
                                let p_key_idxs_start =
                                    l.key_idxs_offsets_per_p[i * num_partitions + p];
                                let p_key_idxs_stop =
                                    l.key_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_key_idxs =
                                    &l.key_idxs_values_per_p[p][p_key_idxs_start..p_key_idxs_stop];
                                if left_is_build {
                                    p_grouper.insert_keys_subset(
                                        keys,
                                        p_key_idxs,
                                        Some(&mut p_group_idxs),
                                    );
                                    let payload = l_keys.1[i].get().await;
                                    p_payload.gather_extend(
                                        &payload,
                                        p_key_idxs,
                                        ShareStrategy::Never,
                                    );
                                } else {
                                    p_grouper.insert_keys_subset(keys, p_key_idxs, None);
                                }
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_keys) {
                            // If we're the last thread to process this set of keys we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(key_drop_q_send.try_send(l));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // We're done, help others out by doing drops.
                    drop(key_drop_q_send); // So we don't deadlock trying to receive from ourselves.
                    while let Ok(l_keys) = key_drop_q_recv.recv().await {
                        drop(l_keys);
                    }

                    groupers.try_set(p, p_grouper).ok().unwrap();
                    payloads
                        .try_set(
                            p,
                            BuildPayload {
                                rows: p_payload.freeze(),
                                group_idxs: p_group_idxs,
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
            drop(arc_keys_per_local_builder);
            drop(key_drop_q_send);

            ASYNC.block_in_place_on(async move {
                for handle in join_handles {
                    handle.await;
                }
            });
        });

        let grouper_per_partition = groupers.try_assume_init().ok().unwrap();
        let marks_per_pipeline = if left_is_build {
            (0..self.local_builders.len())
                .map(|_| {
                    grouper_per_partition
                        .iter()
                        .map(|g| MutableBitmap::from_len_zeroed(g.num_groups() as usize))
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        ProbeState {
            grouper_per_partition,
            payload_per_partition: payloads.try_assume_init().ok().unwrap(),
            marks_per_pipeline,
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
        }
    }
}

/// The rows of one partition of a built left side, and the group of each row.
struct BuildPayload {
    rows: DataFrame,
    group_idxs: Vec<IdxSize>,
}

struct ProbeState {
    grouper_per_partition: Vec<Box<dyn Grouper>>,
    payload_per_partition: Vec<BuildPayload>,
    // When the left side is built: per probe pipeline and partition, the
    // groups a probed key was found for.
    marks_per_pipeline: Vec<Vec<MutableBitmap>>,
    sampled_probe_morsels: BufferedStream,
}

impl ProbeState {
    async fn partition_and_probe(
        mut recv: PortReceiver,
        mut send: PortSender,
        partitions: &[Box<dyn Grouper>],
        partitioner: HashPartitioner,
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let mut probe_match = Vec::new();
        let key_selectors = &params.left_key_selectors;

        while let Ok(morsel) = recv.recv().await {
            let (sf, in_seq, src_token, wait_token) = morsel.into_inner();
            let df = sf.into_df().await;
            if df.height() == 0 {
                continue;
            }

            let hash_keys = select_keys(
                &df,
                key_selectors,
                params,
                params.nulls_equal,
                &state.in_memory_exec_state,
            )
            .await?;

            unsafe {
                let out_df = if params.return_bool {
                    let mut builder = BitmapBuilder::with_capacity(df.height());
                    partitions[0].contains_key_partitioned_groupers(
                        partitions,
                        &hash_keys,
                        &partitioner,
                        params.is_anti,
                        &mut builder,
                    );
                    let mut arr = BooleanArray::from(builder.freeze());
                    if !params.nulls_equal {
                        arr.set_validity(hash_keys.validity().cloned());
                    }
                    let s = BooleanChunked::with_chunk(df[0].name().clone(), arr).into_series();
                    DataFrame::new_unchecked(s.len(), vec![Column::from(s)])
                } else {
                    probe_match.clear();
                    partitions[0].probe_partitioned_groupers(
                        partitions,
                        &hash_keys,
                        &partitioner,
                        params.is_anti,
                        &mut probe_match,
                    );
                    if probe_match.is_empty() {
                        continue;
                    }
                    df.select(params.output_schema.iter_names())?
                        .take_slice_unchecked(&probe_match)
                };

                let mut morsel = Morsel::new_unregistered(out_df, in_seq, src_token.clone());
                if let Some(token) = wait_token {
                    morsel.set_consume_token(token);
                }
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Marks the groups of the built left side that the right rows match.
    async fn partition_and_mark(
        mut recv: PortReceiver,
        partitions: &[Box<dyn Grouper>],
        marks: &mut [MutableBitmap],
        partitioner: HashPartitioner,
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let key_selectors = &params.right_key_selectors;

        while let Ok(morsel) = recv.recv().await {
            let df = morsel.into_df().await;
            if df.height() == 0 {
                continue;
            }

            let hash_keys = select_keys(
                &df,
                key_selectors,
                params,
                params.nulls_equal,
                &state.in_memory_exec_state,
            )
            .await?;
            unsafe {
                partitions[0].mark_groups_partitioned_groupers(
                    partitions,
                    &hash_keys,
                    &partitioner,
                    marks,
                );
            }
        }

        Ok(())
    }

    /// Marks the groups matched by the sampled right morsels, so that the
    /// probe only has to read the right port.
    fn mark_sampled(
        &mut self,
        params: &SemiAntiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        if self.sampled_probe_morsels.is_empty() {
            return Ok(());
        }
        let sampled = core::mem::take(&mut self.sampled_probe_morsels);
        let partitioner = HashPartitioner::new(state.num_pipelines, 0);
        executor::task_scope(|scope| {
            let mut join_handles = Vec::new();
            let receivers = sampled
                .reinsert(state.num_pipelines, None, scope, &mut join_handles)
                .unwrap();
            for (marks, recv) in self.marks_per_pipeline.iter_mut().zip(receivers) {
                join_handles.push(scope.spawn_task(
                    TaskPriority::High,
                    ProbeState::partition_and_mark(
                        recv,
                        &self.grouper_per_partition,
                        marks,
                        partitioner.clone(),
                        params,
                        state,
                    ),
                ));
            }

            ASYNC.block_in_place_on(async move {
                for handle in join_handles {
                    handle.await?;
                }
                PolarsResult::Ok(())
            })
        })
    }

    /// The rows of each partition whose group a probe pipeline marked, or the
    /// other rows for `unmatched`.
    fn output_rows(&mut self, unmatched: bool) -> Vec<Vec<IdxSize>> {
        let payloads = &self.payload_per_partition;
        let mut marks_per_partition: Vec<Vec<MutableBitmap>> =
            (0..payloads.len()).map(|_| Vec::new()).collect();
        for pipeline_marks in self.marks_per_pipeline.drain(..) {
            for (p, marks) in pipeline_marks.into_iter().enumerate() {
                marks_per_partition[p].push(marks);
            }
        }
        RAYON.install(|| {
            marks_per_partition
                .into_par_iter()
                .zip(payloads)
                .map(|(mut marks, payload)| {
                    let Some(mut merged) = marks.pop() else {
                        return Vec::new();
                    };
                    for m in &marks {
                        let mut merged_mut = &mut merged;
                        merged_mut |= m;
                    }
                    if unmatched {
                        merged = !merged;
                    }
                    payload
                        .group_idxs
                        .iter()
                        .enumerate_idx()
                        .filter(|(_, g)| unsafe { merged.get_unchecked(**g as usize) })
                        .map(|(row, _)| row)
                        .collect()
                })
                .collect()
        })
    }
}

impl Drop for ProbeState {
    fn drop(&mut self) {
        RAYON.install(|| {
            // Parallel drop as the state might be quite big.
            self.grouper_per_partition.par_drain(..).for_each(drop);
            self.payload_per_partition.par_drain(..).for_each(drop);
        })
    }
}

struct EmitBuildState {
    partitions: Vec<BuildPayload>,
    rows_per_partition: Vec<Vec<IdxSize>>,
    active_partition_idx: usize,
    offset_in_active_p: usize,
    morsel_seq: MorselSeq,
}

impl EmitBuildState {
    async fn emit(&mut self, send: PortSender, num_pipelines: usize) -> PolarsResult<()> {
        let total_len: usize = self.rows_per_partition.iter().map(|r| r.len()).sum();
        let morsel_size = emit_morsel_size(total_len, num_pipelines);

        let partitions = &self.partitions;
        let rows_per_partition = &self.rows_per_partition;
        let active_partition_idx = &mut self.active_partition_idx;
        let offset_in_active_p = &mut self.offset_in_active_p;
        send_frames(send, &mut self.morsel_seq, || {
            while let Some(p) = partitions.get(*active_partition_idx) {
                let rows = &rows_per_partition[*active_partition_idx];
                if *offset_in_active_p >= rows.len() {
                    *active_partition_idx += 1;
                    *offset_in_active_p = 0;
                    continue;
                }
                let end = (*offset_in_active_p + morsel_size).min(rows.len());
                let idxs = &rows[*offset_in_active_p..end];
                *offset_in_active_p = end;
                return Some(unsafe { p.rows.take_slice_unchecked_impl(idxs, false) });
            }
            None
        })
        .await
    }
}

impl ComputeNode for SemiAntiJoinNode {
    fn name(&self) -> &str {
        match (self.params.return_bool, self.params.is_anti) {
            (false, false) => "semi-join",
            (false, true) => "anti-join",
            (true, false) => "is-in",
            (true, true) => "is-not-in",
        }
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
            self.state = SemiAntiJoinState::Done;
        }

        // If we are sampling and both sides are done/filled, transition to building.
        if let SemiAntiJoinState::Sample(sample_state) = &mut self.state {
            if let Some(build_state) = sample_state.try_transition_to_build(
                recv,
                &mut self.params,
                state,
                &self.spill_ctx,
            )? {
                self.state = SemiAntiJoinState::Build(build_state);
            }
        }

        let build_idx = if self.params.left_is_build == Some(true) {
            0
        } else {
            1
        };
        let probe_idx = 1 - build_idx;

        // If we are building and the build input is done, transition to probing,
        // or finish without reading the probe side when nothing can come out.
        if let SemiAntiJoinState::Build(build_state) = &mut self.state {
            if recv[build_idx] == PortState::Done {
                build_state.publish_runtime_filters(&self.params);
                if self.params.empty_build_gives_empty_output() && build_state.is_empty() {
                    self.state = SemiAntiJoinState::Done;
                } else {
                    let mut probe_state = build_state.finalize(&self.params, &*self.grouper);
                    if self.params.left_is_build() {
                        probe_state.mark_sampled(&self.params, state)?;
                    }
                    self.state = SemiAntiJoinState::Probe(probe_state);
                }
            }
        }

        // If we are probing and the probe input is done, emit the matched (or
        // for an anti join unmatched) build rows if we built the left side,
        // otherwise we're done.
        if let SemiAntiJoinState::Probe(probe_state) = &mut self.state {
            let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
            if samples_consumed && recv[probe_idx] == PortState::Done {
                if self.params.left_is_build() {
                    let rows_per_partition =
                        probe_state.output_rows(self.params.emits_unmatched_build());
                    self.state = SemiAntiJoinState::EmitBuild(EmitBuildState {
                        partitions: core::mem::take(&mut probe_state.payload_per_partition),
                        rows_per_partition,
                        active_partition_idx: 0,
                        offset_in_active_p: 0,
                        morsel_seq: MorselSeq::default(),
                    });
                } else {
                    self.state = SemiAntiJoinState::Done;
                }
            }
        }

        // Finally, check if we are done emitting build rows.
        if let SemiAntiJoinState::EmitBuild(emit_state) = &mut self.state {
            if emit_state.active_partition_idx >= emit_state.partitions.len() {
                self.state = SemiAntiJoinState::Done;
            }
        }

        match &mut self.state {
            SemiAntiJoinState::Sample(sample_state) => {
                send[0] = PortState::Blocked;
                for (idx, left) in [(0, true), (1, false)] {
                    if recv[idx] == PortState::Done {
                        continue;
                    }
                    recv[idx] = if sample_state.len(left) < self.params.sample_limit {
                        PortState::Ready
                    } else {
                        PortState::Blocked
                    };
                }
            },
            SemiAntiJoinState::Build(_) => {
                send[0] = PortState::Blocked;
                if recv[build_idx] != PortState::Done {
                    recv[build_idx] = PortState::Ready;
                }
                if recv[probe_idx] != PortState::Done {
                    recv[probe_idx] = PortState::Blocked;
                }
            },
            SemiAntiJoinState::Probe(probe_state) => {
                if self.params.left_is_build() {
                    // Nothing is sent until the whole right side was probed.
                    send[0] = PortState::Blocked;
                    if recv[probe_idx] != PortState::Done {
                        recv[probe_idx] = PortState::Ready;
                    }
                } else if recv[probe_idx] != PortState::Done {
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
            SemiAntiJoinState::EmitBuild(_) => {
                send[0] = PortState::Ready;
                recv[build_idx] = PortState::Done;
                recv[probe_idx] = PortState::Done;
            },
            SemiAntiJoinState::Done => {
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
            SemiAntiJoinState::Sample { .. } | SemiAntiJoinState::Build { .. }
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
            SemiAntiJoinState::Sample(sample_state) => {
                assert!(send_ports[0].is_none());
                // A side without a port is done.
                let final_len = |left: bool| {
                    let idx = if left { 0 } else { 1 };
                    let len = if recv_ports[idx].is_none() {
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
            SemiAntiJoinState::Build(build_state) => {
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
            SemiAntiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
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
                if self.params.left_is_build() {
                    assert!(send_ports[0].is_none());
                    for (marks, recv) in probe_state.marks_per_pipeline.iter_mut().zip(receivers) {
                        join_handles.push(scope.spawn_task(
                            TaskPriority::High,
                            ProbeState::partition_and_mark(
                                recv,
                                &probe_state.grouper_per_partition,
                                marks,
                                partitioner.clone(),
                                &self.params,
                                state,
                            ),
                        ));
                    }
                } else {
                    let senders = send_ports[0].take().unwrap().parallel();
                    for (recv, send) in receivers.into_iter().zip(senders) {
                        join_handles.push(scope.spawn_task(
                            TaskPriority::High,
                            ProbeState::partition_and_probe(
                                recv,
                                send,
                                &probe_state.grouper_per_partition,
                                partitioner.clone(),
                                &self.params,
                                state,
                            ),
                        ));
                    }
                }
            },
            SemiAntiJoinState::EmitBuild(emit_state) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let send = send_ports[0].take().unwrap().serial();
                join_handles.push(scope.spawn_task(
                    TaskPriority::Low,
                    emit_state.emit(send, state.num_pipelines),
                ));
            },
            SemiAntiJoinState::Done => unreachable!(),
        }
    }
}
