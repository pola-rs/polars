mod build;
mod flush;
mod probe;

use std::sync::Arc;

use polars_async::executor::{self, TaskMetricAggregator};
use polars_core::config;
use polars_core::prelude::*;
use polars_core::runtime::ASYNC;
use polars_core::schema::{Schema, SchemaExt};
use polars_defs::join::{JoinArgs, JoinBuildSide, JoinType, MaintainOrderJoin};
use polars_expr::hash_keys::{HashKeys, KeyEncoder};
use polars_expr::idx_table::{IdxTable, new_idx_table};
use polars_ooc::MostRecentSpillContext;
use polars_plan::plans::options::RuntimeFilter;
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;

use self::build::{BuildState, PartitionBuilder};
use self::flush::FlushState;
use self::probe::{FusedPredicate, ProbeState, ProbeTable};
use super::runtime_filter::RuntimeFilters;
use super::tuning::{BuildStats, JoinTuning};
use super::utils::{
    JoinSampleStats, LEFT_BUILD_MARGIN, REF_KEY_RATIO, UNIQUE_BUILD_MARGIN, UNIQUE_KEY_RATIO,
};
use super::{
    BufferedStream, LOPSIDED_ESTIMATE_FACTOR, LOPSIDED_SAMPLE_FACTOR, build_side_left, sample_sink,
    select_key_columns,
};
use crate::expression::StreamExpr;
use crate::hash_shuffle::store::StoredPartition;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_source::InMemorySourceNode;

struct EquiJoinParams {
    key_encoder: KeyEncoder,
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
    /// See `JoinOptionsIR::estimated_rows`.
    estimated_rows: Option<[u64; 2]>,
    random_state: PlRandomState,
    sample_limit: usize,
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

    /// Should we emit unmatched rows from the left side, or else the right side?
    fn emit_unmatched(&self, left: bool) -> bool {
        if left {
            self.args.how.emits_unmatched_left()
        } else {
            self.args.how.emits_unmatched_right()
        }
    }

    /// Should we emit unmatched rows from the build side?
    fn emit_unmatched_build(&self) -> bool {
        self.emit_unmatched(self.left_is_build.unwrap())
    }

    /// Should we emit unmatched rows from the probe side?
    fn emit_unmatched_probe(&self) -> bool {
        self.emit_unmatched(!self.left_is_build.unwrap())
    }

    /// The payload schema of the left side, or else the right side.
    fn payload_schema(&self, left: bool) -> &Arc<Schema> {
        if left {
            &self.left_payload_schema
        } else {
            &self.right_payload_schema
        }
    }
}

/// One hash partition: its stored frames, build frames as input 0 and probe frames as input 1,
/// and its table while it is in memory.
struct JoinPartition {
    stored: StoredPartition,
    table: Option<ProbeTable>,
    /// The estimated memory of its table, in bytes.
    table_bytes: u64,
}

impl JoinPartition {
    /// The estimated memory of its table beyond its stored build frames, in bytes.
    fn net_table_bytes(&self) -> u64 {
        self.table_bytes.saturating_sub(self.stored.input_size(0))
    }

    /// Starts building its table from its stored build frames.
    fn table_builder(
        &self,
        table: &dyn IdxTable,
        params: &EquiJoinParams,
        stats: &BuildStats,
    ) -> PartitionBuilder {
        let schema = params.payload_schema(params.left_is_build.unwrap()).clone();
        let keys = stats.keys_in(self.stored.num_rows(0)) as usize;
        PartitionBuilder::new(self.stored.index(), table, schema, keys)
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

async fn select_keys_with_columns(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &EquiJoinParams,
    state: &ExecutionState,
) -> PolarsResult<(HashKeys, DataFrame)> {
    let keys = select_key_columns(df, key_selectors, state).await?;
    let hash_keys = params.key_encoder.try_hash_keys(
        &keys,
        params.random_state.clone(),
        params.args.nulls_equal,
    )?;
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

/// The rows of both sides, if one side has many times fewer than the other. A
/// side that ended has its sampled rows; for the others the estimates are used,
/// unless one is below the rows sampled, which proves it wrong.
fn lopsided_estimates(
    estimated_rows: Option<[u64; 2]>,
    sampled_rows: [usize; 2],
    ended: [bool; 2],
) -> Option<(usize, usize)> {
    let mut rows = estimated_rows?.map(|rows| usize::try_from(rows).unwrap_or(usize::MAX));
    for i in 0..2 {
        if ended[i] {
            rows[i] = sampled_rows[i];
        } else if rows[i] < sampled_rows[i] {
            return None;
        }
    }
    let [left_rows, right_rows] = rows;
    let lopsided = |small: usize, large: usize| {
        small
            .checked_mul(LOPSIDED_ESTIMATE_FACTOR)
            .is_some_and(|rows| rows <= large)
    };
    if config::verbose() {
        eprintln!(
            "equi-join sides are incomplete, estimated rows are: {left_rows} vs. {right_rows}"
        );
    }
    (lopsided(left_rows, right_rows) || lopsided(right_rows, left_rows))
        .then_some((left_rows, right_rows))
}

struct SampleState {
    left: Vec<Morsel>,
    left_len: usize,
    right: Vec<Morsel>,
    right_len: usize,
    /// The number of rows each side may sample.
    limits: [Arc<RelaxedCell<usize>>; 2],
    /// The only side being read: the preferred build side of a join with runtime
    /// filters, until it ends or reaches the sample limit. A side that ends is
    /// complete, so its key ranges are published before the other side is read.
    only_side: Option<bool>,
    /// Keeps the sampled morsels spillable.
    spill_ctx: MostRecentSpillContext,
}

impl SampleState {
    fn new(only_side: Option<bool>, spill_ctx: MostRecentSpillContext) -> Self {
        Self {
            left: Vec::new(),
            left_len: 0,
            right: Vec::new(),
            right_len: 0,
            limits: Default::default(),
            only_side,
            spill_ctx,
        }
    }

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
                        let left = JoinSampleStats::from_sample(
                            &self.left,
                            &params.left_key_selectors,
                            Some(&params.left_payload_select),
                            true,
                            &params.key_encoder,
                            params.args.nulls_equal,
                            &params.random_state,
                            params.sample_limit,
                            &state.in_memory_exec_state,
                        )?;
                        let right = JoinSampleStats::from_sample(
                            &self.right,
                            &params.right_key_selectors,
                            Some(&params.right_payload_select),
                            true,
                            &params.key_encoder,
                            params.args.nulls_equal,
                            &params.random_state,
                            params.sample_limit,
                            &state.in_memory_exec_state,
                        )?;

                        let (left_score, right_score) = match lopsided_estimates(
                            params.estimated_rows,
                            [self.left_len, self.right_len],
                            [recv[0] == PortState::Done, recv[1] == PortState::Done],
                        ) {
                            Some((left_rows, right_rows)) => (
                                left.all_rows_build_bytes(left_rows),
                                right.all_rows_build_bytes(right_rows),
                            ),
                            // Both lengths are unknown and assumed equal, unless one side
                            // has unique keys and the other repeats them. The other side is
                            // then taken to reference the unique side's keys, making it
                            // longer by the number of times it repeats each key, and is
                            // penalized by `UNIQUE_BUILD_MARGIN`.
                            None => {
                                let left_unique =
                                    left.min_hash_key_ratio.unwrap_or(0.0) >= UNIQUE_KEY_RATIO;
                                let right_unique =
                                    right.min_hash_key_ratio.unwrap_or(0.0) >= UNIQUE_KEY_RATIO;
                                let mut left_score = left.all_rows_build_bytes(params.sample_limit);
                                let mut right_score =
                                    right.all_rows_build_bytes(params.sample_limit);
                                if left.key_ratio <= REF_KEY_RATIO && right_unique {
                                    left_score *= UNIQUE_BUILD_MARGIN / (left.key_ratio + 1e-6);
                                }
                                if right.key_ratio <= REF_KEY_RATIO && left_unique {
                                    right_score *= UNIQUE_BUILD_MARGIN / (right.key_ratio + 1e-6);
                                }
                                (left_score, right_score)
                            },
                        };
                        if config::verbose() {
                            eprintln!(
                                "estimated build scores are: {left_score:.0} (left, key_ratio={:.4}, min_hash_key_ratio={:?}, key_width={:.1}, row_width={:.1}) vs. {right_score:.0} (right, key_ratio={:.4}, min_hash_key_ratio={:?}, key_width={:.1}, row_width={:.1})",
                                left.key_ratio,
                                left.min_hash_key_ratio,
                                left.key_width,
                                left.row_width,
                                right.key_ratio,
                                right.min_hash_key_ratio,
                                right.key_width,
                                right.row_width,
                            );
                        }
                        left_score * LEFT_BUILD_MARGIN < right_score
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

    /// Hand the keys of a completely sampled side to the runtime filters.
    fn publish_runtime_filters(
        &self,
        left: bool,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let (morsels, key_selectors) = if left {
            (&self.left, &params.left_key_selectors)
        } else {
            (&self.right, &params.right_key_selectors)
        };
        params.runtime_filters.publish_from_sample(
            morsels,
            key_selectors,
            &state.in_memory_exec_state,
        )
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
            state.task_metrics.clone(),
        );
        let mut sampled_probe_morsels = BufferedStream::new(
            "equi-join-right-sample".into(),
            core::mem::take(&mut self.right),
            MorselSeq::default(),
            state.task_metrics.clone(),
        );
        if !left_is_build {
            core::mem::swap(&mut sampled_build_morsels, &mut sampled_probe_morsels);
        }

        let mut build_state = BuildState::new(state.num_pipelines, params, sampled_probe_morsels);

        // Simulate the sample build morsels flowing into the build side.
        if !sampled_build_morsels.is_empty() {
            executor::task_scope(state.task_metrics(), |scope| {
                let mut join_handles = Vec::new();
                let receivers = sampled_build_morsels
                    .reinsert(state.num_pipelines, None, scope, &mut join_handles)
                    .unwrap();

                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::sink(recv, local_builder, params, state, spill_ctx),
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

enum EquiJoinState {
    Sample(SampleState),
    Build(BuildState),
    Probe(ProbeState),
    Flush(FlushState),
    EmitUnmatchedBuildInOrder(InMemorySourceNode),
    Done,
}

pub struct EquiJoinNode {
    state: EquiJoinState,
    params: EquiJoinParams,
    table: Box<dyn IdxTable>,
    spill_ctx: MostRecentSpillContext,
    tuning: JoinTuning,
}

impl EquiJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        left_key_schema: Arc<Schema>,
        right_key_schema: Arc<Schema>,
        key_encoder: KeyEncoder,
        output_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        fused_predicate: Option<(StreamExpr, Arc<Schema>)>,
        runtime_filters: Vec<RuntimeFilter>,
        estimated_rows: Option<[u64; 2]>,
        args: JoinArgs,
        num_pipelines: usize,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
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
                    polars_warn!("can't force left build-side with left-maintaining equi-join");
                }
                Some(false)
            },
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => {
                if args.build_side == Some(JoinBuildSide::ForceRight) {
                    polars_warn!("can't force right build-side with right-maintaining equi-join");
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
        let runtime_filters = RuntimeFilters::new(runtime_filters, key_encoder.schema());

        let left_payload_schema = Arc::new(select_schema(&left_input_schema, &left_payload_select));
        let right_payload_schema =
            Arc::new(select_schema(&right_input_schema, &right_payload_select));

        // The ordered probe would evaluate the fused predicate a partition group at a time.
        assert!(fused_predicate.is_none() || args.maintain_order == MaintainOrderJoin::None);
        let fused_predicate = fused_predicate
            .map(|(expr, schema)| {
                FusedPredicate::new(expr, &schema, &left_payload_schema, &right_payload_schema)
            })
            .transpose()?;

        let table = new_idx_table(&key_encoder);
        let params = EquiJoinParams {
            key_encoder,
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
            estimated_rows,
            random_state: PlRandomState::default(),
            sample_limit,
        };

        let state = if left_is_build.is_some() {
            EquiJoinState::Build(BuildState::new(
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
            let spill_ctx =
                MostRecentSpillContext::new("equi-join-sample".into(), task_metrics.clone());
            EquiJoinState::Sample(SampleState::new(only_side, spill_ctx))
        };

        Ok(Self {
            state,
            params,
            table,
            spill_ctx: MostRecentSpillContext::new("equi-join".into(), task_metrics),
            tuning: JoinTuning::from_env(),
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
                } else {
                    EquiJoinState::Probe(build_state.finalize(
                        &self.params,
                        &*self.table,
                        &self.tuning,
                        state,
                    )?)
                };
            }
        }

        // If we are probing and the probe input is done, emit the unmatched build rows in order
        // if needed, otherwise flush the partitions.
        if let EquiJoinState::Probe(probe_state) = &mut self.state {
            let samples_consumed = probe_state.sampled_probe_morsels.is_empty();
            if samples_consumed && recv[probe_idx] == PortState::Done {
                if self.params.preserve_order_build && self.params.emit_unmatched_build() {
                    let unmatched = probe_state.ordered_unmatched(&self.params);
                    let src = InMemorySourceNode::new(
                        Arc::new(unmatched),
                        probe_state.max_seq_sent.successor(),
                    );
                    self.state = EquiJoinState::EmitUnmatchedBuildInOrder(src);
                } else {
                    let flush = FlushState::new(probe_state, &self.params, &*self.table, state)?;
                    self.state = EquiJoinState::Flush(flush);
                }
            }
        }

        // Finally, check if we are done flushing.
        if let EquiJoinState::Flush(flush) = &self.state {
            if flush.is_finished() {
                self.state = EquiJoinState::Done;
            }
        }

        match &mut self.state {
            EquiJoinState::Sample(sample_state) => {
                send[0] = PortState::Blocked;
                for (idx, left) in [(0, true), (1, false)] {
                    // While both sides are sampled, sampling stops once a side has
                    // `LOPSIDED_SAMPLE_FACTOR` times the rows of the other, done, side.
                    let lopsided =
                        sample_state.only_side.is_none() && recv[1 - idx] == PortState::Done;
                    let limit = if lopsided {
                        let lopsided_limit = sample_state
                            .len(!left)
                            .saturating_mul(LOPSIDED_SAMPLE_FACTOR);
                        self.params.sample_limit.min(lopsided_limit)
                    } else {
                        self.params.sample_limit
                    };
                    sample_state.limits[idx].store(limit);
                    if recv[idx] == PortState::Done {
                        continue;
                    }
                    let open = sample_state.is_open(left);
                    recv[idx] = if open && sample_state.len(left) < limit {
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
            EquiJoinState::Flush(_) => {
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

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            EquiJoinState::Sample(_) => NodeMemoryUsage::WillAccumulate,
            EquiJoinState::Build(_) => NodeMemoryUsage::Accumulating,
            EquiJoinState::Probe(_) => NodeMemoryUsage::HoldingUntilDone,
            EquiJoinState::Flush(_) => NodeMemoryUsage::Draining,
            EquiJoinState::EmitUnmatchedBuildInOrder(src) => src.memory_usage(),
            EquiJoinState::Done => NodeMemoryUsage::Bounded,
        }
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
                if let Some(left_recv) = recv_ports[0].take() {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        sample_sink(
                            left_recv.serial(),
                            &mut sample_state.left,
                            &mut sample_state.left_len,
                            sample_state.limits[0].clone(),
                            sample_state.limits[1].clone(),
                            &sample_state.spill_ctx,
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
                            sample_state.limits[1].clone(),
                            sample_state.limits[0].clone(),
                            &sample_state.spill_ctx,
                        ),
                    ));
                }
            },
            EquiJoinState::Build(build_state) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let receivers = recv_ports[build_idx].take().unwrap().parallel();

                for (local_builder, recv) in build_state.local_builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(
                        TaskPriority::High,
                        BuildState::sink(recv, local_builder, &self.params, state, &self.spill_ctx),
                    ));
                }
            },
            EquiJoinState::Probe(probe_state) => {
                assert!(recv_ports[build_idx].is_none());
                let ProbeState {
                    partitions,
                    writers,
                    sampled_probe_morsels,
                    max_seq_sent,
                    ..
                } = probe_state;
                let senders = send_ports[0].take().unwrap().parallel();
                let receivers = sampled_probe_morsels
                    .reinsert(
                        state.num_pipelines,
                        recv_ports[probe_idx].take(),
                        scope,
                        join_handles,
                    )
                    .unwrap();

                let partitions = &*partitions;
                let mut writers = writers.iter_mut();
                let probe_tasks = receivers
                    .into_iter()
                    .zip(senders)
                    .map(|(recv, send)| {
                        scope.spawn_task(
                            TaskPriority::High,
                            partitions.partition_and_probe(
                                recv,
                                send,
                                writers.next(),
                                &self.params,
                                state,
                            ),
                        )
                    })
                    .collect_vec();

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    for probe_task in probe_tasks {
                        *max_seq_sent = (*max_seq_sent).max(probe_task.await?);
                    }
                    Ok(())
                }));
            },
            EquiJoinState::Flush(flush) => {
                assert!(recv_ports[build_idx].is_none());
                assert!(recv_ports[probe_idx].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                flush.spawn(scope, senders, &self.params, state, join_handles);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lopsided_estimates_are_used() {
        let open = [false, false];
        assert_eq!(
            lopsided_estimates(Some([800, 100]), [100, 100], open),
            Some((800, 100))
        );
        assert_eq!(lopsided_estimates(Some([100, 799]), [100, 100], open), None);
        assert_eq!(lopsided_estimates(None, [100, 100], open), None);
    }

    #[test]
    fn estimates_below_the_sampled_rows_are_not_used() {
        let open = [false, false];
        assert_eq!(
            lopsided_estimates(Some([10_000, 48]), [100, 100], open),
            None
        );
        assert_eq!(
            lopsided_estimates(Some([48, 10_000]), [100, 100], open),
            None
        );
    }

    #[test]
    fn ended_sides_use_their_sampled_rows() {
        assert_eq!(
            lopsided_estimates(Some([20_000, 800]), [100, 100], [true, false]),
            Some((100, 800))
        );
        assert_eq!(
            lopsided_estimates(Some([800, 20_000]), [100, 100], [false, true]),
            Some((800, 100))
        );
        assert_eq!(
            lopsided_estimates(Some([20, 400]), [100, 100], [true, false]),
            None
        );
    }
}
