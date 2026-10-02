use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_expr::groups::new_hash_grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_ooc::{LeastRecentSpillContext, ParameterFreeSpillContext, SpillFrame};
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use super::window::PARTITIONS_PER_PIPELINE;
use crate::morsel::{MorselSeq, SourceToken};

/// Up to this many estimated groups, every pipeline reduces its own morsels and the results are
/// merged. With few groups the hash partitions are too few and too uneven to keep all pipelines
/// busy.
const MAX_LOCAL_GROUPS: usize = 1 << 16;

/// A window that reduces one input column to one value per partition.
pub struct ScalarWindow {
    /// The name of the window column in the output.
    pub name: PlSmallStr,
    /// The input column the reduction reads.
    pub input: PlSmallStr,
    pub reduction: Box<dyn GroupedReduction>,
}

pub struct ScalarWindowParams {
    pub partition_by: Vec<PlSmallStr>,
    pub windows: Vec<ScalarWindow>,
    /// The partition keys and the columns the reductions read.
    pub read_schema: Arc<Schema>,
    pub key_schema: Arc<Schema>,
    pub output_schema: Arc<Schema>,
    pub maintain_order: bool,
}

/// Evaluates scalar windows that share one partitioning. The input morsels are kept. Their rows
/// are hash partitioned on the partition keys, each hash partition is reduced in parallel, and
/// then the morsels are output again with the reduced value of their partition appended.
///
/// The input morsels can be spilled while they are received. The partition keys of the input and
/// the group ids of every row stay in memory, as do the groups and their reduced values.
pub struct ScalarWindowNode {
    params: Arc<ScalarWindowParams>,
    state: ScalarWindowState,
}

enum ScalarWindowState {
    Sink {
        builders: Vec<LocalBuilder>,
        partitioner: HashPartitioner,
        random_state: PlRandomState,
        spill_ctx: LeastRecentSpillContext,
    },
    Replay(Replay),
    Done,
}

#[derive(Default)]
struct LocalBuilder {
    morsels: Vec<(MorselSeq, SpillFrame)>,
    hash_keys: Vec<HashKeys>,
    sketch_per_p: Vec<CardinalitySketch>,
    // The rows of morsels[i] in partition p are
    // idxs_per_p[p][offsets_per_p[i * num_partitions + p]..offsets_per_p[(i + 1) * num_partitions + p]].
    idxs_per_p: Vec<Vec<IdxSize>>,
    offsets_per_p: Vec<usize>,
}

impl LocalBuilder {
    /// The range in `idxs_per_p[p]` of the rows of morsel `i` in partition `p`.
    fn partition_range(&self, i: usize, p: usize) -> std::ops::Range<usize> {
        let num_partitions = self.idxs_per_p.len();
        self.offsets_per_p[i * num_partitions + p]..self.offsets_per_p[(i + 1) * num_partitions + p]
    }
}

impl ScalarWindowNode {
    pub fn new(
        params: Arc<ScalarWindowParams>,
        num_pipelines: usize,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        let num_partitions = num_pipelines * PARTITIONS_PER_PIPELINE;
        let builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                idxs_per_p: vec![Vec::new(); num_partitions],
                offsets_per_p: vec![0; num_partitions],
                sketch_per_p: vec![CardinalitySketch::new(); num_partitions],
                ..Default::default()
            })
            .collect();
        Self {
            params,
            state: ScalarWindowState::Sink {
                builders,
                partitioner: HashPartitioner::new(num_partitions, 0),
                random_state: PlRandomState::default(),
                spill_ctx: LeastRecentSpillContext::new("scalar-window".into(), task_metrics),
            },
        }
    }
}

impl ComputeNode for ScalarWindowNode {
    fn name(&self) -> &str {
        "scalar-window"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done && !matches!(self.state, ScalarWindowState::Done) {
            self.state = ScalarWindowState::Done;
        }

        if let ScalarWindowState::Sink {
            builders,
            partitioner,
            random_state,
            ..
        } = &mut self.state
            && recv[0] == PortState::Done
        {
            let builders = std::mem::take(builders);
            let replay = Replay::new(
                &self.params,
                builders,
                partitioner.num_partitions(),
                random_state,
            )?;
            self.state = ScalarWindowState::Replay(replay);
        }

        match &mut self.state {
            ScalarWindowState::Sink { .. } => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Ready;
                }
                send[0] = PortState::Blocked;
            },
            ScalarWindowState::Replay(replay) => {
                recv[0] = PortState::Done;
                if replay.is_exhausted() {
                    send[0] = PortState::Done;
                    self.state = ScalarWindowState::Done;
                } else {
                    send[0] = PortState::Ready;
                }
            },
            ScalarWindowState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, ScalarWindowState::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        match &mut self.state {
            ScalarWindowState::Sink {
                builders,
                partitioner,
                random_state,
                spill_ctx,
            } => {
                assert!(send_ports[0].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();
                let params = &*self.params;
                let partitioner = &*partitioner;
                let random_state = &*random_state;
                let spill_ctx = &*spill_ctx;

                for (local, mut recv) in builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut morsel) = recv.recv().await {
                            morsel.take_consume_token();
                            {
                                let mut df = morsel.df_mut().await;
                                // SAFETY: the lengths and names of the columns do not change.
                                let columns = unsafe { df.columns_mut_retain_schema() };
                                // The reductions need single-chunk columns.
                                for c in columns.iter_mut().filter(|c| {
                                    c.n_chunks() > 1 && params.read_schema.contains(c.name())
                                }) {
                                    *c = c.rechunk();
                                }
                            }
                            let keys = morsel
                                .df()
                                .await
                                .select(params.partition_by.iter().cloned())?;
                            let hash_keys =
                                HashKeys::from_df(&keys, random_state.clone(), true, false);
                            hash_keys.gen_idxs_per_partition(
                                partitioner,
                                &mut local.idxs_per_p,
                                &mut local.sketch_per_p,
                                None,
                                true,
                            );
                            local
                                .offsets_per_p
                                .extend(local.idxs_per_p.iter().map(|idxs| idxs.len()));
                            local.hash_keys.push(hash_keys);
                            let seq = morsel.seq();
                            let sf = morsel.into_sf();
                            spill_ctx.register(&sf).await;
                            local.morsels.push((seq, sf));
                        }
                        Ok(())
                    }));
                }
            },
            ScalarWindowState::Replay(replay) => {
                assert!(recv_ports[0].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                let replay = &*replay;
                let source_token = SourceToken::new();
                for mut send in senders {
                    let source_token = source_token.clone();
                    join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                        let wait_group = WaitGroup::default();
                        loop {
                            let Some((seq, df)) = replay.next_morsel().await? else {
                                break;
                            };
                            let mut morsel =
                                Morsel::new_unregistered(df, seq, source_token.clone());
                            morsel.set_consume_token(wait_group.token());
                            if send.send(morsel).await.is_err() {
                                break;
                            }
                            wait_group.wait().await;
                            if source_token.stop_requested() {
                                break;
                            }
                        }
                        Ok(())
                    }));
                }
            },
            ScalarWindowState::Done => unreachable!(),
        }
    }
}

/// The reduced input, from which the morsels are output again.
struct Replay {
    params: Arc<ScalarWindowParams>,
    builders: Vec<LocalBuilder>,
    /// The input morsels as (builder, index in builder) in output order.
    order: Vec<(usize, usize)>,
    groups: Groups,
    /// The reduced value of each group for every window.
    values: Vec<Column>,
    next: AtomicUsize,
}

/// The group of every input row.
enum Groups {
    /// Each hash partition has its own groups.
    Partitioned {
        /// The group of each row in `builders[b].idxs_per_p[p]`, as `group_ids[b][p]`.
        group_ids: Vec<Vec<Vec<IdxSize>>>,
        /// The first value of each partition in `values`.
        partition_offsets: Vec<IdxSize>,
    },
    /// The groups are shared by all pipelines.
    Local {
        /// The local group of each row of `builders[b].morsels[i]`, as `group_ids[b][i]`.
        group_ids: Vec<Vec<Vec<IdxSize>>>,
        /// The group of each local group of builder `b`.
        group_of_local_group: Vec<Vec<IdxSize>>,
    },
}

impl Replay {
    fn new(
        params: &Arc<ScalarWindowParams>,
        mut builders: Vec<LocalBuilder>,
        num_partitions: usize,
        random_state: &PlRandomState,
    ) -> PolarsResult<Self> {
        let mut order = Vec::new();
        for (b, builder) in builders.iter().enumerate() {
            order.extend((0..builder.morsels.len()).map(|i| (b, i)));
        }
        if params.maintain_order {
            order.sort_by_key(|(b, i)| builders[*b].morsels[*i].0);
        }

        let mut estimated_groups = 0;
        for p in 0..num_partitions {
            let mut sketch = CardinalitySketch::new();
            for builder in &builders {
                sketch.combine(&builder.sketch_per_p[p]);
            }
            estimated_groups += sketch.estimate();
        }

        let (groups, values) = if estimated_groups <= MAX_LOCAL_GROUPS {
            reduce_locally(params, &builders, random_state)?
        } else {
            reduce_partitions(params, &builders, num_partitions)?
        };
        let values = params
            .windows
            .iter()
            .zip(values)
            .map(|(window, values)| values.with_name(window.name.clone()).into_column())
            .collect();

        // The replay only needs the morsels and, for partitioned groups, the rows per partition.
        for builder in &mut builders {
            builder.hash_keys = Vec::new();
            builder.sketch_per_p = Vec::new();
            if matches!(groups, Groups::Local { .. }) {
                builder.idxs_per_p = Vec::new();
                builder.offsets_per_p = Vec::new();
            }
        }

        Ok(Self {
            params: params.clone(),
            builders,
            order,
            groups,
            values,
            next: AtomicUsize::new(0),
        })
    }

    fn is_exhausted(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.order.len().max(1)
    }

    /// The next input morsel with the window columns appended.
    async fn next_morsel(&self) -> PolarsResult<Option<(MorselSeq, DataFrame)>> {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let seq = MorselSeq::new(n as u64);
        if self.order.is_empty() {
            let df = DataFrame::empty_with_schema(&self.params.output_schema);
            return Ok((n == 0).then_some((seq, df)));
        }
        let Some((b, i)) = self.order.get(n).copied() else {
            return Ok(None);
        };

        let builder = &self.builders[b];
        let mut df = (*builder.morsels[i].1.get().await).clone();
        let group_for_row = match &self.groups {
            Groups::Partitioned {
                group_ids,
                partition_offsets,
            } => {
                let mut group_for_row = vec![0 as IdxSize; df.height()];
                for (p, offset) in partition_offsets.iter().enumerate() {
                    let range = builder.partition_range(i, p);
                    let rows = &builder.idxs_per_p[p][range.clone()];
                    let groups = &group_ids[b][p][range];
                    for (row, group) in rows.iter().zip(groups) {
                        // SAFETY: the rows were generated from this morsel.
                        unsafe { *group_for_row.get_unchecked_mut(*row as usize) = offset + group };
                    }
                }
                group_for_row
            },
            Groups::Local {
                group_ids,
                group_of_local_group,
            } => {
                let group_of_local_group = &group_of_local_group[b];
                group_ids[b][i]
                    .iter()
                    // SAFETY: the local groups are below the number of local groups.
                    .map(|group| unsafe { *group_of_local_group.get_unchecked(*group as usize) })
                    .collect()
            },
        };
        let columns = self
            .values
            .iter()
            // SAFETY: the groups of the rows are below the number of groups.
            .map(|values| unsafe { values.take_slice_unchecked(&group_for_row) })
            .collect::<Vec<_>>();
        // SAFETY: the window columns have the height of the morsel and new names.
        unsafe { df.hstack_mut_unchecked(&columns) };
        Ok(Some((seq, df)))
    }
}

/// Concatenates the values of the partitions of every window.
fn concat_values(
    values_per_partition: Vec<Vec<Series>>,
    num_windows: usize,
) -> PolarsResult<Vec<Series>> {
    let mut values_per_window = vec![Vec::with_capacity(values_per_partition.len()); num_windows];
    for values in values_per_partition {
        for (parts, series) in values_per_window.iter_mut().zip(values) {
            parts.push(series);
        }
    }
    values_per_window
        .into_iter()
        .map(|parts| {
            let mut parts = parts.into_iter();
            let mut values = parts.next().unwrap();
            for part in parts {
                values.append_owned(part)?;
            }
            Ok(values)
        })
        .collect()
}

/// Reduces every hash partition on its own.
fn reduce_partitions(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    num_partitions: usize,
) -> PolarsResult<(Groups, Vec<Series>)> {
    let partitions = RAYON.install(|| {
        (0..num_partitions)
            .into_par_iter()
            .map(|p| reduce_partition(params, builders, p))
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut group_ids = (0..builders.len())
        .map(|_| Vec::with_capacity(num_partitions))
        .collect::<Vec<_>>();
    let mut partition_offsets = Vec::with_capacity(num_partitions);
    let mut num_groups = 0 as IdxSize;
    let mut values = Vec::with_capacity(num_partitions);
    for partition in partitions {
        partition_offsets.push(num_groups);
        num_groups += partition.num_groups;
        for (ids, per_builder) in group_ids.iter_mut().zip(partition.group_ids) {
            ids.push(per_builder);
        }
        values.push(partition.values);
    }
    let values = concat_values(values, params.windows.len())?;
    Ok((
        Groups::Partitioned {
            group_ids,
            partition_offsets,
        },
        values,
    ))
}

struct ReducedPartition {
    num_groups: IdxSize,
    /// The group of each row in `builders[b].idxs_per_p[p]`, as `group_ids[b]`.
    group_ids: Vec<Vec<IdxSize>>,
    values: Vec<Series>,
}

/// Reduces the rows of the input morsels in partition `p`.
fn reduce_partition(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    p: usize,
) -> PolarsResult<ReducedPartition> {
    let mut grouper = new_hash_grouper(params.key_schema.clone());
    let mut reductions = new_reductions(params);

    let mut group_ids = Vec::with_capacity(builders.len());
    for builder in builders {
        let mut ids = Vec::with_capacity(builder.idxs_per_p[p].len());
        for (i, (seq, sf)) in builder.morsels.iter().enumerate() {
            let rows = &builder.idxs_per_p[p][builder.partition_range(i, p)];
            if rows.is_empty() {
                continue;
            }
            let start = ids.len();
            // SAFETY: the rows were generated from this morsel and its keys.
            unsafe { grouper.insert_keys_subset(&builder.hash_keys[i], rows, Some(&mut ids)) };
            let df = sf.get_blocking();
            update_reductions(
                params,
                &mut reductions,
                grouper.num_groups(),
                &df,
                rows,
                &ids[start..],
                seq,
            )?;
        }
        group_ids.push(ids);
    }

    let num_groups = grouper.num_groups();
    let values = finalize_reductions(&mut reductions, num_groups)?;
    Ok(ReducedPartition {
        num_groups,
        group_ids,
        values,
    })
}

/// Reduces the morsels of every builder on its own, then merges the groups of the builders.
fn reduce_locally(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    random_state: &PlRandomState,
) -> PolarsResult<(Groups, Vec<Series>)> {
    struct Local {
        grouper: Box<dyn polars_expr::groups::Grouper>,
        reductions: Vec<Box<dyn GroupedReduction>>,
        group_ids: Vec<Vec<IdxSize>>,
    }

    let locals = RAYON.install(|| {
        builders
            .par_iter()
            .map(|builder| {
                let mut grouper = new_hash_grouper(params.key_schema.clone());
                let mut reductions = new_reductions(params);
                let mut group_ids = Vec::with_capacity(builder.morsels.len());
                let mut all_rows = Vec::new();
                for (i, (seq, sf)) in builder.morsels.iter().enumerate() {
                    let df = sf.get_blocking();
                    let height = df.height() as IdxSize;
                    all_rows.extend(all_rows.len() as IdxSize..height);
                    let rows = &all_rows[..height as usize];
                    let mut ids = Vec::with_capacity(rows.len());
                    // SAFETY: the rows are in-bounds of this morsel and its keys.
                    unsafe {
                        grouper.insert_keys_subset(&builder.hash_keys[i], rows, Some(&mut ids))
                    };
                    update_reductions(
                        params,
                        &mut reductions,
                        grouper.num_groups(),
                        &df,
                        rows,
                        &ids,
                        seq,
                    )?;
                    group_ids.push(ids);
                }
                Ok(Local {
                    grouper,
                    reductions,
                    group_ids,
                })
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut grouper = new_hash_grouper(params.key_schema.clone());
    let mut reductions = new_reductions(params);
    let mut group_ids = Vec::with_capacity(locals.len());
    let mut group_of_local_group = Vec::with_capacity(locals.len());
    for local in locals {
        let keys = local.grouper.get_keys_in_group_order(&params.key_schema);
        let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
        let local_groups = (0..local.grouper.num_groups()).collect::<Vec<_>>();
        let mut groups = Vec::with_capacity(local_groups.len());
        // SAFETY: the local groups are the rows of `keys`.
        unsafe { grouper.insert_keys_subset(&hash_keys, &local_groups, Some(&mut groups)) };
        for (reduction, local_reduction) in reductions.iter_mut().zip(&local.reductions) {
            reduction.resize(grouper.num_groups());
            // SAFETY: the local groups are in-bounds of the local reduction and `groups` of the
            // merged one.
            unsafe { reduction.combine_subset(&**local_reduction, &local_groups, &groups)? };
        }
        group_ids.push(local.group_ids);
        group_of_local_group.push(groups);
    }
    let values = finalize_reductions(&mut reductions, grouper.num_groups())?;
    Ok((
        Groups::Local {
            group_ids,
            group_of_local_group,
        },
        values,
    ))
}

fn new_reductions(params: &ScalarWindowParams) -> Vec<Box<dyn GroupedReduction>> {
    params
        .windows
        .iter()
        .map(|window| window.reduction.new_empty())
        .collect()
}

/// Feeds rows `rows` of `df`, with groups `group_ids`, to the reductions.
fn update_reductions(
    params: &ScalarWindowParams,
    reductions: &mut [Box<dyn GroupedReduction>],
    num_groups: IdxSize,
    df: &DataFrame,
    rows: &[IdxSize],
    group_ids: &[IdxSize],
    seq: &MorselSeq,
) -> PolarsResult<()> {
    for (window, reduction) in params.windows.iter().zip(reductions) {
        reduction.resize(num_groups);
        let column = df.column(&window.input)?;
        // SAFETY: the rows are in-bounds and the group ids are below `num_groups`.
        unsafe { reduction.update_groups_subset(&[column], rows, group_ids, seq.to_u64())? };
    }
    Ok(())
}

fn finalize_reductions(
    reductions: &mut [Box<dyn GroupedReduction>],
    num_groups: IdxSize,
) -> PolarsResult<Vec<Series>> {
    reductions
        .iter_mut()
        .map(|reduction| {
            reduction.resize(num_groups);
            reduction.finalize()
        })
        .collect()
}
