use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_expr::groups::new_hash_grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame, memory_manager};
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

/// The hash partitions are reduced over blocks of about this many rows per partition.
const ROWS_PER_PARTITION_PER_BLOCK: usize = 4096;

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
}

/// Evaluates scalar windows that share one partitioning. The input morsels are kept and reduced in
/// parallel, per pipeline if there are few groups and per hash partition otherwise. Then the
/// morsels are output again with the reduced value of their group appended.
///
/// The input morsels can be spilled until they are output. The partition keys of the input stay in
/// memory until the reduction is done, the group of every row until its morsel is output, and the
/// groups and their reduced values until all morsels are output.
pub struct ScalarWindowNode {
    params: Arc<ScalarWindowParams>,
    spill_ctx: MostRecentSpillContext,
    state: ScalarWindowState,
}

enum ScalarWindowState {
    Sink {
        builders: Vec<LocalBuilder>,
        partitioner: HashPartitioner,
        random_state: PlRandomState,
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
            spill_ctx: MostRecentSpillContext::new("scalar-window".into(), task_metrics),
            state: ScalarWindowState::Sink {
                builders,
                partitioner: HashPartitioner::new(num_partitions, 0),
                random_state: PlRandomState::default(),
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
        let spill_ctx = &self.spill_ctx;
        match &mut self.state {
            ScalarWindowState::Sink {
                builders,
                partitioner,
                random_state,
            } => {
                assert!(send_ports[0].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();
                let params = &*self.params;
                let partitioner = &*partitioner;
                let random_state = &*random_state;

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
                            memory_manager().spill().await;
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
    /// The input morsels in input order. Each is taken once.
    morsels: Vec<Mutex<Option<ReplayMorsel>>>,
    /// The reduced value of each group for every window.
    values: Vec<Column>,
    next: AtomicUsize,
}

struct ReplayMorsel {
    frame: SpillFrame,
    /// The group of each row, an index into the values.
    groups: Vec<IdxSize>,
}

impl Replay {
    fn new(
        params: &Arc<ScalarWindowParams>,
        builders: Vec<LocalBuilder>,
        num_partitions: usize,
        random_state: &PlRandomState,
    ) -> PolarsResult<Self> {
        let mut estimated_groups = 0;
        for p in 0..num_partitions {
            let mut sketch = CardinalitySketch::new();
            for builder in &builders {
                sketch.combine(&builder.sketch_per_p[p]);
            }
            estimated_groups += sketch.estimate();
        }

        let local = estimated_groups <= MAX_LOCAL_GROUPS;
        if polars_config::config().verbose() {
            let how = if local {
                "per pipeline"
            } else {
                "per hash partition"
            };
            eprintln!("[scalar-window]: reduce {how} (estimated groups: {estimated_groups})");
        }
        let (groups, values) = if local {
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

        let mut morsels = builders
            .into_iter()
            .zip(groups)
            .flat_map(|(builder, groups)| builder.morsels.into_iter().zip(groups))
            .collect::<Vec<_>>();
        morsels.sort_by_key(|((seq, _), _)| *seq);
        let morsels = morsels
            .into_iter()
            .map(|((_, frame), groups)| Mutex::new(Some(ReplayMorsel { frame, groups })))
            .collect();

        Ok(Self {
            params: params.clone(),
            morsels,
            values,
            next: AtomicUsize::new(0),
        })
    }

    fn is_exhausted(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.morsels.len().max(1)
    }

    /// Takes the next input morsel and appends the window columns.
    async fn next_morsel(&self) -> PolarsResult<Option<(MorselSeq, DataFrame)>> {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let seq = MorselSeq::new(n as u64);
        if self.morsels.is_empty() {
            let df = DataFrame::empty_with_schema(&self.params.output_schema);
            return Ok((n == 0).then_some((seq, df)));
        }
        let Some(morsel) = self.morsels.get(n) else {
            return Ok(None);
        };
        let ReplayMorsel { frame, groups } = morsel.lock().unwrap().take().unwrap();

        let mut df = frame.into_df().await;
        let columns = self
            .values
            .iter()
            // SAFETY: the groups of the rows are below the number of groups.
            .map(|values| unsafe { values.take_slice_unchecked(&groups) })
            .collect::<Vec<_>>();
        // SAFETY: the window columns have the height of the morsel and new names.
        unsafe { df.hstack_mut_unchecked(&columns) };
        Ok(Some((seq, df)))
    }
}

/// The group of every row of `builders[b].morsels[i]`, as `groups[b][i]`.
type MorselGroups = Vec<Vec<Vec<IdxSize>>>;

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
            // The replay gathers from one chunk much faster than from many.
            Ok(values.rechunk())
        })
        .collect()
}

/// Reduces every hash partition on its own. All partitions read a block of morsels while it is
/// loaded, so every morsel is loaded once.
fn reduce_partitions(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    num_partitions: usize,
) -> PolarsResult<(MorselGroups, Vec<Series>)> {
    struct Partition {
        grouper: Box<dyn polars_expr::groups::Grouper>,
        reductions: Vec<Box<dyn GroupedReduction>>,
        /// The group of each row in `builders[b].idxs_per_p[p]`, as `group_ids[b]`.
        group_ids: Vec<Vec<IdxSize>>,
    }

    let mut partitions = (0..num_partitions)
        .map(|_| Partition {
            grouper: new_hash_grouper(params.key_schema.clone()),
            reductions: new_reductions(params),
            group_ids: vec![Vec::new(); builders.len()],
        })
        .collect::<Vec<_>>();

    // Visit the morsels in input order, the order in which they are also output.
    let mut morsels = builders
        .iter()
        .enumerate()
        .flat_map(|(b, builder)| (0..builder.morsels.len()).map(move |i| (b, i)))
        .collect::<Vec<_>>();
    morsels.sort_by_key(|(b, i)| builders[*b].morsels[*i].0);
    // The groups of a builder are pushed in the order of its morsels.
    debug_assert!(
        builders
            .iter()
            .all(|builder| builder.morsels.is_sorted_by_key(|(seq, _)| *seq))
    );
    let block_rows = num_partitions * ROWS_PER_PARTITION_PER_BLOCK;
    let mut block_start = 0;
    while block_start < morsels.len() {
        let mut block_end = block_start;
        let mut rows_in_block = 0;
        while block_end < morsels.len() && rows_in_block < block_rows {
            let (b, i) = morsels[block_end];
            rows_in_block += builders[b].morsels[i].1.height();
            block_end += 1;
        }
        let block = &morsels[block_start..block_end];

        RAYON.install(|| {
            let dfs = block
                .par_iter()
                .map(|(b, i)| builders[*b].morsels[*i].1.get_blocking())
                .collect::<Vec<_>>();
            partitions
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(p, partition)| {
                    let Partition {
                        grouper,
                        reductions,
                        group_ids,
                    } = partition;
                    for ((b, i), df) in block.iter().zip(&dfs) {
                        let builder = &builders[*b];
                        let rows = &builder.idxs_per_p[p][builder.partition_range(*i, p)];
                        if rows.is_empty() {
                            continue;
                        }
                        let ids = &mut group_ids[*b];
                        let start = ids.len();
                        // SAFETY: the rows were generated from this morsel and its keys.
                        unsafe {
                            grouper.insert_keys_subset(&builder.hash_keys[*i], rows, Some(ids))
                        };
                        update_reductions(
                            params,
                            reductions,
                            grouper.num_groups(),
                            df,
                            rows,
                            &ids[start..],
                            &builder.morsels[*i].0,
                        )?;
                    }
                    PolarsResult::Ok(())
                })
        })?;
        memory_manager().spill_blocking();
        block_start = block_end;
    }

    let partitions = RAYON.install(|| {
        partitions
            .into_par_iter()
            .map(|mut partition| {
                let num_groups = partition.grouper.num_groups();
                let values = finalize_reductions(&mut partition.reductions, num_groups)?;
                Ok((num_groups, values, partition.group_ids))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    // The groups of each partition as `partition_groups[b][p]`.
    let mut partition_groups = (0..builders.len())
        .map(|_| Vec::with_capacity(num_partitions))
        .collect::<Vec<_>>();
    let mut partition_offsets = Vec::with_capacity(num_partitions);
    let mut num_groups = 0 as IdxSize;
    let mut values = Vec::with_capacity(num_partitions);
    for (partition_num_groups, partition_values, group_ids) in partitions {
        partition_offsets.push(num_groups);
        num_groups += partition_num_groups;
        values.push(partition_values);
        for (groups, per_builder) in partition_groups.iter_mut().zip(group_ids) {
            groups.push(per_builder);
        }
    }
    let values = concat_values(values, params.windows.len())?;

    let groups = RAYON.install(|| {
        builders
            .par_iter()
            .zip(partition_groups)
            .map(|(builder, partition_groups)| {
                (0..builder.morsels.len())
                    .map(|i| {
                        let mut groups = vec![0 as IdxSize; builder.morsels[i].1.height()];
                        for (p, offset) in partition_offsets.iter().enumerate() {
                            let range = builder.partition_range(i, p);
                            let rows = &builder.idxs_per_p[p][range.clone()];
                            for (row, group) in rows.iter().zip(&partition_groups[p][range]) {
                                // SAFETY: the rows were generated from this morsel.
                                unsafe {
                                    *groups.get_unchecked_mut(*row as usize) = offset + group
                                };
                            }
                        }
                        groups
                    })
                    .collect()
            })
            .collect()
    });
    Ok((groups, values))
}

/// Reduces the morsels of every builder on its own, then merges the groups of the builders.
fn reduce_locally(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    random_state: &PlRandomState,
) -> PolarsResult<(MorselGroups, Vec<Series>)> {
    struct Local {
        grouper: Box<dyn polars_expr::groups::Grouper>,
        reductions: Vec<Box<dyn GroupedReduction>>,
        groups: Vec<Vec<IdxSize>>,
    }

    let locals = RAYON.install(|| {
        builders
            .par_iter()
            .map(|builder| {
                let mut grouper = new_hash_grouper(params.key_schema.clone());
                let mut reductions = new_reductions(params);
                let mut groups = Vec::with_capacity(builder.morsels.len());
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
                    drop(df);
                    memory_manager().spill_blocking();
                    groups.push(ids);
                }
                Ok(Local {
                    grouper,
                    reductions,
                    groups,
                })
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut grouper = new_hash_grouper(params.key_schema.clone());
    let mut reductions = new_reductions(params);
    let mut groups = Vec::with_capacity(locals.len());
    for local in locals {
        let keys = local.grouper.get_keys_in_group_order(&params.key_schema);
        let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
        let local_groups = (0..local.grouper.num_groups()).collect::<Vec<_>>();
        let mut group_of_local_group = Vec::with_capacity(local_groups.len());
        // SAFETY: the local groups are the rows of `keys`.
        unsafe {
            grouper.insert_keys_subset(&hash_keys, &local_groups, Some(&mut group_of_local_group))
        };
        for (reduction, local_reduction) in reductions.iter_mut().zip(&local.reductions) {
            reduction.resize(grouper.num_groups());
            // SAFETY: the local groups are in-bounds of the local reduction and
            // `group_of_local_group` of the merged one.
            unsafe {
                reduction.combine_subset(
                    &**local_reduction,
                    &local_groups,
                    &group_of_local_group,
                )?
            };
        }
        groups.push((local.groups, group_of_local_group));
    }
    let groups = RAYON.install(|| {
        groups
            .into_par_iter()
            .map(|(mut morsel_groups, group_of_local_group)| {
                for group in morsel_groups.iter_mut().flatten() {
                    // SAFETY: the local groups are below the number of local groups.
                    *group = unsafe { *group_of_local_group.get_unchecked(*group as usize) };
                }
                morsel_groups
            })
            .collect()
    });
    let values = finalize_reductions(&mut reductions, grouper.num_groups())?;
    Ok((groups, values))
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
