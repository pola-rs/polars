use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::groups::{Grouper, new_hash_grouper};
use polars_expr::hash_keys::HashKeys;
use polars_expr::reduce::GroupedReduction;
use polars_ooc::{MostRecentSpillContext, SpillFrame, memory_manager};
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use super::window::{PARTITIONS_PER_PIPELINE, PartitionedMorsels};
use crate::morsel::{MorselSeq, SourceToken};

/// Up to this many estimated groups, every pipeline reduces its own morsels and the results are
/// merged.
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
    pub windows: Vec<ScalarWindow>,
    /// The partition keys and the columns the reductions read.
    pub read_schema: Arc<Schema>,
    /// The partition keys.
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
    /// If there is a memory budget, a block of the partitioned reduction loads at most this many
    /// bytes of input morsels, or one morsel.
    max_block_bytes: Option<usize>,
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

struct LocalBuilder {
    input: PartitionedMorsels,
    /// The hashed keys of `input.morsels[i]`.
    hash_keys: Vec<HashKeys>,
    sketch_per_p: Vec<CardinalitySketch>,
}

impl ScalarWindowNode {
    pub fn new(
        params: Arc<ScalarWindowParams>,
        num_pipelines: usize,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        let num_partitions = num_pipelines * PARTITIONS_PER_PIPELINE;
        let budget = polars_config::config().ooc_memory_budget_bytes();
        let max_block_bytes = (budget != u64::MAX).then_some((budget / 8) as usize);
        let builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                input: PartitionedMorsels::new(num_partitions, max_block_bytes.is_some()),
                hash_keys: Vec::new(),
                sketch_per_p: vec![CardinalitySketch::new(); num_partitions],
            })
            .collect();
        Self {
            params,
            max_block_bytes,
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

        if send[0] == PortState::Done {
            self.state = ScalarWindowState::Done;
        }

        if let ScalarWindowState::Sink {
            builders,
            partitioner,
            random_state,
        } = &mut self.state
            && recv[0] == PortState::Done
        {
            let builders = std::mem::take(builders);
            let replay = Replay::new(
                &self.params,
                builders,
                partitioner.num_partitions(),
                random_state,
                self.max_block_bytes,
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
            } => {
                assert!(send_ports[0].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();
                let params = &*self.params;
                let partitioner = &*partitioner;
                let random_state = &*random_state;
                let spill_ctx = &self.spill_ctx;

                for (local, mut recv) in builders.iter_mut().zip(receivers) {
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            let hash_keys = local
                                .input
                                .push(
                                    morsel,
                                    params.key_schema.iter_names(),
                                    &params.read_schema,
                                    partitioner,
                                    random_state,
                                    &mut local.sketch_per_p,
                                    spill_ctx,
                                )
                                .await?;
                            local.hash_keys.push(hash_keys);
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
        max_block_bytes: Option<usize>,
    ) -> PolarsResult<Self> {
        let estimated_groups_per_p = RAYON.install(|| {
            (0..num_partitions)
                .into_par_iter()
                .map(|p| {
                    let mut sketch = CardinalitySketch::new();
                    for builder in &builders {
                        sketch.combine(&builder.sketch_per_p[p]);
                    }
                    sketch.estimate()
                })
                .collect::<Vec<_>>()
        });
        let estimated_groups = estimated_groups_per_p.iter().sum::<usize>();

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
            reduce_locally(params, &builders, estimated_groups, random_state)?
        } else {
            reduce_partitions(params, &builders, &estimated_groups_per_p, max_block_bytes)?
        };

        let mut morsels = builders
            .into_iter()
            .zip(groups)
            .flat_map(|(builder, groups)| builder.input.morsels.into_iter().zip(groups))
            .collect::<Vec<_>>();
        morsels.sort_by_key(|((seq, _), _)| *seq);
        let morsels = morsels
            .into_iter()
            .map(|((_, frame), groups)| Mutex::new(Some(ReplayMorsel { frame, groups })))
            .collect();

        Ok(Self {
            params: params.clone(),
            morsels,
            values: values.into_columns(),
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

/// The group of every row of `builders[b].input.morsels[i]`, as `groups[b][i]`.
type MorselGroups = Vec<Vec<Vec<IdxSize>>>;

/// A grouper and the reductions of its groups.
struct Reducer {
    grouper: Box<dyn Grouper>,
    reductions: Vec<Box<dyn GroupedReduction>>,
}

impl Reducer {
    fn new(params: &ScalarWindowParams, estimated_groups: usize) -> Self {
        let mut grouper = new_hash_grouper(params.key_schema.clone());
        grouper.reserve(estimated_groups);
        let reductions = params
            .windows
            .iter()
            .map(|window| {
                let mut reduction = window.reduction.new_empty();
                reduction.reserve(estimated_groups);
                reduction
            })
            .collect();
        Self {
            grouper,
            reductions,
        }
    }

    /// Adds rows `rows` of `df` and pushes the group of each row to `group_ids`.
    ///
    /// # Safety
    /// The rows must be in-bounds of `df` and of `hash_keys`, the keys of `df`.
    unsafe fn update(
        &mut self,
        params: &ScalarWindowParams,
        df: &DataFrame,
        hash_keys: &HashKeys,
        rows: &[IdxSize],
        seq: MorselSeq,
        group_ids: &mut Vec<IdxSize>,
    ) -> PolarsResult<()> {
        let start = group_ids.len();
        unsafe {
            self.grouper
                .insert_keys_subset(hash_keys, rows, Some(group_ids))
        };
        let num_groups = self.grouper.num_groups();
        for (window, reduction) in params.windows.iter().zip(&mut self.reductions) {
            reduction.resize(num_groups);
            let column = df.column(&window.input)?;
            // SAFETY: the group ids are below `num_groups`.
            unsafe {
                reduction.update_groups_subset(
                    &[column],
                    rows,
                    &group_ids[start..],
                    seq.to_u64(),
                )?
            };
        }
        Ok(())
    }

    /// The reduced value of every group, with one column per window.
    fn finalize(mut self, params: &ScalarWindowParams) -> PolarsResult<DataFrame> {
        let num_groups = self.grouper.num_groups();
        let columns = params
            .windows
            .iter()
            .zip(&mut self.reductions)
            .map(|(window, reduction)| {
                reduction.resize(num_groups);
                Ok(reduction
                    .finalize()?
                    .with_name(window.name.clone())
                    .into_column())
            })
            .collect::<PolarsResult<_>>()?;
        DataFrame::new(num_groups as usize, columns)
    }
}

/// Reduces every hash partition on its own. All partitions read a block of morsels while it is
/// loaded, so every morsel is loaded once.
fn reduce_partitions(
    params: &ScalarWindowParams,
    builders: &[LocalBuilder],
    estimated_groups_per_p: &[usize],
    max_block_bytes: Option<usize>,
) -> PolarsResult<(MorselGroups, DataFrame)> {
    struct Partition {
        reducer: Reducer,
        /// The group of each row in `builders[b].input.idxs_per_p[p]`, as `group_ids[b]`.
        group_ids: Vec<Vec<IdxSize>>,
    }

    let num_partitions = estimated_groups_per_p.len();
    let mut partitions = RAYON.install(|| {
        estimated_groups_per_p
            .par_iter()
            .enumerate()
            .map(|(p, estimated_groups)| Partition {
                reducer: Reducer::new(params, estimated_groups * 5 / 4),
                group_ids: builders
                    .iter()
                    .map(|builder| Vec::with_capacity(builder.input.idxs_per_p[p].len()))
                    .collect(),
            })
            .collect::<Vec<_>>()
    });

    // Visit the morsels in input order, the order in which they are also output.
    let mut morsels = builders
        .iter()
        .enumerate()
        .flat_map(|(b, builder)| (0..builder.input.morsels.len()).map(move |i| (b, i)))
        .collect::<Vec<_>>();
    morsels.sort_by_key(|(b, i)| builders[*b].input.morsels[*i].0);
    // The groups of a builder are pushed in the order of its morsels.
    debug_assert!(
        builders
            .iter()
            .all(|builder| builder.input.morsels.is_sorted_by_key(|(seq, _)| *seq))
    );
    let block_rows = num_partitions * ROWS_PER_PARTITION_PER_BLOCK;
    let mut block_start = 0;
    let mut num_blocks = 0;
    while block_start < morsels.len() {
        let mut block_end = block_start;
        let mut rows_in_block = 0;
        let mut bytes_in_block = 0;
        while block_end < morsels.len() && rows_in_block < block_rows {
            let (b, i) = morsels[block_end];
            if let Some(max_block_bytes) = max_block_bytes {
                let bytes = builders[b].input.morsel_bytes[i];
                if block_end > block_start && bytes_in_block + bytes > max_block_bytes {
                    break;
                }
                bytes_in_block += bytes;
            }
            rows_in_block += builders[b].input.morsels[i].1.height();
            block_end += 1;
        }
        let block = &morsels[block_start..block_end];

        RAYON.install(|| {
            let dfs = block
                .par_iter()
                .map(|(b, i)| builders[*b].input.morsels[*i].1.get_blocking())
                .collect::<Vec<_>>();
            partitions
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(p, partition)| {
                    for ((b, i), df) in block.iter().zip(&dfs) {
                        let builder = &builders[*b];
                        let rows =
                            &builder.input.idxs_per_p[p][builder.input.partition_range(*i, p)];
                        if rows.is_empty() {
                            continue;
                        }
                        // SAFETY: the rows were generated from this morsel and its keys.
                        unsafe {
                            partition.reducer.update(
                                params,
                                df,
                                &builder.hash_keys[*i],
                                rows,
                                builder.input.morsels[*i].0,
                                &mut partition.group_ids[*b],
                            )?
                        };
                    }
                    PolarsResult::Ok(())
                })
        })?;
        memory_manager().spill_blocking();
        block_start = block_end;
        num_blocks += 1;
    }
    if polars_config::config().verbose() {
        eprintln!(
            "[scalar-window]: reduced {} morsels in {num_blocks} blocks",
            morsels.len()
        );
    }

    let partitions = RAYON.install(|| {
        partitions
            .into_par_iter()
            .map(|partition| Ok((partition.reducer.finalize(params)?, partition.group_ids)))
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    // The groups of each partition as `partition_groups[b][p]`.
    let mut partition_groups = (0..builders.len())
        .map(|_| Vec::with_capacity(num_partitions))
        .collect::<Vec<_>>();
    let mut partition_offsets = Vec::with_capacity(num_partitions);
    let mut num_groups = 0 as IdxSize;
    let mut values = Vec::with_capacity(num_partitions);
    for (partition_values, group_ids) in partitions {
        partition_offsets.push(num_groups);
        num_groups += partition_values.height() as IdxSize;
        values.push(partition_values);
        for (groups, per_builder) in partition_groups.iter_mut().zip(group_ids) {
            groups.push(per_builder);
        }
    }
    let mut values = accumulate_dataframes_vertical_unchecked(values);
    values.rechunk_mut_par();

    let groups = RAYON.install(|| {
        builders
            .par_iter()
            .zip(partition_groups)
            .map(|(builder, partition_groups)| {
                (0..builder.input.morsels.len())
                    .map(|i| {
                        let mut groups = vec![0 as IdxSize; builder.input.morsels[i].1.height()];
                        for (p, offset) in partition_offsets.iter().enumerate() {
                            let range = builder.input.partition_range(i, p);
                            let rows = &builder.input.idxs_per_p[p][range.clone()];
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
    estimated_groups: usize,
    random_state: &PlRandomState,
) -> PolarsResult<(MorselGroups, DataFrame)> {
    let locals = RAYON.install(|| {
        builders
            .par_iter()
            .map(|builder| {
                let mut reducer = Reducer::new(params, 0);
                let mut groups = Vec::with_capacity(builder.input.morsels.len());
                let mut all_rows = Vec::new();
                for ((seq, sf), hash_keys) in builder.input.morsels.iter().zip(&builder.hash_keys) {
                    let df = sf.get_blocking();
                    let height = df.height() as IdxSize;
                    all_rows.extend(all_rows.len() as IdxSize..height);
                    let rows = &all_rows[..height as usize];
                    let mut ids = Vec::with_capacity(rows.len());
                    // SAFETY: the rows are in-bounds of this morsel and its keys.
                    unsafe { reducer.update(params, &df, hash_keys, rows, *seq, &mut ids)? };
                    drop(df);
                    memory_manager().spill_blocking();
                    groups.push(ids);
                }
                Ok((reducer, groups))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut merged = Reducer::new(params, estimated_groups);
    let mut groups = Vec::with_capacity(locals.len());
    for (local, local_morsel_groups) in locals {
        let keys = local.grouper.get_keys_in_group_order(&params.key_schema);
        let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
        let local_groups = (0..local.grouper.num_groups()).collect::<Vec<_>>();
        let mut group_of_local_group = Vec::with_capacity(local_groups.len());
        // SAFETY: the local groups are the rows of `keys`.
        unsafe {
            merged.grouper.insert_keys_subset(
                &hash_keys,
                &local_groups,
                Some(&mut group_of_local_group),
            )
        };
        for (reduction, local_reduction) in merged.reductions.iter_mut().zip(&local.reductions) {
            reduction.resize(merged.grouper.num_groups());
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
        groups.push((local_morsel_groups, group_of_local_group));
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
    Ok((groups, merged.finalize(params)?))
}
