use std::sync::Arc;

use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::groups::new_hash_grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::prelude::WindowExpr;
use polars_ooc::{LeastRecentSpillContext, ParameterFreeSpillContext, SpillFrame};
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use super::compute_node_prelude::*;
use super::in_memory_source::InMemorySourceNode;

/// Only `num_pipelines` partitions are gathered, sorted and evaluated at the same time. More
/// partitions make each one smaller, which lowers the memory used for that work. The input and the
/// result are still held in memory in full.
const PARTITIONS_PER_PIPELINE: usize = 16;

/// How the node outputs its rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WindowOutput {
    /// The input morsels in input order, with the window columns appended.
    OrderedMorsels,
    /// The input morsels in any order, with the window columns appended.
    UnorderedMorsels,
    /// The partitions sorted by partition and order key, with the window columns appended. Only
    /// used when the windows read every input column, so a partition holds whole rows.
    Partitions,
}

pub struct WindowParams {
    partition_by: Vec<PlSmallStr>,
    order_by: Option<(PlSmallStr, SortOptions)>,
    /// The window expressions with their output names.
    exprs: Vec<(PlSmallStr, WindowExpr)>,
    /// The input columns the window expressions read, including the keys.
    read_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    /// Evaluate the rows of a partition in input order.
    ordered_eval: bool,
    output: WindowOutput,
}

impl WindowParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_by: Vec<PlSmallStr>,
        order_by: Option<(PlSmallStr, SortOptions)>,
        exprs: Vec<(PlSmallStr, WindowExpr)>,
        input_schema: &Schema,
        read_schema: Schema,
        output_schema: Arc<Schema>,
        ordered_eval: bool,
        maintain_order: bool,
    ) -> Self {
        let output = if maintain_order {
            WindowOutput::OrderedMorsels
        } else if read_schema.len() == input_schema.len() {
            WindowOutput::Partitions
        } else {
            WindowOutput::UnorderedMorsels
        };
        Self {
            partition_by,
            order_by,
            exprs,
            read_schema: Arc::new(read_schema),
            output_schema,
            ordered_eval,
            output,
        }
    }
}

/// Evaluates window expressions that share one partitioning. The columns the windows read are
/// hash partitioned on the partition keys, and each hash partition is evaluated in parallel.
///
/// The input rows are output with the window columns appended. Without `maintain_order` the rows
/// are output in an unspecified order.
///
/// The input morsels can be spilled while they are received. Evaluation loads the whole input and
/// builds the whole result before the first row is output.
pub struct WindowNode {
    params: Arc<WindowParams>,
    state: WindowState,
}

enum WindowState {
    Sink {
        builders: Vec<LocalBuilder>,
        partitioner: HashPartitioner,
        random_state: PlRandomState,
        spill_ctx: LeastRecentSpillContext,
    },
    Source(InMemorySourceNode),
    Done,
}

#[derive(Default)]
struct LocalBuilder {
    morsels: Vec<(MorselSeq, SpillFrame)>,
    // The rows of morsels[i] in partition p are
    // idxs_per_p[p][offsets_per_p[i * num_partitions + p]..offsets_per_p[(i + 1) * num_partitions + p]].
    idxs_per_p: Vec<Vec<IdxSize>>,
    offsets_per_p: Vec<usize>,
}

impl WindowNode {
    pub fn new(params: Arc<WindowParams>, num_pipelines: usize) -> Self {
        let num_partitions = num_pipelines * PARTITIONS_PER_PIPELINE;
        let builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                morsels: Vec::new(),
                idxs_per_p: vec![Vec::new(); num_partitions],
                offsets_per_p: vec![0; num_partitions],
            })
            .collect();
        Self {
            params,
            state: WindowState::Sink {
                builders,
                partitioner: HashPartitioner::new(num_partitions, 0),
                random_state: PlRandomState::default(),
                spill_ctx: LeastRecentSpillContext::new("window".into()),
            },
        }
    }
}

impl ComputeNode for WindowNode {
    fn name(&self) -> &str {
        "window"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done && !matches!(self.state, WindowState::Done) {
            self.state = WindowState::Done;
        }

        if let WindowState::Sink {
            builders,
            partitioner,
            random_state,
            ..
        } = &mut self.state
            && recv[0] == PortState::Done
        {
            let builders = std::mem::take(builders);
            let out = evaluate_partitions(
                &self.params,
                builders,
                partitioner.num_partitions(),
                random_state,
                &state.in_memory_exec_state,
            )?;
            self.state =
                WindowState::Source(InMemorySourceNode::new(Arc::new(out), MorselSeq::default()));
        }

        match &mut self.state {
            WindowState::Sink { .. } => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Ready;
                }
                send[0] = PortState::Blocked;
            },
            WindowState::Source(source) => {
                recv[0] = PortState::Done;
                source.update_state(&mut [], send, state)?;
            },
            WindowState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, WindowState::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        match &mut self.state {
            WindowState::Sink {
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
                                // Gathers need single-chunk columns.
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
                                &mut [],
                                None,
                                true,
                            );
                            local
                                .offsets_per_p
                                .extend(local.idxs_per_p.iter().map(|idxs| idxs.len()));
                            let seq = morsel.seq();
                            let sf = morsel.into_sf();
                            spill_ctx.register(&sf).await;
                            local.morsels.push((seq, sf));
                        }
                        Ok(())
                    }));
                }
            },
            WindowState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            WindowState::Done => unreachable!(),
        }
    }
}

/// An input morsel with its position in the input.
struct InputMorsel<'a> {
    seq: MorselSeq,
    sf: &'a SpillFrame,
    builder: &'a LocalBuilder,
    /// The index of the morsel in `builder.morsels`.
    idx: usize,
    /// The input row of the first row of the morsel.
    first_row: IdxSize,
}

impl InputMorsel<'_> {
    /// The rows of the morsel in partition `p`.
    fn partition_rows(&self, p: usize) -> &[IdxSize] {
        let b = self.builder;
        let num_partitions = b.idxs_per_p.len();
        let start = b.offsets_per_p[self.idx * num_partitions + p];
        let stop = b.offsets_per_p[(self.idx + 1) * num_partitions + p];
        &b.idxs_per_p[p][start..stop]
    }
}

/// The read columns of the rows in one partition.
struct GatheredPartition {
    df: DataFrame,
    /// The input row of each row of `df`. Empty for [`WindowOutput::Partitions`].
    input_row_for_gathered_row: Vec<IdxSize>,
}

/// The windows of one partition, evaluated on its rows sorted by partition and order key.
struct SortedPartition {
    /// The sorted rows.
    df: DataFrame,
    /// The window columns for the rows of `df`.
    windows: Vec<Column>,
    /// The row of the gathered partition for each row of `df`. `None` if the order did not
    /// change.
    gathered_row_for_sorted_row: Option<IdxCa>,
}

/// The window columns of one partition.
struct PartitionWindows {
    windows: DataFrame,
    /// The input row of each row of `windows`.
    input_row_for_window_row: Vec<IdxSize>,
}

fn evaluate_partitions(
    params: &WindowParams,
    builders: Vec<LocalBuilder>,
    num_partitions: usize,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let mut morsels = builders
        .iter()
        .flat_map(|builder| {
            builder
                .morsels
                .iter()
                .enumerate()
                .map(move |(idx, (seq, sf))| InputMorsel {
                    seq: *seq,
                    sf,
                    builder,
                    idx,
                    first_row: 0,
                })
        })
        .collect::<Vec<_>>();
    if params.ordered_eval || params.output == WindowOutput::OrderedMorsels {
        morsels.sort_by_key(|morsel| morsel.seq);
    }

    let mut height = 0 as IdxSize;
    for morsel in &mut morsels {
        morsel.first_row = height;
        height += morsel.sf.height() as IdxSize;
    }
    if height == 0 {
        return Ok(DataFrame::empty_with_schema(&params.output_schema));
    }

    // The positions of the read columns in the input morsels.
    let read_idxs = {
        let df = morsels[0].sf.get_blocking();
        params
            .read_schema
            .iter_names()
            .map(|name| df.try_get_column_index(name))
            .collect::<PolarsResult<Vec<_>>>()?
    };
    let gather = |p: usize| gather_partition(params, &morsels, &read_idxs, p);

    if params.output == WindowOutput::Partitions {
        // Gather every partition first, so the input morsels can be freed before the partitions
        // are sorted and evaluated.
        let partitions = RAYON.install(|| {
            (0..num_partitions)
                .into_par_iter()
                .map(gather)
                .collect::<Vec<_>>()
        });
        drop(morsels);
        drop(builders);
        return output_sorted_partitions(params, partitions, random_state, state);
    }

    let partitions = RAYON.install(|| {
        (0..num_partitions)
            .into_par_iter()
            .map(|p| {
                let GatheredPartition {
                    df,
                    input_row_for_gathered_row,
                } = gather(p);
                let height = df.height();
                if height == 0 {
                    return Ok(None);
                }
                let SortedPartition {
                    windows,
                    gathered_row_for_sorted_row,
                    ..
                } = evaluate_partition(params, df, random_state, state)?;
                let input_row_for_window_row = match gathered_row_for_sorted_row {
                    None => input_row_for_gathered_row,
                    Some(gathered_rows) => gathered_rows
                        .downcast_as_array()
                        .values()
                        .iter()
                        // SAFETY: the sort only permutes the gathered rows.
                        .map(|row| unsafe {
                            *input_row_for_gathered_row.get_unchecked(*row as usize)
                        })
                        .collect(),
                };
                Ok(Some(PartitionWindows {
                    // SAFETY: the window columns have the same height and unique names.
                    windows: unsafe { DataFrame::new_unchecked(height, windows) },
                    input_row_for_window_row,
                }))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok(append_windows(
        &morsels,
        partitions.into_iter().flatten().collect(),
        height,
    ))
}

/// Gathers the read columns of the rows in partition `p`, in the order of `morsels`.
fn gather_partition(
    params: &WindowParams,
    morsels: &[InputMorsel],
    read_idxs: &[usize],
    p: usize,
) -> GatheredPartition {
    let with_input_rows = params.output != WindowOutput::Partitions;
    let mut builder = DataFrameBuilder::new(params.read_schema.clone());
    let mut input_row_for_gathered_row = Vec::new();
    for morsel in morsels {
        let rows = morsel.partition_rows(p);
        if rows.is_empty() {
            continue;
        }
        let df = morsel.sf.get_blocking();
        let columns = read_idxs.iter().map(|i| df.columns()[*i].clone()).collect();
        // SAFETY: the columns come from one frame and have unique names.
        let df = unsafe { DataFrame::new_unchecked(df.height(), columns) };
        if rows.len() == df.height() {
            builder.extend(&df, ShareStrategy::Never);
        } else {
            // SAFETY: the rows were generated from this morsel.
            unsafe { builder.gather_extend(&df, rows, ShareStrategy::Never) };
        }
        if with_input_rows {
            input_row_for_gathered_row.extend(rows.iter().map(|row| morsel.first_row + row));
        }
    }
    GatheredPartition {
        df: builder.freeze(),
        input_row_for_gathered_row,
    }
}

/// Evaluates the partitions and outputs their sorted rows with the window columns appended.
fn output_sorted_partitions(
    params: &WindowParams,
    partitions: Vec<GatheredPartition>,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let dfs = RAYON.install(|| {
        partitions
            .into_par_iter()
            .filter(|partition| partition.df.height() > 0)
            .map(|partition| {
                let SortedPartition {
                    mut df, windows, ..
                } = evaluate_partition(params, partition.df, random_state, state)?;
                // SAFETY: the window columns have the height of `df` and new names.
                unsafe { df.hstack_mut_unchecked(&windows) };
                Ok(df)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok(accumulate_dataframes_vertical_unchecked(dfs))
}

/// Appends the window columns to the input morsels, each window row on its input row.
fn append_windows(
    morsels: &[InputMorsel],
    partitions: Vec<PartitionWindows>,
    height: IdxSize,
) -> DataFrame {
    let (dfs, input_rows): (Vec<_>, Vec<_>) = partitions
        .into_iter()
        .map(|p| (p.windows, p.input_row_for_window_row))
        .unzip();
    let mut windows = accumulate_dataframes_vertical_unchecked(dfs);
    windows.rechunk_mut_par();
    // SAFETY: every input row is hashed to exactly one partition, gathered once there, and the
    // sort of a partition only permutes its rows. So each input row appears exactly once.
    let window_row_for_input_row = unsafe { invert_input_rows(&input_rows, height) };
    drop(input_rows);

    let dfs = RAYON.install(|| {
        morsels
            .par_iter()
            .map(|morsel| {
                let mut df = morsel.sf.get_blocking().clone();
                let start = morsel.first_row as usize;
                let rows = &window_row_for_input_row[start..start + df.height()];
                // SAFETY: the rows are in-bounds.
                let columns = unsafe { windows.take_slice_unchecked(rows) }.into_columns();
                // SAFETY: the window columns have the height of the morsel and new names.
                unsafe { df.hstack_mut_unchecked(&columns) };
                df
            })
            .collect::<Vec<_>>()
    });
    accumulate_dataframes_vertical_unchecked(dfs)
}

/// Returns the window row for each input row, given the input row of each window row. The window
/// rows are numbered through the partitions in order.
///
/// # Safety
/// The concatenation of `input_row_for_window_row` must be a permutation of `0..height`.
unsafe fn invert_input_rows(
    input_row_for_window_row: &[Vec<IdxSize>],
    height: IdxSize,
) -> Vec<IdxSize> {
    let mut window_row_for_input_row = vec![0 as IdxSize; height as usize];
    let mut starts = Vec::with_capacity(input_row_for_window_row.len());
    let mut start = 0 as IdxSize;
    for input_rows in input_row_for_window_row {
        starts.push(start);
        start += input_rows.len() as IdxSize;
    }
    // SAFETY: the input rows are unique, so the partitions write to disjoint rows.
    let out = unsafe { SyncPtr::new(window_row_for_input_row.as_mut_ptr()) };
    RAYON.install(|| {
        input_row_for_window_row
            .par_iter()
            .zip(starts)
            .for_each(|(input_rows, start)| {
                let out = out.get();
                for (i, row) in input_rows.iter().enumerate() {
                    // SAFETY: the input rows are in `0..height`.
                    unsafe { *out.add(*row as usize) = start + i as IdxSize };
                }
            })
    });
    window_row_for_input_row
}

/// Sorts the rows of one partition by partition and order key and evaluates the windows on them.
fn evaluate_partition(
    params: &WindowParams,
    df: DataFrame,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<SortedPartition> {
    let height = df.height();

    // Dense partition ids, in order of first occurrence.
    let keys = df.select(params.partition_by.iter().cloned())?;
    let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
    let mut grouper = new_hash_grouper(keys.schema().clone());
    let all_rows = (0..height as IdxSize).collect::<Vec<_>>();
    let mut group_idxs = Vec::with_capacity(height);
    // SAFETY: all indices are in-bounds.
    unsafe { grouper.insert_keys_subset(&hash_keys, &all_rows, Some(&mut group_idxs)) };
    let num_groups = grouper.num_groups() as usize;
    drop(all_rows);

    // The rows sorted by the order key, stable if ties must stay in input order.
    let order_idx = match &params.order_by {
        None => None,
        Some((order_by, options)) => {
            let options = SortOptions {
                maintain_order: params.ordered_eval,
                ..*options
            };
            let order_idx = df
                .column(order_by)?
                .as_materialized_series()
                .arg_sort(options);
            Some(order_idx.rechunk().downcast_as_array().values().clone())
        },
    };

    // Stable counting sort on the partition id, visiting the rows in order key order.
    let mut offsets = vec![0 as IdxSize; num_groups + 1];
    for g in &group_idxs {
        // SAFETY: the group ids are below `num_groups`.
        unsafe { *offsets.get_unchecked_mut(*g as usize + 1) += 1 };
    }
    let mut sum = 0;
    for offset in &mut offsets {
        sum += *offset;
        *offset = sum;
    }
    let gathered_row_for_sorted_row = if num_groups == 1 {
        order_idx.map_or_else(|| (0..height as IdxSize).collect(), |idx| idx.to_vec())
    } else {
        let mut gathered_rows = vec![0 as IdxSize; height];
        let mut next = offsets.clone();
        let mut place = |row: IdxSize| {
            // SAFETY: `row` is below `height` and its group id below `num_groups`. A group gets as
            // many positions as it has rows, so `pos` stays below `height`.
            unsafe {
                let pos = next.get_unchecked_mut(*group_idxs.get_unchecked(row as usize) as usize);
                *gathered_rows.get_unchecked_mut(*pos as usize) = row;
                *pos += 1;
            }
        };
        match &order_idx {
            None => (0..height as IdxSize).for_each(&mut place),
            Some(order_idx) => order_idx.iter().copied().for_each(&mut place),
        }
        gathered_rows
    };
    drop(group_idxs);

    let is_identity = gathered_row_for_sorted_row
        .iter()
        .enumerate()
        .all(|(i, row)| i as IdxSize == *row);
    let (df, gathered_row_for_sorted_row) = if is_identity {
        (df, None)
    } else {
        let gathered_rows = IdxCa::from_vec(PlSmallStr::EMPTY, gathered_row_for_sorted_row);
        // SAFETY: `gathered_rows` is a permutation of the rows of `df`.
        let sorted = unsafe { df.take_unchecked(&gathered_rows) };
        drop(df);
        (sorted, Some(gathered_rows))
    };

    let groups = GroupsType::Slice {
        groups: offsets.windows(2).map(|w| [w[0], w[1] - w[0]]).collect(),
        overlapping: false,
        monotonic: true,
    }
    .into_sliceable();

    let windows = params
        .exprs
        .iter()
        .map(|(name, expr)| {
            let out = expr.evaluate_on_sorted_partitions(&df, groups.clone(), state)?;
            Ok(out.with_name(name.clone()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(SortedPartition {
        df,
        windows,
        gathered_row_for_sorted_row,
    })
}
