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

/// More partitions than pipelines keep fewer partitions in memory at the same time.
const PARTITIONS_PER_PIPELINE: usize = 16;

pub struct WindowParams {
    pub partition_by: Vec<PlSmallStr>,
    pub order_by: Option<(PlSmallStr, SortOptions)>,
    /// The window expressions with their output names.
    pub exprs: Vec<(PlSmallStr, WindowExpr)>,
    /// The input columns the window expressions read, including the keys.
    pub read_schema: Arc<Schema>,
    pub output_schema: Arc<Schema>,
    /// Evaluate the rows of a partition in input order.
    pub ordered_eval: bool,
    /// Output the rows in input order.
    pub maintain_order: bool,
}

/// Evaluates window expressions that share one partitioning. The columns the windows read are
/// hash partitioned on the partition keys, and each hash partition is evaluated in parallel.
///
/// The input rows are output with the window columns appended. Without `maintain_order` the rows
/// are output in an unspecified order.
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

fn evaluate_partitions(
    params: &WindowParams,
    builders: Vec<LocalBuilder>,
    num_partitions: usize,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let mut morsels = builders
        .iter()
        .flat_map(|b| {
            b.morsels
                .iter()
                .enumerate()
                .map(move |(i, (seq, sf))| (*seq, b, i, sf))
        })
        .collect::<Vec<_>>();
    if params.ordered_eval || params.maintain_order {
        morsels.sort_by_key(|(seq, ..)| *seq);
    }

    // The row index of the first row of each morsel.
    let mut row_offsets = Vec::with_capacity(morsels.len());
    let mut height = 0 as IdxSize;
    for (.., sf) in &morsels {
        row_offsets.push(height);
        height += sf.height() as IdxSize;
    }
    if height == 0 {
        return Ok(DataFrame::empty_with_schema(&params.output_schema));
    }

    let outputs = RAYON.install(|| {
        (0..num_partitions)
            .into_par_iter()
            .map(|p| {
                let mut builder = DataFrameBuilder::new(params.read_schema.clone());
                let mut row_idx = Vec::new();
                for (k, (_, b, i, sf)) in morsels.iter().enumerate() {
                    let start = b.offsets_per_p[i * num_partitions + p];
                    let stop = b.offsets_per_p[(i + 1) * num_partitions + p];
                    if start == stop {
                        continue;
                    }
                    let df = sf
                        .get_blocking()
                        .select(params.read_schema.iter_names_cloned())?;
                    let idxs = &b.idxs_per_p[p][start..stop];
                    if idxs.len() == df.height() {
                        builder.extend(&df, ShareStrategy::Never);
                    } else {
                        // SAFETY: the indices were generated from this morsel.
                        unsafe { builder.gather_extend(&df, idxs, ShareStrategy::Never) };
                    }
                    row_idx.extend(idxs.iter().map(|j| row_offsets[k] + j));
                }
                evaluate_partition(params, builder.freeze(), row_idx, random_state, state)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let (dfs, row_idxs): (Vec<_>, Vec<_>) = outputs
        .into_iter()
        .filter(|(df, _)| df.height() > 0)
        .unzip();
    let mut windows = accumulate_dataframes_vertical_unchecked(dfs);
    windows.rechunk_mut_par();

    // The row of `windows` for each input row.
    let mut positions = vec![0 as IdxSize; height as usize];
    let mut starts = Vec::with_capacity(row_idxs.len());
    let mut start = 0 as IdxSize;
    for rows in &row_idxs {
        starts.push(start);
        start += rows.len() as IdxSize;
    }
    // SAFETY: the partitions write to disjoint positions.
    let positions_ptr = unsafe { SyncPtr::new(positions.as_mut_ptr()) };
    RAYON.install(|| {
        row_idxs.par_iter().zip(starts).for_each(|(rows, start)| {
            let positions = positions_ptr.get();
            for (i, row) in rows.iter().enumerate() {
                // SAFETY: every input row is in exactly one partition.
                unsafe { *positions.add(*row as usize) = start + i as IdxSize };
            }
        })
    });
    drop(row_idxs);

    let dfs = RAYON.install(|| {
        morsels
            .par_iter()
            .zip(row_offsets)
            .map(|((.., sf), offset)| {
                let mut df = sf.get_blocking().clone();
                let rows = &positions[offset as usize..offset as usize + df.height()];
                let rows = IdxCa::from_slice(PlSmallStr::EMPTY, rows);
                // SAFETY: the positions are in-bounds.
                let columns = unsafe { windows.take_unchecked(&rows) }.into_columns();
                // SAFETY: the window columns have the height of the morsel and new names.
                unsafe { df.hstack_mut_unchecked(&columns) };
                df
            })
            .collect::<Vec<_>>()
    });
    Ok(accumulate_dataframes_vertical_unchecked(dfs))
}

/// Evaluates the windows on the rows of one hash partition.
///
/// Returns the window columns with the rows sorted by partition and order key, and the input row
/// index of each of those rows.
fn evaluate_partition(
    params: &WindowParams,
    df: DataFrame,
    row_idx: Vec<IdxSize>,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<(DataFrame, Vec<IdxSize>)> {
    let height = df.height();
    if height == 0 {
        return Ok((DataFrame::empty(), row_idx));
    }

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
        offsets[*g as usize + 1] += 1;
    }
    for g in 0..num_groups {
        offsets[g + 1] += offsets[g];
    }
    let perm = if num_groups == 1 {
        order_idx.map_or_else(|| (0..height as IdxSize).collect(), |idx| idx.to_vec())
    } else {
        let mut perm = vec![0 as IdxSize; height];
        let mut next = offsets.clone();
        let mut place = |row: IdxSize| {
            let pos = &mut next[group_idxs[row as usize] as usize];
            perm[*pos as usize] = row;
            *pos += 1;
        };
        match &order_idx {
            None => (0..height as IdxSize).for_each(&mut place),
            Some(order_idx) => order_idx.iter().copied().for_each(&mut place),
        }
        perm
    };
    drop(group_idxs);

    let is_identity = perm.iter().enumerate().all(|(i, row)| i as IdxSize == *row);
    let (df, row_idx) = if is_identity {
        (df, row_idx)
    } else {
        let row_idx = perm.iter().map(|i| row_idx[*i as usize]).collect();
        let perm = IdxCa::from_vec(PlSmallStr::EMPTY, perm);
        // SAFETY: `perm` is a permutation of the rows of `df`.
        let sorted = unsafe { df.take_unchecked(&perm) };
        drop(df);
        (sorted, row_idx)
    };

    let groups = GroupsType::Slice {
        groups: offsets.windows(2).map(|w| [w[0], w[1] - w[0]]).collect(),
        overlapping: false,
        monotonic: true,
    }
    .into_sliceable();

    let columns = params
        .exprs
        .iter()
        .map(|(name, expr)| {
            let out = expr.evaluate_on_sorted_partitions(&df, groups.clone(), state)?;
            Ok(out.with_name(name.clone()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: the window columns have the same height and unique names.
    Ok((
        unsafe { DataFrame::new_unchecked(height, columns) },
        row_idx,
    ))
}
