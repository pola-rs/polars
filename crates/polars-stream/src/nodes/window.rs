use std::sync::Arc;

use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
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
use rayon::prelude::*;

use super::compute_node_prelude::*;
use super::in_memory_source::InMemorySourceNode;

pub struct WindowParams {
    pub partition_by: Vec<PlSmallStr>,
    pub order_by: Option<(PlSmallStr, SortOptions)>,
    /// The window expressions with their output names.
    pub exprs: Vec<(PlSmallStr, WindowExpr)>,
    pub input_schema: Arc<Schema>,
    pub output_schema: Arc<Schema>,
    /// Evaluate the rows of a partition in input order.
    pub ordered_eval: bool,
}

/// Evaluates window expressions that share one partitioning. Rows are hash partitioned on the
/// partition keys, and each hash partition is evaluated in parallel. The output order is
/// unspecified.
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
        let num_partitions = num_pipelines;
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
                            // Gathers need single-chunk columns.
                            morsel.df_mut().await.rechunk_mut();
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
    if params.ordered_eval {
        morsels.sort_by_key(|(seq, ..)| *seq);
    }

    let dfs = RAYON.install(|| {
        (0..num_partitions)
            .into_par_iter()
            .map(|p| {
                let mut builder = DataFrameBuilder::new(params.input_schema.clone());
                for (_, b, i, sf) in &morsels {
                    let start = b.offsets_per_p[i * num_partitions + p];
                    let stop = b.offsets_per_p[(i + 1) * num_partitions + p];
                    if start == stop {
                        continue;
                    }
                    let df = sf.get_blocking();
                    let idxs = &b.idxs_per_p[p][start..stop];
                    // SAFETY: the indices were generated from this morsel.
                    unsafe { builder.gather_extend(&df, idxs, ShareStrategy::Never) };
                }
                evaluate_partition(params, builder.freeze(), random_state, state)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    drop(morsels);
    drop(builders);

    let dfs = dfs
        .into_iter()
        .filter(|df| df.height() > 0)
        .collect::<Vec<_>>();
    if dfs.is_empty() {
        return Ok(DataFrame::empty_with_schema(&params.output_schema));
    }
    Ok(accumulate_dataframes_vertical_unchecked(dfs))
}

/// Evaluates the windows on all rows of one hash partition.
fn evaluate_partition(
    params: &WindowParams,
    df: DataFrame,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let height = df.height();
    if height == 0 {
        return Ok(DataFrame::empty_with_schema(&params.output_schema));
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

    // Stable counting sort on the partition id.
    let mut offsets = vec![0 as IdxSize; num_groups + 1];
    for g in &group_idxs {
        offsets[*g as usize + 1] += 1;
    }
    for g in 0..num_groups {
        offsets[g + 1] += offsets[g];
    }
    let mut perm = vec![0 as IdxSize; height];
    let mut next = offsets.clone();
    for (row, g) in group_idxs.iter().enumerate() {
        let pos = &mut next[*g as usize];
        perm[*pos as usize] = row as IdxSize;
        *pos += 1;
    }
    drop(group_idxs);
    drop(next);

    if let Some((order_by, options)) = &params.order_by {
        let order_by = df.column(order_by)?.clone();
        let encoded = _get_rows_encoded_ca(
            PlSmallStr::EMPTY,
            &[order_by],
            &[options.descending],
            &[options.nulls_last],
            false,
        )?;
        let encoded = encoded.downcast_as_array();
        for g in 0..num_groups {
            let group = &mut perm[offsets[g] as usize..offsets[g + 1] as usize];
            if group.len() < 2 {
                continue;
            }
            // SAFETY: the indices are rows of `df`.
            let key = |i: &IdxSize| unsafe { encoded.value_unchecked(*i as usize) };
            if params.ordered_eval {
                group.sort_by(|a, b| key(a).cmp(key(b)));
            } else {
                group.sort_unstable_by(|a, b| key(a).cmp(key(b)));
            }
        }
    }

    let perm = IdxCa::from_vec(PlSmallStr::EMPTY, perm);
    // SAFETY: `perm` is a permutation of the rows of `df`.
    let mut df = unsafe { df.take_unchecked(&perm) };
    drop(perm);

    let groups = GroupsType::Slice {
        groups: offsets
            .windows(2)
            .map(|w| [w[0], w[1] - w[0]])
            .collect::<Vec<_>>()
            .into(),
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
    // SAFETY: the window columns have the height of `df` and new names.
    unsafe { df.hstack_mut_unchecked(&columns) };
    Ok(df)
}
