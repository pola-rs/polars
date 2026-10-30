use std::cmp::Ordering;
use std::ops::{Range, RangeBounds};

use polars_async::executor::{JoinHandle, TaskMetricAggregator, TaskPriority, TaskScope};
use polars_async::primitives::distributor_channel::{self, distributor_channel};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_defs::join::{JoinArgs, JoinType, MaintainOrderJoin};
use polars_ooc::RandomSpillContext;
use polars_ops::frame::merge_join::*;
use polars_utils::UnitVec;
use polars_utils::sort::reorder_cmp;
use rayon::slice::ParallelSliceMut;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::in_memory_source::InMemorySourceNode;
use crate::nodes::joins::utils::SpillFrameSearchBuffer;
use crate::nodes::{ComputeNode, NodeMemoryUsage};
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Build,
    Probe,
    Both,
}

#[derive(Default)]
struct ComputeJoinArenas {
    gather_build: Vec<IdxSize>,
    gather_probe: Vec<IdxSize>,
    gather_probe_unmatched: Vec<IdxSize>,
    df_builders: Option<(DataFrameBuilder, DataFrameBuilder)>,
}

#[derive(Debug)]
pub struct MergeJoinParams {
    pub left: MergeJoinSideParams,
    pub right: MergeJoinSideParams,
    pub output_schema: SchemaRef,
    pub key_descending: bool,
    pub key_nulls_last: bool,
    pub keys_row_encoded: bool,
    pub args: JoinArgs,
}

impl MergeJoinParams {
    pub fn left_is_build(&self) -> bool {
        match self.args.maintain_order {
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => false,
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => true,
            MaintainOrderJoin::None => self.args.how != JoinType::Right,
        }
    }

    pub fn build_params(&self) -> &MergeJoinSideParams {
        match self.left_is_build() {
            true => &self.left,
            false => &self.right,
        }
    }

    pub fn probe_params(&self) -> &MergeJoinSideParams {
        match self.left_is_build() {
            true => &self.right,
            false => &self.left,
        }
    }
}

#[derive(Debug, Default)]
enum MergeJoinState {
    #[default]
    Running,
    FlushInputBuffers,
    EmitUnmatched(InMemorySourceNode),
    Done,
}

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    build_unmerged: SpillFrameSearchBuffer,
    probe_unmerged: SpillFrameSearchBuffer,
    unmatched: Vec<(MorselSeq, DataFrame)>,
    output_seq: MorselSeq,
}

#[derive(Debug)]
pub struct MergeJoinSideParams {
    pub on: Vec<PlSmallStr>,
    pub tmp_key_col: Option<PlSmallStr>,
    pub emit_unmatched: bool,
}

impl MergeJoinSideParams {
    fn key_col(&self) -> &PlSmallStr {
        self.tmp_key_col.as_ref().unwrap_or(&self.on[0])
    }
}

impl MergeJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        output_schema: SchemaRef,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        descending: bool,
        nulls_last: bool,
        keys_row_encoded: bool,
        args: JoinArgs,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> PolarsResult<Self> {
        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on[0]);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on[0]);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert!(left_on.len() == right_on.len());
        assert_eq!(left_key_dtype, right_key_dtype);

        let state = MergeJoinState::Running;
        let left = MergeJoinSideParams {
            on: left_on,
            tmp_key_col: tmp_left_key_col,
            emit_unmatched: matches!(args.how, JoinType::Left | JoinType::Full),
        };
        let right = MergeJoinSideParams {
            on: right_on,
            tmp_key_col: tmp_right_key_col,
            emit_unmatched: matches!(args.how, JoinType::Right | JoinType::Full),
        };
        let params = MergeJoinParams {
            left,
            right,
            output_schema,
            key_descending: descending,
            key_nulls_last: nulls_last,
            keys_row_encoded,
            args,
        };
        let (build_schema, probe_schema) = match params.left_is_build() {
            true => (&left_input_schema, &right_input_schema),
            false => (&right_input_schema, &left_input_schema),
        };
        let build_unmerged = SpillFrameSearchBuffer::empty_with_schema(
            build_schema.clone(),
            RandomSpillContext::new("merge-join-build-buffer".into(), task_metrics.clone()),
        );
        let probe_unmerged = SpillFrameSearchBuffer::empty_with_schema(
            probe_schema.clone(),
            RandomSpillContext::new("merge-join-probe-buffer".into(), task_metrics),
        );
        Ok(MergeJoinNode {
            state,
            params,
            build_unmerged,
            probe_unmerged,
            unmatched: Default::default(),
            output_seq: MorselSeq::default(),
        })
    }
}

impl ComputeNode for MergeJoinNode {
    fn name(&self) -> &str {
        "merge-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        use MergeJoinState::*;

        assert!(recv.len() == 2 && send.len() == 1);

        let input_channels_done = recv.iter().all(|r| *r == PortState::Done);
        let input_buffers_empty = self.build_unmerged.is_empty() && self.probe_unmerged.is_empty();
        let unmatched_buffers_empty = self.unmatched.is_empty();

        if send[0] == PortState::Done {
            self.state = Done;
        }

        if self.params.args.maintain_order == MaintainOrderJoin::None {
            debug_assert!(unmatched_buffers_empty);
        }

        if matches!(self.state, Running) && input_channels_done {
            self.state = FlushInputBuffers;
        }

        if matches!(self.state, FlushInputBuffers) && input_buffers_empty {
            if self.unmatched.is_empty() {
                self.state = Done;
            } else {
                RAYON.install(|| {
                    self.unmatched.par_sort_by_key(|(seq, _df)| *seq);
                });
                let mut all_unmatched = DataFrame::empty_with_schema(&self.params.output_schema);
                for (_seq, df) in self.unmatched.drain(..) {
                    all_unmatched.vstack_mut_owned(df)?;
                }
                let src_node =
                    InMemorySourceNode::new(Arc::new(all_unmatched), self.output_seq.successor());
                self.state = EmitUnmatched(src_node);
            }
        }

        match &mut self.state {
            Running => {
                let recv0_blocked = recv[0] == PortState::Blocked;
                let recv1_blocked = recv[1] == PortState::Blocked;
                let send_blocked = send[0] == PortState::Blocked;
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv0_blocked || recv1_blocked {
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked || send_blocked {
                    recv[0] = PortState::Blocked;
                }
                if recv0_blocked || send_blocked {
                    recv[1] = PortState::Blocked;
                }
            },
            FlushInputBuffers => {
                recv.fill(PortState::Done);
                send[0] = PortState::Ready;
            },
            EmitUnmatched(src_node) => {
                recv.fill(PortState::Done);
                src_node.update_state(&mut [], &mut send[..], state)?;
                if send[0] == PortState::Done {
                    self.state = Done;
                }
            },
            Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }

        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        // Unmatched probe rows are buffered until the end to emit them in order.
        let buffers_unmatched = self.params.args.maintain_order != MaintainOrderJoin::None
            && self.params.probe_params().emit_unmatched;
        match &self.state {
            MergeJoinState::Running if buffers_unmatched => NodeMemoryUsage::Unbounded,
            MergeJoinState::FlushInputBuffers if buffers_unmatched => {
                NodeMemoryUsage::HoldingUntilDone
            },
            MergeJoinState::Running | MergeJoinState::FlushInputBuffers => NodeMemoryUsage::Bounded,
            MergeJoinState::EmitUnmatched(src) => src.memory_usage(),
            MergeJoinState::Done => NodeMemoryUsage::Bounded,
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
        use MergeJoinState::*;

        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        match &mut self.state {
            Running | FlushInputBuffers => {
                let params = &self.params;
                let build_unmerged = &mut self.build_unmerged;
                let probe_unmerged = &mut self.probe_unmerged;
                let unmatched = &mut self.unmatched;
                let mergeable_seq = &mut self.output_seq;
                let build_idx = match self.params.left_is_build() {
                    true => 0,
                    false => 1,
                };
                let recv_build = recv_ports[build_idx].take().map(RecvPort::serial);
                let recv_probe = recv_ports[1 - build_idx].take().map(RecvPort::serial);

                assert!(send_ports[0].is_some());
                let send = send_ports[0].take().unwrap().parallel();
                let (mut distributor, dist_recv) =
                    distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
                let (unmatched_send, mut unmatched_recv) = tokio::sync::mpsc::channel(send.len());
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    find_mergeable_task(
                        recv_build,
                        recv_probe,
                        build_unmerged,
                        probe_unmerged,
                        &mut distributor,
                        params,
                        mergeable_seq,
                    )
                    .await
                }));
                join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                    let unmatched_send = unmatched_send.clone();
                    scope.spawn_task(TaskPriority::High, async move {
                        let mut arenas = ComputeJoinArenas::default();
                        while let Ok((build, probe, seq, source_token)) = recv.recv().await {
                            let output_open = compute_join_and_send(
                                build,
                                probe,
                                seq,
                                source_token,
                                params,
                                &mut arenas,
                                &mut send,
                                unmatched_send.clone(),
                            )
                            .await?;
                            if !output_open {
                                break;
                            }
                        }
                        Ok(())
                    })
                }));
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Some((seq, df)) = unmatched_recv.recv().await {
                        unmatched.push((seq, df));
                    }
                    Ok(())
                }));
            },
            EmitUnmatched(src_node) => {
                assert!(recv_ports[0].is_none());
                assert!(recv_ports[1].is_none());
                assert!(send_ports[0].is_some());
                src_node.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            Done => {
                unreachable!();
            },
        }
    }
}

async fn find_mergeable_task(
    mut recv_build: Option<PortReceiver>,
    mut recv_probe: Option<PortReceiver>,
    build_unmerged: &mut SpillFrameSearchBuffer,
    probe_unmerged: &mut SpillFrameSearchBuffer,
    distributor: &mut distributor_channel::Sender<(
        SpillFrameSearchBuffer,
        SpillFrameSearchBuffer,
        MorselSeq,
        SourceToken,
    )>,
    params: &MergeJoinParams,
    mergeable_seq: &mut MorselSeq,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();

    loop {
        if source_token.stop_requested() {
            build_unmerged
                .stop_and_buffer_from_pipe(recv_build.as_mut())
                .await;
            probe_unmerged
                .stop_and_buffer_from_pipe(recv_probe.as_mut())
                .await;
            return Ok(());
        }

        if recv_build.is_none()
            && build_unmerged.is_empty()
            && recv_probe.is_none()
            && probe_unmerged.is_empty()
        {
            return Ok(());
        }

        let fmp = FindMergeableParams {
            build_done: recv_build.is_none(),
            probe_done: recv_probe.is_none(),
            params,
        };
        match find_mergeable(build_unmerged, probe_unmerged, fmp).await? {
            Ok(partitions) => {
                for (build_mergeable, probe_mergeable) in partitions.into_iter() {
                    if let Err((_, _, _, _)) = distributor
                        .send((
                            build_mergeable,
                            probe_mergeable,
                            *mergeable_seq,
                            source_token.clone(),
                        ))
                        .await
                    {
                        return Ok(());
                    }
                    *mergeable_seq = mergeable_seq.successor();
                }
            },
            Err(NeedMore::Build | NeedMore::Both) if recv_build.is_some() => {
                let Ok(m) = recv_build.as_mut().unwrap().recv().await else {
                    probe_unmerged
                        .stop_and_buffer_from_pipe(recv_probe.as_mut())
                        .await;
                    return Ok(());
                };
                build_unmerged.push_sf(m.into_sf()).await;
            },
            Err(NeedMore::Probe | NeedMore::Both) if recv_probe.is_some() => {
                let Ok(m) = recv_probe.as_mut().unwrap().recv().await else {
                    build_unmerged
                        .stop_and_buffer_from_pipe(recv_build.as_mut())
                        .await;
                    return Ok(());
                };
                probe_unmerged.push_sf(m.into_sf()).await;
            },
            Err(other) => {
                unreachable!("unexpected NeedMore value: {other:?}");
            },
        }
    }
}

/// Returns `false` if the output port is closed.
#[allow(clippy::too_many_arguments)]
async fn compute_join_and_send(
    build: SpillFrameSearchBuffer,
    probe: SpillFrameSearchBuffer,
    seq: MorselSeq,
    source_token: SourceToken,
    params: &MergeJoinParams,
    arenas: &mut ComputeJoinArenas,
    send: &mut PortSender,
    unmatched_send: tokio::sync::mpsc::Sender<(MorselSeq, DataFrame)>,
) -> PolarsResult<bool> {
    let morsel_size = get_ideal_morsel_size();
    let wait_group = WaitGroup::default();

    let mut build = build.into_df().await;
    let mut probe = probe.into_df().await;
    build.rechunk_mut();
    probe.rechunk_mut();

    let mut build_key = build
        .column(params.build_params().key_col())
        .unwrap()
        .as_materialized_series();
    let mut probe_key = probe
        .column(params.probe_params().key_col())
        .unwrap()
        .as_materialized_series();

    #[cfg(feature = "dtype-categorical")]
    let (str_probe_key, str_build_key);
    #[cfg(feature = "dtype-categorical")]
    {
        // Categoricals are lexicographically ordered, not by their physical values.
        if build_key.dtype().is_categorical() {
            str_build_key = build_key.cast(&DataType::String)?;
            build_key = &str_build_key;
        }
        if probe_key.dtype().is_categorical() {
            str_probe_key = probe_key.cast(&DataType::String)?;
            probe_key = &str_probe_key;
        }
    }

    let build_key = build_key.to_physical_repr();
    let probe_key = probe_key.to_physical_repr();

    let mut build_row_offset = 0;
    let mut probe_row_offset = 0;
    let mut probe_last_matched = 0;
    arenas.gather_probe_unmatched.clear();
    while build_row_offset < build.height() || probe_row_offset < probe.height() {
        arenas.gather_build.clear();
        arenas.gather_probe.clear();
        let gather_probe_unmatched = params
            .probe_params()
            .emit_unmatched
            .then_some(&mut arenas.gather_probe_unmatched);
        match_keys(
            &build_key,
            &probe_key,
            &mut arenas.gather_build,
            &mut arenas.gather_probe,
            gather_probe_unmatched,
            params.build_params().emit_unmatched,
            params.key_descending,
            params.args.nulls_equal,
            morsel_size,
            &mut build_row_offset,
            &mut probe_row_offset,
            &mut probe_last_matched,
        );

        let df = gather_and_postprocess(
            build.clone(),
            probe.clone(),
            Some(&arenas.gather_build),
            Some(&arenas.gather_probe),
            &mut arenas.df_builders,
            &params.args,
            &params.left.on,
            &params.right.on,
            params.left_is_build(),
            &params.output_schema,
        )?;
        if df.height() > 0 {
            let mut morsel = Morsel::new_unregistered(df, seq, source_token.clone());
            morsel.set_consume_token(wait_group.token());
            if send.send(morsel).await.is_err() {
                return Ok(false);
            };
            wait_group.wait().await;
        }
    }

    if params.probe_params().emit_unmatched {
        let df_unmatched = gather_and_postprocess(
            build,
            probe,
            None,
            Some(&arenas.gather_probe_unmatched),
            &mut arenas.df_builders,
            &params.args,
            &params.left.on,
            &params.right.on,
            params.left_is_build(),
            &params.output_schema,
        )?;
        if df_unmatched.height() > 0 {
            if params.args.maintain_order == MaintainOrderJoin::None {
                let mut morsel = Morsel::new_unregistered(df_unmatched, seq, source_token.clone());
                morsel.set_consume_token(wait_group.token());
                if send.send(morsel).await.is_err() {
                    return Ok(false);
                }
            } else {
                unmatched_send.send((seq, df_unmatched)).await.unwrap();
            }
            wait_group.wait().await;
        }
    }
    Ok(true)
}

#[derive(Clone, Debug)]
struct FindMergeableParams<'a> {
    build_done: bool,
    probe_done: bool,
    params: &'a MergeJoinParams,
}

async fn find_mergeable(
    build: &mut SpillFrameSearchBuffer,
    probe: &mut SpillFrameSearchBuffer,
    fmp: FindMergeableParams<'_>,
) -> PolarsResult<Result<UnitVec<(SpillFrameSearchBuffer, SpillFrameSearchBuffer)>, NeedMore>> {
    let (build_mergeable, probe_mergeable) =
        match find_mergeable_limiting(build, probe, fmp.clone()).await? {
            Ok((build, probe)) => (build, probe),
            Err(need_more) => return Ok(Err(need_more)),
        };
    assert!(!build_mergeable.is_empty() || !probe_mergeable.is_empty());

    let partitions = find_mergeable_partition(
        &build_mergeable,
        &probe_mergeable,
        fmp.params,
        get_ideal_morsel_size(),
    )
    .await;
    Ok(Ok(partitions))
}

async fn find_mergeable_limiting(
    build: &mut SpillFrameSearchBuffer,
    probe: &mut SpillFrameSearchBuffer,
    fmp: FindMergeableParams<'_>,
) -> PolarsResult<Result<(SpillFrameSearchBuffer, SpillFrameSearchBuffer), NeedMore>> {
    const SEARCH_LIMIT_BUMP_FACTOR: usize = 2;
    let mut search_limit = get_ideal_morsel_size();
    let mut mergeable = find_mergeable_search(build, probe, search_limit, fmp.clone()).await?;
    while match mergeable {
        Err(NeedMore::Build | NeedMore::Both) if search_limit < build.height() => true,
        Err(NeedMore::Probe | NeedMore::Both) if search_limit < probe.height() => true,
        _ => false,
    } {
        // Exponential increase
        search_limit *= SEARCH_LIMIT_BUMP_FACTOR;
        mergeable = find_mergeable_search(build, probe, search_limit, fmp.clone()).await?;
    }
    Ok(mergeable)
}

/// Splits mergeable `build` and `probe` buffers into partitions that can be joined
/// independently.
///
/// Both sides are cut at the same keys, sampled every `morsel_size` rows, so every key
/// lives in one partition and every row is in one partition. A partition holding a single
/// matching key with more than a morsel of output has its build side split further, each
/// part getting the same probe rows.
async fn find_mergeable_partition(
    build: &SpillFrameSearchBuffer,
    probe: &SpillFrameSearchBuffer,
    params: &MergeJoinParams,
    morsel_size: usize,
) -> UnitVec<(SpillFrameSearchBuffer, SpillFrameSearchBuffer)> {
    let build_params = params.build_params();
    let probe_params = params.probe_params();

    let mut cut_keys = Vec::new();
    for idx in (morsel_size..build.height()).step_by(morsel_size) {
        cut_keys.push(get_key(build, build_params, params, idx).await);
    }
    for idx in (morsel_size..probe.height()).step_by(morsel_size) {
        cut_keys.push(get_key(probe, probe_params, params, idx).await);
    }
    cut_keys.sort_by(|l, r| keys_cmp(l, r, params));
    cut_keys.dedup_by(|l, r| keys_cmp(l, r, params).is_eq());

    // Cut before and after the rows of every key, so that a long run of one key gets its
    // own partition.
    let mut cuts = vec![(0, 0)];
    for key in &cut_keys {
        let &(build_prev, probe_prev) = cuts.last().unwrap();
        let lower = (
            binary_search_lower(build, key, build_prev.., params, build_params).await,
            binary_search_lower(probe, key, probe_prev.., params, probe_params).await,
        );
        let upper = (
            binary_search_upper(build, key, lower.0.., params, build_params).await,
            binary_search_upper(probe, key, lower.1.., params, probe_params).await,
        );
        for cut in [lower, upper] {
            if cuts.last() != Some(&cut) {
                cuts.push(cut);
            }
        }
    }
    if cuts.last() != Some(&(build.height(), probe.height())) || cuts.len() == 1 {
        cuts.push((build.height(), probe.height()));
    }

    // Every build part copies its probe rows, so give it enough build rows to keep that copy
    // small next to its output.
    const MIN_SPLIT_BUILD_ROWS: usize = 16;

    let mut partitions = UnitVec::with_capacity(cuts.len() - 1);
    for window in cuts.windows(2) {
        let [(build_start, probe_start), (build_end, probe_end)] = window else {
            unreachable!()
        };
        let build_len = build_end - build_start;
        let probe_len = probe_end - probe_start;
        let probe_part = probe.slice(*probe_start, probe_len);

        let split_build = build_len > MIN_SPLIT_BUILD_ROWS
            && build_len.saturating_mul(probe_len) > morsel_size
            && !probe_params.emit_unmatched
            && is_single_matching_key(
                build,
                *build_start..*build_end,
                probe,
                *probe_start..*probe_end,
                params,
            )
            .await;
        if !split_build {
            partitions.push((build.slice(*build_start, build_len), probe_part));
            continue;
        }

        let chunk_size = (morsel_size / probe_len).max(MIN_SPLIT_BUILD_ROWS);
        for offset in (*build_start..*build_end).step_by(chunk_size) {
            let len = chunk_size.min(build_end - offset);
            partitions.push((build.slice(offset, len), probe_part.clone()));
        }
    }
    partitions
}

/// Whether the given non-empty build and probe rows all have the same key, and that key can
/// match.
async fn is_single_matching_key(
    build: &SpillFrameSearchBuffer,
    build_rows: Range<usize>,
    probe: &SpillFrameSearchBuffer,
    probe_rows: Range<usize>,
    params: &MergeJoinParams,
) -> bool {
    let build_params = params.build_params();
    let probe_params = params.probe_params();
    let key = get_key(build, build_params, params, build_rows.start).await;
    for (buf, sp, idx) in [
        (build, build_params, build_rows.end - 1),
        (probe, probe_params, probe_rows.start),
        (probe, probe_params, probe_rows.end - 1),
    ] {
        if keys_cmp(&key, &get_key(buf, sp, params, idx).await, params).is_ne() {
            return false;
        }
    }
    // Row-encoded keys are read past their validity above.
    let is_null = unsafe { build.get_unchecked(build_params.key_col(), build_rows.start) }
        .await
        .is_null();
    params.args.nulls_equal || !is_null
}

async fn find_mergeable_search(
    build: &mut SpillFrameSearchBuffer,
    probe: &mut SpillFrameSearchBuffer,
    search_limit: usize,
    fmp: FindMergeableParams<'_>,
) -> PolarsResult<Result<(SpillFrameSearchBuffer, SpillFrameSearchBuffer), NeedMore>> {
    let FindMergeableParams {
        build_done,
        probe_done,
        params,
    } = fmp;
    let build_params = params.build_params();
    let probe_params = params.probe_params();

    if build_done && build.is_empty() && !probe_done && probe.is_empty() {
        return Ok(Err(NeedMore::Probe));
    } else if probe_done && probe.is_empty() && !build_done && build.is_empty() {
        return Ok(Err(NeedMore::Build));
    } else if build_done && build.is_empty() {
        let probe_split = probe.split_at(get_ideal_morsel_size());
        return Ok(Ok((build.empty_clone(), probe_split)));
    } else if probe_done && probe.is_empty() {
        let build_split = build.split_at(get_ideal_morsel_size());
        return Ok(Ok((build_split, probe.empty_clone())));
    } else if build.is_empty() && !build_done {
        return Ok(Err(NeedMore::Build));
    } else if probe.is_empty() && !probe_done {
        return Ok(Err(NeedMore::Probe));
    }

    let build_first = get_key(build, build_params, params, 0).await;
    let probe_first = get_key(probe, probe_params, params, 0).await;

    // First return chunks of nulls if there are any
    if !params.args.nulls_equal && !params.key_nulls_last && build_first == AnyValue::Null {
        let build_first_nonnull_idx =
            binary_search_upper(build, &AnyValue::Null, .., params, build_params).await;
        let build_split = build.split_at(build_first_nonnull_idx);
        return Ok(Ok((build_split, probe.empty_clone())));
    }
    if !params.args.nulls_equal && !params.key_nulls_last && probe_first == AnyValue::Null {
        let probe_first_nonnull_idx =
            binary_search_upper(probe, &AnyValue::Null, .., params, probe_params).await;
        let right_split = probe.split_at(probe_first_nonnull_idx);
        return Ok(Ok((build.empty_clone(), right_split)));
    }

    let build_last_idx = usize::min(build.height(), search_limit) - 1;
    let build_last = get_key(build, build_params, params, build_last_idx).await;
    let build_first_incomplete = match build_done {
        false => {
            binary_search_lower(build, &build_last, ..=build_last_idx, params, build_params).await
        },
        true => build.height(),
    };

    let probe_last_idx = usize::min(probe.height(), search_limit) - 1;
    let probe_last = get_key(probe, probe_params, params, probe_last_idx).await;
    let probe_first_incomplete = match probe_done {
        false => {
            binary_search_lower(probe, &probe_last, ..=probe_last_idx, params, probe_params).await
        },
        true => probe.height(),
    };

    if build_first_incomplete == 0 && probe_first_incomplete == 0 {
        debug_assert!(!build_done && !probe_done);
        return Ok(Err(NeedMore::Both));
    } else if build_first_incomplete == 0 {
        debug_assert!(!build_done);
        return Ok(Err(NeedMore::Build));
    } else if probe_first_incomplete == 0 {
        debug_assert!(!probe_done);
        return Ok(Err(NeedMore::Probe));
    }

    let build_last_completed_val =
        get_key(build, build_params, params, build_first_incomplete - 1).await;
    let probe_last_completed_val =
        get_key(probe, probe_params, params, probe_first_incomplete - 1).await;

    let build_mergeable_until; // bound is *exclusive*
    let probe_mergeable_until;
    match keys_cmp(&build_last_completed_val, &probe_last_completed_val, params) {
        Ordering::Equal => {
            build_mergeable_until = build_first_incomplete;
            probe_mergeable_until = probe_first_incomplete;
        },
        Ordering::Less => {
            build_mergeable_until = build_first_incomplete;
            let build_last_mergeable =
                get_key(build, build_params, params, build_mergeable_until - 1).await;
            probe_mergeable_until = binary_search_upper(
                probe,
                &build_last_mergeable,
                ..probe_first_incomplete,
                params,
                probe_params,
            )
            .await;
        },
        Ordering::Greater => {
            probe_mergeable_until = probe_first_incomplete;
            let probe_last_mergeable =
                get_key(probe, probe_params, params, probe_mergeable_until - 1).await;
            build_mergeable_until = binary_search_upper(
                build,
                &probe_last_mergeable,
                ..build_first_incomplete,
                params,
                build_params,
            )
            .await;
        },
    }

    if build_mergeable_until == 0 && probe_mergeable_until == 0 {
        return Ok(Err(NeedMore::Both));
    }

    let build_split = build.split_at(build_mergeable_until);
    let probe_split = probe.split_at(probe_mergeable_until);
    Ok(Ok((build_split, probe_split)))
}

async fn get_key(
    dfsb: &SpillFrameSearchBuffer,
    sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
    idx: usize,
) -> AnyValue<'static> {
    unsafe {
        dfsb.get_bypass_validity(sp.key_col(), idx, params.keys_row_encoded)
            .await
    }
}

async fn binary_search_lower<R: RangeBounds<usize>>(
    dfsb: &SpillFrameSearchBuffer,
    sv: &AnyValue<'_>,
    range: R,
    params: &MergeJoinParams,
    sp: &MergeJoinSideParams,
) -> usize {
    let predicate = |x: &AnyValue<'_>| keys_cmp(sv, x, params).is_le();
    dfsb.binary_search_binary_offset_bypass_validity(
        predicate,
        sp.key_col(),
        range,
        params.keys_row_encoded,
    )
    .await
}

async fn binary_search_upper<R: RangeBounds<usize>>(
    dfsb: &SpillFrameSearchBuffer,
    sv: &AnyValue<'_>,
    range: R,
    params: &MergeJoinParams,
    sp: &MergeJoinSideParams,
) -> usize {
    let predicate = |x: &AnyValue<'_>| keys_cmp(sv, x, params).is_lt();
    dfsb.binary_search_binary_offset_bypass_validity(
        predicate,
        sp.key_col(),
        range,
        params.keys_row_encoded,
    )
    .await
}

fn keys_cmp(lhs: &AnyValue, rhs: &AnyValue, params: &MergeJoinParams) -> Ordering {
    reorder_cmp(lhs, rhs, params.key_descending, params.key_nulls_last)
}

#[cfg(test)]
mod tests {
    use polars_core::runtime::ASYNC;
    use polars_ooc::SpillFrame;

    use super::*;

    const MORSEL_SIZE: usize = 8;

    fn buffer(keys: &[Option<i64>]) -> SpillFrameSearchBuffer {
        let schema = Arc::new(Schema::from_iter([Field::new("k".into(), DataType::Int64)]));
        let mut buf = SpillFrameSearchBuffer::empty_with_schema(
            schema,
            RandomSpillContext::new("test".into(), None),
        );
        for chunk in keys.chunks(5) {
            let df = df!("k" => chunk).unwrap();
            ASYNC.block_on(buf.push_sf(SpillFrame::new_unregistered(df)));
        }
        buf
    }

    fn params(how: JoinType, nulls_equal: bool) -> MergeJoinParams {
        let side = |emit_unmatched| MergeJoinSideParams {
            on: vec!["k".into()],
            tmp_key_col: None,
            emit_unmatched,
        };
        let mut args = JoinArgs::new(how.clone());
        args.nulls_equal = nulls_equal;
        MergeJoinParams {
            left: side(matches!(how, JoinType::Left | JoinType::Full)),
            right: side(matches!(how, JoinType::Right | JoinType::Full)),
            output_schema: Arc::new(Schema::default()),
            key_descending: false,
            key_nulls_last: true,
            keys_row_encoded: false,
            args,
        }
    }

    fn keys(buf: &SpillFrameSearchBuffer) -> Vec<Option<i64>> {
        let df = ASYNC.block_on(buf.clone().into_df());
        df.column("k").unwrap().i64().unwrap().iter().collect()
    }

    fn partition(
        build: &[Option<i64>],
        probe: &[Option<i64>],
        params: &MergeJoinParams,
    ) -> Vec<(Vec<Option<i64>>, Vec<Option<i64>>)> {
        let partitions = ASYNC.block_on(find_mergeable_partition(
            &buffer(build),
            &buffer(probe),
            params,
            MORSEL_SIZE,
        ));
        partitions
            .iter()
            .map(|(build, probe)| (keys(build), keys(probe)))
            .collect()
    }

    fn matches(build: &[Option<i64>], probe: &[Option<i64>], nulls_equal: bool) -> usize {
        let mut count = 0;
        for b in build {
            for p in probe {
                count += usize::from(b == p && (nulls_equal || b.is_some()));
            }
        }
        count
    }

    #[test]
    fn test_partition_cuts_both_sides_at_shared_keys() {
        for descending in [false, true] {
            // Key i appears i % 3 + 1 times on the build side and i % 2 + 1 times on the
            // probe side.
            let mut build: Vec<_> = (0..200)
                .flat_map(|i| vec![Some(i); i as usize % 3 + 1])
                .collect();
            let mut probe: Vec<_> = (50..300)
                .flat_map(|i| vec![Some(i); i as usize % 2 + 1])
                .collect();
            let mut params = params(JoinType::Inner, false);
            if descending {
                build.reverse();
                probe.reverse();
                params.key_descending = true;
            }
            let partitions = partition(&build, &probe, &params);

            let all_build: Vec<_> = partitions.iter().flat_map(|(b, _)| b.clone()).collect();
            let all_probe: Vec<_> = partitions.iter().flat_map(|(_, p)| p.clone()).collect();
            assert_eq!(all_build, build);
            assert_eq!(all_probe, probe);

            let max_part = MORSEL_SIZE + 3;
            for (b, p) in &partitions {
                assert!(b.len() <= max_part && p.len() <= max_part);
            }
            assert!(partitions.len() <= 2 * (build.len() + probe.len()) / MORSEL_SIZE + 2);

            // Every key is in one partition only.
            let key_ranges: Vec<_> = partitions
                .iter()
                .filter_map(|(b, p)| Some((b.iter().chain(p).min()?, b.iter().chain(p).max()?)))
                .collect();
            for pair in key_ranges.windows(2) {
                let [(min_0, max_0), (min_1, max_1)] = pair else {
                    unreachable!()
                };
                assert!(if descending {
                    min_0 > max_1
                } else {
                    max_0 < min_1
                });
            }
        }
    }

    #[test]
    fn test_partition_splits_hot_key() {
        let mut build = vec![Some(0); 200];
        build.push(Some(1));
        let mut probe = vec![Some(0); 150];
        probe.push(Some(1));
        let partitions = partition(&build, &probe, &params(JoinType::Inner, false));

        let found: usize = partitions.iter().map(|(b, p)| matches(b, p, false)).sum();
        assert_eq!(found, matches(&build, &probe, false));
        let probe_rows: usize = partitions.iter().map(|(_, p)| p.len()).sum();
        assert!(probe_rows <= matches(&build, &probe, false) + probe.len());
        assert!(partitions.len() > 2);
    }

    #[test]
    fn test_partition_does_not_split_hot_key_when_probe_emits_unmatched() {
        let build = vec![Some(0); 200];
        let probe = vec![Some(0); 150];
        let partitions = partition(&build, &probe, &params(JoinType::Full, false));
        let probe_rows: usize = partitions.iter().map(|(_, p)| p.len()).sum();
        assert_eq!(probe_rows, probe.len());
    }

    #[test]
    fn test_partition_does_not_split_null_run() {
        let mut build = vec![Some(1), Some(2), Some(3)];
        build.extend(vec![None; 100]);
        let mut probe = vec![Some(2)];
        probe.extend(vec![None; 100]);

        let partitions = partition(&build, &probe, &params(JoinType::Inner, false));
        let probe_rows: usize = partitions.iter().map(|(_, p)| p.len()).sum();
        assert_eq!(probe_rows, probe.len());
        let found: usize = partitions.iter().map(|(b, p)| matches(b, p, false)).sum();
        assert_eq!(found, 1);

        let partitions = partition(&build, &probe, &params(JoinType::Inner, true));
        let found: usize = partitions.iter().map(|(b, p)| matches(b, p, true)).sum();
        assert_eq!(found, matches(&build, &probe, true));
        assert!(partitions.len() > 2);
    }
}
