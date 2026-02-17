use std::cmp::Ordering;

use polars_core::POOL;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_ops::frame::merge_join::*;
use polars_ops::frame::{JoinArgs, JoinType, MaintainOrderJoin};
use polars_utils::UnitVec;
use rayon::slice::ParallelSliceMut;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::{self, distributor_channel};
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_source::InMemorySourceNode;
use crate::nodes::joins::utils::DataFrameSearchBuffer;
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

    pub fn preserve_order_probe(&self) -> bool {
        match &self.args.maintain_order {
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => !self.left_is_build(),
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => self.left_is_build(),
            MaintainOrderJoin::None => false,
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
    build_unmerged: DataFrameSearchBuffer,
    probe_unmerged: DataFrameSearchBuffer,
    unmatched: Vec<(MorselSeq, DataFrame)>,
    output_seq: MorselSeq,
}

#[derive(Debug)]
pub struct MergeJoinSideParams {
    pub input_schema: SchemaRef,
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
    ) -> PolarsResult<Self> {
        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on[0]);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on[0]);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert!(left_on.len() == right_on.len());
        assert_eq!(left_key_dtype, right_key_dtype);

        let state = MergeJoinState::Running;
        let left = MergeJoinSideParams {
            input_schema: left_input_schema.clone(),
            on: left_on,
            tmp_key_col: tmp_left_key_col,
            emit_unmatched: matches!(args.how, JoinType::Left | JoinType::Full),
        };
        let right = MergeJoinSideParams {
            input_schema: right_input_schema.clone(),
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
        let build_unmerged = DataFrameSearchBuffer::empty_with_schema(build_schema.clone());
        let probe_unmerged = DataFrameSearchBuffer::empty_with_schema(probe_schema.clone());
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
                POOL.install(|| {
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
                            compute_join_and_send(
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
    build_unmerged: &mut DataFrameSearchBuffer,
    probe_unmerged: &mut DataFrameSearchBuffer,
    distributor: &mut distributor_channel::Sender<(
        DataFrameSearchBuffer,
        DataFrameSearchBuffer,
        MorselSeq,
        SourceToken,
    )>,
    params: &MergeJoinParams,
    mergeable_seq: &mut MorselSeq,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();

    loop {
        if source_token.stop_requested() {
            stop_and_buffer_pipe_contents(recv_build.as_mut(), build_unmerged).await;
            stop_and_buffer_pipe_contents(recv_probe.as_mut(), probe_unmerged).await;
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
        match find_mergeable(build_unmerged, probe_unmerged, fmp)? {
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
                    stop_and_buffer_pipe_contents(recv_probe.as_mut(), probe_unmerged).await;
                    return Ok(());
                };
                build_unmerged.push_df(m.into_df());
            },
            Err(NeedMore::Probe | NeedMore::Both) if recv_probe.is_some() => {
                let Ok(m) = recv_probe.as_mut().unwrap().recv().await else {
                    stop_and_buffer_pipe_contents(recv_build.as_mut(), build_unmerged).await;
                    return Ok(());
                };
                probe_unmerged.push_df(m.into_df());
            },
            Err(other) => {
                unreachable!("unexpected NeedMore value: {other:?}");
            },
        }
    }
}

/// Tell the sender to this port to stop, and buffer everything that is still in the pipe.
async fn stop_and_buffer_pipe_contents(
    port: Option<&mut PortReceiver>,
    unmerged: &mut DataFrameSearchBuffer,
) {
    let Some(port) = port else {
        return;
    };

    while let Ok(morsel) = port.recv().await {
        morsel.source_token().stop();
        unmerged.push_df(morsel.into_df());
    }
}

#[allow(clippy::too_many_arguments)]
async fn compute_join_and_send(
    build: DataFrameSearchBuffer,
    probe: DataFrameSearchBuffer,
    seq: MorselSeq,
    source_token: SourceToken,
    params: &MergeJoinParams,
    arenas: &mut ComputeJoinArenas,
    send: &mut PortSender,
    unmatched_send: tokio::sync::mpsc::Sender<(MorselSeq, DataFrame)>,
) -> PolarsResult<()> {
    let morsel_size = get_ideal_morsel_size();
    let wait_group = WaitGroup::default();

    let mut build = build.into_df();
    let mut probe = probe.into_df();
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
            let mut morsel = Morsel::new(df, seq, source_token.clone());
            morsel.set_consume_token(wait_group.token());
            if send.send(morsel).await.is_err() {
                return Ok(());
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
                let mut morsel = Morsel::new(df_unmatched, seq, source_token.clone());
                morsel.set_consume_token(wait_group.token());
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }
            } else {
                unmatched_send.send((seq, df_unmatched)).await.unwrap();
            }
            wait_group.wait().await;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct FindMergeableParams<'a> {
    build_done: bool,
    probe_done: bool,
    params: &'a MergeJoinParams,
}

fn find_mergeable(
    build: &mut DataFrameSearchBuffer,
    probe: &mut DataFrameSearchBuffer,
    fmp: FindMergeableParams,
) -> PolarsResult<Result<UnitVec<(DataFrameSearchBuffer, DataFrameSearchBuffer)>, NeedMore>> {
    let (build_mergeable, probe_mergeable) =
        match find_mergeable_limiting(build, probe, fmp.clone())? {
            Ok((build, probe)) => (build, probe),
            Err(need_more) => return Ok(Err(need_more)),
        };
    assert!(!build_mergeable.is_empty() || !probe_mergeable.is_empty());

    let partitions = find_mergeable_partition(build_mergeable, probe_mergeable, fmp)?;
    Ok(Ok(partitions))
}

fn find_mergeable_limiting(
    build: &mut DataFrameSearchBuffer,
    probe: &mut DataFrameSearchBuffer,
    fmp: FindMergeableParams,
) -> PolarsResult<Result<(DataFrameSearchBuffer, DataFrameSearchBuffer), NeedMore>> {
    const SEARCH_LIMIT_BUMP_FACTOR: usize = 2;
    let mut search_limit = get_ideal_morsel_size();
    let mut mergeable = find_mergeable_search(build, probe, search_limit, fmp.clone())?;
    while match mergeable {
        Err(NeedMore::Build | NeedMore::Both) if search_limit < build.height() => true,
        Err(NeedMore::Probe | NeedMore::Both) if search_limit < probe.height() => true,
        _ => false,
    } {
        // Exponential increase
        search_limit *= SEARCH_LIMIT_BUMP_FACTOR;
        mergeable = find_mergeable_search(build, probe, search_limit, fmp.clone())?;
    }
    Ok(mergeable)
}

fn find_mergeable_partition(
    build: DataFrameSearchBuffer,
    probe: DataFrameSearchBuffer,
    fmp: FindMergeableParams,
) -> PolarsResult<UnitVec<(DataFrameSearchBuffer, DataFrameSearchBuffer)>> {
    let morsel_size = get_ideal_morsel_size();

    if fmp.params.preserve_order_probe() || fmp.params.probe_params().emit_unmatched {
        return Ok(UnitVec::from([(build, probe)]));
    }

    let est_out_rows = build.height() * probe.height() + build.height() + probe.height();
    let normal_out_rows = morsel_size.pow(2);
    let partition_count = est_out_rows.div_ceil(normal_out_rows);
    if partition_count <= 1 {
        return Ok(UnitVec::from([(build, probe)]));
    }

    let chunk_size = build.height().div_ceil(partition_count);
    let mut partitions = UnitVec::with_capacity(partition_count);

    // Always make sure that there is at least one partition, even if the build side is empty.
    partitions.push((build.clone().slice(0, chunk_size), probe.clone()));
    let mut offset = chunk_size;
    while offset < build.height() {
        partitions.push((build.clone().slice(offset, chunk_size), probe.clone()));
        offset += chunk_size;
    }

    Ok(partitions)
}

fn find_mergeable_search(
    build: &mut DataFrameSearchBuffer,
    probe: &mut DataFrameSearchBuffer,
    search_limit: usize,
    fmp: FindMergeableParams,
) -> PolarsResult<Result<(DataFrameSearchBuffer, DataFrameSearchBuffer), NeedMore>> {
    let FindMergeableParams {
        build_done,
        probe_done,
        params,
    } = fmp;
    let build_params = params.build_params();
    let probe_params = params.probe_params();
    let build_empty_buf =
        || DataFrameSearchBuffer::empty_with_schema(build_params.input_schema.clone());
    let probe_empty_buf =
        || DataFrameSearchBuffer::empty_with_schema(probe_params.input_schema.clone());
    let build_get = |idx| unsafe {
        build.get_bypass_validity(build_params.key_col(), idx, params.keys_row_encoded)
    };
    let probe_get = |idx| unsafe {
        probe.get_bypass_validity(probe_params.key_col(), idx, params.keys_row_encoded)
    };

    if build_done && build.is_empty() && !probe_done && probe.is_empty() {
        return Ok(Err(NeedMore::Probe));
    } else if probe_done && probe.is_empty() && !build_done && build.is_empty() {
        return Ok(Err(NeedMore::Build));
    } else if build_done && build.is_empty() {
        let probe_split = probe.split_at(get_ideal_morsel_size());
        return Ok(Ok((build_empty_buf(), probe_split)));
    } else if probe_done && probe.is_empty() {
        let build_split = build.split_at(get_ideal_morsel_size());
        return Ok(Ok((build_split, probe_empty_buf())));
    } else if build.is_empty() && !build_done {
        return Ok(Err(NeedMore::Build));
    } else if probe.is_empty() && !probe_done {
        return Ok(Err(NeedMore::Probe));
    }

    let build_first = build_get(0);
    let probe_first = probe_get(0);

    // First return chunks of nulls if there are any
    if !params.args.nulls_equal && !params.key_nulls_last && build_first == AnyValue::Null {
        let build_first_nonnull_idx =
            binary_search_upper(build, &AnyValue::Null, params, build_params);
        let build_split = build.split_at(build_first_nonnull_idx);
        return Ok(Ok((build_split, probe_empty_buf())));
    }
    if !params.args.nulls_equal && !params.key_nulls_last && probe_first == AnyValue::Null {
        let probe_first_nonnull_idx =
            binary_search_upper(probe, &AnyValue::Null, params, probe_params);
        let right_split = probe.split_at(probe_first_nonnull_idx);
        return Ok(Ok((build_empty_buf(), right_split)));
    }

    let build_last_idx = usize::min(build.height(), search_limit);
    let build_last = build_get(build_last_idx - 1);
    let build_first_incomplete = match build_done {
        false => binary_search_lower(build, &build_last, params, build_params),
        true => build.height(),
    };

    let probe_last_idx = usize::min(probe.height(), search_limit);
    let probe_last = probe_get(probe_last_idx - 1);
    let probe_first_incomplete = match probe_done {
        false => binary_search_lower(probe, &probe_last, params, probe_params),
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

    let build_last_completed_val = build_get(build_first_incomplete - 1);
    let probe_last_completed_val = probe_get(probe_first_incomplete - 1);

    let build_mergeable_until; // bound is *exclusive*
    let probe_mergeable_until;
    match keys_cmp(&build_last_completed_val, &probe_last_completed_val, params) {
        Ordering::Equal => {
            build_mergeable_until = build_first_incomplete;
            probe_mergeable_until = probe_first_incomplete;
        },
        Ordering::Less => {
            build_mergeable_until = build_first_incomplete;
            probe_mergeable_until = binary_search_upper(
                probe,
                &build_get(build_mergeable_until - 1),
                params,
                probe_params,
            );
        },
        Ordering::Greater => {
            probe_mergeable_until = probe_first_incomplete;
            build_mergeable_until = binary_search_upper(
                build,
                &probe_get(probe_mergeable_until - 1),
                params,
                build_params,
            );
        },
    }

    if build_mergeable_until == 0 && probe_mergeable_until == 0 {
        return Ok(Err(NeedMore::Both));
    }

    let build_split = build.split_at(build_mergeable_until);
    let probe_split = probe.split_at(probe_mergeable_until);
    Ok(Ok((build_split, probe_split)))
}

fn binary_search_lower(
    dfsb: &DataFrameSearchBuffer,
    sv: &AnyValue,
    params: &MergeJoinParams,
    sp: &MergeJoinSideParams,
) -> usize {
    let predicate = |x: &AnyValue<'_>| keys_cmp(sv, x, params).is_le();
    dfsb.binary_search(predicate, sp.key_col(), params.keys_row_encoded)
}

fn binary_search_upper(
    dfsb: &DataFrameSearchBuffer,
    sv: &AnyValue,
    params: &MergeJoinParams,
    sp: &MergeJoinSideParams,
) -> usize {
    let predicate = |x: &AnyValue<'_>| keys_cmp(sv, x, params).is_lt();
    dfsb.binary_search(predicate, sp.key_col(), params.keys_row_encoded)
}

fn keys_cmp(lhs: &AnyValue, rhs: &AnyValue, params: &MergeJoinParams) -> Ordering {
    match AnyValue::partial_cmp(lhs, rhs).unwrap() {
        Ordering::Equal => Ordering::Equal,
        _ if lhs.is_null() && params.key_nulls_last => Ordering::Greater,
        _ if rhs.is_null() && params.key_nulls_last => Ordering::Less,
        _ if lhs.is_null() => Ordering::Less,
        _ if rhs.is_null() => Ordering::Greater,
        ord if params.key_descending => ord.reverse(),
        ord => ord,
    }
}
