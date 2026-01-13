use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::mem::take;

use arrow::array::Array;
use arrow::array::builder::ShareStrategy;
use arrow::bitmap::MutableBitmap;
use either::{Either, Left, Right};
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::prelude::*;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::TotalOrd;
use polars_utils::{UnitVec, format_pl_smallstr};

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_source::InMemorySourceNode;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

pub const KEY_COL_NAME: &str = "__POLARS_JOIN_KEY_TMP";

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Left,
    Right,
    Both,
    Finished,
}

#[derive(Default)]
struct ComputeJoinArenas {
    gather_left: Vec<IdxSize>,
    gather_right: Vec<IdxSize>,
    matched_probeside: MutableBitmap,
    df_builders: Option<(DataFrameBuilder, DataFrameBuilder)>,
}

#[derive(Debug)]
struct SideParams {
    input_schema: SchemaRef,
    on: Vec<PlSmallStr>,
    key_col: PlSmallStr,
    emit_unmatched: bool,
}

#[derive(Debug)]
struct MergeJoinParams {
    left: SideParams,
    right: SideParams,
    output_schema: SchemaRef,
    key_descending: bool,
    key_nulls_last: bool,
    use_row_encoding: bool,
    args: JoinArgs,
}

impl MergeJoinParams {
    fn left_is_build(&self) -> bool {
        match self.args.maintain_order {
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => false,
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => true,
            MaintainOrderJoin::None if self.args.how == JoinType::Right => false,
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: DataFrameBuffer,
    right_unmerged: DataFrameBuffer,
    unmatched: BTreeMap<MorselSeq, DataFrame>,
    mergeable_seq: MorselSeq,
    max_seq_sent: MorselSeq,
}

#[derive(Debug, Default)]
enum MergeJoinState {
    #[default]
    Running,
    FlushInputBuffers,
    EmitUnmatched(InMemorySourceNode),
    Done,
}

impl MergeJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        output_schema: SchemaRef,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        descending: bool,
        nulls_last: bool,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        assert!(left_on.len() == right_on.len());
        assert!(
            left_input_schema.contains(KEY_COL_NAME) == right_input_schema.contains(KEY_COL_NAME)
        );
        let use_row_encoding = left_input_schema.contains(KEY_COL_NAME);
        let state: MergeJoinState = MergeJoinState::Running;
        let left_key_col;
        let right_key_col;
        if use_row_encoding {
            left_key_col = PlSmallStr::from(KEY_COL_NAME);
            right_key_col = PlSmallStr::from(KEY_COL_NAME);
        } else {
            left_key_col = left_on[0].clone();
            right_key_col = right_on[0].clone();
        }
        let left_unmerged = DataFrameBuffer::empty_with_schema(left_input_schema.clone());
        let right_unmerged = DataFrameBuffer::empty_with_schema(right_input_schema.clone());
        let left = SideParams {
            input_schema: left_input_schema,
            on: left_on,
            key_col: left_key_col,
            emit_unmatched: matches!(args.how, JoinType::Left | JoinType::Full),
        };
        let right = SideParams {
            input_schema: right_input_schema.clone(),
            on: right_on,
            key_col: right_key_col,
            emit_unmatched: matches!(args.how, JoinType::Right | JoinType::Full),
        };
        let params = MergeJoinParams {
            left,
            right,
            output_schema,
            key_descending: descending,
            key_nulls_last: nulls_last,
            use_row_encoding,
            args,
        };
        Ok(MergeJoinNode {
            state,
            params,
            left_unmerged,
            right_unmerged,
            unmatched: Default::default(),
            mergeable_seq: MorselSeq::default(),
            max_seq_sent: MorselSeq::default(),
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

        assert!(recv.len() == 2);
        assert!(send.len() == 1);

        let input_channels_done = recv[0] == PortState::Done && recv[1] == PortState::Done;
        let output_channel_done = send[0] == PortState::Done;
        let input_buffers_empty = self.left_unmerged.is_empty() && self.right_unmerged.is_empty();
        let unmatched_buffers_empty = self.unmatched.is_empty();
        if self.params.args.maintain_order == MaintainOrderJoin::None {
            debug_assert!(unmatched_buffers_empty);
        }

        if output_channel_done {
            self.state = Done;
        } else if !input_channels_done {
            self.state = Running
        } else if input_channels_done && !input_buffers_empty {
            self.state = FlushInputBuffers;
        } else if input_channels_done
            && input_buffers_empty
            && matches!(self.state, Running | FlushInputBuffers)
        {
            let mut all_unmatched = DataFrame::empty_with_schema(&self.params.output_schema);
            for df in take(&mut self.unmatched).into_values() {
                all_unmatched.vstack_mut_owned(df)?;
            }
            let src_node =
                InMemorySourceNode::new(Arc::new(all_unmatched), self.max_seq_sent.successor());
            self.state = EmitUnmatched(src_node);
        } else if input_channels_done && input_buffers_empty && unmatched_buffers_empty {
            self.left_unmerged.clear();
            self.right_unmerged.clear();
            self.unmatched.clear();
            self.state = Done;
        } else {
            unreachable!()
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
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Ready;
            },
            EmitUnmatched(src_node) => {
                debug_assert!(self.left_unmerged.is_empty());
                debug_assert!(self.right_unmerged.is_empty());
                debug_assert!(self.unmatched.is_empty());
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                src_node.update_state(&mut [], &mut send[..], state)?;
                if send[0] == PortState::Done {
                    self.state = Done;
                }
            },
            Done => {
                debug_assert!(self.left_unmerged.is_empty());
                debug_assert!(self.right_unmerged.is_empty());
                debug_assert!(self.unmatched.is_empty());
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
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

        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);

        let params = &self.params;
        let left_unmerged = &mut self.left_unmerged;
        let right_unmerged = &mut self.right_unmerged;
        let unmatched = &mut self.unmatched;
        let mergeable_seq = &mut self.mergeable_seq;
        let max_seq_sent = &mut self.max_seq_sent;

        let mut recv_left = recv_ports[0].take().map(RecvPort::serial);
        let mut recv_right = recv_ports[1].take().map(RecvPort::serial);

        if matches!(self.state, Running | FlushInputBuffers) {
            assert!(send_ports[0].is_some());

            let send = send_ports[0].take().unwrap().parallel();
            let (mut distributor, dist_recv) =
                distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
            let (unmatched_send, mut unmatched_recv) = tokio::sync::mpsc::channel(send.len());
            let (max_seq_sent_send, mut max_seq_sent_recv) = tokio::sync::mpsc::channel(send.len());

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let source_token = SourceToken::new();
                let mut search_limit = get_ideal_morsel_size();

                loop {
                    if source_token.stop_requested() {
                        buffer_unmerged_from_pipe(recv_left.as_mut(), left_unmerged).await;
                        buffer_unmerged_from_pipe(recv_right.as_mut(), right_unmerged).await;
                        return Ok(());
                    }

                    let fmp = FindMergeableParams {
                        left_done: recv_left.is_none(),
                        right_done: recv_right.is_none(),
                        left_params: &params.left,
                        right_params: &params.right,
                        params,
                    };
                    match find_mergeable(left_unmerged, right_unmerged, &mut search_limit, fmp)? {
                        Left(partitions) => {
                            for (left_mergeable, right_mergeable) in partitions.into_iter() {
                                if let Err((_, _, _, _)) = distributor
                                    .send((
                                        left_mergeable,
                                        right_mergeable,
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
                        Right(NeedMore::Left | NeedMore::Both) if recv_left.is_some() => {
                            let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(recv_right.as_mut(), right_unmerged)
                                    .await;
                                return Ok(());
                            };
                            left_unmerged.push_df(m.into_df());
                        },
                        Right(NeedMore::Right | NeedMore::Both) if recv_right.is_some() => {
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(recv_left.as_mut(), left_unmerged).await;
                                return Ok(());
                            };
                            right_unmerged.push_df(m.into_df());
                        },
                        Right(NeedMore::Finished) => {
                            return Ok(());
                        },
                        Right(other) => {
                            unreachable!("unexpected NeedMore value: {other:?}");
                        },
                    }
                }
            }));

            join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                let unmatched_send = unmatched_send.clone();
                let max_seq_sent_send = max_seq_sent_send.clone();
                scope.spawn_task(TaskPriority::High, async move {
                    let mut arenas = ComputeJoinArenas::default();

                    while let Ok((left, right, seq, source_token)) = recv.recv().await {
                        compute_join(
                            left,
                            right,
                            seq,
                            source_token,
                            params,
                            &mut arenas,
                            &mut send,
                            unmatched_send.clone(),
                            max_seq_sent_send.clone(),
                        )
                        .await?;
                    }
                    Ok(())
                })
            }));

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Some(morsel) = unmatched_recv.recv().await {
                    let (df, seq, _st, _) = morsel.into_inner();
                    if let Some(acc) = unmatched.get_mut(&seq) {
                        acc.vstack_mut_owned(df)?;
                    } else {
                        unmatched.insert(seq, df);
                    }
                }
                Ok(())
            }));
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                while let Some(seq) = max_seq_sent_recv.recv().await {
                    *max_seq_sent = (*max_seq_sent).max(seq);
                }
                Ok(())
            }));
        } else if let EmitUnmatched(src_node) = &mut self.state {
            assert!(recv_ports[0].is_none());
            assert!(recv_ports[1].is_none());
            assert!(send_ports[0].is_some());
            src_node.spawn(scope, &mut [], send_ports, state, join_handles);
        } else {
            unreachable!()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn compute_join(
    left: DataFrameBuffer,
    right: DataFrameBuffer,
    seq: MorselSeq,
    source_token: SourceToken,
    params: &MergeJoinParams,
    arenas: &mut ComputeJoinArenas,
    send: &mut PortSender,
    unmatched_send: tokio::sync::mpsc::Sender<Morsel>,
    max_seq_sent_send: tokio::sync::mpsc::Sender<MorselSeq>,
) -> PolarsResult<()> {
    let mut left = left.into_df();
    let mut right = right.into_df();
    left.rechunk_mut();
    right.rechunk_mut();

    let build;
    let build_sp;
    let gather_build;
    let probe;
    let probe_sp;
    let gather_probe;
    if params.left_is_build() {
        build = left.clone();
        build_sp = &params.left;
        gather_build = &mut arenas.gather_left;
        probe = right.clone();
        probe_sp = &params.right;
        gather_probe = &mut arenas.gather_right;
    } else {
        build = right.clone();
        build_sp = &params.right;
        gather_build = &mut arenas.gather_right;
        probe = left.clone();
        probe_sp = &params.left;
        gather_probe = &mut arenas.gather_left;
    }

    let mut build_key = Cow::Borrowed(
        build
            .column(&build_sp.key_col)
            .unwrap()
            .as_materialized_series(),
    );
    let mut probe_key = Cow::Borrowed(
        probe
            .column(&probe_sp.key_col)
            .unwrap()
            .as_materialized_series(),
    );

    #[cfg(feature = "dtype-categorical")]
    {
        // Categoricals are lexicographically ordered, not by their physical values.
        if matches!(build_key.dtype(), DataType::Categorical(_, _)) {
            build_key = Cow::Owned(build_key.cast(&DataType::String)?);
        }
        if matches!(probe_key.dtype(), DataType::Categorical(_, _)) {
            probe_key = Cow::Owned(probe_key.cast(&DataType::String)?);
        }
    }

    let build_key = build_key.to_physical_repr();
    let probe_key = probe_key.to_physical_repr();

    arenas.matched_probeside.clear();
    arenas.matched_probeside.resize(probe_key.len(), false);

    let mut current_offset = 0;
    let mut done = false;
    while !done {
        gather_build.clear();
        gather_probe.clear();
        done = compute_join_dispatch(
            &build_key,
            &probe_key,
            gather_build,
            gather_probe,
            &mut arenas.matched_probeside,
            &mut current_offset,
            build_sp,
            probe_sp,
            params,
        );
        let df = gather_and_postprocess(
            build.clone(),
            probe.clone(),
            gather_build,
            gather_probe,
            &mut arenas.df_builders,
            params,
        )?;
        if df.height() > 0 {
            let morsel = Morsel::new(df, seq, source_token.clone());
            if send.send(morsel).await.is_err() {
                return Ok(());
            };
        }
    }

    if probe_sp.emit_unmatched {
        gather_build.clear();
        gather_probe.clear();
        for (idx, _) in arenas
            .matched_probeside
            .iter()
            .enumerate_idx()
            .filter(|(_, m)| !m)
        {
            gather_build.push(IdxSize::MAX);
            gather_probe.push(idx);
        }

        let df_unmatched = gather_and_postprocess(
            build,
            probe,
            gather_build,
            gather_probe,
            &mut arenas.df_builders,
            params,
        )?;
        if df_unmatched.height() > 0 {
            let morsel = Morsel::new(df_unmatched, seq, source_token.clone());
            if params.args.maintain_order == MaintainOrderJoin::None {
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }
            } else if unmatched_send.send(morsel).await.is_err() {
                panic!("broken pipe");
            }
        }
    }

    if max_seq_sent_send.send(seq).await.is_err() {
        panic!();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_join_dispatch(
    lk: &Series,
    rk: &Series,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    current_offset: &mut usize,
    left_sp: &SideParams,
    right_sp: &SideParams,
    params: &MergeJoinParams,
) -> bool {
    macro_rules! dispatch {
        ($left_key_ca:expr) => {
            compute_join_kernel(
                $left_key_ca,
                rk.as_ref().as_ref(),
                gather_left,
                gather_right,
                matched_right,
                current_offset,
                left_sp,
                right_sp,
                params,
            )
        };
    }

    assert_eq!(lk.dtype(), rk.dtype());
    match lk.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                type PhysCa = ChunkedArray<$T>;
                let lk_ca: &PhysCa  = lk.as_ref().as_ref();
                dispatch!(lk_ca)
            })
        },
        DataType::Boolean => dispatch!(lk.bool().unwrap()),
        DataType::String => dispatch!(lk.str().unwrap()),
        DataType::Binary => dispatch!(lk.binary().unwrap()),
        DataType::BinaryOffset => dispatch!(lk.binary_offset().unwrap()),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(cats, _) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
            let lk_ca: &PhysCa = lk.as_ref().as_ref();
            dispatch!(lk_ca)
        }),
        DataType::Null => compute_join_kernel_nullkeys(
            lk.len(),
            rk.len(),
            gather_left,
            gather_right,
            matched_right,
            current_offset,
            left_sp,
            right_sp,
            params,
        ),
        dt => unimplemented!("merge-join kernel not implemented for {:?}", dt),
    }
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn compute_join_kernel<'a, T: PolarsDataType>(
    left_key: &'a ChunkedArray<T>,
    right_key: &'a ChunkedArray<T>,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    current_offset: &mut usize,
    left_sp: &SideParams,
    right_sp: &SideParams,
    params: &MergeJoinParams,
) -> bool
where
    T::Physical<'a>: TotalOrd,
{
    let morsel_size = get_ideal_morsel_size();

    debug_assert!(gather_left.is_empty());
    debug_assert!(gather_right.is_empty());
    if right_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_key.len());
    }

    let descending = params.key_descending;
    let left_key = left_key.downcast_as_array();
    let right_key = right_key.downcast_as_array();

    let mut iterator = left_key.iter().enumerate().skip(*current_offset).peekable();
    if iterator.peek().is_none() {
        return true;
    }
    let mut skip_ahead_right = 0;
    for (idxl, left_keyval) in iterator {
        if gather_left.len() >= morsel_size {
            return false;
        }
        let left_keyval = left_keyval.as_ref();
        let mut matched = false;
        if params.args.nulls_equal || left_keyval.is_some() {
            for idxr in skip_ahead_right..right_key.len() {
                let right_keyval = unsafe { right_key.get_unchecked(idxr) };
                let right_keyval = right_keyval.as_ref();
                let mut ord: Option<Ordering> = match (&left_keyval, &right_keyval) {
                    (None, None) if params.args.nulls_equal => Some(Ordering::Equal),
                    (Some(l), Some(r)) => Some(TotalOrd::tot_cmp(*l, *r)),
                    _ => None,
                };
                if descending {
                    ord = ord.map(Ordering::reverse);
                }
                if ord == Some(Ordering::Equal) {
                    matched = true;
                    if right_sp.emit_unmatched {
                        matched_right.set(idxr, true);
                    }
                    gather_left.push(idxl as IdxSize);
                    gather_right.push(idxr as IdxSize);
                } else if ord == Some(Ordering::Greater) {
                    skip_ahead_right = idxr;
                } else if ord == Some(Ordering::Less) {
                    break;
                }
            }
        }
        if left_sp.emit_unmatched && !matched {
            gather_left.push(idxl as IdxSize);
            gather_right.push(IdxSize::MAX);
        }
        *current_offset += 1;
    }
    true
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn compute_join_kernel_nullkeys(
    left_n: usize,
    right_n: usize,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    current_offset: &mut usize,
    left_sp: &SideParams,
    right_sp: &SideParams,
    params: &MergeJoinParams,
) -> bool {
    debug_assert!(gather_left.is_empty());
    debug_assert!(gather_right.is_empty());
    if right_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_n);
    }
    if !params.args.nulls_equal {
        return true;
    }

    for idxl in *current_offset..left_n {
        gather_left.push(idxl as IdxSize);
        for idxr in 0..right_n {
            gather_right.push(idxr as IdxSize);
            if right_sp.emit_unmatched {
                matched_right.set(idxr, true);
            }
        }
        if left_sp.emit_unmatched && right_n == 0 {
            gather_right.push(IdxSize::MAX);
        }
        *current_offset += 1;
        if gather_left.len() >= get_ideal_morsel_size() {
            return false;
        }
    }
    true
}

fn gather_and_postprocess(
    build: DataFrame,
    probe: DataFrame,
    gather_build: &[IdxSize],
    gather_probe: &[IdxSize],
    df_builders: &mut Option<(DataFrameBuilder, DataFrameBuilder)>,
    params: &MergeJoinParams,
) -> PolarsResult<DataFrame> {
    let should_coalesce = params.args.should_coalesce();

    let mut left;
    let gather_left;
    let mut right;
    let gather_right;
    if params.left_is_build() {
        left = build;
        gather_left = gather_build;
        right = probe;
        gather_right = gather_probe;
    } else {
        right = build;
        gather_right = gather_build;
        left = probe;
        gather_left = gather_probe;
    }

    // Remove non-payload columns
    for col in left
        .columns()
        .iter()
        .map(Column::name)
        .cloned()
        .collect_vec()
    {
        if params.left.on.contains(&col) && should_coalesce {
            continue;
        }
        if !params.output_schema.contains(&col) {
            left.drop_in_place(&col).unwrap();
        }
    }
    for col in right
        .columns()
        .iter()
        .map(Column::name)
        .cloned()
        .collect_vec()
    {
        if params.left.on.contains(&col) && should_coalesce {
            continue;
        }
        let renamed_col = match left.schema().contains(&col) {
            true => Cow::Owned(format_pl_smallstr!("{}{}", col, params.args.suffix())),
            false => Cow::Borrowed(&col),
        };
        if !params.output_schema.contains(&renamed_col) {
            right.drop_in_place(&col).unwrap();
        }
    }

    if df_builders.is_none() {
        *df_builders = Some((
            DataFrameBuilder::new(left.schema().clone()),
            DataFrameBuilder::new(right.schema().clone()),
        ));
    }

    let (left_build, right_build) = df_builders.as_mut().unwrap();
    if params.right.emit_unmatched {
        left_build.opt_gather_extend(&left, gather_left, ShareStrategy::Never);
    } else {
        unsafe { left_build.gather_extend(&left, gather_left, ShareStrategy::Never) };
    }
    if params.left.emit_unmatched {
        right_build.opt_gather_extend(&right, gather_right, ShareStrategy::Never);
    } else {
        unsafe { right_build.gather_extend(&right, gather_right, ShareStrategy::Never) };
    }

    let mut left = left_build.freeze_reset();
    let mut right = right_build.freeze_reset();

    // Coalsesce the key columns
    if params.args.how == JoinType::Left && should_coalesce {
        for c in &params.left.on {
            if right.schema().contains(c) {
                right.drop_in_place(c.as_str())?;
            }
        }
    } else if params.args.how == JoinType::Right && should_coalesce {
        for c in &params.right.on {
            if left.schema().contains(c) {
                left.drop_in_place(c.as_str())?;
            }
        }
    }

    // Rename any right columns to "{}_right"
    let left_cols: PlHashSet<_> = left.columns().iter().map(Column::name).cloned().collect();
    let right_cols_vec = right.get_column_names_owned();
    let renames = right_cols_vec
        .iter()
        .filter(|c| left_cols.contains(*c))
        .map(|c| {
            let renamed = format_pl_smallstr!("{}{}", c, params.args.suffix());
            (c.as_str(), renamed)
        });
    right.rename_many(renames).unwrap();

    left.hstack_mut(right.columns())?;

    if params.args.how == JoinType::Full && should_coalesce {
        // Coalesce key columns
        for (left_keycol, right_keycol) in
            Iterator::zip(params.left.on.iter(), params.right.on.iter())
        {
            let right_keycol = format_pl_smallstr!("{}{}", right_keycol, params.args.suffix());
            let left_col = left.column(left_keycol).unwrap();
            let right_col = left.column(&right_keycol).unwrap();
            let coalesced = coalesce_columns(&[left_col.clone(), right_col.clone()]).unwrap();
            left.replace(left_keycol, coalesced)
                .unwrap()
                .drop_in_place(&right_keycol)
                .unwrap();
        }
    }

    if should_coalesce {
        for col in &params.right.on {
            let renamed = format_pl_smallstr!("{}{}", col, params.args.suffix());
            if left.schema().contains(&renamed) && !params.output_schema.contains(&renamed) {
                left.drop_in_place(&renamed).unwrap();
            }
        }
    }

    Ok(left)
}

async fn buffer_unmerged_from_pipe(
    port: Option<&mut PortReceiver>,
    unmerged: &mut DataFrameBuffer,
) {
    let Some(port) = port else {
        return;
    };
    let Ok(morsel) = port.recv().await else {
        return;
    };
    morsel.source_token().stop();
    unmerged.push_df(morsel.into_df());

    while let Ok(morsel) = port.recv().await {
        unmerged.push_df(morsel.into_df());
    }
}

#[derive(Clone, Debug)]
struct FindMergeableParams<'sp, 'p> {
    left_done: bool,
    right_done: bool,
    left_params: &'sp SideParams,
    right_params: &'sp SideParams,
    params: &'p MergeJoinParams,
}

fn find_mergeable(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    search_limit: &mut usize,
    fmp: FindMergeableParams,
) -> PolarsResult<Either<UnitVec<(DataFrameBuffer, DataFrameBuffer)>, NeedMore>> {
    let (left_mergeable, right_mergeable) =
        match find_mergeable_limiting(left, right, search_limit, fmp.clone())? {
            Left((left, right)) => (left, right),
            Right(need_more) => return Ok(Right(need_more)),
        };
    assert!(!left_mergeable.is_empty() || !right_mergeable.is_empty());

    let partitions = find_mergeable_partition(left_mergeable, right_mergeable, fmp)?;
    Ok(Left(partitions))
}

fn find_mergeable_limiting(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    search_limit: &mut usize,
    fmp: FindMergeableParams,
) -> PolarsResult<Either<(DataFrameBuffer, DataFrameBuffer), NeedMore>> {
    const SEARCH_LIMIT_BUMP_FACTOR: usize = 2;
    let morsel_size = get_ideal_morsel_size();
    debug_assert!(*search_limit >= morsel_size);
    let mut mergeable = find_mergeable_search(left, right, *search_limit, fmp.clone())?;
    while match mergeable {
        Right(NeedMore::Left | NeedMore::Both) if *search_limit < left.height() => true,
        Right(NeedMore::Right | NeedMore::Both) if *search_limit < right.height() => true,
        _ => false,
    } {
        // Exponential increase
        *search_limit *= SEARCH_LIMIT_BUMP_FACTOR;
        mergeable = find_mergeable_search(left, right, *search_limit, fmp.clone())?;
    }
    if mergeable.is_left() {
        *search_limit = morsel_size;
    }
    Ok(mergeable)
}

fn find_mergeable_partition(
    left: DataFrameBuffer,
    right: DataFrameBuffer,
    fmp: FindMergeableParams,
) -> PolarsResult<UnitVec<(DataFrameBuffer, DataFrameBuffer)>> {
    let morsel_size = get_ideal_morsel_size();
    let maintain_order_left = matches!(
        fmp.params.args.maintain_order,
        MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight
    );
    let maintain_order_right = matches!(
        fmp.params.args.maintain_order,
        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft
    );
    let partition_left = fmp.left_params.emit_unmatched || maintain_order_left;
    let partition_right = fmp.right_params.emit_unmatched || maintain_order_right;
    if partition_left && partition_right {
        // TODO: We may be able to partition a subset of these cases (e.g. for FULL joins)
        return Ok(UnitVec::from([(left, right)]));
    }

    let partition_count = (left.height() * right.height() + left.height() + right.height())
        .div_ceil(morsel_size.pow(2));
    if partition_count <= 1 {
        return Ok(UnitVec::from([(left, right)]));
    }

    let partition_side;
    let broadcast_side;
    let mut partitions = UnitVec::with_capacity(partition_count);
    let mut push_partition: Box<dyn FnMut(DataFrameBuffer, DataFrameBuffer)>;
    if partition_left {
        partition_side = left;
        broadcast_side = right;
        push_partition = Box::new(|part, broad| partitions.push((part, broad)));
    } else {
        partition_side = right;
        broadcast_side = left;
        push_partition = Box::new(|part, broad| partitions.push((broad, part)));
    }

    let chunk_size = partition_side.height().div_ceil(partition_count);
    let mut offset = 0;
    while offset < partition_side.height() - chunk_size {
        let partition_chunk = partition_side.clone().slice(offset, chunk_size);
        push_partition(partition_chunk, broadcast_side.clone());
        offset += chunk_size;
    }
    // Always make sure that there is at least one partition, even if the build side is empty.
    let build_chunk = partition_side.slice(offset, chunk_size);
    push_partition(build_chunk, broadcast_side);

    drop(push_partition);
    Ok(partitions)
}

fn find_mergeable_search(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    search_limit: usize,
    fmp: FindMergeableParams,
) -> PolarsResult<Either<(DataFrameBuffer, DataFrameBuffer), NeedMore>> {
    let FindMergeableParams {
        left_done,
        right_done,
        left_params,
        right_params,
        params,
    } = fmp;
    let left_empty_buf = || DataFrameBuffer::empty_with_schema(left_params.input_schema.clone());
    let right_empty_buf = || DataFrameBuffer::empty_with_schema(right_params.input_schema.clone());
    let left_get = |idx| unsafe { left.get_bypass_validity(&left_params.key_col, idx, params) };
    let right_get = |idx| unsafe { right.get_bypass_validity(&right_params.key_col, idx, params) };

    if left_done && left.is_empty() && right_done && right.is_empty() {
        return Ok(Right(NeedMore::Finished));
    } else if left_done && left.is_empty() && !right_done && right.is_empty() {
        return Ok(Right(NeedMore::Right));
    } else if right_done && right.is_empty() && !left_done && left.is_empty() {
        return Ok(Right(NeedMore::Left));
    } else if left_done && left.is_empty() && !right_params.emit_unmatched {
        // We will never match on the remaining right keys
        right.clear();
        return Ok(Right(NeedMore::Finished));
    } else if right_done && right.is_empty() && !left_params.emit_unmatched {
        // We will never match on the remaining left keys
        left.clear();
        return Ok(Right(NeedMore::Finished));
    } else if left_done && left.is_empty() {
        let right_split = right.split_at(get_ideal_morsel_size());
        return Ok(Left((left_empty_buf(), right_split)));
    } else if right_done && right.is_empty() {
        let left_split = left.split_at(get_ideal_morsel_size());
        return Ok(Left((left_split, right_empty_buf())));
    } else if left.is_empty() && !left_done {
        return Ok(Right(NeedMore::Left));
    } else if right.is_empty() && !right_done {
        return Ok(Right(NeedMore::Right));
    }

    let left_first = left_get(0);
    let right_first = right_get(0);

    // First return chunks of nulls if there are any
    if !params.args.nulls_equal && !params.key_nulls_last && left_first == AnyValue::Null {
        let left_first_nonnull_idx =
            binary_search_upper(left, &AnyValue::Null, params, left_params)?;
        let left_split = left.split_at(left_first_nonnull_idx);
        return Ok(Left((left_split, right_empty_buf())));
    }
    if !params.args.nulls_equal && !params.key_nulls_last && right_first == AnyValue::Null {
        let right_first_nonnull_idx =
            binary_search_upper(right, &AnyValue::Null, params, right_params)?;
        let right_split = right.split_at(right_first_nonnull_idx);
        return Ok(Left((left_empty_buf(), right_split)));
    }

    let left_last_idx = usize::min(left.height(), search_limit);
    let left_last = left_get(left_last_idx - 1);
    let left_first_incomplete = match left_done {
        false => binary_search_lower(left, &left_last, params, left_params)?,
        true => left.height(),
    };

    let right_last_idx = usize::min(right.height(), search_limit);
    let right_last = right_get(right_last_idx - 1);
    let right_first_incomplete = match right_done {
        false => binary_search_lower(right, &right_last, params, right_params)?,
        true => right.height(),
    };

    if left_first_incomplete == 0 && right_first_incomplete == 0 {
        debug_assert!(!left_done && !right_done);
        return Ok(Right(NeedMore::Both));
    } else if left_first_incomplete == 0 {
        debug_assert!(!left_done);
        return Ok(Right(NeedMore::Left));
    } else if right_first_incomplete == 0 {
        debug_assert!(!right_done);
        return Ok(Right(NeedMore::Right));
    }

    let left_last_completed_val = left_get(left_first_incomplete - 1);
    let right_last_completed_val = right_get(right_first_incomplete - 1);

    let left_mergeable_until; // bound is *exclusive*
    let right_mergeable_until;
    let ord = keys_cmp(&left_last_completed_val, &right_last_completed_val, params);
    if ord.is_eq() {
        left_mergeable_until = left_first_incomplete;
        right_mergeable_until = right_first_incomplete;
    } else if ord.is_lt() {
        left_mergeable_until = left_first_incomplete;
        right_mergeable_until = binary_search_upper(
            right,
            &left_get(left_mergeable_until - 1),
            params,
            right_params,
        )?;
    } else if ord.is_gt() {
        right_mergeable_until = right_first_incomplete;
        left_mergeable_until = binary_search_upper(
            left,
            &right_get(right_mergeable_until - 1),
            params,
            left_params,
        )?;
    } else {
        unreachable!();
    }

    if left_mergeable_until == 0 && right_mergeable_until == 0 {
        return Ok(Right(NeedMore::Both));
    }

    let left_split = left.split_at(left_mergeable_until);
    let right_split = right.split_at(right_mergeable_until);
    Ok(Left((left_split, right_split)))
}

unsafe fn series_get_bypass_validity<'a>(
    s: &'a Series,
    index: usize,
    params: &MergeJoinParams,
) -> AnyValue<'a> {
    debug_assert!(index < s.len());
    if params.use_row_encoding {
        let arr = s.binary_offset().unwrap();
        unsafe { arr.get_any_value_bypass_validity(index) }
    } else {
        unsafe { s.get_unchecked(index) }
    }
}

fn binary_search(
    vec: &DataFrameBuffer,
    search_value: &AnyValue,
    is_before: fn(Ordering) -> bool,
    params: &MergeJoinParams,
    sp: &SideParams,
) -> PolarsResult<usize> {
    let mut lower = 0;
    let mut upper = vec.height();
    while lower < upper {
        let mid = (lower + upper) / 2;
        let mid_val = unsafe { vec.get_bypass_validity(&sp.key_col, mid, params) };
        if is_before(keys_cmp(search_value, &mid_val, params)) {
            upper = mid;
        } else {
            lower = mid + 1;
        }
    }
    Ok(lower)
}

fn binary_search_lower(
    vec: &DataFrameBuffer,
    sv: &AnyValue,
    params: &MergeJoinParams,
    sp: &SideParams,
) -> PolarsResult<usize> {
    binary_search(vec, sv, Ordering::is_le, params, sp)
}

fn binary_search_upper(
    vec: &DataFrameBuffer,
    sv: &AnyValue,
    params: &MergeJoinParams,
    sp: &SideParams,
) -> PolarsResult<usize> {
    binary_search(vec, sv, Ordering::is_lt, params, sp)
}

fn keys_cmp(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> Ordering {
    match AnyValue::partial_cmp(left, right) {
        Some(Ordering::Equal) => Ordering::Equal,
        _ if left.is_null() && params.key_nulls_last => Ordering::Greater,
        _ if right.is_null() && params.key_nulls_last => Ordering::Less,
        _ if left.is_null() => Ordering::Less,
        _ if right.is_null() => Ordering::Greater,
        Some(Ordering::Greater) if params.key_descending => Ordering::Less,
        Some(Ordering::Less) if params.key_descending => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Greater,
        Some(Ordering::Less) => Ordering::Less,
        None => unreachable!(),
    }
}

#[derive(Clone)]
struct DataFrameBuffer {
    schema: SchemaRef,
    buf: BTreeMap<usize, DataFrame>,
    total_rows: usize,
    skip_rows: usize,
    frozen: bool,
}

impl fmt::Debug for DataFrameBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.clone().into_df().fmt(f)
    }
}

impl DataFrameBuffer {
    fn empty_with_schema(schema: SchemaRef) -> Self {
        DataFrameBuffer {
            schema,
            buf: BTreeMap::new(),
            total_rows: 0,
            skip_rows: 0,
            frozen: false,
        }
    }

    fn height(&self) -> usize {
        self.total_rows
    }

    unsafe fn get_bypass_validity(
        &self,
        column: &str,
        row_index: usize,
        params: &MergeJoinParams,
    ) -> AnyValue<'_> {
        debug_assert!(row_index < self.total_rows);
        let first_offset = match self.buf.first_key_value() {
            Some((offset, _)) => *offset,
            None => 0,
        };
        let buf_index = self.skip_rows + first_offset + row_index;
        let (df_offset, df) = self.buf.range(..=buf_index).next_back().unwrap();
        let series_index = buf_index - df_offset;
        let series = df.column(column).unwrap().as_materialized_series();
        unsafe { series_get_bypass_validity(series, series_index, params) }
    }

    fn push_df(&mut self, df: DataFrame) {
        assert!(!self.frozen);
        let added_rows = df.height();
        let offset = match self.buf.last_key_value() {
            Some((last_key, last_df)) => last_key + last_df.height(),
            None => 0,
        };
        self.buf.insert(offset, df);
        self.total_rows += added_rows;
    }

    fn split_at(&mut self, mut at: usize) -> Self {
        at = at.clamp(0, self.total_rows);
        let mut left = self.clone();
        left.total_rows = at;
        left.frozen = true;
        self.skip_rows += at;
        self.total_rows -= at;
        self.gc();
        left
    }

    fn slice(mut self, offset: usize, len: usize) -> Self {
        self.skip_rows += offset;
        self.total_rows -= offset;
        self.total_rows = usize::min(self.total_rows, len);
        self.frozen = true;
        self
    }

    fn into_df(self) -> DataFrame {
        let mut acc = DataFrame::empty_with_schema(&self.schema);
        for df in self.buf.into_values() {
            acc.vstack_mut_owned(df).unwrap();
        }
        acc.slice(self.skip_rows as i64, self.total_rows)
    }

    fn gc(&mut self) {
        while let Some((_, df)) = self.buf.first_key_value() {
            if self.skip_rows > df.height() {
                let (_, df) = self.buf.pop_first().unwrap();
                self.skip_rows -= df.height();
            } else {
                break;
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    fn clear(&mut self) {
        assert!(!self.frozen);
        self.buf.clear();
        self.total_rows = 0;
        self.skip_rows = 0;
    }
}
