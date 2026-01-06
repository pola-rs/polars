use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::mem::swap;

use arrow::array::Array;
use arrow::array::builder::ShareStrategy;
use arrow::bitmap::MutableBitmap;
use either::{Either, Left, Right};
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::utils::Container;
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
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

// TODO: [amber] For unmatched rows: gather first and then hstack to a df of nulls
// TODO: [amber] Remove any non-output columns earlier to reduce the
// amount of gathering as much as possible
// TODO: [amber] Do not linearize but accumlate in parallel and then merge all of
// the unmatched dataframes in a state-transition, and then flush them
// TODO: [amber] Make sure that key expressions work

pub const KEY_COL_NAME: &str = "__POLARS_JOIN_KEY_TMP";

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Left,
    Right,
    Both,
    Finished,
}

impl NeedMore {
    fn flip(self) -> Self {
        match self {
            NeedMore::Left => NeedMore::Right,
            NeedMore::Right => NeedMore::Left,
            other => other,
        }
    }
}

#[derive(Debug)]
struct SideParams {
    input_schema: SchemaRef,
    ir_schema: SchemaRef,
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

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: DataFrameBuffer,
    right_unmerged: DataFrameBuffer,
    unmatched: BTreeMap<MorselSeq, DataFrameBuffer>,
    seq: MorselSeq,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum MergeJoinState {
    #[default]
    Running,
    FlushInputBuffers,
    EmitUnmatched,
    Done,
}

impl MergeJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        descending: bool,
        nulls_last: bool,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        assert!(left_on.len() == right_on.len());
        assert!(
            left_input_schema.contains(KEY_COL_NAME) == right_input_schema.contains(KEY_COL_NAME),
        );

        let use_row_encoding = left_input_schema.contains(KEY_COL_NAME);
        let state: MergeJoinState = MergeJoinState::Running;
        let left_key_col;
        let right_key_col;
        let mut left_ir_schema = left_input_schema.clone();
        let mut right_ir_schema = right_input_schema.clone();
        if use_row_encoding {
            left_key_col = PlSmallStr::from(KEY_COL_NAME);
            right_key_col = PlSmallStr::from(KEY_COL_NAME);
            let mut ir_schema = (*left_input_schema).clone();
            ir_schema.insert(left_key_col.clone(), DataType::BinaryOffset);
            left_ir_schema = Arc::new(ir_schema);
            let mut ir_schema = (*right_input_schema).clone();
            ir_schema.insert(right_key_col.clone(), DataType::BinaryOffset);
            right_ir_schema = Arc::new(ir_schema);
        } else {
            left_key_col = left_on[0].clone();
            right_key_col = right_on[0].clone();
        }
        let left = SideParams {
            input_schema: left_input_schema,
            ir_schema: left_ir_schema.clone(),
            on: left_on,
            key_col: left_key_col,
            emit_unmatched: matches!(args.how, JoinType::Left | JoinType::Full),
        };
        let right = SideParams {
            input_schema: right_input_schema,
            ir_schema: right_ir_schema.clone(),
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
            left_unmerged: DataFrameBuffer::empty_with_schema(&left_ir_schema),
            right_unmerged: DataFrameBuffer::empty_with_schema(&right_ir_schema),
            unmatched: Default::default(),
            seq: MorselSeq::default(),
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
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2);
        assert!(send.len() == 1);

        let prev_state = self.state;
        let input_channels_done = recv[0] == PortState::Done && recv[1] == PortState::Done;
        let output_channel_done = send[0] == PortState::Done;
        let input_buffers_empty = self.left_unmerged.is_empty() && self.right_unmerged.is_empty();
        let unmatched_buffers_empty = self.unmatched.is_empty();

        if output_channel_done {
            self.state = MergeJoinState::Done;
        } else if !input_channels_done {
            self.state = MergeJoinState::Running
        } else if input_channels_done && !input_buffers_empty {
            self.state = MergeJoinState::FlushInputBuffers;
        } else if input_channels_done && input_buffers_empty && !unmatched_buffers_empty {
            self.state = MergeJoinState::EmitUnmatched;
        } else if input_channels_done && input_buffers_empty && unmatched_buffers_empty {
            self.state = MergeJoinState::Done;
        } else {
            unreachable!()
        }
        assert!(prev_state <= self.state);

        match self.state {
            MergeJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
            },
            MergeJoinState::FlushInputBuffers | MergeJoinState::EmitUnmatched => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Ready;
            },
            MergeJoinState::Done => {
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        use MergeJoinState::*;

        let params = &self.params;
        let left_unmerged = &mut self.left_unmerged;
        let right_unmerged = &mut self.right_unmerged;
        let unmatched = &mut self.unmatched;
        let seq = &mut self.seq;

        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        let mut recv_left = recv_ports[0].take().map(RecvPort::serial);
        let mut recv_right = recv_ports[1].take().map(RecvPort::serial);

        if recv_left.is_none() && recv_right.is_none() {
            assert!(self.state >= FlushInputBuffers);
        }

        if matches!(self.state, Running | FlushInputBuffers) {
            let send = send_ports[0].take().unwrap().parallel();
            let (mut distributor, dist_recv) =
                distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
            let (unmatched_send, mut unmatched_recv) = tokio::sync::mpsc::channel(send.len());

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let source_token = SourceToken::new();
                let mut search_limit = get_ideal_morsel_size();

                loop {
                    match find_mergeable(
                        left_unmerged,
                        right_unmerged,
                        recv_left.is_none(),
                        recv_right.is_none(),
                        &mut search_limit,
                        params,
                    )? {
                        Left(partitions) => {
                            // if partitions.len() > 1 {
                            //     dbg!(&partitions.len());
                            // }
                            for (left_mergeable, right_mergeable) in partitions.into_iter() {
                                if let Err((_left, _right, _, _)) = distributor
                                    .send((
                                        left_mergeable,
                                        right_mergeable,
                                        *seq,
                                        source_token.clone(),
                                    ))
                                    .await
                                {
                                    panic!();
                                }
                                *seq = seq.successor();
                            }
                        },
                        Right(NeedMore::Left | NeedMore::Both) if recv_left.is_some() => {
                            let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(recv_right.as_mut(), right_unmerged)
                                    .await;
                                break;
                            };
                            left_unmerged.push_df(m.into_df());
                        },
                        Right(NeedMore::Right | NeedMore::Both) if recv_right.is_some() => {
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(recv_left.as_mut(), left_unmerged).await;
                                break;
                            };
                            right_unmerged.push_df(m.into_df());
                        },

                        Right(NeedMore::Finished) => {
                            break;
                        },
                        Right(other) => {
                            unreachable!("unexpected NeedMore value: {other:?}");
                        },
                    }
                }
                Ok(())
            }));

            join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                let unmatched_send = unmatched_send.clone();
                scope.spawn_task(TaskPriority::High, async move {
                    let mut arenas = Arenas::default();

                    while let Ok((left, right, seq, source_token)) = recv.recv().await {
                        let mut left_df = left.clone().into_df();
                        let mut right_df = right.clone().into_df();

                        left_df.rechunk_mut();
                        right_df.rechunk_mut();

                        compute_join(
                            left.clone(),
                            right.clone(),
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
                while let Some(morsel) = unmatched_recv.recv().await {
                    let (df, seq, _st, _) = morsel.into_inner();
                    if let Some(buf) = unmatched.get_mut(&seq) {
                        buf.push_df(df);
                    } else {
                        let mut buf = DataFrameBuffer::empty_with_schema(df.schema());
                        buf.push_df(df);
                        unmatched.insert(seq, buf);
                    }
                }
                Ok(())
            }));
        } else if self.state == MergeJoinState::EmitUnmatched {
            assert!(recv_ports[0].is_none());
            assert!(recv_ports[1].is_none());
            assert!(send_ports[0].is_some());
            let mut send = send_ports[0].take().unwrap().serial();

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Some((_, mut buf)) = unmatched.pop_first() {
                    while !buf.is_empty() {
                        let df = buf.split_at(get_ideal_morsel_size()).into_df();
                        let morsel = Morsel::new(df, *seq, SourceToken::new());
                        if let Err(_morsel) = send.send(morsel).await {
                            panic!();
                        }
                    }
                }
                Ok(())
            }));
        }
    }
}

fn find_mergeable(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    left_done: bool,
    right_done: bool,
    search_limit: &mut usize,
    p: &MergeJoinParams,
) -> PolarsResult<Either<UnitVec<(DataFrameBuffer, DataFrameBuffer)>, NeedMore>> {
    const SEARCH_LIMIT_BUMP_FACTOR: usize = 2;
    let morsel_size = get_ideal_morsel_size();
    debug_assert!(*search_limit >= morsel_size);
    let mut mergeable = find_mergeable_flip(left, right, left_done, right_done, *search_limit, p)?;
    while match mergeable {
        Right(NeedMore::Left | NeedMore::Both) if *search_limit < left.len() => true,
        Right(NeedMore::Right | NeedMore::Both) if *search_limit < right.len() => true,
        _ => false,
    } {
        // Exponential increase
        *search_limit *= SEARCH_LIMIT_BUMP_FACTOR;
        mergeable = find_mergeable_flip(left, right, left_done, right_done, *search_limit, p)?;
    }
    if mergeable.is_left() {
        *search_limit /= SEARCH_LIMIT_BUMP_FACTOR;
        if *search_limit < morsel_size {
            *search_limit = morsel_size;
        }
    }

    Ok(mergeable)
}

fn find_mergeable_flip(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    left_done: bool,
    right_done: bool,
    search_limit: usize,
    p: &MergeJoinParams,
) -> PolarsResult<Either<UnitVec<(DataFrameBuffer, DataFrameBuffer)>, NeedMore>> {
    if p.args.how == JoinType::Right {
        // dbg!(left.len() as f32 / 100000.0, right.len() as f32 / 100000.0);
        // let tick = std::time::Instant::now();
        let ok = find_mergeable_partition(
            right,
            left,
            right_done,
            left_done,
            search_limit,
            &p.right,
            &p.left,
            p,
        )?;
        // let find_mergeable = tick.elapsed();
        // dbg!(find_mergeable);
        Ok(match ok {
            Left(mut partitions) => {
                partitions.iter_mut().for_each(|(x1, x2)| swap(x1, x2));
                Left(partitions)
            },
            Right(side) => Right(side.flip()),
        })
    } else {
        // dbg!(left.len() as f32 / 100000.0, right.len() as f32 / 100000.0);
        // let tick = std::time::Instant::now();
        let ok = find_mergeable_partition(
            left,
            right,
            left_done,
            right_done,
            search_limit,
            &p.left,
            &p.right,
            p,
        )?;
        // let find_mergeable = tick.elapsed();
        // dbg!(find_mergeable);
        Ok(match ok {
            Left(partitions) => Left(partitions),
            other => other,
        })
    }
}

fn find_mergeable_partition(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    left_done: bool,
    right_done: bool,
    limit: usize,
    left_params: &SideParams,
    right_params: &SideParams,
    params: &MergeJoinParams,
) -> PolarsResult<Either<UnitVec<(DataFrameBuffer, DataFrameBuffer)>, NeedMore>> {
    let morsel_size = get_ideal_morsel_size();
    let mergeable = find_mergeable_search(
        left,
        right,
        left_done,
        right_done,
        limit,
        left_params,
        right_params,
        params,
    )?;
    let (left, right) = match mergeable {
        Left((left, right)) => (left, right),
        Right(need_more) => return Ok(Right(need_more)),
    };
    assert!(
        !left.is_empty() || !right.is_empty(),
        "search result is empty"
    );
    let chunks_count =
        (left.len() * right.len() + left.len() + right.len()).div_ceil(morsel_size.pow(2));
    if chunks_count <= 1 {
        return Ok(Left(UnitVec::from([(left, right)])));
    }
    let chunk_size = left.len().div_ceil(chunks_count);
    let mut offset = 0;
    let mut partitioned = UnitVec::with_capacity(chunks_count);
    while offset < left.len() - offset {
        let left_chunk = left.clone().slice(offset, chunk_size);
        partitioned.push((left_chunk, right.clone()));
        offset += chunk_size;
    }
    let left_chunk = left.slice(offset, chunk_size);
    partitioned.push((left_chunk, right));
    Ok(Left(partitioned))
}

fn find_mergeable_search(
    left: &mut DataFrameBuffer,
    right: &mut DataFrameBuffer,
    left_done: bool,
    right_done: bool,
    limit: usize,
    left_params: &SideParams,
    right_params: &SideParams,
    params: &MergeJoinParams,
) -> PolarsResult<Either<(DataFrameBuffer, DataFrameBuffer), NeedMore>> {
    loop {
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
            return Ok(Left((
                DataFrameBuffer::empty_with_schema(&left_params.ir_schema),
                right_split,
            )));
        } else if right_done && right.is_empty() {
            let left_split = left.split_at(get_ideal_morsel_size());
            return Ok(Left((
                left_split,
                DataFrameBuffer::empty_with_schema(&right_params.ir_schema),
            )));
        } else if left.is_empty() && !left_done {
            return Ok(Right(NeedMore::Left));
        } else if right.is_empty() && !right_done {
            return Ok(Right(NeedMore::Right));
        }

        let left_first = left.get_bypass_validity(&left_params.key_col, 0, params);
        let right_first = right.get_bypass_validity(&right_params.key_col, 0, params);

        // First return chunks of nulls if there are any
        if !params.args.nulls_equal && !params.key_nulls_last && left_first == AnyValue::Null {
            let left_first_nonnull_idx =
                binary_search_upper(&left, &AnyValue::Null, params, left_params)?;
            let left_split = left.split_at(left_first_nonnull_idx);
            return Ok(Left((
                left_split,
                DataFrameBuffer::empty_with_schema(&right_params.ir_schema),
            )));
        }
        if !params.args.nulls_equal && !params.key_nulls_last && right_first == AnyValue::Null {
            let right_first_nonnull_idx =
                binary_search_upper(&right, &AnyValue::Null, params, right_params)?;
            let right_split = right.split_at(right_first_nonnull_idx);
            return Ok(Left((
                DataFrameBuffer::empty_with_schema(&left_params.ir_schema),
                right_split,
            )));
        }

        let left_last_idx = usize::min(left.len(), limit);
        let left_last = left.get_bypass_validity(&left_params.key_col, left_last_idx - 1, params);
        let right_last_idx = usize::min(right.len(), limit);
        let right_last =
            right.get_bypass_validity(&right_params.key_col, right_last_idx - 1, params);

        let left_first_incomplete = match left_done {
            false => binary_search_lower(left, &left_last, params, &left_params)?,
            true => left.len(),
        };

        let right_first_incomplete = match right_done {
            false => binary_search_lower(right, &right_last, params, &right_params)?,
            true => right.len(),
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

        let left_last_completed_val =
            left.get_bypass_validity(&left_params.key_col, left_first_incomplete - 1, params);
        let right_last_completed_val =
            right.get_bypass_validity(&right_params.key_col, right_first_incomplete - 1, params);

        let left_mergable_until; // bound is *exclusive*
        let right_mergable_until;
        if keys_eq(&left_last_completed_val, &right_last_completed_val, params) {
            left_mergable_until = left_first_incomplete;
            right_mergable_until = right_first_incomplete;
        } else if keys_lt(&left_last_completed_val, &right_last_completed_val, params) {
            left_mergable_until = left_first_incomplete;
            right_mergable_until = binary_search_upper(
                right,
                &left.get_bypass_validity(&left_params.key_col, left_mergable_until - 1, params),
                params,
                right_params,
            )?;
        } else if keys_gt(&left_last_completed_val, &right_last_completed_val, params) {
            right_mergable_until = right_first_incomplete;
            left_mergable_until = binary_search_upper(
                left,
                &right.get_bypass_validity(&right_params.key_col, right_mergable_until - 1, params),
                params,
                left_params,
            )?;
        } else {
            unreachable!();
        }

        if left_mergable_until == 0 && right_mergable_until == 0 {
            return Ok(Right(NeedMore::Both));
        }

        let left_split = left.split_at(left_mergable_until);
        let right_split = right.split_at(right_mergable_until);
        return Ok(Left((left_split, right_split)));
    }
}

fn series_get_bypass_validity<'a>(
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
    op: fn(&AnyValue, &AnyValue, &MergeJoinParams) -> bool,
    params: &MergeJoinParams,
    sp: &SideParams,
) -> PolarsResult<usize> {
    let mut lower = 0;
    let mut upper = vec.len();
    while lower < upper {
        let mid = (lower + upper) / 2;
        let mid_val = vec.get_bypass_validity(&sp.key_col, mid, params);
        if op(search_value, &mid_val, params) {
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
    binary_search(vec, sv, keys_le, params, sp)
}

fn binary_search_upper(
    vec: &DataFrameBuffer,
    sv: &AnyValue,
    params: &MergeJoinParams,
    sp: &SideParams,
) -> PolarsResult<usize> {
    binary_search(vec, sv, keys_lt, params, sp)
}

#[derive(Default)]
struct Arenas {
    gather_left: Vec<IdxSize>,
    gather_right: Vec<IdxSize>,
    matched_probeside: MutableBitmap,
    df_builders: Option<(DataFrameBuilder, DataFrameBuilder)>,
}

async fn compute_join(
    left: DataFrameBuffer,
    right: DataFrameBuffer,
    seq: MorselSeq,
    source_token: SourceToken,
    params: &MergeJoinParams,
    arenas: &mut Arenas,
    matched_send: &mut PortSender,
    unmatched_send: tokio::sync::mpsc::Sender<Morsel>,
) -> PolarsResult<()> {
    // TODO [amber] LEFT HERE
    // The main gap in the profile is due to the state transition from Running
    // to FlushInputBuffers.  I don't know if we can do anything about that.
    // I think the next step is to remove the non-payload colunms before gathering.
    //
    // Good luck! <3

    let mut left_sp = &params.left;
    let mut right_sp = &params.right;
    let mut left = left.into_df();
    let mut right = right.into_df();

    left.rechunk_mut();
    right.rechunk_mut();

    let left_key = left
        .column(&params.left.key_col)
        .unwrap()
        .as_materialized_series();
    let right_key = right
        .column(&params.right.key_col)
        .unwrap()
        .as_materialized_series();

    let right_build_maintain_order = matches!(
        params.args.maintain_order,
        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft,
    );
    let right_build_optimization = params.args.maintain_order == MaintainOrderJoin::None
        && right_key.null_count() > left_key.null_count();
    let right_is_build = right_build_maintain_order || right_build_optimization;

    arenas.matched_probeside.clear();
    if right_is_build {
        arenas.matched_probeside.resize(left_key.len(), false);
    } else {
        arenas.matched_probeside.resize(right_key.len(), false);
    }

    let mut current_offset = 0;
    let mut done = false;
    while !done {
        arenas.gather_left.clear();
        arenas.gather_right.clear();
        if right_is_build {
            done = compute_join_dispatch(
                right_key,
                left_key,
                &mut arenas.gather_right,
                &mut arenas.gather_left,
                &mut arenas.matched_probeside,
                right_sp,
                left_sp,
                params,
                &mut current_offset,
            );
        } else {
            done = compute_join_dispatch(
                left_key,
                right_key,
                &mut arenas.gather_left,
                &mut arenas.gather_right,
                &mut arenas.matched_probeside,
                left_sp,
                right_sp,
                params,
                &mut current_offset,
            );
        }

        let mut df = Default::default();
        gather_and_postprocess(
            &mut df,
            &left,
            &right,
            &mut arenas.gather_left,
            &mut arenas.gather_right,
            &mut arenas.df_builders,
            params,
        )?;

        if !df.is_empty() {
            let morsel = Morsel::new(df, seq, source_token.clone());
            if matched_send.send(morsel).await.is_err() {
                panic!("broken pipe");
            };
        }
    }

    arenas.gather_left.clear();
    arenas.gather_right.clear();
    if right_is_build {
        swap(&mut arenas.gather_left, &mut arenas.gather_right);
        swap(&mut left_sp, &mut right_sp);
    }
    if right_sp.emit_unmatched {
        for (idx, _) in arenas
            .matched_probeside
            .iter()
            .enumerate_idx()
            .filter(|(_, m)| !m)
        {
            arenas.gather_left.push(IdxSize::MAX);
            arenas.gather_right.push(idx);
        }
    }
    if right_is_build {
        swap(&mut arenas.gather_left, &mut arenas.gather_right);
        swap(&mut left_sp, &mut right_sp);
    }

    let mut df_unmatched = Default::default();
    gather_and_postprocess(
        &mut df_unmatched,
        &left,
        &right,
        &mut arenas.gather_left,
        &mut arenas.gather_right,
        &mut arenas.df_builders,
        params,
    )?;
    if !df_unmatched.is_empty() {
        let morsel = Morsel::new(df_unmatched, seq, source_token.clone());
        if unmatched_send.send(morsel).await.is_err() {
            panic!("broken pipe");
        }
    }

    Ok(())
}

fn gather_and_postprocess(
    df: &mut DataFrame,
    left: &DataFrame,
    right: &DataFrame,
    left_gather: &mut Vec<IdxSize>,
    right_gather: &mut Vec<IdxSize>,
    df_builders: &mut Option<(DataFrameBuilder, DataFrameBuilder)>,
    params: &MergeJoinParams,
) -> PolarsResult<()> {
    // Remove the added row-encoded key columns
    let mut left = left.clone();
    let mut right = right.clone();
    if params.use_row_encoding {
        left = left.drop(&params.left.key_col).unwrap();
        right = right.drop(&params.right.key_col).unwrap();
    }

    if df_builders.is_none() {
        *df_builders = Some((
            DataFrameBuilder::new(left.schema().clone()),
            DataFrameBuilder::new(right.schema().clone()),
        ));
    }

    let (left_build, right_build) = df_builders.as_mut().unwrap();
    left_build.opt_gather_extend(&left, &left_gather, ShareStrategy::Never);
    right_build.opt_gather_extend(&right, &right_gather, ShareStrategy::Never);

    let mut left = left_build.freeze_reset();
    let mut right = right_build.freeze_reset();

    // Coalsesce the key columns
    if params.args.how == JoinType::Left && params.args.should_coalesce() {
        for c in &params.left.on {
            right.drop_in_place(c.as_str())?;
        }
    } else if params.args.how == JoinType::Right && params.args.should_coalesce() {
        for c in &params.right.on {
            left.drop_in_place(c.as_str())?;
        }
    }

    // Rename any right columns to "{}_right"
    rename_right_columns(&left, &mut right, params)?;

    left.hstack_mut(&right.get_columns())?;
    if params.args.how == JoinType::Full && params.args.should_coalesce() {
        for (left_keycol, right_keycol) in
            Iterator::zip(params.left.on.iter(), params.right.on.iter())
        {
            let right_keycol = format_pl_smallstr!("{}{}", right_keycol, params.args.suffix());
            let left_col = left.column(&left_keycol).unwrap();
            let right_col = left.column(&right_keycol).unwrap();
            let coalesced = coalesce_columns(&[left_col.clone(), right_col.clone()]).unwrap();
            left.replace(&left_keycol, coalesced.take_materialized_series())
                .unwrap()
                .drop_in_place(&right_keycol)
                .unwrap();
        }
    }

    *df = drop_non_output_columns(&left, params)?;
    Ok(())
}

fn compute_join_dispatch(
    lk: &Series,
    rk: &Series,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    left_sp: &SideParams,
    right_sp: &SideParams,
    params: &MergeJoinParams,
    current_offset: &mut usize,
) -> bool {
    macro_rules! dispatch {
        ($left_key:expr, $right_key:expr) => {
            compute_join_kernel(
                $left_key,
                $right_key,
                gather_left,
                gather_right,
                matched_right,
                left_sp,
                right_sp,
                params,
                current_offset,
            )
        };
    }

    debug_assert_eq!(lk.dtype(), rk.dtype());
    match lk.dtype() {
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => dispatch!(lk.i8().unwrap(), rk.i8().unwrap()),
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => dispatch!(lk.i16().unwrap(), rk.i16().unwrap()),
        DataType::Int32 => dispatch!(lk.i32().unwrap(), rk.i32().unwrap()),
        DataType::Int64 => dispatch!(lk.i64().unwrap(), rk.i64().unwrap()),
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => dispatch!(lk.i128().unwrap(), rk.i128().unwrap()),
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => dispatch!(lk.u8().unwrap(), rk.u8().unwrap()),
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => dispatch!(lk.u16().unwrap(), rk.u16().unwrap()),
        DataType::UInt32 => dispatch!(lk.u32().unwrap(), rk.u32().unwrap()),
        DataType::UInt64 => dispatch!(lk.u64().unwrap(), rk.u64().unwrap()),
        #[cfg(feature = "dtype-u128")]
        DataType::UInt128 => dispatch!(lk.u128().unwrap(), rk.u128().unwrap()),
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => dispatch!(lk.f16().unwrap(), rk.f16().unwrap()),
        DataType::Float32 => dispatch!(lk.f32().unwrap(), rk.f32().unwrap()),
        DataType::Float64 => dispatch!(lk.f64().unwrap(), rk.f64().unwrap()),
        #[cfg(feature = "dtype-date")]
        DataType::Date => dispatch!(lk.date().unwrap().physical(), rk.date().unwrap().physical()),
        #[cfg(feature = "dtype-time")]
        DataType::Time => dispatch!(lk.time().unwrap().physical(), rk.time().unwrap().physical()),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, _) => dispatch!(
            lk.datetime().unwrap().physical(),
            rk.datetime().unwrap().physical()
        ),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => dispatch!(
            lk.duration().unwrap().physical(),
            rk.duration().unwrap().physical()
        ),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => dispatch!(
            lk.decimal().unwrap().physical(),
            rk.decimal().unwrap().physical()
        ),
        #[cfg(feature = "dtype-categorical")]
        dt @ (DataType::Enum(_, _) | DataType::Categorical(_, _)) => {
            match dt.cat_physical().unwrap() {
                CategoricalPhysical::U8 => {
                    dispatch!(lk.cat8().unwrap().physical(), rk.cat8().unwrap().physical())
                },
                CategoricalPhysical::U16 => dispatch!(
                    lk.cat16().unwrap().physical(),
                    rk.cat16().unwrap().physical()
                ),
                CategoricalPhysical::U32 => dispatch!(
                    lk.cat32().unwrap().physical(),
                    rk.cat32().unwrap().physical()
                ),
            }
        },
        DataType::BinaryOffset => {
            dispatch!(lk.binary_offset().unwrap(), rk.binary_offset().unwrap())
        },
        dt => unimplemented!("merge join kernel not implemented for {:?}", dt),
    }
}

fn compute_join_kernel<'a, T: PolarsDataType>(
    left_key: &'a ChunkedArray<T>,
    right_key: &'a ChunkedArray<T>,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    left_sp: &SideParams,
    right_sp: &SideParams,
    params: &MergeJoinParams,
    current_offset: &mut usize,
) -> bool
where
    T::Physical<'a>: TotalOrd,
{
    debug_assert!(gather_left.is_empty());
    debug_assert!(gather_right.is_empty());
    if right_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_key.len());
    }

    let descending = params.key_descending;
    let left_key = left_key.downcast_as_array();
    let right_key = right_key.downcast_as_array();

    let mut iterator = left_key.iter().skip(*current_offset).peekable();
    if iterator.peek().is_none() {
        return true;
    }
    let mut skip_ahead_right = 0;
    for (idxl, left_keyval) in iterator.enumerate() {
        if gather_left.len() >= get_ideal_morsel_size() {
            return false;
        }
        let left_keyval = left_keyval.as_ref();
        let mut matched = false;
        if params.args.nulls_equal || left_keyval.is_some() {
            for idxr in skip_ahead_right..right_key.len() {
                let right_keyval = right_key.get(idxr);
                let right_keyval = right_keyval.as_ref();
                let mut ord = match (&left_keyval, &right_keyval) {
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
        if (left_sp.emit_unmatched) && !(matched) {
            gather_left.push(idxl as IdxSize);
            gather_right.push(IdxSize::MAX);
        }
        *current_offset += 1;
    }
    true
}

fn rename_right_columns(
    left: &DataFrame,
    right: &mut DataFrame,
    params: &MergeJoinParams,
) -> PolarsResult<()> {
    let left_cols: PlHashSet<PlSmallStr> = left
        .get_column_names()
        .into_iter()
        .cloned()
        .collect::<PlHashSet<_>>();
    for col in right.get_column_names_owned() {
        if left_cols.contains(&col) {
            let new_name = format_pl_smallstr!("{}{}", col, params.args.suffix());
            right.rename(&col, new_name).unwrap(); // FIXME: [amber] Potential quadratic behavior
        }
    }
    Ok(())
}

fn drop_non_output_columns(df: &DataFrame, params: &MergeJoinParams) -> PolarsResult<DataFrame> {
    let mut drop_cols = PlHashSet::with_capacity(df.width());
    for col in df.get_column_names() {
        if !key_is_in_output(col, params) {
            drop_cols.insert(col.clone());
        }
    }
    Ok(df.drop_many_amortized(&drop_cols))
}

fn key_is_in_output(col_name: &PlSmallStr, params: &MergeJoinParams) -> bool {
    params.output_schema.contains(col_name)
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

fn keys_eq(left: &AnyValue, right: &AnyValue, _params: &MergeJoinParams) -> bool {
    left == right
}

fn keys_lt(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    if keys_eq(left, right, params) {
        false
    } else if params.key_nulls_last {
        if left.is_null() {
            false
        } else if right.is_null() {
            true
        } else if params.key_descending {
            left > right
        } else {
            left < right
        }
    } else {
        if left.is_null() {
            true
        } else if right.is_null() {
            false
        } else if params.key_descending {
            left > right
        } else {
            left < right
        }
    }
}

fn keys_le(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    keys_lt(left, right, params) || keys_eq(left, right, params)
}

fn keys_gt(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    !keys_le(left, right, params)
}

#[derive(Clone)]
struct DataFrameBuffer {
    schema: Schema,
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
    fn empty_with_schema(schema: &Schema) -> Self {
        DataFrameBuffer {
            schema: schema.clone(),
            buf: BTreeMap::new(),
            total_rows: 0,
            skip_rows: 0,
            frozen: false,
        }
    }

    fn len(&self) -> usize {
        self.total_rows
    }

    fn get_bypass_validity(
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
        series_get_bypass_validity(series, series_index, params)
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
        self.debug_check_stats_correct();
    }

    fn split_at(&mut self, mut at: usize) -> Self {
        at = at.clamp(0, self.total_rows);
        let mut left = self.clone();
        left.total_rows = at;
        left.frozen = true;
        left.debug_check_stats_correct();
        self.skip_rows += at;
        self.total_rows -= at;
        self.gc();
        self.debug_check_stats_correct();
        left
    }

    fn slice(mut self, offset: usize, len: usize) -> Self {
        self.skip_rows += offset;
        self.total_rows -= offset;
        self.total_rows = usize::min(self.total_rows, len);
        self.frozen = true;
        self.debug_check_stats_correct();
        self
    }

    fn into_df(self) -> DataFrame {
        self.debug_check_stats_correct();
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
        self.debug_check_stats_correct();
    }

    #[cfg(debug_assertions)]
    fn debug_check_stats_correct(&self) {
        debug_assert!(
            self.frozen
                || self.buf.values().map(|df| df.height()).sum::<usize>() - self.skip_rows
                    == self.total_rows
        );
        debug_assert!(self.buf.values().all(|df| &**df.schema() == &self.schema));
        let mut last: Option<(usize, DataFrame)> = None;
        for (key, df) in self.buf.iter() {
            if let Some((last_key, last_df)) = last {
                debug_assert!(last_key + last_df.height() == *key);
            }
            last = Some((*key, df.clone()));
        }
    }
}
