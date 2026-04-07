use std::collections::VecDeque;

use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_ops::frame::{AsOfOptions, AsofJoin, AsofStrategy, JoinArgs, JoinType};
use polars_ops::series::{rle_lengths, rle_lengths_helper_ca};
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;
use polars_utils::scratch_vec::ScratchVec;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel as dc;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::{DataFrameSearchBuffer, stop_and_buffer_pipe_contents};
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Debug)]
pub struct AsOfJoinSideParams {
    pub on: PlSmallStr,
    pub tmp_key_col: Option<PlSmallStr>,
    pub by: Vec<PlSmallStr>,
}

impl AsOfJoinSideParams {
    fn key_col(&self) -> &PlSmallStr {
        self.tmp_key_col.as_ref().unwrap_or(&self.on)
    }
}

#[derive(Debug)]
struct AsOfJoinParams {
    output_schema: SchemaRef,
    left: AsOfJoinSideParams,
    right: AsOfJoinSideParams,
    args: JoinArgs,
}

impl AsOfJoinParams {
    fn as_of_options(&self) -> &AsOfOptions {
        let JoinType::AsOf(ref options) = self.args.how else {
            unreachable!("incorrect join type");
        };
        options
    }
}

#[derive(Debug, Default, PartialEq)]
enum AsOfJoinState {
    #[default]
    Running,
    FlushInputBuffer,
    Done,
}

#[derive(Debug)]
pub struct AsOfJoinNode {
    params: AsOfJoinParams,
    state: AsOfJoinState,
    /// We may need to stash a morsel on the left side whenever we do not
    /// have enough data on the right side, but the right side is empty.
    /// In these cases, we stash that morsel here.
    left_buffer: VecDeque<(DataFrame, MorselSeq)>,
    /// Buffer of the live range of right AsOf join rows.
    right_buffer: DataFrameSearchBuffer,
}

impl AsOfJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        output_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        left_by: Option<Vec<PlSmallStr>>,
        right_by: Option<Vec<PlSmallStr>>,
        args: JoinArgs,
    ) -> Self {
        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert_eq!(left_key_dtype, right_key_dtype);
        let left = AsOfJoinSideParams {
            on: left_on,
            tmp_key_col: tmp_left_key_col,
            by: left_by.unwrap_or_default(),
        };
        let right = AsOfJoinSideParams {
            on: right_on,
            tmp_key_col: tmp_right_key_col,
            by: right_by.unwrap_or_default(),
        };

        let params = AsOfJoinParams {
            output_schema,
            left,
            right,
            args,
        };
        AsOfJoinNode {
            params,
            state: AsOfJoinState::default(),
            left_buffer: Default::default(),
            right_buffer: DataFrameSearchBuffer::empty_with_schema(right_input_schema),
        }
    }
}

impl ComputeNode for AsOfJoinNode {
    fn name(&self) -> &str {
        "asof-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = AsOfJoinState::Done;
        }

        if self.state == AsOfJoinState::Running && recv[0] == PortState::Done {
            self.state = AsOfJoinState::FlushInputBuffer;
        }

        if self.state == AsOfJoinState::FlushInputBuffer && self.left_buffer.is_empty() {
            self.state = AsOfJoinState::Done;
        }

        let recv0_blocked = recv[0] == PortState::Blocked;
        let recv1_blocked = recv[1] == PortState::Blocked;
        let send_blocked = send[0] == PortState::Blocked;
        match self.state {
            AsOfJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv0_blocked {
                    recv[1] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked {
                    recv[0] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[0] = PortState::Blocked;
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::FlushInputBuffer => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv1_blocked {
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::Done => {
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        match &self.state {
            AsOfJoinState::Running | AsOfJoinState::FlushInputBuffer => {
                let params = &self.params;
                let recv_left = match self.state {
                    AsOfJoinState::Running => Some(recv_ports[0].take().unwrap().serial()),
                    _ => None,
                };
                let recv_right = recv_ports[1].take().map(RecvPort::serial);
                let send = send_ports[0].take().unwrap().parallel();
                let (distributor, dist_recv) =
                    dc::distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
                let left_buffer = &mut self.left_buffer;
                let right_buffer = &mut self.right_buffer;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    distribute_work_task(
                        recv_left,
                        recv_right,
                        distributor,
                        left_buffer,
                        right_buffer,
                        params,
                    )
                    .await
                }));

                join_handles.extend(dist_recv.into_iter().zip(send).map(|(recv, send)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        compute_and_emit_task(recv, send, params).await
                    })
                }));
            },
            AsOfJoinState::Done => {
                unreachable!();
            },
        }
    }
}

async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    left_buffer: &mut VecDeque<(DataFrame, MorselSeq)>,
    right_buffer: &mut DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();
    let right_done = recv_right.is_none();

    loop {
        if source_token.stop_requested() {
            stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df, seq| {
                left_buffer.push_back((df, seq))
            })
            .await;
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df, _| {
                right_buffer.push_df(df)
            })
            .await;
            return Ok(());
        }

        let (left_df, seq, st) = if let Some((df, seq)) = left_buffer.pop_front() {
            (df, seq, source_token.clone())
        } else if let Some(ref mut recv) = recv_left
            && let Ok(m) = recv.recv().await
        {
            let (df, seq, st, _) = m.into_inner();
            (df, seq, st)
        } else {
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df, _| {
                right_buffer.push_df(df)
            })
            .await;
            return Ok(());
        };

        while need_more_right_side(&left_df, right_buffer, params)? && !right_done {
            if let Some(ref mut recv) = recv_right
                && let Ok(morsel_right) = recv.recv().await
            {
                right_buffer.push_df(morsel_right.into_df());
            } else {
                // The right pipe is empty at this stage, we will need to wait for
                // a new stage and try again.
                left_buffer.push_back((left_df, seq));
                stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df, seq| {
                    left_buffer.push_back((df, seq))
                })
                .await;
                return Ok(());
            }
        }

        distributor
            .send((left_df.clone(), right_buffer.clone(), seq, st))
            .await
            .unwrap();
        prune_right_side(&left_df, right_buffer, params)?;
    }
}

/// Do we need more values on the right side before we can compute the AsOf join
/// between the right side and the complete left side?
fn need_more_right_side(
    left: &DataFrame,
    right: &DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<bool> {
    if left.height() == 0 {
        return Ok(false);
    } else if right.height() == 0 {
        return Ok(true);
    }

    let options = params.as_of_options();

    let mut start = 0;
    let mut end = right.height();
    for (left_by, right_by) in Iterator::zip(params.left.by.iter(), params.right.by.iter()) {
        let left_by_col = left.column(left_by)?.as_materialized_series();
        // SAFETY: We checked earlier that the dataframes are not empty
        let left_last_group = unsafe { left_by_col.get_unchecked(left.height() - 1) };
        start = right.binary_search(|x| x >= &left_last_group, right_by, start..end, false);
        end = right.binary_search(|x| x > &left_last_group, right_by, start..end, false)
    }

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: We checked earlier that the dataframes are not empty
    let left_last_val = unsafe { left_key.get_unchecked(left_key.len() - 1) };
    let right_range_end = match (options.strategy, options.allow_eq) {
        (AsofStrategy::Forward, true) => right.binary_search(
            |x| *x >= left_last_val,
            params.right.key_col(),
            start..end,
            false,
        ),
        (AsofStrategy::Forward, false) | (AsofStrategy::Backward, true) => right.binary_search(
            |x| *x > left_last_val,
            params.right.key_col(),
            start..end,
            false,
        ),
        (AsofStrategy::Backward, false) | (AsofStrategy::Nearest, _) => {
            let first_greater = right.binary_search(
                |x| *x > left_last_val,
                params.right.key_col(),
                start..end,
                false,
            );
            if first_greater >= right.height() {
                return Ok(true);
            }
            // In the Backward/Nearest cases, there may be a chunk of consecutive equal
            // values following the match value on the left side.  In this case, the AsOf
            // join is greedy and should until the *end* of that chunk.

            // SAFETY: We just checked that right_range_end is in bounds
            let fst_greater_val =
                unsafe { right.get_unchecked(params.right.key_col(), first_greater) };
            right.binary_search(
                |x| *x > fst_greater_val,
                params.right.key_col(),
                first_greater..end,
                false,
            )
        },
    };
    Ok(right_range_end >= right.height())
}

fn prune_right_side(
    left: &DataFrame,
    right: &mut DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    if left.height() == 0 || right.height() == 0 {
        return Ok(());
    }

    let mut start = 0;
    let mut end = right.height();
    for (left_by, right_by) in Iterator::zip(params.left.by.iter(), params.right.by.iter()) {
        let left_by_col = left.column(left_by)?.as_materialized_series();
        // SAFETY: We checked earlier that the dataframes are not empty
        let left_last_group = unsafe { left_by_col.get_unchecked(left.height() - 1) };
        start = right.binary_search(|x| x >= &left_last_group, right_by, start..end, false);
        end = right.binary_search(|x| x > &left_last_group, right_by, start..end, false)
    }

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: We checked earlier that the dataframes are not empty
    let left_first_val = unsafe { left_key.get_unchecked(0) };
    let right_range_start = right
        .binary_search(
            |x| *x >= left_first_val,
            params.right.key_col(),
            start..end,
            false,
        )
        .saturating_sub(1);
    right.split_at(right_range_start);
    Ok(())
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    mut send: PortSender,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let mut left_lengths_scratch = ScratchVec::default();
    let mut right_lengths_scratch = ScratchVec::default();
    while let Ok((left_df, right_dfsb, seq, st)) = dist_recv.recv().await {
        let out = compute_asof_join(
            left_df,
            right_dfsb,
            &mut left_lengths_scratch,
            &mut right_lengths_scratch,
            params,
        )?;
        let morsel = Morsel::new(out, seq, st);
        if send.send(morsel).await.is_err() {
            return Ok(());
        }
    }
    Ok(())
}

fn compute_asof_join(
    left_df: DataFrame,
    right_dfsb: DataFrameSearchBuffer,
    left_lengths_scratch: &mut ScratchVec<IdxSize>,
    right_lengths_scratch: &mut ScratchVec<IdxSize>,
    params: &AsOfJoinParams,
) -> PolarsResult<DataFrame> {
    const ROW_ENCODED_COL_NAME: PlSmallStr = PlSmallStr::from_static("__PL_ASOF_JOIN_BY");

    let options = params.as_of_options();
    let right_df = right_dfsb.into_df();
    let any_key_is_temporary_col =
        params.left.tmp_key_col.is_some() || params.right.tmp_key_col.is_some();
    let coalesce = any_key_is_temporary_col || params.args.should_coalesce();

    let left_group = if params.left.by.len() == 1 {
        Some(left_df.column(&params.left.by[0])?)
    } else {
        None
    };
    let right_group = if params.right.by.len() == 1 {
        Some(right_df.column(&params.right.by[0])?)
    } else {
        None
    };

    let left_key = left_df.column(params.left.key_col())?;
    let right_key = right_df.column(params.right.key_col())?;

    if params.left.by.is_empty() || params.right.by.is_empty() {
        let mut out = AsofJoin::_join_asof(
            &left_df,
            &right_df,
            left_key.as_materialized_series(),
            right_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            params.args.suffix.clone(),
            None,
            coalesce,
            options.allow_eq,
            options.check_sortedness,
        )?;

        // Drop any temporary key columns that were added
        for tmp_key_col in [&params.left.tmp_key_col, &params.right.tmp_key_col] {
            if let Some(tmp_col) = tmp_key_col
                && out.schema().contains(tmp_col)
            {
                out.drop_in_place(tmp_col)?;
            }
        }

        // If the join key passed to _join_asof() was a temporary key column,
        // we still need to coalesce the real 'on' columns ourselves.
        if any_key_is_temporary_col
            && params.args.should_coalesce()
            && params.left.on == params.right.on
        {
            let right_on_name = format_pl_smallstr!("{}{}", params.right.on, params.args.suffix());
            out.drop_in_place(&right_on_name)?;
        }

        return Ok(out);
    }

    let mut left_lengths = left_lengths_scratch.get();
    let mut right_lengths = right_lengths_scratch.get();

    // Computing of left lengths
    let mut row_encoded_left = None;
    if params.left.by.len() > 1 {
        let by = params
            .left
            .by
            .iter()
            .map(|name| left_df.column(name).unwrap().clone())
            .collect_vec();
        let descending = vec![false; params.left.by.len()];
        let nulls_last = vec![false; params.left.by.len()];
        let row_encoded =
            _get_rows_encoded_ca(ROW_ENCODED_COL_NAME, &by, &descending, &nulls_last, true)?;
        rle_lengths_helper_ca(&row_encoded, &mut left_lengths);
        row_encoded_left = Some(row_encoded);
    } else if params.left.by.len() == 1 {
        rle_lengths(left_df.column(&params.left.by[0])?, &mut left_lengths)?;
    }

    // Computing of right lengths
    let mut row_encoded_right = None;
    if params.right.by.len() > 1 {
        let by = params
            .right
            .by
            .iter()
            .map(|name| right_df.column(name).unwrap().clone())
            .collect_vec();
        let descending = vec![false; params.right.by.len()];
        let nulls_last = vec![false; params.right.by.len()];
        let row_encoded = _get_rows_encoded_ca(
            "__PL_ASOF_JOIN_BY".into(),
            &by,
            &descending,
            &nulls_last,
            true,
        )?;
        rle_lengths_helper_ca(&row_encoded, &mut right_lengths);
        row_encoded_right = Some(row_encoded);
    } else if params.right.by.len() == 1 {
        rle_lengths(right_df.column(&params.right.by[0])?, &mut right_lengths)?;
    }

    let left_group_idxs = left_lengths.iter().scan(0, |state, len| {
        let start = *state;
        *state += *len as usize;
        Some((start, *len as usize))
    });
    let mut right_group_idxs = right_lengths.iter().scan(0, |state, len| {
        let start = *state;
        *state += *len as usize;
        Some((start, *len as usize))
    });

    let mut acc = DataFrame::empty_with_arc_schema(params.output_schema.clone());
    // When right is empty there are no groups to pull from, but we still need to
    // emit all left rows with null right columns.  Use sentinel values; the while
    // loop below is guarded against accessing an empty right_df.
    let (mut right_start, mut right_len) = right_group_idxs.next().unwrap_or((0, 0));

    let left_group_is_none =
        |idx, row_encoded: Option<&ChunkedArray<BinaryOffsetType>>| match params.left.by.len() {
            0 => false,
            1 => unsafe { left_group.unwrap_unchecked().get_unchecked(idx) }.is_null(),
            _ => unsafe { row_encoded.unwrap_unchecked().get_unchecked(idx).is_none() },
        };
    let right_group_is_none =
        |idx, row_encoded: Option<&ChunkedArray<BinaryOffsetType>>| match params.right.by.len() {
            0 => false,
            1 => unsafe { right_group.unwrap_unchecked().get_unchecked(idx) }.is_null(),
            _ => unsafe { row_encoded.unwrap_unchecked().get_unchecked(idx).is_none() },
        };
    let right_is_lt_left =
        |left_idx,
         right_idx,
         row_encoded_left: Option<&ChunkedArray<BinaryOffsetType>>,
         row_encoded_right: Option<&ChunkedArray<BinaryOffsetType>>| match params
            .right
            .by
            .len()
        {
            0 => false,
            1 => unsafe {
                right_group.unwrap_unchecked().get_unchecked(right_idx)
                    < left_group.unwrap_unchecked().get_unchecked(left_idx)
            },
            _ => unsafe {
                row_encoded_right
                    .unwrap_unchecked()
                    .get_unchecked(right_idx)
                    < row_encoded_left.unwrap_unchecked().get_unchecked(left_idx)
            },
        };

    for (left_start, left_len) in left_group_idxs {
        if left_group_is_none(left_start, row_encoded_left.as_ref()) {
            // None group can never match on the right side
            right_len = 0;
        } else if right_df.height() > 0 {
            // Only advance the right-group cursor when right is non-empty;
            // accessing right_df when empty would be out of bounds.
            while right_group_is_none(right_start, row_encoded_right.as_ref())
                || right_is_lt_left(
                    left_start,
                    right_start,
                    row_encoded_left.as_ref(),
                    row_encoded_right.as_ref(),
                )
            {
                if let Some((rs, rl)) = right_group_idxs.next() {
                    right_start = rs;
                    right_len = rl;
                } else {
                    // Right group does not exist. We still want to emit nulls, so
                    // we just use an empty dataframe as the right side.
                    right_len = 0;
                    break;
                };
            }
        }
        let left_chunk = left_df.slice(left_start as i64, left_len);
        let right_chunk = right_df.slice(right_start as i64, right_len);
        let left_chunk_key = left_key.slice(left_start as i64, left_len);
        let right_chunk_key = right_key.slice(right_start as i64, right_len);

        let mut out = AsofJoin::_join_asof(
            &left_chunk,
            &right_chunk,
            left_chunk_key.as_materialized_series(),
            right_chunk_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            params.args.suffix.clone(),
            None,
            coalesce,
            options.allow_eq,
            options.check_sortedness,
        )?;

        // Drop any temporary key columns that were added
        for tmp_key_col in [&params.left.tmp_key_col, &params.right.tmp_key_col] {
            if let Some(tmp_col) = tmp_key_col
                && out.schema().contains(tmp_col)
            {
                out.drop_in_place(tmp_col)?;
            }
        }

        // If the join key passed to _join_asof() was a temporary key column,
        // we still need to coalesce the real 'on' columns ourselves.
        if any_key_is_temporary_col
            && params.args.should_coalesce()
            && params.left.on == params.right.on
        {
            let right_on_name = format_pl_smallstr!("{}{}", params.right.on, params.args.suffix());
            out.drop_in_place(&right_on_name)?;
        }

        // When we have joined on groups, we still need to remove the group columns
        // from the right input dataframe.
        for col in &params.right.by {
            let with_suffix = format_pl_smallstr!("{}{}", col, params.args.suffix());
            if out.get_column_index(&with_suffix).is_some() {
                out.drop_in_place(&with_suffix)?;
            } else {
                out.drop_in_place(&col)?;
            }
        }

        acc.vstack_mut_owned_unchecked(out);
    }
    Ok(acc)
}
