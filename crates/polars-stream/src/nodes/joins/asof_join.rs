use std::cmp::Ordering;
use std::collections::VecDeque;

use AsofStrategy::*;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::distributor_channel as dc;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::utils::{Container, accumulate_dataframes_vertical_unchecked};
use polars_ops::frame::is_sorted::DataFrameIsSorted;
use polars_ops::frame::{
    _check_asof_columns, _finish_join, _join_asof_dispatch, AsOfOptions, AsofStrategy, JoinArgs,
    JoinType,
};
use polars_ops::series::{rle_lengths, rle_lengths_helper_ca};
use polars_utils::itertools::Itertools;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::sort::reorder_cmp;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::{DataFrameSearchBuffer, stop_and_buffer_pipe_contents};
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

const ROW_ENCODED_COL_NAME: PlSmallStr = PlSmallStr::from_static("__PL_ASOF_JOIN_BY");

#[derive(Debug)]
pub struct AsOfJoinSideParams {
    pub on: PlSmallStr,
    pub tmp_key_col: Option<PlSmallStr>,
}

impl AsOfJoinSideParams {
    fn key_col(&self) -> &PlSmallStr {
        self.tmp_key_col.as_ref().unwrap_or(&self.on)
    }
}

#[derive(Debug)]
struct AsOfJoinParams {
    left: AsOfJoinSideParams,
    right: AsOfJoinSideParams,
    by_descending: Vec<bool>,
    by_nulls_last: Vec<bool>,
    args: JoinArgs,
}

impl AsOfJoinParams {
    fn as_of_options(&self) -> &AsOfOptions {
        let JoinType::AsOf(ref options) = self.args.how else {
            unreachable!();
        };
        options
    }

    fn left_by(&self) -> &[PlSmallStr] {
        self.as_of_options()
            .left_by
            .as_ref()
            .map_or(&[], |x| &x[..])
    }

    fn right_by(&self) -> &[PlSmallStr] {
        self.as_of_options()
            .right_by
            .as_ref()
            .map_or(&[], |x| &x[..])
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
    left_buffer: VecDeque<DataFrame>,
    /// Buffer of the live range of right AsOf join rows.
    right_buffer: DataFrameSearchBuffer,
    output_seq: MorselSeq,
    // Slot to store the last row of the previous left morsel. Used to check
    // that the left side is sorted across morsel boundaries.
    last_non_null_row_left: Option<DataFrame>,
    last_non_null_row_right: Option<DataFrame>,
}

impl AsOfJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        by_descending: Option<Vec<bool>>,
        by_nulls_last: Option<Vec<bool>>,
        args: JoinArgs,
    ) -> Self {
        let JoinType::AsOf(ref options) = args.how else {
            unreachable!();
        };
        assert!({
            let by_len = || options.left_by.as_ref().unwrap().len();
            let by_all_none = options.left_by.is_none()
                && options.right_by.is_none()
                && by_descending.is_none()
                && by_nulls_last.is_none();
            by_all_none
                || (options.right_by.as_ref().unwrap().len() == by_len()
                    && by_descending.as_ref().unwrap().len() == by_len()
                    && by_nulls_last.as_ref().unwrap().len() == by_len())
        });

        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert_eq!(left_key_dtype, right_key_dtype);
        let left = AsOfJoinSideParams {
            on: left_on,
            tmp_key_col: tmp_left_key_col,
        };
        let right = AsOfJoinSideParams {
            on: right_on,
            tmp_key_col: tmp_right_key_col,
        };

        let params = AsOfJoinParams {
            left,
            right,
            by_descending: by_descending.unwrap_or_default(),
            by_nulls_last: by_nulls_last.unwrap_or_default(),
            args,
        };
        AsOfJoinNode {
            params,
            state: AsOfJoinState::default(),
            left_buffer: Default::default(),
            right_buffer: DataFrameSearchBuffer::empty_with_schema(right_input_schema),
            output_seq: Default::default(),
            last_non_null_row_left: None,
            last_non_null_row_right: None,
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
                let output_seq = &mut self.output_seq;
                let last_non_null_row_left = &mut self.last_non_null_row_left;
                let last_non_null_row_right = &mut self.last_non_null_row_right;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    distribute_work_task(
                        recv_left,
                        recv_right,
                        distributor,
                        left_buffer,
                        right_buffer,
                        output_seq,
                        last_non_null_row_left,
                        last_non_null_row_right,
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

#[allow(clippy::too_many_arguments)]
async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    left_buffer: &mut VecDeque<DataFrame>,
    right_buffer: &mut DataFrameSearchBuffer,
    output_seq: &mut MorselSeq,
    last_non_null_row_left: &mut Option<DataFrame>,
    last_non_null_row_right: &mut Option<DataFrame>,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();
    let right_done = recv_right.is_none();

    loop {
        if source_token.stop_requested() {
            stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df| left_buffer.push_back(df))
                .await;
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df| right_buffer.push_df(df))
                .await;
            return Ok(());
        }

        let (left_df, st) = if let Some(df) = left_buffer.pop_front() {
            (df, source_token.clone())
        } else if let Some(ref mut recv) = recv_left
            && let Ok(m) = recv.recv().await
        {
            let (df, _, st, _) = m.into_inner();
            (df, st)
        } else {
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df| right_buffer.push_df(df))
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
                left_buffer.push_front(left_df);
                stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df| {
                    left_buffer.push_back(df)
                })
                .await;
                return Ok(());
            }
        }

        if params.as_of_options().check_sortedness {
            check_left_continuity(last_non_null_row_left, &left_df, params)?;
        }

        if !params.as_of_options().check_sortedness {
            // If we need to check sortedness, we cannot prune the right side
            // yet, because the worker task still needs to check the internal
            // sortedness of this right chunk.
            prune_right_side(&left_df, right_buffer, 0, last_non_null_row_right, params)?;
        }
        if distributor
            .send((left_df.clone(), right_buffer.clone(), *output_seq, st))
            .await
            .is_err()
        {
            return Ok(());
        }
        *output_seq = output_seq.successor();
        prune_right_side(
            &left_df,
            right_buffer,
            left_df.height().saturating_sub(1),
            last_non_null_row_right,
            params,
        )?;
    }
}

/// Check that the first row of the DataFrame is in order with respect to the value in prev_row.
fn check_left_continuity(
    last_non_null_row: &mut Option<DataFrame>,
    df: &DataFrame,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    if df.height() == 0 {
        return Ok(());
    }

    let key_col_name = params.left.key_col();
    let sorted_by_cols = params.left_by().iter().chain([key_col_name]);
    let sorted_by_descending = params.by_descending.iter().chain([&false]);
    let sorted_by_nulls_last = params.by_nulls_last.iter().chain([&false]);
    let project = df.select(sorted_by_cols.clone())?;
    let first_non_null_at = project.column(key_col_name)?.first_non_null();
    let first_non_null_row = first_non_null_at.map(|at| project.slice(at as i64, 1));
    check_continuity(
        last_non_null_row.clone(),
        first_non_null_row,
        sorted_by_cols,
        sorted_by_descending,
        sorted_by_nulls_last,
    )?;

    // Store the last non-null row of this DataFrame for the next check.
    if let Some(pos) = project.column(key_col_name)?.last_non_null() {
        *last_non_null_row = Some(project.slice((pos) as i64, 1));
    }
    Ok(())
}

fn check_right_continuity(
    last_non_null_row: &mut Option<DataFrame>,
    dfsb: &DataFrameSearchBuffer,
    split_at_idx: usize,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let key_col_name = params.right.key_col();
    let sorted_by_cols = params.right_by().iter().chain([key_col_name]);
    let project = dfsb.select(sorted_by_cols.clone());
    let df = project.into_df();
    let sorted_by_descending = params.by_descending.iter().chain([&false]);
    let sorted_by_nulls_last = params.by_nulls_last.iter().chain([&false]);
    let before_split = df.slice(0, split_at_idx);
    let after_split = df.slice(split_at_idx as i64, df.height() - split_at_idx);
    let last_non_null = before_split.column(key_col_name)?.last_non_null();
    let first_non_null = after_split.column(key_col_name)?.first_non_null();
    let before_split_point_row = last_non_null
        .map(|pos| before_split.slice(pos as i64, 1))
        .or(last_non_null_row.clone());
    let after_split_point_row = first_non_null.map(|pos| after_split.slice(pos as i64, 1));
    check_continuity(
        before_split_point_row,
        after_split_point_row,
        sorted_by_cols,
        sorted_by_descending,
        sorted_by_nulls_last,
    )?;

    // Store the last non-null row of this DataFrame for the next check.
    if let Some(pos) = df.column(key_col_name)?.last_non_null() {
        *last_non_null_row = Some(after_split.slice(pos as i64, 1));
    }

    Ok(())
}

fn check_continuity<'a, IS, IB>(
    before_split_point_row: Option<DataFrame>,
    after_split_point_row: Option<DataFrame>,
    sorted_by_cols: IS,
    sorted_by_descending: IB,
    sorted_by_nulls_last: IB,
) -> PolarsResult<()>
where
    IS: Iterator<Item = &'a PlSmallStr> + Clone,
    IB: Iterator<Item = &'a bool> + Clone,
{
    if let Some(before_split_point) = before_split_point_row
        && let Some(after_split_point) = after_split_point_row
    {
        let across_split_point = before_split_point.vstack(&after_split_point)?;
        if !across_split_point.is_sorted(
            &sorted_by_cols.cloned().collect_vec(),
            &sorted_by_descending.cloned().collect_vec(),
            &sorted_by_nulls_last.cloned().collect_vec(),
        )? {
            return Err(not_sorted_err());
        }
    }
    Ok(())
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
    let by_iter = Iterator::zip(params.left_by().iter(), params.right_by().iter());
    let reorder_iter = Iterator::zip(params.by_descending.iter(), params.by_nulls_last.iter());
    for ((left_by, right_by), (descending, nulls_last)) in Iterator::zip(by_iter, reorder_iter) {
        let left_by_col = left.column(left_by)?.as_materialized_series();
        // SAFETY: We checked earlier that the dataframes are not empty
        let left_last_group = unsafe { left_by_col.get_unchecked(left.height() - 1) };
        let cmp =
            move |a: &AnyValue<'_>, b: &AnyValue<'_>| reorder_cmp(a, b, *descending, *nulls_last);
        start = right.binary_search(|x| cmp(x, &left_last_group).is_ge(), right_by, start..end);
        end = right.binary_search(|x| cmp(x, &left_last_group).is_gt(), right_by, start..end);
        if start >= right.height() {
            return Ok(true);
        } else if end < right.height() {
            return Ok(false);
        }
    }

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: We checked earlier that the dataframes are not empty
    let left_last_val = unsafe { left_key.get_unchecked(left_key.len() - 1) };
    let right_range_end = match (options.strategy, options.allow_eq) {
        (Forward, true) | (Backward, false) => {
            right.binary_search(|x| *x >= left_last_val, params.right.key_col(), start..end)
        },
        (Forward, false) | (Backward, true) => {
            right.binary_search(|x| *x > left_last_val, params.right.key_col(), start..end)
        },
        (Nearest, _) => {
            let first_greater =
                right.binary_search(|x| *x > left_last_val, params.right.key_col(), start..end);
            if first_greater >= right.height() {
                return Ok(true);
            }
            // In the Nearest case, there may be a chunk of consecutive equal
            // values following the match value on the left side.  In this case,
            // the AsOf join is greedy and should until the *end* of that chunk.

            // SAFETY: We just checked that first_greater is in bounds
            let first_greater_val =
                unsafe { right.get_unchecked(params.right.key_col(), first_greater) };
            right.binary_search(
                |x| *x > first_greater_val,
                params.right.key_col(),
                first_greater..end,
            )
        },
    };
    Ok(right_range_end >= right.height())
}

/// Prune right-side rows that are no longer needed using a specific left row as the
/// pruning reference point.
fn prune_right_side(
    left: &DataFrame,
    right: &mut DataFrameSearchBuffer,
    left_row_idx: usize,
    last_non_null_row: &mut Option<DataFrame>,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    if left.height() == 0 || right.height() == 0 {
        return Ok(());
    }

    let mut start = 0;
    let mut end = right.height();
    let by_iter = Iterator::zip(params.left_by().iter(), params.right_by().iter());
    let reorder_iter = Iterator::zip(params.by_descending.iter(), params.by_nulls_last.iter());
    for ((left_by, right_by), (descending, nulls_last)) in Iterator::zip(by_iter, reorder_iter) {
        let left_by_col = left.column(left_by)?.as_materialized_series();
        // SAFETY: We checked earlier that the dataframes are not empty
        let group_val = unsafe { left_by_col.get_unchecked(left_row_idx) };
        let cmp =
            move |a: &AnyValue<'_>, b: &AnyValue<'_>| reorder_cmp(a, b, *descending, *nulls_last);
        start = right.binary_search(|x| cmp(x, &group_val).is_ge(), right_by, start..end);
        end = right.binary_search(|x| cmp(x, &group_val).is_gt(), right_by, start..end);
    }

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: We checked earlier that the dataframes are not empty
    let key_val = unsafe { left_key.get_unchecked(left_row_idx) };
    let mut right_range_start =
        right.binary_search(|x| *x >= key_val, params.right.key_col(), start..end);
    if matches!(params.as_of_options().strategy, Backward | Nearest) {
        right_range_start = right_range_start.saturating_sub(1).max(start);
    }

    if params.as_of_options().check_sortedness {
        check_right_continuity(last_non_null_row, right, right_range_start, params)?;
    }
    right.split_at(right_range_start);
    Ok(())
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    mut send: PortSender,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let wait_group = WaitGroup::default();
    let mut scratch1 = ScratchVec::default();
    let mut scratch2 = ScratchVec::default();
    while let Ok((left_df, right_dfsb, seq, st)) = dist_recv.recv().await {
        let out = compute_asof_join(left_df, right_dfsb, params, &mut scratch1, &mut scratch2)?;
        let mut morsel = Morsel::new(out, seq, st);
        morsel.set_consume_token(wait_group.token());
        if send.send(morsel).await.is_err() {
            return Ok(());
        }
        wait_group.wait().await;
    }
    Ok(())
}

/// Encodes the "by" group columns of a DataFrame for sorted group traversal.
///
/// Provides run-length group boundaries and value comparison, abstracting over
/// the single-column and multi-column cases.
struct ByGroups<'a> {
    kind: GroupEncodingKind,
    /// Run-lengths of consecutive equal group values (in sorted order).
    run_lengths: &'a mut Vec<IdxSize>,
}

enum GroupEncodingKind {
    /// Single by-column stored directly.
    Single(Column),
    /// Multiple by-columns merged into a row-encoded binary column for comparison.
    Multi(ChunkedArray<BinaryOffsetType>),
}

impl<'a> ByGroups<'a> {
    fn find_groups(
        df: &DataFrame,
        by: &[PlSmallStr],
        run_lengths: &'a mut ScratchVec<IdxSize>,
        params: &AsOfJoinParams,
    ) -> PolarsResult<Self> {
        let run_lengths = run_lengths.get();
        let kind = match by {
            [col_name] => {
                let col = df.column(col_name)?;
                rle_lengths(col, run_lengths)?;
                GroupEncodingKind::Single(col.clone())
            },
            _ => {
                let cols = by
                    .iter()
                    .map(|n| df.column(n).unwrap().clone())
                    .collect_vec();
                let encoded = _get_rows_encoded_ca(
                    ROW_ENCODED_COL_NAME,
                    &cols,
                    &params.by_descending,
                    &params.by_nulls_last,
                    true,
                )?;
                rle_lengths_helper_ca(&encoded, run_lengths);
                GroupEncodingKind::Multi(encoded)
            },
        };
        Ok(Self { kind, run_lengths })
    }

    /// Returns true if the group value at `idx` is null.
    ///
    /// # Safety
    /// `idx` must be a valid row index.
    unsafe fn is_null_at(&self, idx: usize) -> bool {
        match &self.kind {
            GroupEncodingKind::Single(col) => unsafe { col.get_unchecked(idx) }.is_null(),
            GroupEncodingKind::Multi(enc) => unsafe { enc.get_unchecked(idx) }.is_none(),
        }
    }

    /// Lexicographically compare group `idx` with group `other_idx`.
    ///
    /// # Safety
    /// Both indices must be valid row indices within their respective encodings.
    unsafe fn cmp_at(
        &self,
        self_idx: usize,
        other: &Self,
        other_idx: usize,
        params: &AsOfJoinParams,
    ) -> Ordering {
        match (&self.kind, &other.kind) {
            (GroupEncodingKind::Single(s), GroupEncodingKind::Single(o)) => reorder_cmp(
                &unsafe { s.get_unchecked(self_idx) },
                &unsafe { o.get_unchecked(other_idx) },
                params.by_descending[0],
                params.by_nulls_last[0],
            ),
            (GroupEncodingKind::Multi(s), GroupEncodingKind::Multi(o)) => reorder_cmp(
                &unsafe { s.get_unchecked(self_idx) },
                &unsafe { o.get_unchecked(other_idx) },
                false,
                false,
            ),
            _ => unreachable!("mismatched GroupEncoding kinds"),
        }
    }

    /// Iterates over groups as `(start_row, row_count)` pairs.
    fn iter_groups(&self) -> impl Iterator<Item = (usize, usize)> {
        self.run_lengths.iter().scan(0usize, |offset, &len| {
            let start = *offset;
            *offset += len as usize;
            Some((start, len as usize))
        })
    }
}

fn drop_columns(
    left_df: &mut DataFrame,
    right_df: &mut DataFrame,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    // Drop any temporary join-key columns that were added before calling _join_asof.
    if let Some(ref col) = params.left.tmp_key_col
        && left_df.schema().contains(col)
    {
        left_df.drop_in_place(col)?;
    }
    if let Some(ref col) = params.right.tmp_key_col
        && right_df.schema().contains(col)
    {
        right_df.drop_in_place(col)?;
    }

    // Only return one set of group columns in the result.
    for col in params.right_by() {
        right_df.drop_in_place(col)?;
    }

    // Coalesce the key column.
    if params.args.should_coalesce() && params.left.on == params.right.on {
        right_df.drop_in_place(&params.right.on)?;
    }
    Ok(())
}

/// Calls `_join_asof` on a left/right pair and cleans up the output via
/// [`finalize_output`].
fn join_asof_ungrouped(
    mut left_df: DataFrame,
    mut right_df: DataFrame,
    left_key: &Series,
    right_key: &Series,
    options: &AsOfOptions,
    params: &AsOfJoinParams,
) -> PolarsResult<DataFrame> {
    _check_asof_columns(
        left_key,
        right_key,
        options.tolerance.is_some(),
        false, // Sortedness is already checked
        false,
    )?;

    let take_idx = _join_asof_dispatch(
        left_key,
        right_key,
        options.strategy,
        options.tolerance.clone().map(Scalar::into_value),
        options.allow_eq,
    )?;
    drop_columns(&mut left_df, &mut right_df, params)?;
    // SAFETY: _join_asof_dispatch only returns in-bounds indices.
    let right_df = unsafe { right_df.take_unchecked(&take_idx) };
    _finish_join(left_df, right_df, params.args.suffix.clone())
}

fn compute_asof_join(
    mut left_df: DataFrame,
    right_dfsb: DataFrameSearchBuffer,
    params: &AsOfJoinParams,
    left_lengths: &mut ScratchVec<IdxSize>,
    right_lengths: &mut ScratchVec<IdxSize>,
) -> PolarsResult<DataFrame> {
    let mut right_df = right_dfsb.into_df();
    let options = params.as_of_options();
    let left_key = left_df.column(params.left.key_col())?.to_physical_repr();
    let right_key = right_df
        .column(params.right.key_col())?
        .clone()
        .to_physical_repr();

    if params.as_of_options().check_sortedness {
        check_df_sorted(&left_df, params.left_by(), &params.left.on, params)?;
        check_df_sorted(&right_df, params.right_by(), &params.right.on, params)?;
    }

    if params.left_by().is_empty() {
        return join_asof_ungrouped(
            left_df,
            right_df,
            left_key.as_materialized_series(),
            right_key.as_materialized_series(),
            options,
            params,
        );
    }

    let left_groups = ByGroups::find_groups(&left_df, params.left_by(), left_lengths, params)?;
    let right_groups = ByGroups::find_groups(&right_df, params.right_by(), right_lengths, params)?;
    let cmp_at = |left_idx, right_idx| unsafe {
        ByGroups::cmp_at(&left_groups, left_idx, &right_groups, right_idx, params)
    };
    let left_groups_iter = left_groups.iter_groups();
    let mut right_groups_iter = right_groups.iter_groups();
    let mut right_group = right_groups_iter.next();

    drop_columns(&mut left_df, &mut right_df, params)?;

    let mut out_right = Vec::with_capacity(left_groups.run_lengths.len());
    for (left_start, left_group_len) in left_groups_iter {
        let right_chunk_len =
            // SAFETY: the iterator results will be in-bounds.
            if unsafe { left_groups.is_null_at(left_start) }  {
                // If the left group is null, then it will have no matches on the right side.
                0
            } else {
                // Advance the right cursor past null groups and any groups that are
                // ordered before the left group.
                let should_bump_right_group =  |right_start| unsafe {
                    right_groups.is_null_at(right_start) || cmp_at(left_start, right_start).is_gt()
                };
                while let Some((right_start, _)) = right_group
                    && should_bump_right_group(right_start)
                {
                    right_group = right_groups_iter.next();
                }
                if let Some((right_start, right_len)) = right_group
                    && cmp_at(left_start, right_start).is_eq()
                {
                    right_len
                } else {
                    // The left group is not present in the right group.
                    0
                }
            };

        let right_start = right_group.map_or(0, |(start, _len)| start);
        let group_left_key = left_key.slice(left_start as i64, left_group_len);
        let group_right_key = right_key.slice(right_start as i64, right_chunk_len);

        let take_idx = _join_asof_dispatch(
            group_left_key.as_materialized_series(),
            group_right_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            options.allow_eq,
        )?;

        out_right.push(
            right_df
                .slice(right_start as i64, right_chunk_len)
                .take(&take_idx)?,
        );
    }

    let initial = DataFrame::empty_with_arc_schema(right_df.schema().clone());
    let out_right =
        accumulate_dataframes_vertical_unchecked([initial].into_iter().chain(out_right));
    _finish_join(left_df, out_right, params.args.suffix.clone())
}

fn check_df_sorted(
    dataframe: &DataFrame,
    by: &[PlSmallStr],
    on: &PlSmallStr,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let sorted_by_cols = by.iter().chain([on]);
    let sorted_by_descending = params.by_descending.iter().chain([&false]);
    let sorted_by_nulls_last = params.by_nulls_last.iter().chain([&false]);
    if !dataframe.is_sorted(
        &sorted_by_cols.cloned().collect_vec(),
        &sorted_by_descending.cloned().collect_vec(),
        &sorted_by_nulls_last.cloned().collect_vec(),
    )? {
        return Err(not_sorted_err());
    }
    Ok(())
}

fn not_sorted_err() -> PolarsError {
    polars_err!(InvalidOperation: "argument in operation 'asof_join' is not sorted, please sort the 'expr/series/column' first")
}
