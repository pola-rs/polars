use std::collections::VecDeque;
use std::mem::swap;
use std::ops::BitOr;

use arrow::array::builder::ShareStrategy;
use either::{Either, Left, Right};
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::prelude::*;
use polars_plan::prelude::*;
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::pipe::{PortReceiver, RecvPort, SendPort};

// TODO: [amber] Use `_get_rows_encoded` (with `descending`) to encode the keys
// TODO: [amber] Use a distributor to distribute the merge-join work to a pool of workers
// BUG: [amber] Currently the join may process sub-chunks and not produce "cartesian product"
// results when there are chunks of equal keys
// TODO: [amber] I think I want to have the inner join dispatch to another streaming
// equi join node rather than dispatching to in-memory engine

const KEY_COL_NAME: &str = "__POLARS_JOIN_KEY";

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Left,
    Right,
    Both,
    Finished,
}

impl BitOr for NeedMore {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        use NeedMore::*;
        match (self, rhs) {
            (Left, Left) => Left,
            (Right, Right) => Right,
            (Finished, Finished) => Finished,
            (Finished, other) | (other, Finished) => other,
            (Left, Right) | (Right, Left) | (Both, _) | (_, Both) => Both,
        }
    }
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
    descending: Vec<bool>,
    nulls_last: Vec<bool>,
    emit_unmatched: bool,
}

#[derive(Debug)]
struct MergeJoinParams {
    left: SideParams,
    right: SideParams,
    output_schema: SchemaRef,
    key_descending: bool,
    key_nulls_last: bool, // Only valid when using a single-column key
    args: JoinArgs,
}

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: VecDeque<DataFrame>,
    left_unmerged_nulls: VecDeque<DataFrame>,
    right_unmerged: VecDeque<DataFrame>,
    right_unmerged_nulls: VecDeque<DataFrame>,
    seq: MorselSeq,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum MergeJoinState {
    #[default]
    Running,
    Flushing,
    Done,
}

impl MergeJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        left_sortedness: IRSorted,
        right_sortedness: IRSorted,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        assert!(left_on.len() == right_on.len());
        assert!(left_sortedness.0.len() == left_on.len());
        assert!(right_sortedness.0.len() == right_on.len());

        let state: MergeJoinState = MergeJoinState::Running;
        let left_key_col;
        let right_key_col;
        let mut left_ir_schema = left_input_schema.clone();
        let mut right_ir_schema = right_input_schema.clone();
        if left_on.len() > 1 {
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
        let left_descending = left_sortedness
            .0
            .iter()
            .map(|s| s.descending.unwrap())
            .collect_vec();
        let right_descending = right_sortedness
            .0
            .iter()
            .map(|s| s.descending.unwrap())
            .collect_vec();
        assert!(left_descending.len() == right_descending.len());
        let is_descending = |s: &IRSorted| {
            (s.0.len() == 1)
                .then(|| s.0[0].descending.unwrap())
                .unwrap_or(false)
        };
        let key_descending = is_descending(&left_sortedness);
        assert!(key_descending == is_descending(&right_sortedness));
        let left_nulls_last = left_sortedness
            .0
            .iter()
            .map(|s| s.nulls_last.unwrap())
            .collect_vec();
        let right_nulls_last = right_sortedness
            .0
            .iter()
            .map(|s| s.nulls_last.unwrap())
            .collect_vec();
        let is_nulls_last = |s: &IRSorted| {
            (s.0.len() == 1)
                .then(|| s.0[0].nulls_last.unwrap())
                .unwrap_or(false)
        };
        let key_nulls_last = is_nulls_last(&left_sortedness);
        assert!(key_nulls_last == is_nulls_last(&right_sortedness));
        let left = SideParams {
            input_schema: left_input_schema,
            ir_schema: left_ir_schema,
            on: left_on,
            key_col: left_key_col,
            descending: left_descending,
            nulls_last: left_nulls_last,
            emit_unmatched: matches!(args.how, JoinType::Left | JoinType::Full),
        };
        let right = SideParams {
            input_schema: right_input_schema,
            ir_schema: right_ir_schema,
            on: right_on,
            key_col: right_key_col,
            descending: right_descending,
            nulls_last: right_nulls_last,
            emit_unmatched: matches!(args.how, JoinType::Right | JoinType::Full),
        };
        let params = MergeJoinParams {
            left,
            right,
            output_schema,
            key_descending,
            key_nulls_last,
            args,
        };
        Ok(MergeJoinNode {
            state,
            params,
            left_unmerged: Default::default(),
            right_unmerged: Default::default(),
            left_unmerged_nulls: Default::default(),
            right_unmerged_nulls: Default::default(),
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
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = MergeJoinState::Done;
        } else if recv[0] == PortState::Done
            && recv[1] == PortState::Done
            && (!self.left_unmerged.is_empty()
                || !self.left_unmerged_nulls.is_empty()
                || !self.right_unmerged.is_empty()
                || !self.right_unmerged_nulls.is_empty())
        {
            self.state = MergeJoinState::Flushing;
        } else if recv[0] == PortState::Done && recv[1] == PortState::Done {
            self.state = MergeJoinState::Done;
        }

        match self.state {
            MergeJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
            },
            MergeJoinState::Flushing => {
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
        let left_unmerged_nulls = &mut self.left_unmerged_nulls;
        let right_unmerged = &mut self.right_unmerged;
        let right_unmerged_nulls = &mut self.right_unmerged_nulls;
        let seq = &mut self.seq;

        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        let mut recv_left = recv_ports[0].take().map(RecvPort::serial);
        let mut recv_right = recv_ports[1].take().map(RecvPort::serial);
        let send = send_ports[0].take().unwrap().parallel();

        debug_assert!(
            (recv_left.is_none() && recv_right.is_none())
                == (self.state == MergeJoinState::Flushing)
        );

        let (mut distributor, dist_recv) =
            distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        if matches!(self.state, Running | Flushing) {
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                // Partitioner task; worst-case complexity O(n)

                let source_token = SourceToken::new();

                loop {
                    match dbg!(find_mergeable(
                        left_unmerged,
                        left_unmerged_nulls,
                        right_unmerged,
                        right_unmerged_nulls,
                        recv_left.is_none(),
                        recv_right.is_none(),
                        params,
                    )?) {
                        Left((left_mergeable, right_mergeable)) => {
                            let left_mergeable =
                                Morsel::new(left_mergeable, *seq, source_token.clone());
                            *seq = seq.successor();
                            distributor
                                .send((left_mergeable, right_mergeable))
                                .await
                                .unwrap(); // TODO [amber] Can this error be handled?
                        },
                        Right(NeedMore::Left | NeedMore::Both) if recv_left.is_some() => {
                            let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_right.as_mut(),
                                    right_unmerged,
                                    right_unmerged_nulls,
                                    &params.right,
                                    &params,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            add_key_column(&mut df, &params.left, params)?;
                            let (non_nulls, nulls) = split_null_key_rows(df, &params.left, params);
                            push_back_opt(left_unmerged, non_nulls);
                            push_back_opt(left_unmerged_nulls, nulls);
                        },
                        Right(NeedMore::Right | NeedMore::Both) if recv_right.is_some() => {
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_left.as_mut(),
                                    left_unmerged,
                                    left_unmerged_nulls,
                                    &params.left,
                                    &params,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            add_key_column(&mut df, &params.right, params)?;
                            let (non_nulls, nulls) = split_null_key_rows(df, &params.left, params);
                            push_back_opt(right_unmerged, non_nulls);
                            push_back_opt(right_unmerged_nulls, nulls);
                        },

                        Right(NeedMore::Finished) => {
                            dbg!("breaking because we are finished");
                            dbg!(
                                left_unmerged,
                                left_unmerged_nulls,
                                right_unmerged,
                                right_unmerged_nulls
                            );
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
                scope.spawn_task(TaskPriority::High, async move {
                    while let Ok((morsel, right)) = recv.recv().await {
                        let (left, seq, source_token, _) = morsel.into_inner();
                        let joined = compute_join(left, right, params)?;
                        let morsel = Morsel::new(joined, seq, source_token);
                        if send.send(morsel).await.is_err() {
                            break;
                        }
                    }
                    Ok(())
                })
            }));
        }
    }
}

fn find_mergeable(
    l: &mut VecDeque<DataFrame>,
    l_nulls: &mut VecDeque<DataFrame>,
    r: &mut VecDeque<DataFrame>,
    r_nulls: &mut VecDeque<DataFrame>,
    l_done: bool,
    r_done: bool,
    p: &MergeJoinParams,
) -> PolarsResult<Either<(DataFrame, DataFrame), NeedMore>> {
    dbg!(&l, &l_nulls, &r_nulls, &r, l_done, r_done);
    if p.args.how == JoinType::Right {
        let mergable = find_mergeable_inner(r, l, &p.right, &p.left, p, r_done, l_done)?;
        if let Left((df2, df1)) = mergable {
            return Ok(Left((df1, df2)));
        }
        let mergeable_nulls = dbg!(find_mergeable_nulls(
            r_nulls, l_nulls, &p.right, &p.left, p, r_done, l_done
        )?);
        if let Left((df2, df1)) = mergeable_nulls {
            return Ok(Left((df1, df2)));
        }
        Ok(Right(
            mergable.unwrap_right().flip() | mergeable_nulls.unwrap_right().flip(),
        ))
    } else {
        let mergable = find_mergeable_inner(l, r, &p.left, &p.right, p, l_done, r_done)?;
        if mergable.is_left() {
            return Ok(mergable);
        }
        let mergeable_nulls = dbg!(find_mergeable_nulls(
            l_nulls, r_nulls, &p.left, &p.right, p, l_done, r_done
        )?);
        if mergeable_nulls.is_left() {
            return Ok(mergeable_nulls);
        }
        Ok(Right(
            mergable.unwrap_right() | mergeable_nulls.unwrap_right(),
        ))
    }
}

fn find_mergeable_inner(
    left: &mut VecDeque<DataFrame>,
    right: &mut VecDeque<DataFrame>,
    left_params: &SideParams,
    right_params: &SideParams,
    params: &MergeJoinParams,
    left_done: bool,
    right_done: bool,
) -> PolarsResult<Either<(DataFrame, DataFrame), NeedMore>> {
    loop {
        if left_done && left.is_empty() && right_done && right.is_empty() {
            return (Ok(Right(NeedMore::Finished)));
        } else if left_done && left.is_empty() {
            // [amber] Also done here in case of LEFT join, right?
            return (Ok(Left((
                DataFrame::empty_with_schema(&left_params.ir_schema),
                right.pop_front().unwrap(),
            ))));
        } else if right_done && right.is_empty() {
            return (Ok(Left((
                left.pop_front().unwrap(),
                DataFrame::empty_with_schema(&right_params.ir_schema),
            ))));
        } else if left_done && left.len() <= 1 && right_done && right.len() <= 1 {
            return (Ok(Left((
                left.pop_front().unwrap(),
                right.pop_front().unwrap(),
            ))));
        } else if left.is_empty() && !left_done {
            return (Ok(Right(NeedMore::Left)));
        } else if right.is_empty() && !right_done {
            return (Ok(Right(NeedMore::Right)));
        }

        let Some(left_df) = left.front() else {
            return (Ok(Right(NeedMore::Left)));
        };
        let Some(right_df) = right.front() else {
            return (Ok(Right(NeedMore::Right)));
        };

        if left_df.is_empty() {
            left.pop_front();
            continue;
        }
        if right_df.is_empty() {
            right.pop_front();
            continue;
        }

        dbg!(&left_df);
        dbg!(&right_df);

        let left_keys = left_df
            .column(&left_params.key_col)?
            .as_materialized_series();
        let right_keys = right_df
            .column(&right_params.key_col)?
            .as_materialized_series();

        let left_last = left_keys.get(left_keys.len() - 1)?;
        let left_first_incomplete = match left_done && left.len() <= 1 {
            false => binary_search_lower(left_keys, &left_last, params)?,
            true => left_keys.len(),
        };
        if left_first_incomplete == 0 && left.len() > 1 {
            vstack_head(left, left_params, params);
            continue;
        } else if left_first_incomplete == 0 {
            debug_assert!(!left_done);
            return (Ok(Right(NeedMore::Left)));
        }

        let right_last = right_keys.get(right_keys.len() - 1)?;
        let right_first_incomplete = match right_done && right.len() <= 1 {
            false => binary_search_lower(right_keys, &right_last, params)?,
            true => right_keys.len(),
        };
        if right_first_incomplete == 0 && right.len() > 1 {
            vstack_head(right, right_params, params);
            continue;
        } else if right_first_incomplete == 0 {
            debug_assert!(!right_done);
            return (Ok(Right(NeedMore::Right)));
        }

        let left_last_completed_val = left_keys.get(left_first_incomplete - 1)?;
        let right_last_completed_val = right_keys.get(right_first_incomplete - 1)?;
        let left_mergable_until; // bound is *exclusive*
        let right_mergable_until;
        if keys_eq(&left_last_completed_val, &right_last_completed_val, params) {
            left_mergable_until = left_first_incomplete;
            right_mergable_until = right_first_incomplete;
        } else if keys_lt(&left_last_completed_val, &right_last_completed_val, params) {
            left_mergable_until = left_first_incomplete;
            right_mergable_until =
                binary_search_upper(right_keys, &left_keys.get(left_mergable_until - 1)?, params)?;
        } else if keys_gt(&left_last_completed_val, &right_last_completed_val, params) {
            right_mergable_until = right_first_incomplete;
            left_mergable_until = binary_search_upper(
                left_keys,
                &right_keys.get(right_mergable_until - 1)?,
                params,
            )?;
        } else {
            unreachable!();
        }

        if left_mergable_until == 0 && left.len() > 1 {
            vstack_head(left, left_params, params);
            continue;
        } else if right_mergable_until == 0 && right.len() > 1 {
            vstack_head(right, right_params, params);
            continue;
        } else if left_mergable_until == 0 && right_mergable_until == 0 {
            return (Ok(Right(NeedMore::Both)));
        }

        let (left_df, left_rest) = left
            .pop_front()
            .unwrap()
            .split_at(left_mergable_until as i64);
        if !left_rest.is_empty() {
            left.push_front(left_rest);
        }
        let (right_df, right_rest) = right
            .pop_front()
            .unwrap()
            .split_at(right_mergable_until as i64);
        if !right_rest.is_empty() {
            right.push_front(right_rest);
        }
        return (Ok(Left((left_df, right_df))));
    }
}

fn find_mergeable_nulls(
    l: &mut VecDeque<DataFrame>,
    r: &mut VecDeque<DataFrame>,
    l_params: &SideParams,
    r_params: &SideParams,
    params: &MergeJoinParams,
    l_done: bool,
    r_done: bool,
) -> PolarsResult<Either<(DataFrame, DataFrame), NeedMore>> {
    let need_more_left = match dbg!(find_mergeable_nulls_side(l, l_done)?) {
        Left(df1) => {
            let df2 = DataFrame::empty_with_schema(&r_params.ir_schema);
            return Ok(Left((df1, df2)));
        },
        Right(need_more) => need_more,
    };
    let need_more_right = match dbg!(find_mergeable_nulls_side(r, r_done)?) {
        Left(df2) => {
            let df1 = DataFrame::empty_with_schema(&l_params.ir_schema);
            return Ok(Left((df1, df2)));
        },
        Right(need_more) => need_more.flip(),
    };
    Ok(Right(need_more_left | need_more_right))
}

fn find_mergeable_nulls_side(
    unmerged: &mut VecDeque<DataFrame>,
    done: bool,
) -> PolarsResult<Either<DataFrame, NeedMore>> {
    loop {
        if done && unmerged.is_empty() {
            return Ok(Right(NeedMore::Finished));
        }
        let Some(df) = unmerged.pop_front() else {
            return Ok(Right(NeedMore::Left));
        };
        if df.is_empty() {
            continue;
        }
        return Ok(Left(df));
    }
}

fn binary_search(
    s: &Series,
    search_value: &AnyValue,
    op: fn(&AnyValue, &AnyValue, &MergeJoinParams) -> bool,
    params: &MergeJoinParams,
) -> PolarsResult<usize> {
    let mut lower = 0;
    let mut upper = s.len();
    while lower < upper {
        let mid = (lower + upper) / 2;
        let mid_val = s.get(mid)?;
        if op(search_value, &mid_val, params) {
            upper = mid;
        } else {
            lower = mid + 1;
        }
    }
    return Ok(lower);
}

fn binary_search_lower(s: &Series, sv: &AnyValue, params: &MergeJoinParams) -> PolarsResult<usize> {
    binary_search(s, sv, keys_le, params)
}

fn binary_search_upper(s: &Series, sv: &AnyValue, params: &MergeJoinParams) -> PolarsResult<usize> {
    binary_search(s, sv, keys_lt, params)
}

fn vstack_head(unmerged: &mut VecDeque<DataFrame>, sp: &SideParams, params: &MergeJoinParams) {
    let is_sorted = match params.key_descending {
        false => IsSorted::Ascending,
        true => IsSorted::Descending,
    };
    let mut df = unmerged.pop_front().unwrap();
    let col_idx = df.get_column_index(&sp.key_col).unwrap();
    df.vstack_mut_owned(unmerged.pop_front().unwrap()).unwrap();
    unsafe {
        let col = &mut df.get_columns_mut()[col_idx];
        col.set_sorted_flag(is_sorted);
    };
    unmerged.push_front(df);
}

fn compute_join(
    mut left: DataFrame,
    mut right: DataFrame,
    params: &MergeJoinParams,
) -> PolarsResult<DataFrame> {
    let join_type = &params.args.how;
    let right_is_build = *join_type == JoinType::Right
        || (*join_type == JoinType::Inner
            && matches!(
                params.args.maintain_order,
                MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft,
            ));

    left.rechunk_mut();
    right.rechunk_mut();

    // TODO: [amber] Remove any non-output columns earlier to reduce the
    // amount of gathering as much as possible

    let mut left_key = left.column(&params.left.key_col).unwrap();
    let mut right_key = right.column(&params.right.key_col).unwrap();

    if right_is_build {
        swap(&mut left_key, &mut right_key);
    }
    let mut left_gather_idxs = Vec::new();
    let mut right_gather_idxs = Vec::new();
    // TODO: [amber] This is still quadratic
    for (idxl, left_key) in left_key.as_materialized_series().iter().enumerate() {
        let mut matched = false;
        for (idxr, right_key) in right_key.as_materialized_series().iter().enumerate() {
            if keys_eq(&left_key, &right_key, &params) {
                matched = true;
                left_gather_idxs.push(idxl as IdxSize);
                right_gather_idxs.push(idxr as IdxSize);
            }
        }
        if !matched && matches!(params.args.how, JoinType::Left | JoinType::Right) {
            left_gather_idxs.push(idxl as IdxSize);
            right_gather_idxs.push(IdxSize::MAX);
        }
    }
    if right_is_build {
        swap(&mut left_gather_idxs, &mut right_gather_idxs);
    }

    // Remove the added row-encoded key columns
    if params.left.on.len() > 1 {
        left = left.drop(&params.left.key_col).unwrap();
        right = right.drop(&params.right.key_col).unwrap();
    }

    let mut left_build = DataFrameBuilder::new(left.schema().clone());
    let mut right_build = DataFrameBuilder::new(right.schema().clone());
    left_build.opt_gather_extend(&left, &left_gather_idxs, ShareStrategy::Never);
    right_build.opt_gather_extend(&right, &right_gather_idxs, ShareStrategy::Never);
    left = left_build.freeze();
    right = right_build.freeze();

    // Coalsesce the key columns (and rename them)
    if params.args.how == JoinType::Left {
        for c in &params.left.on {
            right.drop_in_place(c.as_str())?;
        }
    } else if params.args.how == JoinType::Right {
        for c in &params.right.on {
            left.drop_in_place(c.as_str())?;
        }
    }

    // Rename any right columns to "{}_right"
    rename_right_columns(&left, &mut right, params)?;

    left.hstack_mut(&right.get_columns())?;
    let df = drop_non_output_columns(&left, params)?;
    Ok(df)
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

fn add_key_column(
    df: &mut DataFrame,
    sp: &SideParams,
    params: &MergeJoinParams,
) -> PolarsResult<()> {
    debug_assert_eq!(*df.schema(), sp.input_schema);
    let broadcast_nulls = !params.args.nulls_equal;
    if sp.on.len() > 1 {
        let columns = sp
            .on
            .iter()
            .map(|c| df.column(c).unwrap())
            .cloned()
            .collect_vec();
        let key = row_encode::_get_rows_encoded_ca(
            sp.key_col.clone(),
            &columns,
            &sp.descending,
            &sp.nulls_last,
            broadcast_nulls,
        )?
        .into_column();
        df.hstack_mut(&[key])?;
    }
    debug_assert!(*df.schema() == sp.ir_schema);
    Ok(())
}

fn split_null_key_rows(
    df: DataFrame,
    sp: &SideParams,
    params: &MergeJoinParams,
) -> (Option<DataFrame>, Option<DataFrame>) {
    let key_column = df.column(&sp.key_col).unwrap();
    let null_count = key_column.null_count();
    if params.args.nulls_equal || null_count == 0 {
        (Some(df), None)
    } else if null_count == key_column.len() {
        (None, Some(df))
    } else {
        let key = df.column(&sp.key_col).unwrap();
        let non_nulls = df.filter(&key.is_not_null()).unwrap();
        let nulls = df.filter(&key.is_null()).unwrap();
        (Some(non_nulls), Some(nulls))
    }
}

async fn buffer_unmerged_from_pipe(
    port: Option<&mut PortReceiver>,
    unmerged: &mut VecDeque<DataFrame>,
    unmerged_nulls: &mut VecDeque<DataFrame>,
    sp: &SideParams,
    params: &MergeJoinParams,
) {
    let mut buffer = move |mut df: DataFrame| {
        add_key_column(&mut df, sp, params).unwrap();
        let (non_nulls, nulls) = split_null_key_rows(df, sp, params);
        push_back_opt(unmerged, non_nulls);
        push_back_opt(unmerged_nulls, nulls);
    };

    let Some(port) = port else {
        return;
    };
    let Ok(morsel) = port.recv().await else {
        return;
    };
    morsel.source_token().stop();

    buffer(morsel.into_df());
    while let Ok(morsel) = port.recv().await {
        buffer(morsel.into_df());
    }
}

fn keys_eq(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    if left.is_null() && right.is_null() {
        params.args.nulls_equal
    } else {
        left == right
    }
}

fn keys_lt(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    if !params.args.nulls_equal {
        debug_assert!(!left.is_null() && !right.is_null(), "nulls are not ordered")
    }
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

fn push_back_opt<T>(vec: &mut VecDeque<T>, opt: Option<T>) {
    if let Some(x) = opt {
        vec.push_back(x);
    }
}
