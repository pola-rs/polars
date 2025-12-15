use std::collections::VecDeque;
use std::mem::swap;

use arrow::array::builder::ShareStrategy;
use either::{Either, Left, Right};
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::prelude::*;
use polars_plan::prelude::*;
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
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
    descending: Vec<bool>,
    nulls_last: Vec<bool>,
}

#[derive(Debug)]
struct MergeJoinParams {
    left: SideParams,
    right: SideParams,
    output_schema: SchemaRef,
    key_descending: bool,
    key_nulls_last: bool,
    args: JoinArgs,
}

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: VecDeque<DataFrame>,
    right_unmerged: VecDeque<DataFrame>,
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
            left_key_col = PlSmallStr::from("__POLARS_JOIN_KEY");
            right_key_col = PlSmallStr::from("__POLARS_JOIN_KEY");
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
        };
        let right = SideParams {
            input_schema: right_input_schema,
            ir_schema: right_ir_schema,
            on: right_on,
            key_col: right_key_col,
            descending: right_descending,
            nulls_last: right_nulls_last,
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
            seq: MorselSeq::default(),
        })
    }

    fn compute_join(
        &self,
        mut left: DataFrame,
        mut right: DataFrame,
        _state: &StreamingExecutionState,
    ) -> PolarsResult<DataFrame> {
        let join_type = &self.params.args.how;
        let right_is_build = *join_type == JoinType::Right
            || (*join_type == JoinType::Inner
                && matches!(
                    self.params.args.maintain_order,
                    MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft,
                ));

        if self.params.left.on.len() > 1 && !self.params.args.nulls_equal {
            left.drop_in_place(&self.params.left.key_col)?;
            append_key_column(&mut left, &self.params.left, true)?;
        }
        if self.params.right.on.len() > 1 && !self.params.args.nulls_equal {
            right.drop_in_place(&self.params.right.key_col)?;
            append_key_column(&mut right, &self.params.right, true)?;
        }

        left.rechunk_mut();
        right.rechunk_mut();

        // TODO: [amber] Remove any non-output columns earlier to reduce the
        // amount of gathering as much as possible
        // TODO: [amber] Do a cardinality sketch to estimate the output size?

        let mut left_key = left.column(&self.params.left.key_col).unwrap();
        let mut right_key = right.column(&self.params.right.key_col).unwrap();

        if right_is_build {
            swap(&mut left_key, &mut right_key);
        }
        let mut left_gather_idxs = Vec::new();
        let mut right_gather_idxs = Vec::new();
        // TODO: [amber] This is still quadratic
        for (idxl, left_key) in left_key.as_materialized_series().iter().enumerate() {
            let mut matched = false;
            for (idxr, right_key) in right_key.as_materialized_series().iter().enumerate() {
                if keys_eq(&left_key, &right_key, &self.params) {
                    matched = true;
                    left_gather_idxs.push(idxl as IdxSize);
                    right_gather_idxs.push(idxr as IdxSize);
                }
            }
            if !matched && matches!(self.params.args.how, JoinType::Left | JoinType::Right) {
                left_gather_idxs.push(idxl as IdxSize);
                right_gather_idxs.push(IdxSize::MAX);
            }
        }
        if right_is_build {
            swap(&mut left_gather_idxs, &mut right_gather_idxs);
        }

        // Remove the added row-encoded key columns
        if self.params.left.on.len() > 1 {
            left = left.drop(&self.params.left.key_col).unwrap();
            right = right.drop(&self.params.right.key_col).unwrap();
        }

        let mut left_build = DataFrameBuilder::new(left.schema().clone());
        let mut right_build = DataFrameBuilder::new(right.schema().clone());
        left_build.opt_gather_extend(&left, &left_gather_idxs, ShareStrategy::Never);
        right_build.opt_gather_extend(&right, &right_gather_idxs, ShareStrategy::Never);
        left = left_build.freeze();
        right = right_build.freeze();

        // Coalsesce the key columns (and rename them)
        if self.params.args.how == JoinType::Left {
            for c in &self.params.left.on {
                right.drop_in_place(c.as_str())?;
            }
        } else if self.params.args.how == JoinType::Right {
            for c in &self.params.right.on {
                left.drop_in_place(c.as_str())?;
            }
        }

        // Rename any right columns to "{}_right"
        self.rename_right_columns(&left, &mut right)?;

        // for (left_on, right_on) in self.params.left_on.iter().zip(self.params.right_on.iter()) {
        //     let left_col = df_left.column(left_on)?;
        //     let right_col = df_right.column(right_on)?;
        //     coalesce_columns(s)
        //     df_left.replace(left_on, coalesced.into_column())?;
        // }

        // let left = self.drop_non_output_columns(left)?;
        // let right = self.drop_non_output_columns(right)?;

        left.hstack_mut(&right.get_columns())?;
        let df = self.drop_non_output_columns(&left)?;
        Ok(df)
    }

    fn rename_right_columns(&self, left: &DataFrame, right: &mut DataFrame) -> PolarsResult<()> {
        let left_cols: PlHashSet<PlSmallStr> = left
            .get_column_names()
            .into_iter()
            .cloned()
            .collect::<PlHashSet<_>>();
        for col in right.get_column_names_owned() {
            if left_cols.contains(&col) {
                let new_name = format_pl_smallstr!("{}{}", col, self.params.args.suffix());
                right.rename(&col, new_name).unwrap(); // FIXME: [amber] Potential quadratic behavior
            }
        }
        Ok(())
    }

    fn drop_non_output_columns(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let mut drop_cols = PlHashSet::with_capacity(df.width());
        for col in df.get_column_names() {
            if !self.key_is_in_output(col) {
                drop_cols.insert(col.clone());
            }
        }
        Ok(df.drop_many_amortized(&drop_cols))
    }

    fn key_is_in_output(&self, col_name: &PlSmallStr) -> bool {
        self.params.output_schema.contains(col_name)
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
            && (!self.left_unmerged.is_empty() || !self.right_unmerged.is_empty())
        {
            self.state = MergeJoinState::Flushing;
        } else if recv[0] == PortState::Done
            && self.left_unmerged.is_empty()
            && recv[1] == PortState::Done
        {
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
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        use MergeJoinState::*;

        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        let mut recv_left = recv_ports[0].take().map(RecvPort::serial);
        let mut recv_right = recv_ports[1].take().map(RecvPort::serial);
        let mut send = send_ports[0].take().unwrap().serial();

        debug_assert!(
            (recv_left.is_none() && recv_right.is_none())
                == (self.state == MergeJoinState::Flushing)
        );

        if matches!(self.state, Running | Flushing) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                loop {
                    match dbg!(find_mergeable(
                        &mut self.left_unmerged,
                        &mut self.right_unmerged,
                        recv_left.is_none(),
                        recv_right.is_none(),
                        &self.params,
                    )?) {
                        Left((left_chunk, right_chunk)) => {
                            dbg!("MARK 1");
                            // [amber] TODO: put this into a distributor instead of computing the join serially
                            let joined = self.compute_join(left_chunk, right_chunk, state)?;
                            let source_token = SourceToken::new();
                            let morsel = Morsel::new(joined, self.seq, source_token);
                            send.send(morsel).await.unwrap(); // TODO [amber] Can this error be handled?
                            self.seq = self.seq.successor();
                        },
                        Right(NeedMore::Left | NeedMore::Both) if recv_left.is_some() => {
                            dbg!("MARK 2");
                            // Need more left data
                            let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_right.as_mut(),
                                    &mut self.right_unmerged,
                                    &self.params.right,
                                    &self.params,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            append_key_column(&mut df, &self.params.left, false)?;
                            self.left_unmerged.push_back(df);
                        },
                        Right(NeedMore::Right | NeedMore::Both) if recv_right.is_some() => {
                            dbg!("MARK 3");
                            // Need more right data
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_left.as_mut(),
                                    &mut self.left_unmerged,
                                    &self.params.left,
                                    &self.params,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            append_key_column(&mut df, &self.params.right, false)?;
                            self.right_unmerged.push_back(df);
                        },

                        Right(NeedMore::Finished) => {
                            dbg!("breaking because we are finished");
                            break;
                        },
                        Right(other) => {
                            panic!("Unexpected NeedMore value: {other:?}");
                        },
                    }
                }
                Ok(())
            }));
        }
    }
}

fn find_mergeable(
    left: &mut VecDeque<DataFrame>,
    right: &mut VecDeque<DataFrame>,
    left_done: bool,
    right_done: bool,
    params: &MergeJoinParams,
) -> PolarsResult<Either<(DataFrame, DataFrame), NeedMore>> {
    if params.args.how == JoinType::Right {
        let res = find_mergeable_inner(
            right,
            left,
            &params.right,
            &params.left,
            params,
            right_done,
            left_done,
        )?;
        return Ok(match res {
            Left((right_df, left_df)) => Left((left_df, right_df)),
            Right(side) => Right(side.flip()),
        });
    } else {
        find_mergeable_inner(
            left,
            right,
            &params.left,
            &params.right,
            params,
            left_done,
            right_done,
        )
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
        dbg!(left_done);
        dbg!(right_done);
        dbg!(&left);
        dbg!(&right);

        if left_done && left.is_empty() && right_done && right.is_empty() {
            return dbg!(Ok(Right(NeedMore::Finished)));
        } else if left_done && left.is_empty() {
            // [amber] Also done here in case of LEFT join, right?
            return dbg!(Ok(Left((
                DataFrame::empty_with_schema(&left_params.ir_schema),
                right.pop_front().unwrap(),
            ))));
        } else if right_done && right.is_empty() {
            return dbg!(Ok(Left((
                left.pop_front().unwrap(),
                DataFrame::empty_with_schema(&right_params.ir_schema),
            ))));
        } else if left_done && left.len() <= 1 && right_done && right.len() <= 1 {
            return dbg!(Ok(Left((
                left.pop_front().unwrap(),
                right.pop_front().unwrap(),
            ))));
        } else if left.is_empty() && !left_done {
            return dbg!(Ok(Right(NeedMore::Left)));
        } else if right.is_empty() && !right_done {
            return dbg!(Ok(Right(NeedMore::Right)));
        }

        let Some(left_df) = left.front() else {
            return dbg!(Ok(Right(NeedMore::Left)));
        };
        let Some(right_df) = right.front() else {
            return dbg!(Ok(Right(NeedMore::Right)));
        };

        if left_df.is_empty() {
            left.pop_front();
            continue;
        }
        if right_df.is_empty() {
            right.pop_front();
            continue;
        }

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
            return dbg!(Ok(Right(NeedMore::Left)));
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
            return dbg!(Ok(Right(NeedMore::Right)));
        }

        dbg!(left_first_incomplete);
        dbg!(right_first_incomplete);

        let left_last_completed_val = left_keys.get(left_first_incomplete - 1)?;
        let right_last_completed_val = right_keys.get(right_first_incomplete - 1)?;
        dbg!(&left_last_completed_val);
        dbg!(&right_last_completed_val);
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

        dbg!(left_mergable_until);
        dbg!(right_mergable_until);

        if left_mergable_until == 0 && left.len() > 1 {
            vstack_head(left, left_params, params);
            continue;
        } else if right_mergable_until == 0 && right.len() > 1 {
            vstack_head(right, right_params, params);
            continue;
        } else if left_mergable_until == 0 && right_mergable_until == 0 {
            return dbg!(Ok(Right(NeedMore::Both)));
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
        return dbg!(Ok(Left((left_df, right_df))));
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
    dbg!(&unmerged[0], &unmerged[1]);
    let mut df = unmerged.pop_front().unwrap();
    let col_idx = df.get_column_index(&sp.key_col).unwrap();
    df.vstack_mut_owned(unmerged.pop_front().unwrap()).unwrap();
    unsafe {
        let col = &mut df.get_columns_mut()[col_idx];
        col.set_sorted_flag(is_sorted);
    };
    unmerged.push_front(df);
}

fn append_key_column(
    df: &mut DataFrame,
    sp: &SideParams,
    broadcast_nulls: bool,
) -> PolarsResult<()> {
    debug_assert_eq!(*df.schema(), sp.input_schema);
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

async fn buffer_unmerged_from_pipe(
    port: Option<&mut PortReceiver>,
    unmerged: &mut VecDeque<DataFrame>,
    sp: &SideParams,
    params: &MergeJoinParams,
) {
    let Some(port) = port else {
        return;
    };
    let Ok(morsel) = port.recv().await else {
        return;
    };
    morsel.source_token().stop();
    let mut df = morsel.into_df();
    append_key_column(&mut df, sp, false).unwrap();
    unmerged.push_back(df);

    while let Ok(morsel) = port.recv().await {
        let mut df = morsel.into_df();
        append_key_column(&mut df, sp, false).unwrap();
        unmerged.push_back(df);
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

fn keys_ge(left: &AnyValue, right: &AnyValue, params: &MergeJoinParams) -> bool {
    !keys_lt(left, right, params)
}
