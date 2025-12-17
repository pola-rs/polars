use std::collections::{BTreeMap, VecDeque};
use std::mem::swap;

use arrow::array::builder::ShareStrategy;
use arrow::bitmap::MutableBitmap;
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
use crate::async_primitives::morsel_linearizer::MorselLinearizer;
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
    key_nulls_last: bool,
    args: JoinArgs,
}

#[derive(Debug)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: VecDeque<DataFrame>,
    right_unmerged: VecDeque<DataFrame>,
    unmatched: BTreeMap<MorselSeq, DataFrame>,
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

        dbg!(&self.state);
        if recv_left.is_none() && recv_right.is_none() {
            assert!(self.state >= FlushInputBuffers);
        }

        if matches!(self.state, Running | FlushInputBuffers) {
            let send = send_ports[0].take().unwrap().parallel();
            let (mut distributor, dist_recv) =
                distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
            let (mut unmatched_linearizer, unmatched_send) =
                MorselLinearizer::new(dist_recv.len(), 1);

            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let source_token = SourceToken::new();

                loop {
                    match find_mergeable(
                        left_unmerged,
                        right_unmerged,
                        recv_left.is_none(),
                        recv_right.is_none(),
                        params,
                    )? {
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
                                    &params.right,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            add_key_column(&mut df, &params.left, false)?;
                            left_unmerged.push_back(df);
                        },
                        Right(NeedMore::Right | NeedMore::Both) if recv_right.is_some() => {
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_left.as_mut(),
                                    left_unmerged,
                                    &params.left,
                                )
                                .await;
                                break;
                            };
                            let mut df = m.into_df();
                            add_key_column(&mut df, &params.right, false)?;
                            right_unmerged.push_back(df);
                        },

                        Right(NeedMore::Finished) => {
                            dbg!("breaking because we are finished");
                            break;
                        },
                        Right(other) => {
                            unreachable!("unexpected NeedMore value: {other:?}");
                        },
                    }
                }
                Ok(())
            }));

            join_handles.extend(dist_recv.into_iter().zip(send).zip(unmatched_send).map(
                |((mut recv, mut send), mut unmatched_inserter)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        while let Ok((morsel, right)) = recv.recv().await {
                            let (left, seq, source_token, _wt) = morsel.into_inner();
                            let (matched, unmatched) = compute_join(left, right, params)?;
                            if !matched.is_empty() {
                                let morsel = Morsel::new(matched, seq, source_token.clone());
                                if send.send(morsel).await.is_err() {
                                    break;
                                }
                            }
                            if !unmatched.is_empty() {
                                let morsel = Morsel::new(unmatched, seq, source_token.clone());
                                if unmatched_inserter.insert(morsel).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Ok(())
                    })
                },
            ));

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Some(morsel) = unmatched_linearizer.get().await {
                    let (df, seq, _st, _) = morsel.into_inner();
                    if let Some(append_to) = unmatched.get_mut(&seq) {
                        append_to.vstack_mut_owned(df)?;
                    } else {
                        unmatched.insert(seq, df);
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
                while let Some((btree_key, df)) = unmatched.pop_first() {
                    let morsel = Morsel::new(df, *seq, SourceToken::new());
                    if let Err(morsel) = send.send(morsel).await {
                        unmatched.insert(btree_key, morsel.into_df());
                        break;
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
    p: &MergeJoinParams,
) -> PolarsResult<Either<(DataFrame, DataFrame), NeedMore>> {
    if p.args.how == JoinType::Right {
        let ok = find_mergeable_inner(right, left, &p.right, &p.left, p, right_done, left_done)?;
        return Ok(match ok {
            Left((right_df, left_df)) => Left((left_df, right_df)),
            Right(side) => Right(side.flip()),
        });
    } else {
        find_mergeable_inner(left, right, &p.left, &p.right, p, left_done, right_done)
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
    dbg!(&left, &right);
    loop {
        if left_done && left.is_empty() && right_done && right.is_empty() {
            return Ok(Right(NeedMore::Finished));
        } else if left_done && left.is_empty() && !right_params.emit_unmatched {
            // We will never match on the remaining right keys
            right.clear();
            return Ok(Right(NeedMore::Finished));
        } else if right_done && right.is_empty() && !left_params.emit_unmatched {
            // We will never match on the remaining left keys
            left.clear();
            return Ok(Right(NeedMore::Finished));
        } else if left_done && left.is_empty() {
            return Ok(Left((
                DataFrame::empty_with_schema(&left_params.ir_schema),
                right.pop_front().unwrap(),
            )));
        } else if right_done && right.is_empty() {
            return Ok(Left((
                left.pop_front().unwrap(),
                DataFrame::empty_with_schema(&right_params.ir_schema),
            )));
        } else if left_done && left.len() <= 1 && right_done && right.len() <= 1 {
            return Ok(Left((
                left.pop_front().unwrap(),
                right.pop_front().unwrap(),
            )));
        } else if left.is_empty() && !left_done {
            return Ok(Right(NeedMore::Left));
        } else if right.is_empty() && !right_done {
            return Ok(Right(NeedMore::Right));
        }

        let Some(left_df) = left.front() else {
            return Ok(Right(NeedMore::Left));
        };
        let Some(right_df) = right.front() else {
            return Ok(Right(NeedMore::Right));
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
            return Ok(Right(NeedMore::Left));
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
            return Ok(Right(NeedMore::Right));
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
            return Ok(Right(NeedMore::Both));
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
        return Ok(Left((left_df, right_df)));
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
) -> PolarsResult<(DataFrame, DataFrame)> {
    let mut left_sp = &params.left;
    let mut right_sp = &params.right;
    let right_is_build = matches!(
        params.args.maintain_order,
        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft,
    );

    // Add row-encoded key columns if needed
    if params.left.on.len() > 1 && !params.args.nulls_equal {
        left.drop_in_place(&params.left.key_col)?;
        add_key_column(&mut left, &params.left, true)?;
    }
    if params.right.on.len() > 1 && !params.args.nulls_equal {
        right.drop_in_place(&params.right.key_col)?;
        add_key_column(&mut right, &params.right, true)?;
    }

    left.rechunk_mut();
    right.rechunk_mut();

    // TODO: [amber] Remove any non-output columns earlier to reduce the
    // amount of gathering as much as possible

    let mut left_key = left.column(&params.left.key_col).unwrap();
    let mut right_key = right.column(&params.right.key_col).unwrap();
    if right_is_build {
        swap(&mut left_key, &mut right_key);
        swap(&mut left_sp, &mut right_sp);
    }
    let mut right_matched = MutableBitmap::from_len_zeroed(right_key.len());
    let mut left_gather = Vec::new();
    let mut right_gather = Vec::new();
    let mut right_unmatched_gather = Vec::new();
    let mut skip_ahead = 0;
    for (idxl, left_keyval) in left_key.as_materialized_series().iter().enumerate() {
        let mut matched = false;
        for idxr in skip_ahead..right_key.len() {
            let right_keyval = right_key.get(idxr).unwrap();
            let both_valid = !left_keyval.is_null() && !right_keyval.is_null();
            if keys_eq(&left_keyval, &right_keyval, &params) {
                matched = true;
                right_matched.set(idxr, true);
                left_gather.push(idxl as IdxSize);
                right_gather.push(idxr as IdxSize);
            } else if both_valid && keys_lt(&right_keyval, &left_keyval, params) {
                skip_ahead = idxr;
            } else if both_valid && keys_gt(&right_keyval, &left_keyval, params) {
                break;
            }
        }
        if left_sp.emit_unmatched && !matched {
            left_gather.push(idxl as IdxSize);
            right_gather.push(IdxSize::MAX);
        }
    }
    if right_sp.emit_unmatched {
        for (idxr, _) in right_matched.iter().enumerate().filter(|(_, m)| !m) {
            right_unmatched_gather.push(idxr as IdxSize);
        }
    }

    let mut left_unmatched_gather = vec![IdxSize::MAX; right_unmatched_gather.len()];
    if right_is_build {
        swap(&mut left_gather, &mut right_gather);
        swap(&mut left_unmatched_gather, &mut right_unmatched_gather);
        swap(&mut left_sp, &mut right_sp);
    }

    // Remove the added row-encoded key columns
    if params.left.on.len() > 1 {
        left = left.drop(&params.left.key_col).unwrap();
        right = right.drop(&params.right.key_col).unwrap();
    }

    let mut df_main = Default::default();
    let mut df_unmatched = Default::default();
    for (df, lg, rg) in [
        (&mut df_main, left_gather, right_gather),
        (
            &mut df_unmatched,
            left_unmatched_gather,
            right_unmatched_gather,
        ),
    ] {
        let mut left_build = DataFrameBuilder::new(left.schema().clone());
        let mut right_build = DataFrameBuilder::new(right.schema().clone());
        left_build.opt_gather_extend(&left, &lg, ShareStrategy::Never);
        right_build.opt_gather_extend(&right, &rg, ShareStrategy::Never);
        let mut left = left_build.freeze();
        let mut right = right_build.freeze();

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
    }

    Ok((df_main, df_unmatched))
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

fn add_key_column(df: &mut DataFrame, sp: &SideParams, broadcast_nulls: bool) -> PolarsResult<()> {
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
) {
    let Some(port) = port else {
        return;
    };
    let Ok(morsel) = port.recv().await else {
        return;
    };
    morsel.source_token().stop();
    let mut df = morsel.into_df();
    add_key_column(&mut df, sp, false).unwrap();
    unmerged.push_back(df);

    while let Ok(morsel) = port.recv().await {
        let mut df = morsel.into_df();
        add_key_column(&mut df, sp, false).unwrap();
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
