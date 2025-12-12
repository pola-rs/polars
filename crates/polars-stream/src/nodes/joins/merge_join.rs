use std::collections::VecDeque;
use std::mem::swap;
use std::ops::Not;

use arrow::Either;
// use arrow::Either::{Left as Either::Left, Right as Either::Right};
use arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_ops::prelude::*;
use polars_plan::prelude::*;
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools;
use polars_utils::parma::raw::Key;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::pipe::{PortReceiver, RecvPort, SendPort};

// Do row encoding in lowering and then we will have one key column per input
// Probably Gijs has already done something like this somewhere

// TODO: [amber] Use `_get_rows_encoded` (with `descending`) to encode the keys
// TODO: [amber] Consider extending the HashKeys to support sorted data
// TODO: [amber] Use a distributor to distribute the merge-join work to a pool of workers
// BUG: [amber] Currently the join may process sub-chunks and not produce "cartesian product"
// results when there are chunks of equal keys
// TODO: [amber] I think I want to have the inner join dispatch to another streaming
// equi join node rather than dispatching to in-memory engine

#[derive(Debug)]
enum Side {
    Left,
    Right,
}

impl Not for Side {
    type Output = Side;

    fn not(self) -> Self::Output {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }
}

#[derive(Debug)]
struct SideParams {
    input_schema: SchemaRef,
    ir_schema: SchemaRef,
    on: Vec<PlSmallStr>,
    key_col: PlSmallStr,
    sortedness: IRSorted,
}

#[derive(Debug)]
struct MergeJoinParams {
    left: SideParams,
    right: SideParams,
    output_schema: SchemaRef,
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
        let left = SideParams {
            input_schema: left_input_schema,
            ir_schema: left_ir_schema,
            on: left_on,
            key_col: left_key_col,
            sortedness: left_sortedness,
        };
        let right = SideParams {
            input_schema: right_input_schema,
            ir_schema: right_ir_schema,
            on: right_on,
            key_col: right_key_col,
            sortedness: right_sortedness,
        };
        let params = MergeJoinParams {
            left,
            right,
            output_schema,
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
        let is_right_join = *join_type == JoinType::Right;

        left.rechunk_mut();
        right.rechunk_mut();

        // TODO: [amber] Remove any non-output columns earlier to reduce the
        // amount of gathering as much as possible
        // TODO: [amber] Do a cardinality sketch to estimate the output size?

        let mut left_key = left.column(&self.params.left.key_col).unwrap();
        let mut right_key = right.column(&self.params.right.key_col).unwrap();

        if is_right_join {
            swap(&mut left_key, &mut right_key);
        }
        let mut left_gather_idxs = Vec::new();
        let mut right_gather_idxs = Vec::new();
        for (idxl, left_key) in left_key.as_materialized_series().iter().enumerate() {
            let mut matched = false;
            for (idxr, right_key) in right_key.as_materialized_series().iter().enumerate() {
                if keys_equal(&left_key, &right_key, self.params.args.nulls_equal) {
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
        if is_right_join {
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

        if matches!(self.state, Running | Flushing) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                loop {
                    let flush = self.state == Flushing;
                    match find_mergeable(
                        &mut self.left_unmerged,
                        &mut self.right_unmerged,
                        flush,
                        &self.params,
                    )? {
                        Either::Left((left_chunk, right_chunk)) => {
                            // [amber] TODO: put this into a distributor instead of computing the join serially
                            let joined = self.compute_join(left_chunk, right_chunk, state)?;
                            let source_token = SourceToken::new();
                            let morsel = Morsel::new(joined, self.seq, source_token);
                            dbg!(&morsel);
                            send.send(morsel).await.unwrap(); // TODO [amber] Can this error be handled?
                            self.seq = self.seq.successor();
                        },
                        Either::Right(Side::Left) if self.state == Flushing => {
                            self.right_unmerged.clear();
                            return Ok(());
                        },
                        Either::Right(Side::Left) if recv_left.is_some() => {
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
                            append_key_columns(&mut df, &self.params.left, &self.params)?;
                            self.left_unmerged.push_back(df);
                        },
                        Either::Right(Side::Right) if recv_right.is_some() => {
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
                            append_key_columns(&mut df, &self.params.right, &self.params)?;
                            self.right_unmerged.push_back(df);
                        },
                        Either::Right(_) => break,
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
    flush: bool,
    params: &MergeJoinParams,
) -> PolarsResult<Either<(DataFrame, DataFrame), Side>> {
    if params.args.how == JoinType::Right {
        let res = find_mergeable_inner(right, left, &params.right, &params.left, flush)?;
        return Ok(match res {
            Either::Left((right_df, left_df)) => Either::Left((left_df, right_df)),
            Either::Right(side) => Either::Right(!side),
        });
    } else {
        find_mergeable_inner(left, right, &params.left, &params.right, flush)
    }
}

fn find_mergeable_inner(
    left: &mut VecDeque<DataFrame>,
    right: &mut VecDeque<DataFrame>,
    left_params: &SideParams,
    right_params: &SideParams,
    flush: bool,
) -> PolarsResult<Either<(DataFrame, DataFrame), Side>> {
    loop {
        if left.is_empty() && flush {
            right.clear();
            return Ok(Either::Right(Side::Left));
        }
        if left.is_empty() {
            return Ok(Either::Right(Side::Left));
        }
        if right.is_empty() && flush {
            return Ok(Either::Left((
                left.pop_front().unwrap(),
                DataFrame::empty_with_schema(&*right_params.ir_schema),
            )));
        }
        if right.is_empty() {
            return Ok(Either::Right(Side::Right));
        }

        let left_df = &left[0];
        let right_df = &right[0];

        if left_df.height() == 0 {
            left.pop_front();
            continue;
        } else if right_df.height() == 0 {
            right.pop_front();
            continue;
        }

        let left_key_val = left_df
            .column(&left_params.key_col)?
            .as_materialized_series_maintain_scalar();
        let right_key_val = right_df
            .column(&right_params.key_col)?
            .as_materialized_series_maintain_scalar();
        let left_key_min = left_key_val.slice(0, 1);
        let left_key_max = left_key_val.slice(left_key_val.len() as i64 - 1, 1);
        let right_key_min = right_key_val.slice(0, 1);
        let right_key_max = right_key_val.slice(right_key_val.len() as i64 - 1, 1);

        if left_key_min == left_key_max && left.len() >= 2 {
            vstack_head(left);
            continue;
        } else if right_key_min == right_key_max && right.len() >= 2 {
            vstack_head(right);
            continue;
        } else if left_key_min == left_key_max && !flush {
            return Ok(Either::Right(Side::Left));
        } else if right_key_min == right_key_max && !flush {
            return Ok(Either::Right(Side::Right));
        }

        let global_max = if left_key_max.get(0)? < right_key_max.get(0)? {
            left_key_max
        } else {
            right_key_max
        };
        let (left_cutoff, right_cutoff) = if flush {
            (
                find_item_offset(&left_key_val, &global_max, SearchSortedSide::Right, false)? + 1,
                find_item_offset(&right_key_val, &global_max, SearchSortedSide::Right, false)? + 1,
            )
        } else {
            (
                find_item_offset(&left_key_val, &global_max, SearchSortedSide::Left, false)?,
                find_item_offset(&right_key_val, &global_max, SearchSortedSide::Left, false)?,
            )
        };

        let left_df = left.pop_front().unwrap();
        let right_df = right.pop_front().unwrap();
        let (left_df, left_rest) = left_df.split_at(left_cutoff);
        let (right_df, right_rest) = right_df.split_at(right_cutoff);
        left.push_front(left_rest);
        right.push_front(right_rest);

        debug_assert!(left_df.height() > 0 || right_df.height() > 0);
        debug_assert!(*left_df.schema() == left_params.ir_schema);
        debug_assert!(*right_df.schema() == right_params.ir_schema);
        return Ok(Either::Left((left_df, right_df)));
    }
}

fn vstack_head(unmerged: &mut VecDeque<DataFrame>) {
    let mut df = unmerged.pop_front().unwrap();
    df.vstack_mut_owned(unmerged.pop_front().unwrap()).unwrap();
    unmerged.push_front(df);
}

fn find_item_offset(
    s: &Series,
    search_value: &Series,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<i64> {
    // debug_assert!(s.get_flags().is_sorted_ascending());
    let pos = search_sorted(&s, &search_value, side, descending).unwrap();
    Ok(pos.iter().next().unwrap_or(Some(0)).unwrap() as i64)
}

fn keys_equal(left: &AnyValue, right: &AnyValue, nulls_equal: bool) -> bool {
    if left.is_null() && right.is_null() {
        nulls_equal
    } else {
        left == right
    }
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
    append_key_columns(&mut df, sp, params).unwrap();
    unmerged.push_back(df);

    while let Ok(morsel) = port.recv().await {
        let mut df = morsel.into_df();
        append_key_columns(&mut df, sp, params).unwrap();
        unmerged.push_back(df);
    }
}

fn append_key_columns(
    df: &mut DataFrame,
    sp: &SideParams,
    params: &MergeJoinParams,
) -> PolarsResult<()> {
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
            &vec![false; columns.len()], // TODO: [amber]
            &vec![false; columns.len()], // TODO: [amber]
            !params.args.nulls_equal,
        )?
        .into_column();
        df.hstack_mut(&[key])?;
    }
    Ok(())
}
