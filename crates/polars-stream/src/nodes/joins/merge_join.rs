use std::collections::VecDeque;

use arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_ops::prelude::*;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::io_sources::multi_scan::reader_interface::output;
use crate::pipe::{PortReceiver, RecvPort, SendPort};

// Do row encoding in lowering and then we will have one key column per input
// Proably Gijs has already done something like this somewhere

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

#[derive(Default)]
struct MergeJoinParams {
    left_input_schema: Arc<Schema>,
    right_input_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    left_key: PlSmallStr,
    right_key: PlSmallStr,
    args: JoinArgs,
    random_state: PlRandomState,
}

#[derive(Default)]
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
        left_key: PlSmallStr,
        right_key: PlSmallStr,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        let state = MergeJoinState::Running;
        let params = MergeJoinParams {
            left_input_schema,
            right_input_schema,
            output_schema,
            left_key,
            right_key,
            args,
            random_state: PlRandomState::default(),
        };
        Ok(MergeJoinNode {
            state,
            params,
            ..Default::default()
        })
    }

    fn compute_join(
        &self,
        left: &mut DataFrame,
        right: &mut DataFrame,
        state: &StreamingExecutionState,
    ) -> PolarsResult<DataFrame> {
        left.rechunk_mut();
        right.rechunk_mut();

        // TODO: [amber] Do a cardinality sketch to estimate the output size?
        let left_key_col = left.column(&self.params.left_key).unwrap();
        let right_key_col = right.column(&self.params.right_key).unwrap();

        let left = self.drop_non_output_columns(left)?;
        let right = self.drop_non_output_columns(right)?;

        let mut left_build = DataFrameBuilder::new(left.schema().clone());
        let mut right_build = DataFrameBuilder::new(right.schema().clone());

        let mut left_gather_idxs = Vec::new();
        let mut right_gather_idxs = Vec::new();
        for (idxl, left_key) in left_key_col.phys_iter().enumerate() {
            let mut matched = false;
            for (idxr, right_key) in right_key_col.phys_iter().enumerate() {
                if keys_equal(&left_key, &right_key, self.params.args.nulls_equal) {
                    matched = true;
                    left_gather_idxs.push(idxl as IdxSize);
                    right_gather_idxs.push(idxr as IdxSize);
                }
            }
            if !matched {
                left_gather_idxs.push(idxl as IdxSize);
                right_gather_idxs.push(IdxSize::MAX);
            }
        }

        left_build.opt_gather_extend(&left, &left_gather_idxs, ShareStrategy::Never);
        right_build.opt_gather_extend(&right, &right_gather_idxs, ShareStrategy::Never);
        let mut df_left = left_build.freeze();
        let df_right = right_build.freeze();
        df_left.hstack_mut(&df_right.get_columns())?;
        Ok(df_left)
    }

    fn drop_non_output_columns(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let mut drop_cols = PlHashSet::with_capacity(df.width());
        for col in df.get_column_names() {
            if !self.params.output_schema.contains(&col) {
                drop_cols.insert(col.clone());
            }
        }
        Ok(df.drop_many_amortized(&drop_cols))
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
                    match pop_mergable(
                        &mut self.left_unmerged,
                        &self.params.left_key,
                        &mut self.right_unmerged,
                        &self.params.right_key,
                        flush,
                        &self.params,
                    )? {
                        Ok((mut left_chunk, mut right_chunk)) => {
                            // [amber] TODO: put this into a distributor instead of computing the join serially
                            let joined =
                                self.compute_join(&mut left_chunk, &mut right_chunk, state)?;
                            let source_token = SourceToken::new();
                            let morsel = Morsel::new(joined, self.seq, source_token);
                            send.send(morsel).await.unwrap(); // TODO [amber] Can this error be handled?
                            self.seq = self.seq.successor();
                        },
                        Err(Side::Left) if self.state == Flushing => {
                            self.right_unmerged.clear();
                            return Ok(());
                        },
                        Err(Side::Left) if recv_left.is_some() => {
                            // Need more left data
                            let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_right.as_mut(),
                                    &mut self.right_unmerged,
                                )
                                .await;
                                break;
                            };
                            let df = m.into_df();
                            self.left_unmerged.push_back(df);
                        },
                        Err(Side::Right) if recv_right.is_some() => {
                            // Need more right data
                            let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                                buffer_unmerged_from_pipe(
                                    recv_left.as_mut(),
                                    &mut self.left_unmerged,
                                )
                                .await;
                                break;
                            };
                            let df = m.into_df();
                            self.right_unmerged.push_back(df);
                        },
                        Err(_) => break,
                    }
                }
                Ok(())
            }));
        }
    }
}

fn pop_mergable(
    left: &mut VecDeque<DataFrame>,
    left_key: &str,
    right: &mut VecDeque<DataFrame>,
    right_key: &str,
    flush: bool,
    params: &MergeJoinParams,
) -> PolarsResult<Result<(DataFrame, DataFrame), Side>> {
    fn vstack_head(unmerged: &mut VecDeque<DataFrame>) {
        let mut df = unmerged.pop_front().unwrap();
        df.vstack_mut_owned(unmerged.pop_front().unwrap()).unwrap();
        unmerged.push_front(df);
    }

    // TODO: [amber] LEFT HERE
    // Start handling nulls, or implement the different kinds of joins.
    // Right join can be done by swapping left and right inputs to this function.
    // But I would really like to do that in lowering instead of putting a bunch
    // of bookkeeping code here.

    // TODO: [amber] Eat all the None keys at the front

    loop {
        if left.is_empty() && flush {
            right.clear();
            return Ok(Err(Side::Left));
        }
        if left.is_empty() {
            return Ok(Err(Side::Left));
        }
        if right.is_empty() && flush {
            return Ok(Ok((left.pop_front().unwrap(), DataFrame::empty())));
        }
        if right.is_empty() {
            return Ok(Err(Side::Right));
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

        let left_key_val = left_df.column(left_key)?;
        let right_key_val = right_df.column(right_key)?;
        let left_key_min = left_key_val.get(0)?;
        let left_key_max = left_key_val.get(left_key_val.len() - 1)?;
        let right_key_max = right_key_val.get(right_key_val.len() - 1)?;
        let right_key_min = right_key_val.get(0)?;

        if left_key_min == left_key_max && left.len() >= 2 {
            vstack_head(left);
            continue;
        } else if right_key_min == right_key_max && right.len() >= 2 {
            vstack_head(right);
            continue;
        } else if left_key_min == left_key_max && !flush {
            return Ok(Err(Side::Left));
        } else if right_key_min == right_key_max && !flush {
            return Ok(Err(Side::Right));
        }

        let global_max = if left_key_max < right_key_max {
            left_key_max
        } else {
            right_key_max
        };
        let (left_cutoff, right_cutoff) = if flush {
            (
                find_item_offset(left_key_val, &global_max, SearchSortedSide::Right, false)? + 1,
                find_item_offset(right_key_val, &global_max, SearchSortedSide::Right, false)? + 1,
            )
        } else {
            (
                find_item_offset(left_key_val, &global_max, SearchSortedSide::Left, false)?,
                find_item_offset(right_key_val, &global_max, SearchSortedSide::Left, false)?,
            )
        };

        let left_df = left.pop_front().unwrap();
        let right_df = right.pop_front().unwrap();
        let (left_df, left_rest) = left_df.split_at(left_cutoff);
        let (right_df, right_rest) = right_df.split_at(right_cutoff);
        left.push_front(left_rest);
        right.push_front(right_rest);

        debug_assert!(left_df.height() > 0 || right_df.height() > 0);
        return Ok(Ok((left_df, right_df)));
    }
}

fn find_item_offset(
    column: &Column,
    search_value: &AnyValue,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<i64> {
    debug_assert!(column.get_flags().is_sorted_ascending());
    let s = column.as_materialized_series_maintain_scalar();
    let sv = Series::from_any_values(PlSmallStr::EMPTY, &[search_value.clone()], true)?;
    let pos = search_sorted(&s, &sv, side, descending).unwrap();
    Ok(pos.iter().next().unwrap_or(Some(0)).unwrap() as i64)
}

async fn buffer_unmerged_from_pipe(
    port: Option<&mut PortReceiver>,
    unmerged: &mut VecDeque<DataFrame>,
) {
    let Some(port) = port else {
        return;
    };
    let Ok(morsel) = port.recv().await else {
        return;
    };
    morsel.source_token().stop();
    unmerged.push_back(morsel.into_df());
    while let Ok(morsel) = port.recv().await {
        unmerged.push_back(morsel.into_df());
    }
}

fn keys_equal(left: &AnyValue, right: &AnyValue, nulls_equal: bool) -> bool {
    if left.is_null() && right.is_null() {
        nulls_equal
    } else {
        left == right
    }
}
