use std::collections::VecDeque;

use arrow::legacy::kernels::sorted_join::left;
use futures::SinkExt;
use polars_compute::gather::binary;
use polars_core::prelude::*;
use polars_expr::hash_keys::HashKeys;
use polars_expr::state::ExecutionState;
use polars_ops::prelude::*;
use polars_plan::dsl::Expr;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::joins::equi_join::compute_payload_selector;
use crate::pipe::{RecvPort, SendPort};

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
    left_key: PlSmallStr,
    right_key: PlSmallStr,
    args: JoinArgs,
    random_state: PlRandomState,
}

type BufferChunk = (DataFrame, DataFrame); // (keys, payload)

#[derive(Default)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: VecDeque<BufferChunk>,
    right_unmerged: VecDeque<BufferChunk>,
    seq: MorselSeq,
}

#[derive(Default, PartialEq, Eq)]
enum MergeJoinState {
    #[default]
    Running,
    Done,
}

impl MergeJoinNode {
    pub fn new(left_key: PlSmallStr, right_key: PlSmallStr, args: JoinArgs) -> PolarsResult<Self> {
        let state = MergeJoinState::Running;
        let params = MergeJoinParams {
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
        }

        if recv[0] == PortState::Done
            && self.left_unmerged.is_empty()
            && recv[1] == PortState::Done
            && self.right_unmerged.is_empty()
        {
            self.state = MergeJoinState::Done;
        }

        match self.state {
            MergeJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
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
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        let mut recv_left = recv_ports[0].take().map(RecvPort::serial);
        let mut recv_right = recv_ports[1].take().map(RecvPort::serial);
        let mut send = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            loop {
                match dbg!(pop_mergable(
                    &mut self.left_unmerged,
                    &mut self.right_unmerged
                )?) {
                    Ok(((left_payload, left_keys), (right_payload, right_keys))) => {
                        let left_df = left_payload.hstack(left_keys.get_columns())?;
                        let right_df = right_payload.hstack(right_keys.get_columns())?;
                        let left_on = left_keys
                            .get_columns()
                            .iter()
                            .map(|c| c.name().clone())
                            .collect::<Vec<_>>();
                        let right_on = right_keys
                            .get_columns()
                            .iter()
                            .map(|c| c.name().clone())
                            .collect::<Vec<_>>();
                        dbg!(&left_on, &right_on);
                        let joined = DataFrame::join(
                            &left_df,
                            &right_df,
                            left_on,
                            right_on,
                            self.params.args.clone(),
                            None,
                        )?;
                        dbg!(&joined);
                        let m = Morsel::new(joined, self.seq, SourceToken::new());
                        send.send(m).await.unwrap();
                        self.seq = self.seq.successor();
                    },
                    Err(Side::Left) if recv_left.is_some() => {
                        // Need more left data
                        let Ok(m) = recv_left.as_mut().unwrap().recv().await else {
                            break;
                        };
                        let df = m.into_df();
                        dbg!(&df);
                        // let keys =
                        //     select_keys(&df, &self.params, &state.in_memory_exec_state).await?;
                        // self.left_unmerged.push_back((df, keys));
                    },
                    Err(Side::Right) if recv_right.is_some() => {
                        // Need more right data
                        let Ok(m) = recv_right.as_mut().unwrap().recv().await else {
                            break;
                        };
                        let df = m.into_df();
                        dbg!(&df);
                        // let keys =
                        //     select_keys(&df, &self.params, &state.in_memory_exec_state).await?;
                        // self.right_unmerged.push_back((df, keys));
                    },
                    Err(_) => {
                        eprintln!("out of data");
                        dbg!(&self.left_unmerged, &self.right_unmerged);
                        self.state = MergeJoinState::Done;
                        break;
                    },
                }
            }
            // TODO [amber] Should we now tell the other sender to stop sending,
            // and then eat all of the remaining data in the other stream?

            Ok(())
        }));
    }
}

fn pop_mergable(
    left: &mut VecDeque<BufferChunk>,
    right: &mut VecDeque<BufferChunk>,
) -> PolarsResult<Result<(BufferChunk, BufferChunk), Side>> {
    if left.is_empty() {
        return Ok(Err(Side::Left));
    }
    if right.is_empty() {
        return Ok(Err(Side::Right));
    }

    loop {
        let (mut left_df, mut left_keys) = left.pop_front().unwrap();
        let (mut right_df, mut right_keys) = right.pop_front().unwrap();

        let left_key_last = left_keys.tail(Some(1));
        let right_key_last = right_keys.tail(Some(1));
        assert!(!left_key_last.is_empty() && !right_key_last.is_empty());
        let left_lt_last_mask = left_keys.get_columns()[0].lt(&left_key_last.get_columns()[0])?; // TODO: [amber] multiple key columns
        let right_lt_last_mask = right_keys.get_columns()[0].lt(&right_key_last.get_columns()[0])?; // TODO: [amber] multiple key columns

        let left_end_offset = left_lt_last_mask
            .downcast_as_array()
            .values()
            .leading_ones();
        let right_end_offset = right_lt_last_mask
            .downcast_as_array()
            .values()
            .leading_ones();

        dbg!(&left_keys, &right_keys);
        dbg!(&left_end_offset, &right_end_offset);

        let left_key_columns = left_keys.get_columns();
        let right_key_columns = right_keys.get_columns();
        let left_tail = left_keys.slice((left_end_offset - 1) as i64, 1);
        let right_tail = right_keys.slice((right_end_offset - 1) as i64, 1);

        let max_key = if left_tail.get_columns()[0].get(0)? <= right_tail.get_columns()[0].get(0)? {
            left_tail.get_columns()
        } else {
            right_tail.get_columns()
        };
        dbg!(&max_key);

        let left_overlap_end = find_item_offset(left_key_columns, max_key)?;
        let right_overlap_end = find_item_offset(right_key_columns, max_key)?;
        let left_head = left_keys.head(Some(1));
        let right_head = right_keys.head(Some(1));
        let left_search_value = left_head.get_columns();
        let right_search_value = right_head.get_columns();
        let left_first_chunk_end = find_item_offset(left_key_columns, left_search_value)?;
        let right_first_chunk_end = find_item_offset(right_key_columns, right_search_value)?;
        debug_assert!(!(left_overlap_end == 0 && right_overlap_end == 0));

        // We may need to wait until we have a complete chunk of equal keys on the "smaller" side
        if right_overlap_end == 0 || left_end_offset == 0 {
            let Some((left_ext_df, left_ext_keys)) = left.pop_front() else {
                left.push_front((left_df, left_keys));
                right.push_front((right_df, right_keys));
                return Ok(Err(Side::Left));
            };
            left_df.vstack_mut(&left_ext_df)?;
            left_keys.vstack_mut(&left_ext_keys)?;
            left.push_front((left_df, left_keys));
            right.push_front((right_df, right_keys));
            continue;
        } else if left_overlap_end == 0 || right_end_offset == 0 {
            let Some((right_ext_df, right_ext_keys)) = right.pop_front() else {
                left.push_front((left_df, left_keys));
                right.push_front((right_df, right_keys));
                return Ok(Err(Side::Right));
            };
            right_df.vstack_mut(&right_ext_df)?;
            right_keys.vstack_mut(&right_ext_keys)?;
            left.push_front((left_df, left_keys));
            right.push_front((right_df, right_keys));
            continue;
        }

        let left_split_offset = left_overlap_end.min(left_first_chunk_end) as i64;
        let right_split_offset = right_overlap_end.min(right_first_chunk_end) as i64;
        debug_assert!(0 < left_split_offset && left_split_offset < left_keys.height() as i64);
        debug_assert!(0 < right_split_offset && right_split_offset < right_keys.height() as i64);
        let (left_df_mergeable, left_df_remain) = left_df.split_at(left_split_offset);
        let (left_keys_mergeable, left_keys_remain) = left_keys.split_at(left_split_offset);
        let (right_df_mergeable, right_df_remain) = right_df.split_at(right_split_offset);
        let (right_keys_mergeable, right_keys_remain) = right_keys.split_at(right_split_offset);

        if left_keys_remain.height() > 0 {
            left.push_front((left_df_remain, left_keys_remain));
        }
        if right_keys_remain.height() > 0 {
            right.push_front((right_df_remain, right_keys_remain));
        }

        return Ok(Ok((
            (left_df_mergeable, left_keys_mergeable),
            (right_df_mergeable, right_keys_mergeable),
        )));
    }
}

fn find_item_offset(columns: &[Column], search_value: &[Column]) -> PolarsResult<IdxSize> {
    let mut descending_iter = columns.iter().map(|c| c.get_flags().is_sorted_descending());
    assert!(columns.len() == search_value.len() && columns.len() == descending_iter.len());

    let mut upper_bound = columns[0].len() as IdxSize;
    for (idx, column) in columns.iter().enumerate() {
        let descending = descending_iter.next().unwrap();
        let s = column
            .slice(0, (upper_bound) as usize)
            .as_materialized_series_maintain_scalar();
        let sv = &search_value[idx]
            .as_materialized_series_maintain_scalar()
            .slice(0, (upper_bound) as usize);
        let ca_hi = search_sorted(&s, sv, SearchSortedSide::Right, descending).unwrap();
        upper_bound = ca_hi.iter().next().unwrap_or(Some(0)).unwrap();
    }
    Ok(upper_bound)
}
