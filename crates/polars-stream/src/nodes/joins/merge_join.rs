use std::collections::VecDeque;

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

#[derive(Default)]
struct MergeJoinParams {
    left_key_schema: Arc<Schema>,
    left_key_selectors: Vec<StreamExpr>,
    #[allow(dead_code)]
    right_key_schema: Arc<Schema>,
    right_key_selectors: Vec<StreamExpr>,
    left_payload_select: Vec<Option<PlSmallStr>>,
    right_payload_select: Vec<Option<PlSmallStr>>,
    left_payload_schema: Arc<Schema>,
    right_payload_schema: Arc<Schema>,
    args: JoinArgs,
    random_state: PlRandomState,
}

async fn select_keys(
    df: &DataFrame,
    key_selectors: &[StreamExpr],
    params: &MergeJoinParams,
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    let mut key_columns = Vec::new();
    for selector in key_selectors {
        key_columns.push(selector.evaluate(df, state).await?.into_column());
    }
    DataFrame::new_with_broadcast_len(key_columns, df.height())
}

type BufferChunk = (DataFrame, DataFrame); // (keys, payload)

#[derive(Default)]
pub struct MergeJoinNode {
    state: MergeJoinState,
    params: MergeJoinParams,
    left_unmerged: VecDeque<BufferChunk>,
    right_unmerged: VecDeque<BufferChunk>,
}

#[derive(Default, PartialEq, Eq)]
enum MergeJoinState {
    #[default]
    Running,
    Done,
}

impl MergeJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        left_key_schema: Arc<Schema>,
        right_key_schema: Arc<Schema>,
        unique_key_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        let left_payload_select = compute_payload_selector(
            &left_input_schema,
            &right_input_schema,
            &left_key_schema,
            &right_key_schema,
            true,
            &args,
        )?;
        let right_payload_select = compute_payload_selector(
            &right_input_schema,
            &left_input_schema,
            &right_key_schema,
            &left_key_schema,
            false,
            &args,
        )?;
        let state = MergeJoinState::Running;
        let params = MergeJoinParams {
            left_key_schema,
            left_key_selectors,
            right_key_schema,
            right_key_selectors,
            left_payload_select,
            right_payload_select,
            left_payload_schema: unique_key_schema.clone(),
            right_payload_schema: unique_key_schema,
            args,
            random_state: PlRandomState::default(),
        };
        Ok(MergeJoinNode {
            state,
            params,
            left_unmerged: VecDeque::new(),
            right_unmerged: VecDeque::new(),
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
                let left_has_data = !self.left_unmerged.is_empty() || recv_left.is_some();
                let right_has_data = !self.right_unmerged.is_empty() || recv_right.is_some();

                // while left_has_data && right_has_data || (recv_left.is_some() && recv_right.is_some()) {
                //     // First merge all umerged data
                // }

                if let Some(((left_df, left_keys), (right_df, right_keys))) = dbg!(pop_mergable(
                    &mut self.left_unmerged,
                    &mut self.right_unmerged
                )?) {
                    // Perform merge join on left_df and right_df
                    send.send(Morsel::new(
                        left_df.clone(), // [amber] Placeholder, replace with actual joined DataFrame
                        MorselSeq::default(),
                        SourceToken::default(),
                    ))
                    .await
                    .unwrap();
                } else {
                    // Absorb some data
                    let left_m = recv_left.as_mut().unwrap().recv().await.unwrap();
                    let right_m = recv_right.as_mut().unwrap().recv().await.unwrap();
                    let df_left = left_m.df();
                    let df_right = right_m.df();
                    let left_keys = select_keys(
                        df_left,
                        &self.params.left_key_selectors,
                        &self.params,
                        &state.in_memory_exec_state,
                    )
                    .await?;
                    let right_keys = select_keys(
                        df_right,
                        &self.params.right_key_selectors,
                        &self.params,
                        &state.in_memory_exec_state,
                    )
                    .await?;

                    self.left_unmerged.push_back((df_left.clone(), left_keys));
                    self.right_unmerged
                        .push_back((df_right.clone(), right_keys));
                };
            }

            Ok(())
        }));

        // [amber] LEFT HERE
        //
        // There is no polars-ops merge-join kernel.  But also,
        // I do not really know whether we implement this streaming
        // node by composing merge-join in indivudual morsels.
        //
        // That's what you'll have to figure out first,
        // probably on a piece of paper.
        //
        // Good luck Amber! <3

        // send.send(msg).await?;
        // }
    }
    //     Ok(())
    // }));
}

fn pop_mergable(
    left: &mut VecDeque<BufferChunk>,
    right: &mut VecDeque<BufferChunk>,
) -> PolarsResult<Option<(BufferChunk, BufferChunk)>> {
    if left.is_empty() || right.is_empty() {
        return Ok(None);
    }

    loop {
        let (mut left_df, mut left_keys) = left.pop_front().unwrap();
        let (mut right_df, mut right_keys) = right.pop_front().unwrap();
        let left_key_columns = left_keys.get_columns();
        let right_key_columns = right_keys.get_columns();
        let left_tail = left_keys.tail(Some(1));
        let right_tail = right_keys.tail(Some(1));
        let left_search_value = right_tail.get_columns();
        let right_search_value = left_tail.get_columns();
        let left_overlap_end = find_item_offset(left_key_columns, left_search_value)?;
        let right_overlap_end = find_item_offset(right_key_columns, right_search_value)?;
        debug_assert!(!(left_overlap_end == 0 && right_overlap_end == 0));

        // We may need to wait until we have a complete chunk of equal keys on the "smaller" side
        if right_overlap_end == 0 {
            let Some((left_ext_df, left_ext_keys)) = left.pop_front() else {
                left.push_front((left_df, left_keys));
                right.push_front((right_df, right_keys));
                return Ok(None);
            };
            left_df.vstack_mut(&left_ext_df)?;
            left_keys.vstack_mut(&left_ext_keys)?;
            left.push_front((left_df, left_keys));
            right.push_front((right_df, right_keys));
            continue;
        } else if left_overlap_end == 0 {
            let Some((right_ext_df, right_ext_keys)) = right.pop_front() else {
                left.push_front((left_df, left_keys));
                right.push_front((right_df, right_keys));
                return Ok(None);
            };
            right_df.vstack_mut(&right_ext_df)?;
            right_keys.vstack_mut(&right_ext_keys)?;
            left.push_front((left_df, left_keys));
            right.push_front((right_df, right_keys));
            continue;
        }

        let (left_df_mergeable, left_df_remain) = left_df.split_at(left_overlap_end as i64);
        let (left_keys_mergeable, left_keys_remain) = left_keys.split_at(left_overlap_end as i64);
        let (right_df_mergeable, right_df_remain) = right_df.split_at(right_overlap_end as i64);
        let (right_keys_mergeable, right_keys_remain) =
            right_keys.split_at(right_overlap_end as i64);

        if left_keys_remain.height() > 0 {
            left.push_front((left_df_remain, left_keys_remain));
        }
        if right_keys_remain.height() > 0 {
            right.push_front((right_df_remain, right_keys_remain));
        }

        return Ok(Some((
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
