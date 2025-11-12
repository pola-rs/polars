use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::GroupsType;
use polars_core::schema::Schema;
use polars_core::series::ChunkCompareEq;
use polars_error::{PolarsError, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_ops::series::rle_lengths;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};

pub struct SortedGroupBy {
    buf_df: DataFrame,

    seq: MorselSeq,

    key: PlSmallStr,
    aggs: Arc<[(PlSmallStr, StreamExpr)]>,
}
impl SortedGroupBy {
    pub fn new(
        key: PlSmallStr,
        aggs: Arc<[(PlSmallStr, StreamExpr)]>,
        input_schema: Arc<Schema>,
    ) -> Self {
        let buf_df = DataFrame::empty_with_arc_schema(input_schema.clone());
        Self {
            buf_df,
            seq: MorselSeq::default(),
            key,
            aggs,
        }
    }

    async fn evaluate_one(
        key: &str,
        aggs: &[(PlSmallStr, StreamExpr)],
        state: &ExecutionState,
        idxs: &mut Vec<IdxSize>,
        df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let column = df.column(key).unwrap();
        rle_lengths(column, idxs).unwrap();

        let mut offset = 0;
        let groups = GroupsType::Slice {
            groups: idxs
                .iter()
                .map(|i| {
                    let start = offset;
                    offset += i;
                    [start, *i]
                })
                .collect(),
            overlapping: false,
        }
        .into_sliceable();

        let mut offset = 0;
        idxs.iter_mut().for_each(|idx| {
            let v = *idx;
            *idx = offset;
            offset += v;
        });

        let mut columns = Vec::with_capacity(1 + aggs.len());
        columns.push(unsafe { column.take_slice_unchecked(idxs) });

        for (name, agg) in aggs.iter() {
            let mut agg = agg.evaluate_on_groups(&df, &groups, state).await?;
            let agg = agg.finalize();
            columns.push(agg.with_name(name.clone()));
        }

        Ok(unsafe { DataFrame::new_no_checks(idxs.len(), columns) })
    }
}

impl ComputeNode for SortedGroupBy {
    fn name(&self) -> &str {
        "sorted-group-by"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
        } else if recv[0] == PortState::Done {
            if self.buf_df.is_empty() {
                send[0] = PortState::Done;
            } else {
                send[0] = PortState::Ready;
            }
        } else {
            recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        let Some(recv) = recv_ports[0].take() else {
            // We no longer have to receive data. Finalize and send all remaining data.
            assert!(!self.buf_df.is_empty());
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let df = Self::evaluate_one(
                    &self.key,
                    &self.aggs,
                    &state.in_memory_exec_state,
                    &mut Vec::new(),
                    std::mem::take(&mut self.buf_df),
                )
                .await?;

                _ = send
                    .send(Morsel::new(df, self.seq.successor(), SourceToken::new()))
                    .await;

                Ok(())
            }));
            return;
        };

        let mut recv = recv.serial();
        let send = send_ports[0].take().unwrap().parallel();

        let (mut distributor, rxs) =
            distributor_channel::<Morsel>(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        // Worker tasks.
        //
        // These evaluate the aggregations.
        join_handles.extend(rxs.into_iter().zip(send).map(|(mut rx, mut tx)| {
            let wg = WaitGroup::default();
            let key = self.key.clone();
            let aggs = self.aggs.clone();
            let state = state.in_memory_exec_state.split();
            let mut idxs = Vec::<IdxSize>::new();
            scope.spawn_task(TaskPriority::High, async move {
                while let Ok(mut morsel) = rx.recv().await {
                    morsel = morsel
                        .async_try_map::<PolarsError, _, _>(async |df| {
                            Self::evaluate_one(&key, &aggs, &state, &mut idxs, df).await
                        })
                        .await?;
                    morsel.set_consume_token(wg.token());

                    if tx.send(morsel).await.is_err() {
                        break;
                    }
                    wg.wait().await;
                }

                Ok(())
            })
        }));

        // Distributor task.
        //
        // This finds boundaries to distribute to worker threads over.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let (df, seq, source_token, wait_token) = morsel.into_inner();
                self.seq = seq;
                drop(wait_token);

                if df.height() == 0 {
                    continue;
                }

                self.buf_df.vstack_mut_owned(df).unwrap();

                let buf_key_column = self.buf_df.column(&self.key).unwrap();
                let fst = buf_key_column.get(0).unwrap();
                let lst = buf_key_column.get(buf_key_column.len() - 1).unwrap();

                if fst == lst {
                    continue;
                }

                let mut last_group_size = buf_key_column
                    .tail(Some(1))
                    .equal_missing(buf_key_column)
                    .unwrap();
                last_group_size.rechunk_mut();
                let last_group_size = last_group_size.downcast_as_array().values().trailing_ones();

                let offset = self.buf_df.height() - last_group_size;

                let df;
                (df, self.buf_df) = self.buf_df.split_at(offset as i64);

                if distributor
                    .send(Morsel::new(df, seq, source_token))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            Ok(())
        }));
    }
}
