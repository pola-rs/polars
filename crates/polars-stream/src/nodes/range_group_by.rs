use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::GroupsType;
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

pub struct RangeGroupBy {
    buffer: Option<(DataFrame, MorselSeq, SourceToken)>,
    key: PlSmallStr,
    aggs: Arc<[(PlSmallStr, StreamExpr)]>,
}
impl RangeGroupBy {
    pub fn new(key: PlSmallStr, aggs: Arc<[(PlSmallStr, StreamExpr)]>) -> Self {
        Self {
            buffer: None,
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
        let column = df.column(&key).unwrap();
        rle_lengths(&column, idxs).unwrap();

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
        columns.push(unsafe { column.take_slice_unchecked(&idxs) });

        for (name, agg) in aggs.iter() {
            let mut agg = agg.evaluate_on_groups(&df, &groups, &state).await?;
            let agg = agg.finalize();
            columns.push(agg.with_name(name.clone()));
        }

        Ok(unsafe { DataFrame::new_no_checks(idxs.len(), columns) })
    }
}

impl ComputeNode for RangeGroupBy {
    fn name(&self) -> &str {
        "range-group-by"
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
            self.buffer.take();
        } else if recv[0] == PortState::Done {
            if self.buffer.is_some() {
                send[0] = PortState::Ready;
            } else {
                send[0] = PortState::Done;
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
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let (df, seq, src) = self.buffer.take().unwrap();
                let df = Self::evaluate_one(
                    &self.key,
                    &self.aggs,
                    &state.in_memory_exec_state,
                    &mut Vec::new(),
                    df,
                )
                .await?;

                _ = send.send(Morsel::new(df, seq, src)).await;
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
            while self.buffer.is_none()
                && let Ok(m) = recv.recv().await
            {
                if m.df().height() == 0 {
                    continue;
                }
                let (df, seq, source, _) = m.into_inner();
                self.buffer = Some((df, seq, source));
            }
            
            if self.buffer.is_none() {
                return Ok(());
            }

            let (buf_df, buf_seq, buf_src) = self.buffer.as_mut().unwrap();

            while let Ok(mut morsel) = recv.recv().await {
                if morsel.df().height() == 0 {
                    continue;
                }

                drop(morsel.take_consume_token());

                let buf_key_column = buf_df.column(&self.key).unwrap();
                let morsel_key_column = morsel.df().column(&self.key).unwrap();
                let buf_val = buf_key_column.get(buf_key_column.len() - 1).unwrap();
                let morsel_key_val = morsel_key_column.get(morsel_key_column.len() - 1).unwrap();

                if buf_val == morsel_key_val {
                    buf_df.vstack_mut_owned(morsel.into_df()).unwrap();
                    continue;
                }

                let mut num_eq_to_buf_last = buf_key_column
                    .tail(Some(1))
                    .equal_missing(morsel_key_column)
                    .unwrap();
                num_eq_to_buf_last.rechunk_mut();
                let num_eq_to_buf_last = num_eq_to_buf_last
                    .downcast_as_array()
                    .values()
                    .leading_ones();

                morsel = morsel.map(|df| {
                    let (head, tail) = df.split_at(num_eq_to_buf_last as i64);
                    buf_df.vstack_mut_owned(head).unwrap();
                    tail
                });

                let (m_df, m_seq, m_src, _) = morsel.into_inner();
                let df = std::mem::replace(buf_df, m_df);
                let seq = std::mem::replace(buf_seq, m_seq);
                let src = std::mem::replace(buf_src, m_src);

                if distributor.send(Morsel::new(df, seq, src)).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
