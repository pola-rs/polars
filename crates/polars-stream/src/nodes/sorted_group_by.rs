use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::GroupsType;
use polars_core::schema::Schema;
use polars_core::series::IsSorted;
use polars_error::{PolarsError, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_ops::series::{SearchSortedSide, rle_lengths, search_sorted};
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

    slice: Option<(IdxSize, IdxSize)>,
}
impl SortedGroupBy {
    pub fn new(
        key: PlSmallStr,
        aggs: Arc<[(PlSmallStr, StreamExpr)]>,
        slice: Option<(IdxSize, IdxSize)>,
        input_schema: Arc<Schema>,
    ) -> Self {
        let buf_df = DataFrame::empty_with_arc_schema(input_schema.clone());
        Self {
            buf_df,
            seq: MorselSeq::default(),
            key,
            aggs,
            slice,
        }
    }

    async fn evaluate_one(
        key: &str,
        aggs: &[(PlSmallStr, StreamExpr)],
        state: &ExecutionState,
        idxs: &mut Vec<IdxSize>,
        df: DataFrame,
        windows_slice: (IdxSize, IdxSize),
    ) -> PolarsResult<DataFrame> {
        let column = df.column(key).unwrap();
        rle_lengths(column, idxs).unwrap();

        let windows_offset = windows_slice.0 as usize;
        let windows_length = (windows_slice.1 as usize).min(idxs.len() - windows_offset);

        let df_offset = idxs[..windows_offset].iter().sum::<IdxSize>();
        let df_height = idxs[windows_offset..][..windows_length]
            .iter()
            .sum::<IdxSize>();

        let df = df.slice(df_offset as i64, df_height as usize);
        idxs.truncate(windows_offset + windows_length);
        idxs.drain(..windows_offset);

        let mut offset = 0;
        let groups = idxs
            .iter()
            .map(|i| {
                let start = offset;
                offset += i;
                [start, *i]
            })
            .collect();
        let groups = GroupsType::new_slice(groups, false, true).into_sliceable();

        let mut offset = 0;
        idxs.iter_mut().for_each(|idx| {
            let v = *idx;
            *idx = offset;
            offset += v;
        });

        let mut columns = Vec::with_capacity(1 + aggs.len());
        let column = df.column(key).unwrap();
        columns.push(unsafe { column.take_slice_unchecked(idxs) });

        for (name, agg) in aggs.iter() {
            let mut agg = agg.evaluate_on_groups(&df, &groups, state).await?;
            let agg = agg.finalize();
            columns.push(agg.with_name(name.clone()));
        }

        Ok(unsafe { DataFrame::new_unchecked(idxs.len(), columns) })
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

        if self.slice.is_some_and(|(_, l)| l == 0) {
            recv[0] = PortState::Done;
            send[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
        }

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
        } else if recv[0] == PortState::Done {
            if self.buf_df.height() == 0 {
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
            assert!(self.buf_df.height() > 0);
            assert!(self.slice.is_none_or(|(_, l)| l > 0));
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let df = Self::evaluate_one(
                    &self.key,
                    &self.aggs,
                    &state.in_memory_exec_state,
                    &mut Vec::new(),
                    std::mem::take(&mut self.buf_df),
                    self.slice.unwrap_or((0, IdxSize::MAX)),
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

        let (mut distributor, rxs) = distributor_channel::<(Morsel, (IdxSize, IdxSize))>(
            send.len(),
            *DEFAULT_DISTRIBUTOR_BUFFER_SIZE,
        );

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
                while let Ok((mut morsel, windows_slice)) = rx.recv().await {
                    morsel = morsel
                        .async_try_map::<PolarsError, _, _>(async |df| {
                            Self::evaluate_one(&key, &aggs, &state, &mut idxs, df, windows_slice)
                                .await
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
            while let Ok(morsel) = recv.recv().await
                && self.slice.is_none_or(|(_, l)| l > 0)
            {
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

                let mut buf_key_column = buf_key_column.as_materialized_series().clone();
                buf_key_column.set_sorted_flag(IsSorted::Ascending);

                let descending = fst > lst;
                let num_flushable = search_sorted(
                    &buf_key_column,
                    &buf_key_column.tail(Some(1)),
                    SearchSortedSide::Left,
                    descending,
                )
                .unwrap();
                let num_flushable = unsafe { num_flushable.get_unchecked(0) }.unwrap();

                let df;
                (df, self.buf_df) = self.buf_df.split_at(num_flushable as i64);

                let mut windows_offset = 0;
                let mut windows_length = IdxSize::MAX;

                if let Some((offset, length)) = self.slice.as_mut() {
                    let buf_key_column = buf_key_column.head(Some(num_flushable as usize));

                    // Since `buf_key_column` is flagged as sorted, this is simply a linear scan.
                    let num_uniq_values = buf_key_column.n_unique()? as IdxSize;

                    // Fast path: Slice allows skipping the entire morsel.
                    if *offset >= num_uniq_values {
                        *offset -= num_uniq_values;
                        continue;
                    }

                    windows_offset = *offset;
                    windows_length = *length;

                    let num_skipped_values = (*offset).min(num_uniq_values);
                    *offset -= num_skipped_values;
                    *length = (*length).saturating_sub(num_uniq_values - num_skipped_values);
                }

                if distributor
                    .send((
                        Morsel::new(df, seq, source_token),
                        (windows_offset, windows_length),
                    ))
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
