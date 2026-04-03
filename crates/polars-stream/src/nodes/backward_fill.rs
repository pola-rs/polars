use polars_core::prelude::{Column, DataType, FillNullStrategy};
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{MorselSeq, SourceToken, get_ideal_morsel_size};

pub struct BackwardFillNode {
    dtype: DataType,

    /// Maximum number of consecutive nulls to fill.
    limit: IdxSize,

    /// Sequence counter for output morsels emitted by the serial thread.
    seq: MorselSeq,

    /// Count of trailing nulls from previous morsels not yet emitted. These are waiting for a
    /// future non-null value to potentially fill them or to exceed the limit.
    pending_nulls: IdxSize,

    /// Column name.
    col_name: PlSmallStr,
}

impl BackwardFillNode {
    pub fn new(limit: Option<IdxSize>, dtype: DataType, col_name: PlSmallStr) -> Self {
        Self {
            limit: limit.unwrap_or(IdxSize::MAX),
            dtype,
            seq: MorselSeq::default(),
            pending_nulls: 0,
            col_name,
        }
    }
}

impl ComputeNode for BackwardFillNode {
    fn name(&self) -> &str {
        "backward_fill"
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
            self.pending_nulls = 0;
        } else if recv[0] == PortState::Done {
            // We may still have pending nulls to flush as actual nulls.
            if self.pending_nulls > 0 {
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let recv = recv_ports[0].take();
        let send = send_ports[0].take().unwrap();

        let limit = self.limit;
        let dtype = self.dtype.clone();
        let pending_nulls = &mut self.pending_nulls;
        let seq = &mut self.seq;
        let col_name = self.col_name.clone();

        let Some(recv) = recv else {
            // Input exhausted. Flush remaining pending_nulls as actual nulls.
            if *pending_nulls == 0 {
                return;
            }

            let pending = *pending_nulls;
            let mut send = send.serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let source_token = SourceToken::new();
                let morsel_size = get_ideal_morsel_size();
                let mut remaining = pending as usize;
                while remaining > 0 {
                    let chunk_size = morsel_size.min(remaining);
                    let df = Column::full_null(col_name.clone(), chunk_size, &dtype).into_frame();
                    if send
                        .send(Morsel::new(df, *seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    *seq = seq.successor();
                    remaining -= chunk_size;
                }
                Ok(())
            }));

            *pending_nulls = 0;
            return;
        };

        let mut receiver = recv.serial();
        let senders = send.parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        // Serial thread: handles serial state and sends morsel without backward_fill to parallel
        // workers.
        let serial_dtype = dtype.clone();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let dtype = serial_dtype;
            let source_token = SourceToken::new();
            let ideal_morsel_size = get_ideal_morsel_size() as IdxSize;

            while let Ok(morsel) = receiver.recv().await {
                let column = &morsel.df()[0];
                let height = column.len();
                if height == 0 {
                    continue;
                }

                let null_count = column.null_count();
                if null_count == height {
                    *pending_nulls += height as IdxSize;
                }

                // Flush pending nulls that exceed the limit as already-final null morsels.
                // This also covers the all-null case above.
                while *pending_nulls > limit {
                    let chunk_size = ideal_morsel_size.min(*pending_nulls - limit);
                    let col = Column::full_null(col_name.clone(), chunk_size as usize, &dtype);
                    let null_morsel = Morsel::new(col.into_frame(), *seq, source_token.clone());

                    *seq = seq.successor();
                    *pending_nulls -= chunk_size;
                    if distributor.send(null_morsel).await.is_err() {
                        return Ok(());
                    }
                }

                if null_count == height {
                    // Fast path: all nulls.
                    continue;
                }

                let new_pending_nulls = if null_count == 0 {
                    0
                } else {
                    // Note: unwrap is fine as `null_count != height`.
                    let trailing_nulls = height - column.last_non_null().unwrap() - 1;
                    (trailing_nulls as IdxSize).min(limit)
                };

                let mut column = if new_pending_nulls > 0 {
                    // Remove new pending nulls.
                    column.slice(0, column.len() - new_pending_nulls as usize)
                } else {
                    column.clone()
                };
                if *pending_nulls > 0 {
                    // Prepend the old pending nulls.
                    let mut c =
                        Column::full_null(col_name.clone(), *pending_nulls as usize, &dtype);
                    c.append_owned(column)?;
                    column = c;
                }

                let morsel = Morsel::new(column.into_frame(), *seq, source_token.clone());

                *seq = seq.successor();
                *pending_nulls = new_pending_nulls;
                if distributor.send(morsel).await.is_err() {
                    return Ok(());
                }
            }

            Ok(())
        }));

        // Parallel worker threads: Apply fill null and emit.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                while let Ok(mut morsel) = recv.recv().await {
                    let col = &morsel.df()[0];
                    if col.has_nulls() {
                        *morsel.df_mut() = col
                            .fill_null(FillNullStrategy::Backward(Some(limit)))?
                            .into_frame();
                    }
                    morsel.set_consume_token(wait_group.token());
                    if send.send(morsel).await.is_err() {
                        break;
                    }
                    wait_group.wait().await;
                }

                Ok(())
            }));
        }
    }
}
