use polars_core::prelude::{AnyValue, Column, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_error::PolarsResult;
use polars_ops::series::{InterpolationMethod, interpolate};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{MorselSeq, SourceToken, get_ideal_morsel_size};

pub struct InterpolateNode {
    method: InterpolationMethod,
    /// dtype of the input column — used to build `full_null` prefix columns in the serial thread
    /// before handing off to `interpolate()`, which may cast (e.g. Int64 -> Float64).
    input_dtype: DataType,
    /// dtype of the output column — used to emit boundary nulls in the flush phase.
    output_dtype: DataType,
    col_name: PlSmallStr,

    /// Sequence counter for output morsels emitted by the serial thread.
    seq: MorselSeq,

    /// The last non-null value seen across morsels. `AnyValue::Null` if none seen yet.
    last_non_null: AnyValue<'static>,

    /// Number of trailing nulls seen since `last_non_null`, not yet emitted.
    /// These are waiting for a future non-null value (the right endpoint) before
    /// they can be interpolated.
    pending_nulls: IdxSize,
}

impl InterpolateNode {
    pub fn new(
        method: InterpolationMethod,
        input_dtype: DataType,
        output_dtype: DataType,
        col_name: PlSmallStr,
    ) -> Self {
        Self {
            method,
            input_dtype,
            output_dtype,
            col_name,
            seq: MorselSeq::default(),
            last_non_null: AnyValue::Null,
            pending_nulls: 0,
        }
    }
}

impl ComputeNode for InterpolateNode {
    fn name(&self) -> &str {
        "interpolate"
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
            // We may still have pending trailing nulls to flush as boundary nulls.
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

        let method = self.method;
        let input_dtype = self.input_dtype.clone();
        let output_dtype = self.output_dtype.clone();
        let col_name = self.col_name.clone();
        let pending_nulls = &mut self.pending_nulls;
        let last_non_null = &mut self.last_non_null;
        let seq = &mut self.seq;

        let Some(recv) = recv else {
            // Input exhausted. Flush pending trailing nulls as actual nulls.
            debug_assert!(*pending_nulls > 0);

            let source_token = SourceToken::new();
            let mut send = send.serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let morsel_size = get_ideal_morsel_size();
                while *pending_nulls > 0 && !source_token.stop_requested() {
                    let chunk_size = morsel_size.min(*pending_nulls as usize);
                    let df =
                        Column::full_null(col_name.clone(), chunk_size, &output_dtype).into_frame();
                    if send
                        .send(Morsel::new(df, *seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    *seq = seq.successor();
                    *pending_nulls -= chunk_size as IdxSize;
                }
                Ok(())
            }));
            return;
        };

        let mut receiver = recv.serial();
        let senders = send.parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        // Serial receive and state handling thread.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                let (df, _, source_token, _) = morsel.into_inner();
                let mut columns = df.into_columns();
                assert_eq!(columns.len(), 1);
                let column = columns.pop().unwrap();
                let height = column.len();

                let Some(last_non_null_idx) = column.last_non_null() else {
                    // All nulls: accumulate into pending and wait for a right endpoint.
                    *pending_nulls += height as IdxSize;
                    continue;
                };

                // Everything until the last run of nulls is ready to be sent.
                let ready_values = column.slice(0, last_non_null_idx + 1);

                if distributor
                    .send((
                        *seq,
                        source_token,
                        *pending_nulls,
                        last_non_null.clone(),
                        ready_values,
                    ))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                *seq = seq.successor();

                *last_non_null = column.get(last_non_null_idx).unwrap().into_static();
                *pending_nulls = (height - 1 - last_non_null_idx) as IdxSize;
            }

            Ok(())
        }));

        // Parallel worker threads.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            let input_dtype = input_dtype.clone();
            let output_dtype = output_dtype.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                while let Ok((seq, source_token, pending_nulls, last_non_null, mut column)) =
                    recv.recv().await
                {
                    // Add back in the last non-null value and the pending nulls.
                    //
                    // If we have only seen nulls until now, last_non_null=Null and its just an
                    // extra null at the start.
                    let has_prepended = pending_nulls > 0 || !last_non_null.is_null();
                    if has_prepended {
                        let mut c = Column::new_scalar(
                            column.name().clone(),
                            Scalar::new(input_dtype.clone(), last_non_null),
                            1,
                        );
                        c.append_owned(Column::full_null(
                            column.name().clone(),
                            pending_nulls as usize,
                            &input_dtype,
                        ))?;
                        c.append_owned(column)?;
                        column = c;
                    }

                    // Interpolate if necessary.
                    column = if column.has_nulls() {
                        interpolate(column.as_materialized_series(), method).into_column()
                    } else {
                        column.cast(&output_dtype)?
                    };

                    // If there were pending nulls, we know that the previous morsel already
                    // emitted our first value. We are only using it to enable interpolation here.
                    // Slice it off and don't double emit it.
                    if has_prepended {
                        column = column.slice(1, usize::MAX);
                    }

                    let mut morsel = Morsel::new(column.into_frame(), seq, source_token.clone());
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
