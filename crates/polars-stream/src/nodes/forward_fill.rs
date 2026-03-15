use polars_core::prelude::{AnyValue, Column, DataType, FillNullStrategy, Scalar};
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;

pub struct ForwardFillNode {
    dtype: DataType,

    /// Last valid value seen. Equals `AnyValue::Null` i.f.f. no valid value has yet been seen.
    last: AnyValue<'static>,

    /// Maximum number of nulls to fill in until seeing a valid value.
    limit: IdxSize,
    /// Amount of nulls that have been filled in since seeing a valid value.
    consecutive_nulls: IdxSize,
}

impl ForwardFillNode {
    pub fn new(limit: Option<IdxSize>, dtype: DataType) -> Self {
        Self {
            limit: limit.unwrap_or(IdxSize::MAX),
            dtype,
            last: AnyValue::Null,
            consecutive_nulls: 0,
        }
    }
}

impl ComputeNode for ForwardFillNode {
    fn name(&self) -> &str {
        "forward_fill"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        let mut receiver = recv_ports[0].take().unwrap().serial();
        let senders = send_ports[0].take().unwrap().parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let limit = self.limit;
        let last = &mut self.last;
        let consecutive_nulls = &mut self.consecutive_nulls;

        // Serial receiver thread: determines the last non-null value and consecutive null
        // count for each morsel, then distributes (morsel, last, consecutive_nulls) to workers.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                if morsel.df().height() == 0 {
                    continue;
                }

                let column = &morsel.df()[0];
                let height = column.len();
                let null_count = column.null_count();

                let morsel_last = last.clone();
                let morsel_consecutive_nulls = *consecutive_nulls;

                if null_count == height {
                    // All null.
                    *consecutive_nulls += height as IdxSize;
                } else if let Some(idx) = column.last_non_null() {
                    // Some nulls.
                    *last = column.get(idx).unwrap().into_static();
                    *consecutive_nulls = (height - 1 - idx) as IdxSize;
                } else {
                    // All valid.
                    *last = column.get(height - 1).unwrap().into_static();
                    *consecutive_nulls = 0;
                }
                *consecutive_nulls = IdxSize::min(*consecutive_nulls, limit);

                if distributor
                    .send((morsel, morsel_last, morsel_consecutive_nulls))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            Ok(())
        }));

        // Parallel worker threads: perform the actual fill / fast paths.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            let dtype = self.dtype.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();

                while let Ok((morsel, last, consecutive_nulls)) = recv.recv().await {
                    let mut morsel = morsel.try_map(|df| {
                        let column = &df[0];
                        let height = column.len();
                        let null_count = column.null_count();
                        let name = column.name().clone();

                        // Remaining fill limit for the start morsel.
                        let leading_limit = limit.saturating_sub(consecutive_nulls) as usize;

                        let out = if null_count == 0
                            || (null_count == height && (last.is_null() || leading_limit == 0))
                        {
                            // Fast path: output = input.
                            column.clone()
                        } else if null_count == height {
                            // Fast path: input is all nulls.
                            let mut out = Column::new_scalar(
                                name,
                                Scalar::new(dtype.clone(), last),
                                height.min(leading_limit),
                            );
                            if leading_limit < height {
                                out.append_owned(Column::full_null(
                                    PlSmallStr::EMPTY,
                                    height - leading_limit,
                                    &dtype,
                                ))?;
                            }
                            out
                        } else if last.is_null()
                            || leading_limit == 0
                            || unsafe { !column.get_unchecked(0).is_null() }
                        {
                            // Faster path: result is equal to performing a normal `forward_fill` on
                            // the column.
                            column.fill_null(FillNullStrategy::Forward(Some(limit as IdxSize)))?
                        } else {
                            // Output = concat[
                            //     repeat_n(last, min(leading, leading_limit)),
                            //     repeat_n(NULL, leading - min(leading, leading_limit)),
                            //     forward_fill(column[leading..]),
                            // ]

                            // @Performance. If you want to make this fully optimal (although it is
                            // likely overkill), you can implement a kernel of `forward_fill` with a
                            // `init` value. This would remove the need for these appends.
                            let leading = column.first_non_null().unwrap();
                            let fill_last_count = leading_limit.min(leading);
                            let mut out = Column::new_scalar(
                                name.clone(),
                                Scalar::new(dtype.clone(), last),
                                fill_last_count,
                            );
                            if fill_last_count < leading {
                                out.append_owned(Column::full_null(
                                    name,
                                    leading - fill_last_count,
                                    &dtype,
                                ))?;
                            }

                            let mut tail = column.slice(leading as i64, height - leading);
                            if tail.has_nulls() {
                                tail = tail
                                    .fill_null(FillNullStrategy::Forward(Some(limit as IdxSize)))?;
                            }
                            out.append_owned(tail)?;
                            out
                        };

                        PolarsResult::Ok(out.into_frame())
                    })?;
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
