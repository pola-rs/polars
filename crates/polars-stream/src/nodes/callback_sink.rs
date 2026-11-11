use std::num::NonZeroUsize;

use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_core::frame::DataFrame;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_ops::frame::{first_run_len, split_by_key};
use polars_plan::prelude::PlanCallback;
use polars_utils::pl_str::PlSmallStr;

use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

pub struct CallbackSinkNode {
    function: PlanCallback<DataFrame, bool>,
    maintain_order: bool,
    split_by: Option<PlSmallStr>,

    buffer: DataFrame,
    chunk_size: Option<NonZeroUsize>,
    is_done: bool,
}

impl CallbackSinkNode {
    pub fn new(
        function: PlanCallback<DataFrame, bool>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
        split_by: Option<PlSmallStr>,
    ) -> Self {
        Self {
            function,
            maintain_order,
            split_by,

            buffer: DataFrame::empty(),
            chunk_size,
            is_done: false,
        }
    }
}

impl ComputeNode for CallbackSinkNode {
    fn name(&self) -> &str {
        "sink_batches"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.is_empty());

        if self.is_done || recv[0] == PortState::Done {
            recv[0] = PortState::Done;

            // Flush the last buffer
            if self.buffer.height() > 0 && !self.is_done {
                let function = self.function.clone();
                let df = std::mem::take(&mut self.buffer);
                if let Some(chunk_size) = self.chunk_size {
                    assert!(df.height() <= chunk_size.get());
                }
                state.spawn_subphase_task(async move {
                    ASYNC
                        .spawn_blocking(move || function.call(df))
                        .await
                        .unwrap()?;
                    Ok(())
                });
                return Ok(());
            }
        } else {
            recv[0] = PortState::Ready;
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
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let mut recv = recv_ports[0]
            .take()
            .unwrap()
            .serial_with_maintain_order(self.maintain_order);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while !self.is_done
                && let Ok(m) = recv.recv().await
            {
                let (sf, _, _, consume_token) = m.into_inner();
                let morsel = sf.into_df().await;

                if let Some(split_by) = self.split_by.clone() {
                    if morsel.height() == 0 {
                        drop(consume_token);
                        continue;
                    }

                    let key = split_by.as_str();
                    let mut complete = Vec::new();

                    if self.buffer.height() == 0 {
                        let (mut groups, pending) = split_by_key(&morsel, key)?;
                        self.buffer = pending;
                        complete.append(&mut groups);
                    } else {
                        let continues = {
                            let last = self.buffer.column(key)?.get(self.buffer.height() - 1)?;
                            let first = morsel.column(key)?.get(0)?;
                            last == first
                        };

                        if !continues {
                            complete.push(std::mem::take(&mut self.buffer));
                            let (mut groups, pending) = split_by_key(&morsel, key)?;
                            self.buffer = pending;
                            complete.append(&mut groups);
                        } else {
                            let run = first_run_len(&morsel, key)?;
                            if run == morsel.height() {
                                self.buffer.vstack_mut_owned(morsel)?;
                            } else {
                                let head = morsel.slice(0, run);
                                let mut group = std::mem::take(&mut self.buffer);
                                group.vstack_mut_owned(head)?;
                                complete.push(group);

                                let rest = morsel.slice(run as i64, morsel.height() - run);
                                let (mut groups, pending) = split_by_key(&rest, key)?;
                                self.buffer = pending;
                                complete.append(&mut groups);
                            }
                        }
                    }

                    for df in complete {
                        let function = self.function.clone();
                        let should_stop = ASYNC
                            .spawn_blocking(move || function.call(df))
                            .await
                            .unwrap()?;

                        if should_stop {
                            self.is_done = true;
                            break;
                        }
                    }
                } else {
                    // @NOTE: This also performs schema validation.
                    self.buffer.vstack_mut_owned(morsel)?;

                    while self.buffer.height() > 0
                        && self
                            .chunk_size
                            .is_none_or(|chunk_size| self.buffer.height() >= chunk_size.get())
                    {
                        let chunk_size = self.chunk_size.map_or(usize::MAX, Into::into);

                        let df;
                        (df, self.buffer) = self
                            .buffer
                            .split_at(self.buffer.height().min(chunk_size) as i64);

                        let function = self.function.clone();
                        let should_stop = ASYNC
                            .spawn_blocking(move || function.call(df))
                            .await
                            .unwrap()?;

                        if should_stop {
                            self.is_done = true;
                            break;
                        }
                    }
                }
                drop(consume_token);
                // Increase the backpressure. Only free up a pipeline when the morsel has been
                // processed in its entirety.
            }

            Ok(())
        }));
    }
}
